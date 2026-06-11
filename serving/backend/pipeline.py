"""
MedicalPipeline — orchestrates the full analysis for one CT scan.

Execution order (tất cả trong 1 lần gọi Modal duy nhất via infer_batch):
  1. radiology screening
  2. [if nodule] radiology detail   ← run_if: nodule_presence == "Yes"
  3. cardiology
  4. oncology
  5. finding_impression

Lý do dùng infer_batch thay vì 5 lần gọi riêng:
  vLLM KV prefix cache của 85 ảnh CT (≈2GB) sẽ bị evict nếu request của
  scan khác chen vào giữa các bước — infer_batch giữ toàn bộ pipeline trong
  1 lần gọi, không có gián đoạn.

Backend build sẵn tất cả prompts (formatter ở đây, không ở Modal),
Modal chỉ nhận raw prompt + run `llm.generate()` theo đúng thứ tự.

CT images đi qua S3 làm trung gian:
  backend convert .dcm → PNG → upload S3 (serving.backend.storage.s3) → tạo
  presigned URL → gửi URL cho Modal; Modal tự tải ảnh từ URL đó.
"""

import asyncio
import io
import logging
import os
import tempfile
import time
import zipfile

import httpx
from PIL import Image

logger = logging.getLogger(__name__)

from agents.data_prep.clinical_text import clinical_to_text
from agents.data_prep.ct_processor import dicom_dir_to_slices
from agents.cardiology.formatter import CardiologyInferenceFormatter
from agents.cardiology.formatter import parse_inference_output as cardio_parse
from agents.finding_impression.formatter import FindingImpressionInferenceFormatter
from agents.finding_impression.formatter import parse_inference_output as fi_parse
from agents.oncology.formatter import OncologyInferenceFormatter
from agents.oncology.formatter import parse_inference_output as onco_parse
from agents.radiology.formatter import RadiologyInferenceFormatter
from agents.radiology.formatter import parse_inference_output as radio_parse
from agents.verification.formatter import VerificationInferenceFormatter
from agents.verification.formatter import parse_inference_output as verif_parse

from .highlighter_client import highlight_many
from .modal_client import call_modal_pipeline
from .storage.s3 import S3Storage

MAX_SLICES = 85

# Module-level formatter instances (stateless, reusable)
_screening_fmt = RadiologyInferenceFormatter(step="screening")
_detail_fmt    = RadiologyInferenceFormatter(step="detail")
_cardio_fmt    = CardiologyInferenceFormatter()
_onco_fmt      = OncologyInferenceFormatter()
_fi_fmt        = FindingImpressionInferenceFormatter()
_verif_fmt     = VerificationInferenceFormatter()


def _slices_from_dicom_zip(zip_bytes: bytes) -> list[Image.Image]:
    """Giải nén .zip chứa 1 CT series (.dcm) → convert sang HU volume → PIL RGB slices."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_dir)
        return dicom_dir_to_slices(tmp_dir, MAX_SLICES)


class MedicalPipeline:

    def __init__(self):
        self.modal_pipeline_url = os.environ["MODAL_PIPELINE_URL"]
        self.storage = S3Storage()
        self._uploaded: dict[str, int] = {}  # cache_key → số slice đã có trên S3

    async def _ensure_images_on_s3(self, cache_key: str, dicom_zip_bytes: bytes) -> list[str]:
        """Convert .dcm → PNG → upload S3 (nếu chưa có), trả về danh sách presigned URL."""
        if cache_key in self._uploaded:
            n_slices = self._uploaded[cache_key]
            logger.info("[S3] HIT — %d slices đã có sẵn trên S3, bỏ qua convert + upload", n_slices)
        else:
            logger.info("[CT] Unzipping → DICOM → HU volume → slicing...")
            t0 = time.perf_counter()
            images = _slices_from_dicom_zip(dicom_zip_bytes)
            logger.info("[CT] Sliced: %d slices, size=%s, elapsed=%.1fs",
                        len(images), images[0].size, time.perf_counter() - t0)

            t0 = time.perf_counter()
            n_slices = await asyncio.to_thread(self.storage.upload_images, cache_key, images)
            self._uploaded[cache_key] = n_slices
            logger.info("[S3] Uploaded %d slices → s3://%s/ct-slices/%s, elapsed=%.1fs",
                        n_slices, self.storage.bucket, cache_key, time.perf_counter() - t0)

        return self.storage.presigned_urls(cache_key, n_slices)

    async def get_analysis(self, pid: str, series_uid: str) -> dict | None:
        """Load lại kết quả phân tích đã lưu — dùng cho chatbot /ask."""
        cache_key = f"{pid}/{series_uid}"
        return await asyncio.to_thread(self.storage.load_analysis, cache_key)

    async def list_analyses(self) -> list[dict]:
        """Liệt kê tất cả bệnh nhân đã có kết quả phân tích lưu trên S3."""
        return await asyncio.to_thread(self.storage.list_analyses)

    async def run(
        self,
        pid: str,
        series_uid: str,
        clinical_data: dict,
        dicom_zip_bytes: bytes,
    ) -> dict:
        t_total = time.perf_counter()
        cache_key = f"{pid}/{series_uid}"
        logger.info("=== Pipeline start  pid=%s  series=%s  zip_size=%.1fMB ===",
                    pid, series_uid, len(dicom_zip_bytes) / 1e6)

        # Đã phân tích trước đó (lưu trên S3) → trả về luôn, bỏ qua Modal pipeline
        cached_result = await asyncio.to_thread(self.storage.load_analysis, cache_key)
        if cached_result is not None:
            logger.info("=== Pipeline SKIP — kết quả đã có sẵn trên S3 (cache_key=%s) ===", cache_key)
            return cached_result

        image_urls = await self._ensure_images_on_s3(cache_key, dicom_zip_bytes)

        logger.info("[Prompt] Clinical: %s", clinical_to_text(clinical_data)[:120])

        # Build tất cả prompts trước (formatter ở backend, không ở Modal)
        tasks = [
            {
                "id":             "screening",
                "adapter":        "radiology",
                "prompt":         _screening_fmt.format_prompt(clinical_data),
                "max_new_tokens": 512,
            },
            {
                "id":             "detail",
                "adapter":        "radiology",
                "prompt":         _detail_fmt.format_prompt(clinical_data),
                "max_new_tokens": 512,
                # chỉ chạy nếu screening phát hiện nodule
                "run_if": {"task_id": "screening", "field": "nodule_presence", "value": "Yes"},
            },
            {
                "id":             "cardiology",
                "adapter":        "cardiology",
                "prompt":         _cardio_fmt.format_prompt(clinical_data),
                "max_new_tokens": 512,
            },
            {
                "id":             "oncology",
                "adapter":        "oncology",
                "prompt":         _onco_fmt.format_prompt(clinical_data),
                "max_new_tokens": 512,
            },
            {
                "id":                 "finding_impression",
                "adapter":            "finding_impression",
                "prompt":             _fi_fmt.format_prompt(),
                "max_new_tokens":     1024,
                "repetition_penalty": 1.2,
            },
        ]

        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Call 1 — 4 specialist agents
            raw = await call_modal_pipeline(
                client, self.modal_pipeline_url, tasks, cache_key, image_urls
            )

            r = raw["results"]
            radio_screen = radio_parse(r["screening"], "screening")
            radio_detail = radio_parse(r["detail"], "detail") if r.get("detail") else None
            cardio       = cardio_parse(r["cardiology"])
            onco         = onco_parse(r["oncology"])
            fi           = fi_parse(r["finding_impression"])
            logger.info("[RAW] finding_impression:\n%s", r.get("finding_impression", ""))

            # Call 2 — Agent 5 Part 1: MedGemma base (no LoRA) xác minh lại qua ảnh CT
            # Images đã có trong RAM cache của Modal từ call 1 → không cần gửi image_urls lại
            verif_prompt = _verif_fmt.format_prompt(
                radio_screen, radio_detail, cardio, onco, fi, clinical_data
            )
            logger.info("[Agent5/Part1] Building verification task — prompt_len=%d chars",
                        len(verif_prompt))
            raw_verif = await call_modal_pipeline(
                client,
                self.modal_pipeline_url,
                [{"id": "verification", "adapter": "base",
                  "prompt": verif_prompt, "max_new_tokens": 1024,
                  "repetition_penalty": 1.3}],
                cache_key,
                image_urls,  # Call 2 may land on a different container — always pass S3 URLs
            )
            verif_raw = raw_verif["results"].get("verification") or ""
            logger.info("[RAW] verification:\n%s", verif_raw)
            verif = verif_parse(verif_raw)

        # Highlight các cụm từ quan trọng trong free-text (OpenAI, text-only,
        # không liên quan ảnh CT) — fail-safe, lỗi thì giữ nguyên text gốc
        highlighted = await highlight_many({
            "findings":    fi["findings"],
            "impression":  fi["impression"],
            "verification": verif["analysis"],
        })

        result = {
            "radiology": {
                "screening": {"answer": radio_screen["answer"]},
                "detail":    {"answer": radio_detail["answer"]} if radio_detail else None,
            },
            "cardiology": {
                "answer": cardio["answer"],
            },
            "oncology": {
                "answer": onco["answer"],
            },
            "finding_impression": {
                "findings":   highlighted["findings"],
                "impression": highlighted["impression"],
            },
            "verification": {
                "analysis": highlighted["verification"],
            },
        }

        # Lưu kết quả lên S3 — chatbot /ask sau này load lại làm context
        await asyncio.to_thread(self.storage.save_analysis, cache_key, result)

        logger.info("=== Pipeline done  total=%.1fs ===", time.perf_counter() - t_total)

        return result
