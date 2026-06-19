import gc
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from botocore.exceptions import ClientError
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

try:
    from serving.backend.highlighter_client import call_llm_json
except Exception:
    try:
        from backend.highlighter_client import call_llm_json
    except Exception:
        try:
            from ..highlighter_client import call_llm_json
        except Exception:
            from highlighter_client import call_llm_json

router = APIRouter(prefix="/voxtell", tags=["voxtell"])
logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=1800.0, write=300.0, pool=30.0)
_s3_client = None


def _voxtell_mode() -> str:
    mode = os.getenv("VOXTELL_BACKEND_MODE", "proxy").strip().lower()
    return "s3_direct" if mode in {"s3_direct", "direct", "modal_s3_direct"} else "proxy"


def _is_s3_direct() -> bool:
    return _voxtell_mode() == "s3_direct"


# ---------------------------------------------------------------------------
# Proxy mode: local/main backend -> Modal GPU app
# ---------------------------------------------------------------------------

def _get_modal_base_url() -> str:
    base_url = os.getenv("VOXTELL_MODAL_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise HTTPException(status_code=500, detail="VOXTELL_MODAL_BASE_URL is not configured.")
    return base_url


def _modal_url(path: str) -> str:
    return f"{_get_modal_base_url()}{path}"


def _raise_modal_error(status_code: int, body: bytes) -> None:
    text = body.decode("utf-8", errors="replace")
    detail = text[:1000] if text else "VoxTell Modal request failed."
    raise HTTPException(status_code=status_code, detail=detail)


async def _proxy_volume(pid: str, series_uid: str):
    # VOXTELL_MODAL_BASE_URL must be the Modal app URL including /api, e.g.
    # https://...-voxtell-s3-direct-web.modal.run/api
    # External Modal path then becomes /api/api/v1/voxtell/volume/...
    upstream_path = f"/api/v1/voxtell/volume/{quote(pid, safe='')}/{quote(series_uid, safe='')}"

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        try:
            logger.info("VOXTELL_PROXY_VOLUME upstream=%s", _modal_url(upstream_path))
            upstream = await client.get(_modal_url(upstream_path))
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Cannot reach VoxTell Modal service: {exc}") from exc

    if upstream.status_code >= 400:
        _raise_modal_error(upstream.status_code, upstream.content)

    headers = {}
    content_disposition = upstream.headers.get("content-disposition")
    if content_disposition:
        headers["content-disposition"] = content_disposition

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "application/gzip",
        headers=headers,
    )


async def _proxy_predict(pid: str, series_uid: str, prompt: str):
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        try:
            logger.info("VOXTELL_PROXY_PREDICT upstream=%s", _modal_url("/api/v1/voxtell/predict"))
            upstream = await client.post(
                _modal_url("/api/v1/voxtell/predict"),
                data={"pid": pid, "series_uid": series_uid, "prompt": prompt},
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Cannot reach VoxTell Modal service: {exc}") from exc

    if upstream.status_code >= 400:
        _raise_modal_error(upstream.status_code, upstream.content)

    headers = {}
    content_disposition = upstream.headers.get("content-disposition")
    if content_disposition:
        headers["content-disposition"] = content_disposition

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "application/gzip",
        headers=headers,
    )


async def _proxy_predict_bytes(pid: str, series_uid: str, prompt: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
        try:
            logger.info("VOXTELL_PROXY_PREDICT_BYTES upstream=%s prompt=%r", _modal_url("/api/v1/voxtell/predict"), prompt)
            upstream = await client.post(
                _modal_url("/api/v1/voxtell/predict"),
                data={"pid": pid, "series_uid": series_uid, "prompt": prompt},
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Cannot reach VoxTell Modal service: {exc}") from exc

    if upstream.status_code >= 400:
        _raise_modal_error(upstream.status_code, upstream.content)

    filename = f"voxtell_{pid}_{'_'.join(prompt.split())[:80] or 'mask'}.nii.gz"
    content_disposition = upstream.headers.get("content-disposition") or ""
    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        filename = match.group(1)

    return upstream.content, filename


# ---------------------------------------------------------------------------
# S3-direct mode: Modal GPU app -> S3 -> dcm2niix -> VoxTell
# ---------------------------------------------------------------------------

def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        region = os.getenv("AWS_REGION", "").strip() or None
        _s3_client = boto3.client("s3", region_name=region)
    return _s3_client


def _require_bucket() -> str:
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="S3_BUCKET_NAME is not configured.")
    return bucket


def _dicom_s3_key_candidates(pid: str, series_uid: str) -> list[str]:
    base = f"dicom-raw/{pid}/{series_uid}"
    return [base] if base.lower().endswith(".zip") else [f"{base}.zip", base]


def _download_patient_dicom_zip(pid: str, series_uid: str, output_path: str) -> str:
    bucket = _require_bucket()
    client = _get_s3_client()
    last_error: Exception | None = None

    for key in _dicom_s3_key_candidates(pid, series_uid):
        try:
            logger.info("S3_GET_START bucket=%s key=%s", bucket, key)
            client.download_file(bucket, key, output_path)
            logger.info(
                "S3_GET_DONE bucket=%s key=%s path=%s size=%s",
                bucket,
                key,
                output_path,
                os.path.getsize(output_path),
            )
            return key
        except ClientError as exc:
            last_error = exc
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                continue
            logger.exception("S3_GET_FAILED bucket=%s key=%s", bucket, key)
            raise HTTPException(status_code=500, detail=f"Cannot download DICOM zip from S3: {exc}") from exc

    tried = ", ".join(_dicom_s3_key_candidates(pid, series_uid))
    raise HTTPException(
        status_code=404,
        detail=f"DICOM zip not found in S3. Tried: {tried}" + (f". Last error: {last_error}" if last_error else ""),
    )


def _safe_extract_zip(zip_path: str, extract_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        root = os.path.abspath(extract_dir)
        for member in zip_file.infolist():
            target = os.path.abspath(os.path.join(extract_dir, member.filename))
            if not target.startswith(root + os.sep) and target != root:
                raise HTTPException(status_code=400, detail="Invalid zip path detected.")
        zip_file.extractall(extract_dir)


def _find_dicom_dir(root: str) -> str | None:
    import pydicom

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith("._")]
        for filename in filenames:
            if filename.startswith("._"):
                continue
            file_path = os.path.join(dirpath, filename)
            if filename.lower().endswith(".dcm"):
                return dirpath
            try:
                pydicom.dcmread(file_path, stop_before_pixels=True)
                return dirpath
            except Exception:
                continue
    return None


def convert_dicom_to_nifti(dicom_dir: str, output_dir: str) -> str:
    logger.info("DCM2NIIX_START input=%s output=%s", dicom_dir, output_dir)
    result = subprocess.run(
        ["dcm2niix", "-z", "y", "-f", "converted", "-o", output_dir, dicom_dir],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error("DCM2NIIX_FAILED stderr=%s stdout=%s", result.stderr, result.stdout)
        raise RuntimeError(f"dcm2niix failed: {result.stderr}")

    nifti_files = glob.glob(os.path.join(output_dir, "*.nii.gz"))
    if not nifti_files:
        logger.error("DCM2NIIX_NO_OUTPUT stdout=%s stderr=%s", result.stdout, result.stderr)
        raise RuntimeError("dcm2niix produced no NIfTI output")

    nifti_path = max(nifti_files, key=os.path.getsize)
    logger.info("DCM2NIIX_DONE output=%s size=%s", nifti_path, os.path.getsize(nifti_path))
    return nifti_path


def _patient_dicom_zip_to_nifti(pid: str, series_uid: str, work_dir: str) -> tuple[str, str]:
    zip_path = os.path.join(work_dir, "dicom.zip")
    s3_key = _download_patient_dicom_zip(pid, series_uid, zip_path)

    extract_dir = os.path.join(work_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    try:
        logger.info("UNZIP_START path=%s output=%s", zip_path, extract_dir)
        _safe_extract_zip(zip_path, extract_dir)
        logger.info("UNZIP_DONE output=%s", extract_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid DICOM zip file in S3.") from exc

    dicom_dir = _find_dicom_dir(extract_dir)
    if dicom_dir is None:
        raise HTTPException(status_code=400, detail="No DICOM files found in S3 zip.")

    nifti_out_dir = os.path.join(work_dir, "nifti")
    os.makedirs(nifti_out_dir, exist_ok=True)

    try:
        nifti_path = convert_dicom_to_nifti(dicom_dir, nifti_out_dir)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return nifti_path, s3_key


def _cleanup_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        logger.warning("Failed to remove temp file: %s", path, exc_info=True)


def _validate_patient_params(pid: str, series_uid: str) -> tuple[str, str]:
    pid = pid.strip()
    series_uid = series_uid.strip()
    if not pid or not series_uid:
        raise HTTPException(status_code=400, detail="pid and series_uid are required.")
    return pid, series_uid


# ---------------------------------------------------------------------------
# VoxTell auto prompt: report -> shared LLM client -> Top-3 + prompt
# ---------------------------------------------------------------------------

AUTO_PROMPT_SYSTEM_PROMPT = """
You are a radiology report triage assistant for chest CT.

Read the provided analysis report JSON and choose exactly 3 clinically important,
segmentation-relevant details.

Rules:
1. Choose findings that are visible on CT and could guide segmentation.
2. Prefer concrete radiology findings over risk scores or prognosis labels.
3. Prefer lesion/type, size, morphology, and anatomical location.
4. Ignore patient demographics, smoking history, administrative text, and general risk scores.
5. Do not choose negative findings such as "no pleural effusion" or "no suspicious nodule".
6. Return exactly 3 concise strings in descending clinical importance.
7. Do NOT create the final VoxTell prompt. The backend will map your Top-3 to a VoxTell-friendly prompt.

Return valid JSON only:
{
  "top3": ["detail 1", "detail 2", "detail 3"]
}
""".strip()


IMPORTANT_SEGMENTATION_KEYWORDS = [
    "nodule",
    "mass",
    "lesion",
    "tumor",
    "opacity",
    "consolidation",
    "effusion",
    "atelectasis",
    "emphysema",
    "infiltrate",
    "infiltration",
    "ground-glass",
    "ground glass",
    "pleural",
    "pulmonary",
    "lung",
    "lobe",
    "lobar",
    "mediastinal",
    "lymphadenopathy",
]

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnegative for\b",
    r"\bnot seen\b",
    r"\bnot identified\b",
    r"\bno evidence of\b",
    r"\bno suspicious\b",
]


def _clean_report_text(value: Any) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = re.sub(r"<hl[^>]*>", "", text)
    text = text.replace("</hl>", "")
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _get_pipeline_from_app_state(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "get_analysis"):
        return pipeline

    pipeline = getattr(request.app.state, "medical_pipeline", None)
    if pipeline is not None and hasattr(pipeline, "get_analysis"):
        return pipeline

    return None


def _build_llm_analysis_context(analysis: dict[str, Any]) -> str:
    payload = {
        "radiology": analysis.get("radiology"),
        "finding_impression": analysis.get("finding_impression"),
        "verification": analysis.get("verification"),
    }
    return _compact_json(payload)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_auto_prompt_payload(raw: str) -> tuple[list[str], str]:
    payload = json.loads(_strip_code_fences(raw))

    raw_top3 = payload.get("top3") or payload.get("targets") or []
    if not isinstance(raw_top3, list):
        raise ValueError("LLM field `top3` must be a list.")

    top3: list[str] = []
    seen: set[str] = set()
    for item in raw_top3:
        if isinstance(item, dict):
            value = item.get("detail") or item.get("finding") or item.get("text") or item.get("prompt") or ""
        else:
            value = str(item)

        cleaned = _clean_report_text(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            top3.append(cleaned)
            seen.add(key)
        if len(top3) == 3:
            break

    if not top3:
        raise ValueError("LLM returned no usable top3 details.")

    # Important: VoxTell works better with short class-level prompts.
    # LLM selects Top-3 for clinical relevance; backend maps those Top-3
    # into one VoxTell-friendly prompt such as "lung nodule".
    prompt = _build_voxtell_prompt_from_top3(top3)

    return top3[:3], prompt[:500]


async def _extract_top3_and_prompt_via_llm(analysis: dict[str, Any]) -> tuple[list[str], str]:
    user_text = (
        "Read this chest CT analysis report JSON and return exactly 3 clinically important "
        "segmentation-relevant details. Return JSON with only the top3 field.\n\n"
        f"REPORT_JSON:\n{_build_llm_analysis_context(analysis)}"
    )
    raw = await call_llm_json(
        system_prompt=AUTO_PROMPT_SYSTEM_PROMPT,
        user_text=user_text,
        max_tokens=1024,
        temperature=0.0,
        timeout=30.0,
    )
    return _parse_auto_prompt_payload(raw)


def _is_meaningful_value(value: Any) -> bool:
    text = _clean_report_text(value)
    if not text:
        return False
    return text.lower() not in {"none", "no", "normal", "unknown", "n/a", "na", "null", "false"}


def _has_segmentation_keyword(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in IMPORTANT_SEGMENTATION_KEYWORDS)


def _looks_negative_sentence(text: str) -> bool:
    lower = text.lower().strip()
    return _has_segmentation_keyword(lower) and any(re.search(pattern, lower) for pattern in NEGATION_PATTERNS)


def _stringify_structured_item(key: str, value: Any) -> str:
    label = key.replace("_", " ").strip()
    cleaned = _clean_report_text(value)
    return f"{label}: {cleaned}" if label else cleaned


def _collect_structured_candidates(analysis: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    radiology = analysis.get("radiology") or {}
    screening_answer = ((radiology.get("screening") or {}).get("answer") or {})
    detail_answer = ((radiology.get("detail") or {}).get("answer") or {}) if radiology.get("detail") else {}

    for source in [detail_answer, screening_answer]:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if not _is_meaningful_value(value):
                continue
            if _has_segmentation_keyword(f"{key} {value}"):
                candidates.append(_stringify_structured_item(key, value))

    nodule_presence = screening_answer.get("nodule_presence") if isinstance(screening_answer, dict) else None
    if isinstance(nodule_presence, str) and nodule_presence.strip().lower() == "yes":
        candidates.append("nodule presence: Yes")

    return candidates


def _split_report_sentences(text: str) -> list[str]:
    text = _clean_report_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?:\s*[-•]\s+)", text)
    return [p.strip(" ;,.") for p in parts if p.strip(" ;,.")]


def _collect_free_text_candidates(analysis: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    finding_impression = analysis.get("finding_impression") or {}
    verification = analysis.get("verification") or {}

    free_text_sources = [
        finding_impression.get("impression"),
        finding_impression.get("findings"),
        verification.get("analysis") if isinstance(verification, dict) else verification,
    ]

    for source in free_text_sources:
        for sentence in _split_report_sentences(source or ""):
            if _looks_negative_sentence(sentence):
                continue
            if _has_segmentation_keyword(sentence):
                candidates.append(sentence)

    return candidates


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        cleaned = _clean_report_text(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _extract_top3_for_voxtell_rule_fallback(analysis: dict[str, Any]) -> list[str]:
    candidates = []
    candidates.extend(_collect_structured_candidates(analysis))
    candidates.extend(_collect_free_text_candidates(analysis))
    return _dedupe_keep_order(candidates)[:3]


VOXTELL_FRIENDLY_PROMPT_RULES: list[tuple[list[str], str, int]] = [
    # Strong, tested prompts first. For the current case, both "lung nodule"
    # and "pulmonary nodule" produce non-empty masks; "right lower lobe lung nodule"
    # and over-specific nodule prompts can produce empty masks.
    (["nodule", "pulmonary nodule", "solid nodule"], "lung nodule", 100),
    (["mass", "tumor"], "lung mass", 95),
    (["lesion"], "lung lesion", 90),
    (["pleural effusion", "effusion"], "pleural effusion", 85),
    (["consolidation"], "lung consolidation", 80),
    (["ground-glass", "ground glass", "opacity"], "lung opacity", 75),
    (["atelectasis"], "atelectasis", 70),
    (["emphysema", "emphysematous"], "emphysema", 45),
    (["reticulation", "reticular"], "lung reticulation", 40),
    (["lymphadenopathy", "lymph node"], "mediastinal lymphadenopathy", 35),
]


def _map_top3_to_voxtell_prompt(top3: list[str]) -> str:
    """Map LLM-selected Top-3 findings to a short VoxTell-friendly prompt.

    Top-3 is used for clinical selection/explainability.
    The final prompt is intentionally short because VoxTell empirically works
    better with class-level prompts like "lung nodule" than long descriptions.
    """
    best_prompt = ""
    best_score = -1.0

    for index, item in enumerate(top3):
        lower = _clean_report_text(item).lower()
        if not lower:
            continue

        for patterns, prompt, priority in VOXTELL_FRIENDLY_PROMPT_RULES:
            if any(pattern in lower for pattern in patterns):
                # Preserve LLM order while still preferring reliable target classes.
                score = priority - (index * 0.5)
                if score > best_score:
                    best_prompt = prompt
                    best_score = score

    if best_prompt:
        return best_prompt

    return "lung abnormality"


def _build_voxtell_prompt_from_top3(top3: list[str]) -> str:
    if not top3:
        return "lung abnormality"
    return _map_top3_to_voxtell_prompt(top3)


async def _build_auto_prompt_from_saved_analysis(request: Request, pid: str, series_uid: str) -> tuple[list[str], str, str]:
    pipeline = _get_pipeline_from_app_state(request)
    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "MedicalPipeline is not available in app.state. "
                "Check server.py: app.state.pipeline or app.state.medical_pipeline must be initialized."
            ),
        )

    analysis = await pipeline.get_analysis(pid, series_uid)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="No analysis found for this patient/series. Please run /api/v1/analyze first.",
        )

    try:
        top3, prompt = await _extract_top3_and_prompt_via_llm(analysis)
        logger.info("VOXTELL_AUTO_PROMPT_LLM_OK pid=%s series=%s top3=%s prompt=%r", pid, series_uid, top3, prompt)
        return top3, prompt, "llm"
    except Exception as exc:
        logger.exception("VOXTELL_AUTO_PROMPT_LLM_FAILED pid=%s series=%s fallback=rule error=%s", pid, series_uid, exc)

    top3 = _extract_top3_for_voxtell_rule_fallback(analysis)
    prompt = _build_voxtell_prompt_from_top3(top3)
    return top3, prompt, "rule_fallback"


def _metadata_header(value: Any) -> str:
    return quote(json.dumps(value, ensure_ascii=False, default=str), safe="")


def _make_voxtell_binary_response(
    *,
    content: bytes,
    filename: str,
    pid: str,
    series_uid: str,
    prompt: str,
    top3: list[str] | None = None,
    source: str | None = None,
    candidate_logs: list[dict[str, Any]] | None = None,
):
    headers = {
        "content-disposition": f'attachment; filename="{filename}"',
        "x-voxtell-prompt": quote(prompt, safe=""),
        "access-control-expose-headers": (
            "content-disposition, x-voxtell-prompt, x-voxtell-source, "
            "x-voxtell-top3, x-voxtell-candidate-logs"
        ),
    }
    if source:
        headers["x-voxtell-source"] = source
    if top3 is not None:
        headers["x-voxtell-top3"] = _metadata_header(top3)
    if candidate_logs is not None:
        headers["x-voxtell-candidate-logs"] = _metadata_header(candidate_logs)

    return Response(
        content=content,
        media_type="application/gzip",
        headers=headers,
    )


async def _predict_bytes_with_prompt(request: Request, pid: str, series_uid: str, prompt: str) -> tuple[bytes, str]:
    if not _is_s3_direct():
        return await _proxy_predict_bytes(pid, series_uid, prompt)

    predictor = getattr(request.app.state, "voxtell_predictor", None)
    if predictor is None:
        raise HTTPException(status_code=500, detail="VoxTell model not loaded.")

    with tempfile.TemporaryDirectory() as temp_dir:
        nifti_path, s3_key = _patient_dicom_zip_to_nifti(pid, series_uid, temp_dir)

        logger.info(
            "VOXTELL_PREDICT_START pid=%s series=%s key=%s prompt=%r nifti=%s",
            pid,
            series_uid,
            s3_key,
            prompt,
            nifti_path,
        )

        try:
            import torch
            from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reader_writer = NibabelIOWithReorient()
            img, props = reader_writer.read_images([nifti_path])
            segmentations = predictor.predict_single_image(img, [prompt])
            seg_result = segmentations[0]

            output_path = os.path.join(temp_dir, "segmentation.nii.gz")
            reader_writer.write_seg(seg_result, output_path, props)

            safe_prompt = "_".join(prompt.split())[:80] or "mask"
            filename = f"voxtell_{pid}_{safe_prompt}.nii.gz"
            content = Path(output_path).read_bytes()

            logger.info(
                "VOXTELL_PREDICT_DONE pid=%s series=%s prompt=%r output=%s size=%s",
                pid,
                series_uid,
                prompt,
                output_path,
                len(content),
            )

            return content, filename
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("VOXTELL_PREDICT_FAILED: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _run_voxtell_prediction_with_prompt(request: Request, pid: str, series_uid: str, prompt: str):
    content, filename = await _predict_bytes_with_prompt(request, pid, series_uid, prompt)
    return _make_voxtell_binary_response(
        content=content,
        filename=filename,
        pid=pid,
        series_uid=series_uid,
        prompt=prompt,
    )


def _mask_stats_from_nifti_bytes(content: bytes) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(tmp_path)
        data = img.get_fdata()

        nonzero = int(np.count_nonzero(data))
        max_value = float(np.nanmax(data)) if data.size else 0.0
        total = int(data.size)
        ratio = float(nonzero / total) if total else 0.0

        return {
            "shape": list(data.shape),
            "nonzero": nonzero,
            "max": max_value,
            "total": total,
            "ratio": ratio,
            "is_nonempty": nonzero > 0 and max_value > 0,
        }
    finally:
        _cleanup_file(tmp_path)


def _candidate_is_acceptable(stats: dict[str, Any]) -> bool:
    if not stats.get("is_nonempty"):
        return False

    ratio = float(stats.get("ratio") or 0.0)
    nonzero = int(stats.get("nonzero") or 0)

    # Avoid selecting huge accidental masks for a lesion workflow.
    # Keep this permissive because organs can still be requested manually.
    if nonzero < int(os.getenv("VOXTELL_AUTO_MIN_NONZERO", "10")):
        return False
    if ratio > float(os.getenv("VOXTELL_AUTO_MAX_MASK_RATIO", "0.20")):
        return False

    return True


def _generate_prompt_candidates_from_top3(top3: list[str], max_candidates: int | None = None) -> list[str]:
    max_candidates = max_candidates or int(os.getenv("VOXTELL_AUTO_MAX_CANDIDATES", "5"))
    joined = " ; ".join(top3).lower()
    primary = top3[0] if top3 else ""
    primary_lower = primary.lower()

    candidates: list[str] = []

    def add(prompt: str):
        prompt = _clean_report_text(prompt)
        if prompt and prompt.lower() not in {p.lower() for p in candidates}:
            candidates.append(prompt)

    # Keep high-success prompts first. Empirically, the local tests showed:
    # "lung nodule" and "pulmonary nodule" work, while over-specific/location prompts may be empty.
    if "nodule" in joined:
        add("lung nodule")
        add("pulmonary nodule")
        if "right lower lobe" in primary_lower:
            add("right lower lobe lung nodule")
        elif "left lower lobe" in primary_lower:
            add("left lower lobe lung nodule")
        elif "right upper lobe" in primary_lower:
            add("right upper lobe lung nodule")
        elif "left upper lobe" in primary_lower:
            add("left upper lobe lung nodule")
        if "solid" in primary_lower:
            add("solid pulmonary nodule")

    if any(x in joined for x in ["mass", "tumor", "carcinoma", "cancer"]):
        add("lung tumor")
        add("lung mass")

    if "pleural effusion" in joined or "effusion" in joined:
        add("pleural effusion")

    if "consolidation" in joined:
        add("lung consolidation")

    if "ground-glass" in joined or "ground glass" in joined or "opacity" in joined:
        add("lung opacity")

    if "atelectasis" in joined:
        add("atelectasis")

    if "emphysema" in joined or "emphysematous" in joined:
        add("emphysema")

    # Add the primary clinical phrase last as a transparent but risky candidate.
    # It is useful for real report-derived prompts, but should not block simpler working prompts.
    add(primary)

    if not candidates:
        add(_build_voxtell_prompt_from_top3(top3))

    return candidates[:max_candidates]


async def _run_auto_candidate_segmentation(
    request: Request,
    pid: str,
    series_uid: str,
    top3: list[str],
    source: str,
):
    candidates = _generate_prompt_candidates_from_top3(top3)
    candidate_logs: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    logger.info(
        "VOXTELL_AUTO_CANDIDATES_START pid=%s series=%s source=%s candidates=%s top3=%s",
        pid,
        series_uid,
        source,
        candidates,
        top3,
    )

    last_error: Exception | None = None
    for index, prompt in enumerate(candidates, start=1):
        try:
            content, filename = await _predict_bytes_with_prompt(request, pid, series_uid, prompt)
            stats = _mask_stats_from_nifti_bytes(content)
            accepted = _candidate_is_acceptable(stats)

            log_item = {
                "rank": index,
                "prompt": prompt,
                "filename": filename,
                "accepted": accepted,
                **stats,
            }
            candidate_logs.append(log_item)

            logger.info(
                "VOXTELL_AUTO_CANDIDATE_RESULT pid=%s series=%s prompt=%r accepted=%s nonzero=%s ratio=%s",
                pid,
                series_uid,
                prompt,
                accepted,
                stats.get("nonzero"),
                stats.get("ratio"),
            )

            if accepted and selected is None:
                selected = {
                    "content": content,
                    "filename": filename,
                    "prompt": prompt,
                    "stats": stats,
                }
                if os.getenv("VOXTELL_AUTO_TRY_ALL_CANDIDATES", "0").strip().lower() not in {"1", "true", "yes"}:
                    break
        except Exception as exc:
            last_error = exc
            logger.exception(
                "VOXTELL_AUTO_CANDIDATE_FAILED pid=%s series=%s prompt=%r error=%s",
                pid,
                series_uid,
                prompt,
                exc,
            )
            candidate_logs.append(
                {
                    "rank": index,
                    "prompt": prompt,
                    "accepted": False,
                    "error": str(exc),
                }
            )

    if selected is None:
        detail = {
            "message": "VoxTell auto segmentation produced no acceptable mask for all candidate prompts.",
            "top3": top3,
            "candidates": candidate_logs,
        }
        if last_error:
            detail["last_error"] = str(last_error)
        raise HTTPException(status_code=422, detail=detail)

    logger.info(
        "VOXTELL_AUTO_SELECTED pid=%s series=%s prompt=%r stats=%s candidates=%s",
        pid,
        series_uid,
        selected["prompt"],
        selected["stats"],
        candidate_logs,
    )

    return _make_voxtell_binary_response(
        content=selected["content"],
        filename=selected["filename"],
        pid=pid,
        series_uid=series_uid,
        prompt=selected["prompt"],
        top3=top3,
        source=source,
        candidate_logs=candidate_logs,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/volume/{pid}/{series_uid:path}")
async def get_voxtell_volume(pid: str, series_uid: str):
    pid, series_uid = _validate_patient_params(pid, series_uid)

    if not _is_s3_direct():
        return await _proxy_volume(pid, series_uid)

    with tempfile.TemporaryDirectory() as temp_dir:
        nifti_path, s3_key = _patient_dicom_zip_to_nifti(pid, series_uid, temp_dir)
        final_path = os.path.join(tempfile.gettempdir(), f"voxtell_volume_{pid}_{os.urandom(8).hex()}.nii.gz")
        shutil.copy(nifti_path, final_path)

    logger.info("VOLUME_READY pid=%s series=%s key=%s file=%s", pid, series_uid, s3_key, final_path)
    return FileResponse(
        final_path,
        media_type="application/gzip",
        filename=f"{pid}_{series_uid}.nii.gz",
        background=BackgroundTask(_cleanup_file, final_path),
    )


@router.post("/prompt-auto")
async def voxtell_prompt_auto(
    request: Request,
    pid: Annotated[str, Form()],
    series_uid: Annotated[str, Form()],
):
    pid, series_uid = _validate_patient_params(pid, series_uid)
    top3, prompt, source = await _build_auto_prompt_from_saved_analysis(request, pid, series_uid)
    prompt_candidates = _generate_prompt_candidates_from_top3(top3)

    logger.info(
        "VOXTELL_AUTO_PROMPT pid=%s series=%s source=%s top3=%s mapped_prompt=%r candidates=%s",
        pid,
        series_uid,
        source,
        top3,
        prompt,
        prompt_candidates,
    )

    return {
        "source": source,
        "pid": pid,
        "series_uid": series_uid,
        "top3": top3,
        "prompt": prompt,
        "prompt_candidates": prompt_candidates,
    }


@router.post("/predict-auto")
async def voxtell_predict_auto(
    request: Request,
    pid: Annotated[str, Form()],
    series_uid: Annotated[str, Form()],
):
    pid, series_uid = _validate_patient_params(pid, series_uid)
    top3, prompt, source = await _build_auto_prompt_from_saved_analysis(request, pid, series_uid)

    logger.info(
        "VOXTELL_AUTO_PREDICT_CANDIDATE_MODE pid=%s series=%s source=%s top3=%s initial_prompt=%r",
        pid,
        series_uid,
        source,
        top3,
        prompt,
    )

    return await _run_auto_candidate_segmentation(request, pid, series_uid, top3, source)


@router.post("/predict")
async def voxtell_predict(
    request: Request,
    pid: Annotated[str, Form()],
    series_uid: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
):
    pid, series_uid = _validate_patient_params(pid, series_uid)
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required.")

    return await _run_voxtell_prediction_with_prompt(request, pid, series_uid, prompt)
