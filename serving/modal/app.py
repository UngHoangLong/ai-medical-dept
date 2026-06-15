"""
Modal inference app — MedGemma 1.5-4B + 4 LoRA adapters — served via vLLM.

CT image loading:
  - Backend convert .dcm → PNG, upload lên S3, gửi presigned `image_urls`
    + `cache_key` ("{pid}/{series_uid}") trong request đầu tiên của 1 scan
  - Modal tải ảnh từ S3 đúng 1 lần, giữ trong RAM cache (dict theo cache_key)
    — các agent tiếp theo (detail, cardiology, oncology, Q&A...) trên cùng
    scan chỉ cần gửi cache_key, không cần tải lại từ S3

vLLM handles concurrent requests natively via its internal scheduler.
"""

import base64
import io
import json
import re
import time

import modal
from pydantic import BaseModel


# ─── Pydantic models cho infer_batch ─────────────────────────────────────────

class RunIf(BaseModel):
    """Chỉ chạy task này nếu 1 field trong output của task trước khớp value."""
    task_id: str   # id của task trước cần check
    field: str     # JSON field trong output text của task đó
    value: str     # giá trị mong đợi (string exact match)

class PipelineTask(BaseModel):
    id: str
    adapter: str
    prompt: str
    max_new_tokens: int = 512
    repetition_penalty: float = 1.0
    run_if: RunIf | None = None

class PipelineRequest(BaseModel):
    cache_key: str
    tasks: list[PipelineTask]
    image_urls: list[str] | None = None
    ct_slices: list[str] | None = None


def _check_run_if(run_if: RunIf, results: dict) -> bool:
    """Kiểm tra điều kiện run_if bằng cách parse JSON từ output text của task trước."""
    prev = results.get(run_if.task_id)
    if not prev:
        return False
    m = re.search(r"\{[^{}]*\}", prev, re.DOTALL)
    if not m:
        return False
    try:
        return json.loads(m.group()).get(run_if.field) == run_if.value
    except Exception:
        return False

# ─── HuggingFace repo IDs ─────────────────────────────────────────────────
BASE_MODEL_ID = "unsloth/medgemma-1.5-4b-it"
MAX_SEQ_LEN   = 26000   # matches training: 85 CT slices × image tokens + text

ADAPTER_IDS: dict[str, str] = {
    "radiology":          "UngLong/medgemma-radiology-lora",
    "cardiology":         "ChonJohn171105/medgemma-cardiology-lora",
    "oncology":           "Phiphi216/medgemma-oncology-lora-v2",
    "finding_impression": "rimine/test-medgemma-1.5-4b-ct-rate-finding-impression",
}

MODEL_DIR = "/model-cache"

# ─── Modal resources ──────────────────────────────────────────────────────
app = modal.App("ai-medical-dept")

model_volume = modal.Volume.from_name("medgemma-weights", create_if_missing=True)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm>=0.8.0",
        "transformers>=4.47.0",   # AutoProcessor for chat template (CPU only)
        "pillow>=10.0.0",
        "huggingface_hub>=0.23.0",
        "numpy>=1.26.0",
        "requests>=2.31.0",       # tải ảnh CT từ presigned S3 URL
    )
)


# ─── Inference class ──────────────────────────────────────────────────────
@app.cls(
    gpu="A100-80GB",
    image=inference_image,
    volumes={
        MODEL_DIR: model_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    scaledown_window=300,
    max_containers=1,   # 1 container duy nhất — giữ image cache + KV cache cho toàn session
    timeout=900,   # 15 min — covers cold start (vLLM load ~6min) + inference
)
@modal.concurrent(max_inputs=4)

class MedicalInference:

    @modal.enter()
    def load(self):
        import os
        from vllm import LLM
        from transformers import AutoProcessor
        from huggingface_hub import snapshot_download

        # RAM cache: cache_key → list[PIL.Image] — tránh tải lại từ S3 nhiều
        # lần trong cùng 1 phiên phân tích (screening → detail → agent khác)
        self._image_cache: dict[str, list] = {}

        os.environ["HF_HOME"] = MODEL_DIR
        # lưu torch.compile cache vào Volume (persist qua các lần cold start)
        # → bỏ qua được ~107s Dynamo transform + graph compile ở lần sau
        os.environ["VLLM_CACHE_ROOT"] = f"{MODEL_DIR}/vllm_cache"
        token = os.environ["HF_TOKEN"]

        for name, hf_id in ADAPTER_IDS.items():
            adapter_path = f"{MODEL_DIR}/adapters/{name}"
            if not os.path.exists(adapter_path):
                print(f"Downloading adapter [{name}]: {hf_id}")
                snapshot_download(
                    repo_id=hf_id,
                    local_dir=adapter_path,
                    token=token,
                )

        self.processor = AutoProcessor.from_pretrained(
            BASE_MODEL_ID,
            token=token,
            cache_dir=MODEL_DIR,
        )
        self.processor.tokenizer.model_max_length = MAX_SEQ_LEN

        print("Starting vLLM engine...")
        self.llm = LLM(
            model=BASE_MODEL_ID,           # HuggingFace repo của base model
            enable_lora=True,              # cho phép load LoRA adapter per-request
            max_lora_rank=16,              # rank tối đa của adapter — phải khớp với lúc train
            max_model_len=MAX_SEQ_LEN,     # context window tối đa: 85 slices × 256 tokens + text
            gpu_memory_utilization=0.9,    # dùng 90% VRAM cho model + KV cache
            download_dir=MODEL_DIR,        # lưu model weights vào Volume (tránh download lại)
            max_num_seqs=1,                # chỉ xử lý 1 request tại một thời điểm — tránh batch
                                           # 2 CT scans cùng lúc gây KV cache explosion
            enable_prefix_caching=True,    # cache KV activations của phần ảnh CT (85 images prefix)
                                           # → agents 2,3,4,5 trên cùng scan không recompute vision prefix
        )

        self._adapters = {
            name: (idx + 1, f"{MODEL_DIR}/adapters/{name}")
            for idx, name in enumerate(ADAPTER_IDS)
        }

        model_volume.commit()
        print("vLLM engine ready.")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def infer(self, req: dict) -> dict:
        """
        Request body:
            adapter        : "radiology" | "cardiology" | "oncology" | "finding_impression"
            prompt         : str
            cache_key      : str              — "{pid}/{series_uid}", always required
            image_urls     : list[str] | None — presigned S3 URL cho từng slice PNG;
                                                gửi ở lần gọi đầu tiên của 1 scan,
                                                bỏ qua ở các lần sau (RAM cache hit)
            ct_slices      : list[str] | None — base64 PNG; chỉ dùng cho warmup
                                                (không có scan thật để lấy URL)
            max_new_tokens : int              — optional, default 512

        Response body:
            adapter  : str
            output   : str
            cached   : bool — True if images were served from RAM cache
        """
        import requests
        from PIL import Image
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        adapter: str        = req["adapter"]
        prompt: str         = req["prompt"]
        cache_key: str      = req["cache_key"]
        image_urls: list    = req.get("image_urls") or []
        slices_b64: list    = req.get("ct_slices") or []
        max_new_tokens: int = req.get("max_new_tokens", 512)

        # ── Resolve CT images (RAM cache theo cache_key) ─────────────────
        cache_hit = cache_key in self._image_cache

        if cache_hit:
            images = self._image_cache[cache_key]
        elif image_urls:
            images = [
                Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("RGB")
                for url in image_urls
            ]
            self._image_cache[cache_key] = images
        elif slices_b64:
            images = [
                Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
                for s in slices_b64
            ]
            self._image_cache[cache_key] = images
        else:
            raise ValueError(f"Cache miss for '{cache_key}' and no image_urls/ct_slices provided.")

        # ── Build chat template ──────────────────────────────────────────
        # Khớp ĐÚNG PrefetchVisionDataCollator lúc fine-tune: N ảnh liên tiếp
        # (không nhãn "SLICE i" xen giữa — model chưa từng thấy dạng đó khi
        # train LoRA, chèn vào sẽ làm lệch cấu trúc chuỗi token ảnh).
        content = [{"type": "image"} for _ in range(len(images))]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        formatted_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # ── Inference ────────────────────────────────────────────────────
        lora_int_id, lora_path = self._adapters[adapter]
        outputs = self.llm.generate(
            {"prompt": formatted_text, "multi_modal_data": {"image": images}},
            sampling_params=SamplingParams(temperature=0.0, max_tokens=max_new_tokens),
            lora_request=LoRARequest(
                lora_name=adapter,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
            ),
        )

        return {
            "adapter": adapter,
            "output": outputs[0].outputs[0].text,
            "cached": cache_hit,
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    def infer_batch(self, req: PipelineRequest) -> dict:
        """
        Chạy toàn bộ pipeline (tối đa 5 agent) trong 1 lần gọi duy nhất.

        Lợi ích so với gọi `infer` 5 lần riêng:
          - vLLM KV prefix cache của 85 ảnh CT giữ nguyên suốt cả pipeline
            (không bị evict bởi request của scan khác chen vào giữa các bước)
          - Ít HTTP round-trip hơn (1 vs 5)

        Backend build sẵn tất cả prompts rồi gửi 1 lần. Mỗi task có thể có
        điều kiện `run_if` — task chỉ chạy nếu 1 field trong output JSON của
        task trước khớp value mong đợi (dùng cho radiology detail: chỉ chạy
        khi screening trả nodule_presence == "Yes").

        Request body: PipelineRequest (xem schema trong docs)
        Response body:
            results : dict[task_id → output_text | null]
            cached  : bool
        """
        import requests as _req_lib
        from PIL import Image
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        cache_key   = req.cache_key
        image_urls  = req.image_urls or []
        slices_b64  = req.ct_slices or []

        # ── Resolve CT images (RAM cache) ────────────────────────────────
        cache_hit = cache_key in self._image_cache
        if cache_hit:
            images = self._image_cache[cache_key]
        elif image_urls:
            images = [
                Image.open(io.BytesIO(_req_lib.get(url, timeout=60).content)).convert("RGB")
                for url in image_urls
            ]
            self._image_cache[cache_key] = images
        elif slices_b64:
            images = [
                Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
                for s in slices_b64
            ]
            self._image_cache[cache_key] = images
        else:
            raise ValueError(f"Cache miss for '{cache_key}' and no image_urls/ct_slices provided.")

        # Content prefix chung — build 1 lần, tái dùng cho mọi task
        # (KV prefix cache của vLLM sẽ nhận ra phần ảnh giống nhau → reuse)
        img_content = [{"type": "image"} for _ in range(len(images))]

        results: dict[str, str | None] = {}

        for task in req.tasks:
            # Kiểm tra điều kiện — bỏ qua task nếu không thoả mãn
            if task.run_if and not _check_run_if(task.run_if, results):
                results[task.id] = None
                continue

            content = img_content + [{"type": "text", "text": task.prompt}]
            messages = [{"role": "user", "content": content}]
            formatted_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )

            if task.adapter == "base":
                lora_req = None
            else:
                lora_int_id, lora_path = self._adapters[task.adapter]
                lora_req = LoRARequest(
                    lora_name=task.adapter,
                    lora_int_id=lora_int_id,
                    lora_path=lora_path,
                )

            outputs = self.llm.generate(
                {"prompt": formatted_text, "multi_modal_data": {"image": images}},
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=task.max_new_tokens,
                    repetition_penalty=task.repetition_penalty,
                ),
                lora_request=lora_req,
            )
            results[task.id] = outputs[0].outputs[0].text

        return {"results": results, "cached": cache_hit}


_DUMMY_PNG_B64 = (  # 1×1 white pixel — đủ để pass qua bước load ảnh, không cần CT thật
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4"
    "nGNgAAIAAAUAAen63NgAAAAASUVORK5CYII="
)


@app.local_entrypoint()
def warmup():
    """
    Trigger cold start chủ động — gửi HTTP POST thẳng đến web endpoint.

    `infer` được decorate bằng @modal.fastapi_endpoint nên là "webhook function":
    Modal CẤM gọi qua .remote() (chỉ cho gọi qua URL HTTP hoặc .local() — mà
    .local() chạy ngay trên máy local, không đụng tới container/GPU remote).
    Nên cách duy nhất để "chọc" container remote start lên là gửi 1 request
    HTTP thật tới web URL đã deploy. @modal.enter() (load model) luôn chạy
    trước khi infer() xử lý request, nên chỉ cần request tới là container warm.

    Usage: modal run serving/modal/app.py::warmup
    """
    import json
    import os
    import urllib.request

    from dotenv import load_dotenv
    load_dotenv()

    url = os.environ["MODAL_ENDPOINT_URL"]
    payload = json.dumps({
        "adapter": "radiology",
        "prompt": "ping",
        "cache_key": "warmup/ping",
        "ct_slices": [_DUMMY_PNG_B64],
        "max_new_tokens": 4,
    }).encode()

    print(f"[warmup] POST {url} ...")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        print(f"[warmup] Container started — status={resp.status}")

# build image: modal deploy serving/modal/app.py


# Warm up: modal run serving/modal/app.py::warmup


# Check logs app (log của mọi containers): modal app logs ai-medical-dept  

# Check container để lấy "container-id": modal container list

# Nếu muốn tắt ngay (không cần đợi scaledown_window=300s):
# modal container stop <container-id>

# Hoặc tắt cả app luôn nếu xong việc hẳn:
# modal app stop ai-medical-dept

#---------------------------------------------------

# CÁC CÂU LỆNH ĐĂNG NHẬP VÀ SETUP TÀI KHOẢN MỚI

# modal token new

# modal secret create huggingface-token HF_TOKEN=<your_token>

# modal deploy serving/modal/app.py
