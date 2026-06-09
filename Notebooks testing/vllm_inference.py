import json
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "opencv-python")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster model transfers
            "VLLM_LOG_STATS_INTERVAL": "1",  # more frequent metrics logging
            "HF_TOKEN": "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"
        }
    )
)

MODEL_NAME = "unsloth/medgemma-1.5-4b-it"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
FAST_BOOT = False


app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60 
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"A100:{N_GPU}",
    scaledown_window=10 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=30,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import json
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    # add model-specific configuration
    cmd += [
        # skip multimedia support, just language
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 85, 'video': 0, 'audio': 0})}'",
    ]
    
    cmd += [
        # use the same context window for all models, even if they don't need it, to avoid recompilations
        "--max_num_seqs",
        "5",
    ]
    
    cmd += [
        "--max_model_len",
        "25808",
    ]
    
    cmd += [
        "--enable-lora"
    ]
    
    # Khai báo tất cả các LoRA modules trên cùng một tham số, cách nhau bằng dấu cách
    cmd += [
        "--lora-modules",
        "findings_and_impressions_lora=rimine/test-medgemma-1.5-4b-ct-rate-finding-impression "
        "oncology_lora=Phiphi216/medgemma-oncology-lora "
        "cardiology_lora=ChonJohn171105/medgemma-cardiology-lora "
        "radiology_lora=UngLong/medgemma-radiology-lora"
    ]

    # Ép vLLM tận dụng tối đa VRAM cho KV Cache khổng lồ của ảnh
    cmd += [
        "--gpu-memory-utilization", "0.92" 
    ]
    print(*cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


