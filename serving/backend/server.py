"""
FastAPI backend for AI Medical Department.

Modes:
- proxy (default): local/main backend does not load VoxTell. It proxies
  /api/v1/voxtell/* to VOXTELL_MODAL_BASE_URL.
- s3_direct: Modal GPU backend loads VoxTell, downloads DICOM zip from S3,
  converts with dcm2niix, and runs VoxTell segmentation directly.

Set VOXTELL_BACKEND_MODE=s3_direct only inside the Modal GPU app.
"""

import logging
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from serving.backend.pipeline import MedicalPipeline
from serving.backend.routes import router


def _voxtell_mode() -> str:
    mode = os.getenv("VOXTELL_BACKEND_MODE", "proxy").strip().lower()
    return "s3_direct" if mode in {"s3_direct", "direct", "modal_s3_direct"} else "proxy"


def _is_s3_direct() -> bool:
    return _voxtell_mode() == "s3_direct"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("serving.backend.pipeline").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("serving.backend")

app = FastAPI(title="AI Medical Department", version="0.1.0")

raw_origins = os.getenv("CORS_ORIGINS", "*")
allow_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
allow_credentials = "*" not in allow_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_VOXTELL_MODEL_DIR = os.path.abspath(
    os.path.join(PROJECT_ROOT, "models", "voxtell_v1.1")
)
VOXTELL_MODEL_DIR = os.getenv("VOXTELL_MODEL_DIR", DEFAULT_VOXTELL_MODEL_DIR)


@app.on_event("startup")
async def startup() -> None:
    """Initialize MedicalPipeline and conditionally load VoxTell on Modal GPU."""
    app.state.pipeline = MedicalPipeline()
    await app.state.pipeline.connect()
    logger.info("MedicalPipeline connected.")

    app.state.voxtell_predictor = None
    mode = _voxtell_mode()
    logger.info("VoxTell backend mode: %s", mode)

    if not _is_s3_direct():
        logger.info("VoxTell is not loaded locally; /voxtell routes proxy to VOXTELL_MODAL_BASE_URL.")
        return

    if os.getenv("DISABLE_VOXTELL", "0") == "1":
        logger.warning("VoxTell loading disabled by DISABLE_VOXTELL=1.")
        return

    try:
        import torch
        from voxtell.inference.predictor import VoxTellPredictor

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading VoxTell model from %s on %s...", VOXTELL_MODEL_DIR, device)
        predictor = VoxTellPredictor(model_dir=VOXTELL_MODEL_DIR, device=device)
        predictor.perform_everything_on_device = False
        app.state.voxtell_predictor = predictor
        logger.info("VoxTell model loaded successfully.")
    except Exception as exc:
        app.state.voxtell_predictor = None
        logger.exception("Error loading VoxTell model: %s", exc)


@app.on_event("shutdown")
async def shutdown() -> None:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        await pipeline.close()
        logger.info("MedicalPipeline closed.")


@app.get("/health")
async def health() -> dict[str, str]:
    if _is_s3_direct():
        predictor = getattr(app.state, "voxtell_predictor", None)
        return {
            "status": "ok",
            "voxtell": "loaded" if predictor is not None else "not_loaded",
            "voxtell_mode": "s3_direct",
        }

    voxtell_modal_base_url = os.getenv("VOXTELL_MODAL_BASE_URL", "").strip()
    return {
        "status": "ok",
        "voxtell": "modal_proxy",
        "voxtell_mode": "proxy",
        "voxtell_modal": "configured" if voxtell_modal_base_url else "missing_env",
    }


app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
