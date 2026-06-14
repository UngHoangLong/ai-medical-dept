"""
FastAPI backend for AI Medical Department.

Production responsibilities:
- Mount AI Medical API routes under /api/v1.
- Initialize MedicalPipeline lifecycle.
- VoxTell CT Viewer endpoints are mounted through routes/voxtell.py and proxy to Modal.
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

from serving.backend.pipeline import MedicalPipeline
from serving.backend.routes import router

load_dotenv()

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


@app.on_event("startup")
async def startup() -> None:
    """
    Initialize the main medical pipeline.

    VoxTell is intentionally not loaded in this backend.
    The /api/v1/voxtell/* routes proxy requests to the Modal service
    configured by VOXTELL_MODAL_BASE_URL.
    """
    app.state.pipeline = MedicalPipeline()
    await app.state.pipeline.connect()
    logger.info("MedicalPipeline connected.")


@app.on_event("shutdown")
async def shutdown() -> None:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        await pipeline.close()
        logger.info("MedicalPipeline closed.")


@app.get("/health")
async def health() -> dict[str, str]:
    voxtell_modal_base_url = os.getenv("VOXTELL_MODAL_BASE_URL", "").strip()
    return {
        "status": "ok",
        "voxtell": "modal_proxy",
        "voxtell_modal": "configured" if voxtell_modal_base_url else "missing_env",
    }


app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)