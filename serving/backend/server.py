"""
FastAPI backend — runs on VPS, orchestrates calls to Modal inference endpoint.

Endpoints:
  POST /api/v1/analyze  — full pipeline (radiology + cardiology + oncology + report)
  GET  /health          — health check
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("serving.backend.pipeline").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from serving.backend.pipeline import MedicalPipeline
from serving.backend.routes import router

load_dotenv()

app = FastAPI(title="AI Medical Department", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    app.state.pipeline = MedicalPipeline()


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(router, prefix="/api/v1")
