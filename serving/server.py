"""
FastAPI inference server.
POST /analyze  — runs all 3 specialist agents + report
POST /ask      — doctor interface Q&A
GET  /health   — health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from serving.api.routes import router
from serving.lora_manager import LoraManager

app = FastAPI(title="AI Medical Department", version="0.1.0")

# Loaded once at startup, shared across requests
lora_manager: LoraManager | None = None


@app.on_event("startup")
async def startup():
    global lora_manager
    lora_manager = LoraManager.from_config("configs/model.yaml")
    app.state.lora_manager = lora_manager


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(router, prefix="/api/v1")
