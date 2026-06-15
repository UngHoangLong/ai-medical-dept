from fastapi import APIRouter

from serving.backend.routes.analyze import router as analyze_router
from serving.backend.routes.stt import router as stt_router

router = APIRouter()
router.include_router(analyze_router)
router.include_router(stt_router)

# uvicorn serving.backend.server:app --port 8000

