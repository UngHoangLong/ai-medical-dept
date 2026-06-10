from fastapi import APIRouter

from serving.backend.routes.analyze import router as analyze_router

router = APIRouter()
router.include_router(analyze_router)

# uvicorn serving.backend.server:app --port 8000

