import logging
import os
# pyrefly: ignore [missing-import]
import httpx
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

# Lưu ý: prefix "/api/v1" được thêm tự động tại server.py, ở đây chỉ cần định nghĩa endpoint "/stt"
router = APIRouter(tags=["Speech-to-Text"])

MODAL_STT_URL = os.getenv("MODAL_STT_URL")

@router.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if not MODAL_STT_URL:
        logger.error("MODAL_STT_URL environmental variable is not set")
        raise HTTPException(status_code=500, detail="STT service is not configured on the server")
        
    # 1. Đọc file âm thanh gửi lên từ Frontend
    audio_content = await file.read()
    
    # 2. Dùng HTTP Client không đồng bộ gửi dữ liệu sang Modal AI
    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            logger.info("Sending audio to STT service at: %s", MODAL_STT_URL)
            response = await client.post(
                MODAL_STT_URL,
                content=audio_content
            )
            if response.status_code != 200:
                logger.error("Modal STT error: %d - %s", response.status_code, response.text)
                raise HTTPException(status_code=500, detail="Modal AI Service Error")
                
            return response.json() # Trả về {"text": "..."} cho Frontend
            
        except httpx.RequestError as e:
            logger.error("Failed to connect to Modal STT: %s", str(e))
            raise HTTPException(status_code=503, detail="STT Service Unavailable")
