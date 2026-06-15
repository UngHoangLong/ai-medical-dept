import io
import modal
from fastapi import Request
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libgomp1")
    .pip_install(
        "torch",
        "faster-whisper", 
        "torchaudio", 
        "fastapi[standard]",
        "transformers",
        "safetensors"
    )
)

app = modal.App(name="medical-stt-service", image=image)

@app.cls(cpu=2.0, memory=2048) # Chạy bằng CPU (2 cores, 2GB RAM) là đủ cho bản Base
class MedicalSTT:
    @modal.enter()
    def load_model(self):
        import os
        import torch
        import ctranslate2
        # Đã sửa lỗi chính tả import từ 'faster-whisper' thành 'faster_whisper'
        # pyrefly: ignore [missing-import]
        from faster_whisper import WhisperModel
        
        model_id = "Kaivalya1993/whisper-base-medical-en"
        output_dir = "/root/ct2_model_whisper_base_medical"
        
        # Kiểm tra nếu mô hình chưa được convert sang định dạng CTranslate2 thì tiến hành convert
        if not os.path.exists(output_dir):
            print(f"Đang tự động chuyển đổi {model_id} sang định dạng CTranslate2...")
            converter = ctranslate2.converters.TransformersConverter(
                model_id,
                copy_files=["tokenizer.json"]
            )
            # Tiến hành convert và lưu vào thư mục cục bộ trên container với kiểu số thực float32 (mượt trên CPU)
            converter.convert(output_dir, quantization="float32", force=True)
            print("Chuyển đổi mô hình thành công!")

        # Load mô hình trực tiếp từ thư mục đã convert nội bộ
        self.model = WhisperModel(output_dir, device="cpu", compute_type="float32")

    @modal.fastapi_endpoint(method="POST")
    async def transcribe(self, request: Request):
        audio_bytes = await request.body()
        audio_file = io.BytesIO(audio_bytes)
        
        segments, info = self.model.transcribe(
            audio_file, 
            beam_size=5
        )
        text = "".join([segment.text for segment in segments])
        
        return {"text": text.strip()}
