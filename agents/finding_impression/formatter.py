"""
Finding & Impression formatters — sinh radiology report (Findings + Impression)
trực tiếp từ ảnh CT (MedGemma 1.5 LoRA).

Khác biệt CĂN BẢN so với 3 agent kia (radiology/cardiology/oncology):
  - Output là văn bản narrative tự do (giống report bác sĩ viết tay),
    KHÔNG phải JSON Q&A có allowed-values.
  - Prompt CỐ ĐỊNH — KHÔNG nhúng clinical_data hay kết quả của 3 specialist
    khác vào trong (đã verify NGUYÊN VĂN từ HF: rimine/ct-rate-medgemma-ready,
    model chỉ nhìn ảnh CT + 1 câu lệnh chung, không có context nào khác).

⚠️ Hàm `_build_fi_prompt()` cũ trong pipeline.py build prompt bằng cách nhúng
kết quả 3 agent kia vào — đó là TRAIN/SERVING MISMATCH (model chưa từng thấy
format đó lúc fine-tune), không được dùng. Prompt đúng là TRAINING_PROMPT dưới đây.
"""

import re

# Prompt CỐ ĐỊNH — khớp NGUYÊN VĂN dataset rimine/ct-rate-medgemma-ready
TRAINING_PROMPT = (
    "Analyze the provided contiguous block of CT slices carefully. Generate a "
    "detailed section for Findings and provide a final clinical Impression "
    "based on the visual evidence for this CT volume scan."
)


# ===========================================================================
# TRAINING formatter — dùng khi build dataset ready
# ===========================================================================

class FindingImpressionFormatter:
    """Training — prompt cố định, answer = report narrative gốc (Findings/Impression)."""

    def format_sample(self, record: dict) -> dict:
        return {
            "prompt": TRAINING_PROMPT,
            "answer": record["response"],
            "keys":   record.get("keys"),
            "pids":   record.get("pids"),
        }


# ===========================================================================
# INFERENCE formatter — dùng cho backend
# ===========================================================================

class FindingImpressionInferenceFormatter:
    """Inference — dùng cho backend. Prompt CỐ ĐỊNH, không cần clinical_data."""

    def format_prompt(self) -> str:
        return TRAINING_PROMPT


# ---------------------------------------------------------------------------
# Parse output inference (backend dùng)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"Findings\s*:\s*(?P<findings>.*?)\n\s*Impression\s*:\s*(?P<impression>.*)",
    re.IGNORECASE | re.DOTALL,
)


def parse_inference_output(text: str) -> dict:
    """Tách report narrative thành 2 section: findings / impression.
    Fallback: nếu không match format chuẩn → trả raw text làm findings.
    """
    m = _SECTION_RE.search(text)
    if m:
        return {
            "findings":   m.group("findings").strip(),
            "impression": m.group("impression").strip(),
            "raw": text,
        }
    # Fallback — format lạ hoặc thiếu Impression: trả raw để bác sĩ tự đọc
    raw = text.strip()
    return {"findings": raw if raw else None, "impression": None, "raw": text}
