"""
Radiology formatters — build prompt cho Radiology agent (MedGemma 1.5 LoRA).

2 bước (sequential inference, khớp cách fine-tune):
  - Screening : chest_abn_54–61 + nodule_presence  (8 task, mọi scan)
  - Detail    : nodule_location/attenuation/margin/size (4 task, chỉ khi nodule_presence=Yes)

2 chế độ prompt:
  - TRAINING  : prompt + JSON answer ngắn — KHỚP với build_radiology_ready.py đã train.
  - INFERENCE : prompt yêu cầu model GIẢI THÍCH reasoning chi tiết (dựa imaging + clinical),
                rồi xuất JSON kết luận ở cuối. Phần reasoning đến từ base model (không train) →
                có thể không định vị slice chính xác (limitation).

allowed values khớp ĐÚNG answer_dict trong openm3chest-labels-v2.
"""

import json
import random


# ---------------------------------------------------------------------------
# Task definitions — question + allowed values (khớp answer_dict trong data)
# ---------------------------------------------------------------------------

SCREENING_TASKS = [
    "chest_abn_54", "chest_abn_55", "chest_abn_56", "chest_abn_57",
    "chest_abn_58", "chest_abn_59", "chest_abn_61", "nodule_presence",
]
DETAIL_TASKS = [
    "nodule_location", "nodule_attenuation", "nodule_margin", "nodule_size",
]

# question: mô tả finding để model biết hỏi gì
# allowed : list giá trị hợp lệ — PHẢI khớp answer_dict đã dùng lúc train
TASK_INFO: dict[str, dict] = {
    "chest_abn_54":    {"q": "Is there any atelectasis, segmental or greater?",
                        "allowed": ["No", "Yes"]},
    "chest_abn_55":    {"q": "Is there any pleural thickening or effusion?",
                        "allowed": ["No", "Yes"]},
    "chest_abn_56":    {"q": "Is there a non-calcified mass or adenopathy >=10mm "
                             "in the hilar/mediastinal region?",
                        "allowed": ["No", "Yes"]},
    "chest_abn_57":    {"q": "Is there any chest wall abnormality "
                             "(bone destruction or metastasis)?",
                        "allowed": ["No", "Yes"]},
    "chest_abn_58":    {"q": "Is there any consolidation?",
                        "allowed": ["No", "Yes"]},
    "chest_abn_59":    {"q": "Is there any emphysema?",
                        "allowed": ["No", "Yes"]},
    "chest_abn_61":    {"q": "Is there any fibrosis, honeycombing, reticular/"
                             "reticulonodular opacities, or scarring?",
                        "allowed": ["No", "Yes"]},
    "nodule_presence": {"q": "Is there any lung nodule?",
                        "allowed": ["No", "Yes"]},
    "nodule_location": {"q": "In which lobe is the nodule located?",
                        "allowed": ["Right Upper Lobe", "Right Middle Lobe",
                                    "Right Lower Lobe", "Left Upper Lobe", "Left Lower Lobe"]},
    "nodule_attenuation": {"q": "What is the predominant attenuation of the nodule?",
                           "allowed": ["Solid", "Ground Glass", "Others"]},
    "nodule_margin":   {"q": "What is the nodule margin type?",
                        "allowed": ["Spiculated (Stellate)", "Smooth",
                                    "Poorly defined", "Unable to determine"]},
    "nodule_size":     {"q": "What is the nodule size?",
                        "allowed": ["<=4mm", "4~6mm", "6~8mm",
                                    "8~15mm", "15~30mm", ">30mm"]},
}

# int label → text (để decode ground truth lúc build training data)
LABEL_MAP: dict[str, dict[int, str]] = {
    t: {i: v for i, v in enumerate(info["allowed"])}
    for t, info in TASK_INFO.items()
}

ROLE = "You are a helpful radiology assistant."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_clinical(clinical_data) -> dict:
    if isinstance(clinical_data, str):
        return json.loads(clinical_data)
    return clinical_data or {}


def _questions_block(tasks: list[str]) -> str:
    """Tạo block câu hỏi kèm allowed values, khớp format đã train."""
    lines = []
    for t in tasks:
        allowed = " | ".join(TASK_INFO[t]["allowed"])
        lines.append(f'- "{t}": {TASK_INFO[t]["q"]} (allowed: {allowed})')
    return "\n".join(lines)


def _keys_str(tasks: list[str]) -> str:
    return ", ".join(f'"{t}"' for t in tasks)


def _decode_labels(labels: dict, tasks: list[str]) -> dict[str, str]:
    return {t: LABEL_MAP[t][int(labels[t])] for t in tasks if t in labels}


# ===========================================================================
# TRAINING formatters — khớp build_radiology_ready.py (prompt + JSON ngắn)
# ===========================================================================

def _training_prompt(clinical_text: str, tasks: list[str]) -> str:
    return (
        f"{ROLE} Analyze the provided chest CT scan slices together with "
        "the patient's clinical record.\n\n"
        f"[PATIENT CLINICAL RECORD]\n{clinical_text}\n\n"
        "[QUESTIONS]\nFor each item below, choose exactly one of its allowed values:\n"
        f"{_questions_block(tasks)}\n\n"
        "[OUTPUT FORMAT]\nRespond ONLY with a single JSON object (no extra text) "
        f"using exactly these keys: {{{_keys_str(tasks)}}}."
    )


class RadiologyScreeningFormatter:
    """Training — Step 1 screening (8 task)."""
    TASKS = SCREENING_TASKS

    def format_sample(self, record: dict) -> dict:
        clinical = _parse_clinical(record.get("clinical_data"))
        from agents.data_prep.clinical_text import clinical_to_text
        answer = _decode_labels(record["labels"], self.TASKS)
        return {
            "prompt": _training_prompt(clinical_to_text(clinical), self.TASKS),
            "answer": json.dumps(answer, ensure_ascii=False),
            "keys":   record["keys"],
            "pids":   record["pids"],
        }


class RadiologyDetailFormatter:
    """Training — Step 2 detail (4 task)."""
    TASKS = DETAIL_TASKS

    def format_sample(self, record: dict) -> dict:
        clinical = _parse_clinical(record.get("clinical_data"))
        from agents.data_prep.clinical_text import clinical_to_text
        answer = _decode_labels(record["labels"], self.TASKS)
        return {
            "prompt": _training_prompt(clinical_to_text(clinical), self.TASKS),
            "answer": json.dumps(answer, ensure_ascii=False),
            "keys":   record["keys"],
            "pids":   record["pids"],
        }


# ===========================================================================
# INFERENCE formatters — reasoning chi tiết + JSON kết luận ở cuối
# ===========================================================================


class RadiologyInferenceFormatter:
    """
    Inference — dùng cho backend. Prompt khớp training format → JSON-only output.
    """

    def __init__(self, step: str):
        assert step in ("screening", "detail")
        self.step  = step
        self.tasks = SCREENING_TASKS if step == "screening" else DETAIL_TASKS

    def format_prompt(self, clinical_data) -> str:
        from agents.data_prep.clinical_text import clinical_to_text
        clinical_text = clinical_to_text(_parse_clinical(clinical_data))
        return _training_prompt(clinical_text, self.tasks)


# ---------------------------------------------------------------------------
# Parse output inference (backend dùng)
# ---------------------------------------------------------------------------

def parse_inference_output(text: str, step: str) -> dict:
    """
    Parse JSON-only output từ model (training format).
    Thử ```json fence trước, fallback {} object.
    """
    import re
    tasks = SCREENING_TASKS if step == "screening" else DETAIL_TASKS

    answer = {}
    fences = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fences[-1] if fences else None
    if candidate is None:
        objs = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
        candidate = objs[-1] if objs else None
    if candidate:
        try:
            answer = json.loads(candidate)
        except Exception:
            answer = {}

    answer = {t: answer[t] for t in tasks if t in answer}
    return {"answer": answer, "raw": text}
