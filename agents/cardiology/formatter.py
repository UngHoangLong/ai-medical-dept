"""
Cardiology formatters — build prompt cho Cardiology agent (MedGemma 1.5 LoRA).

1 bước duy nhất (khớp cách fine-tune — 2 task hỏi gộp trong cùng 1 lượt,
không sequential như radiology screening/detail):
  - CVD_diagnosis : có bất thường tim mạch đáng chú ý không  (No | Yes)
  - CVD_mortality : nguy cơ tử vong do bệnh tim mạch         (Low risk | High risk)

Prompt template + allowed values khớp ĐÚNG dataset đã dùng để fine-tune
(HF: ChonJohn171105/cardiology_ready_finetune) — verify trực tiếp từ
datasets-server API, ví dụ response thật: {"CVD_diagnosis": "Yes", "CVD_mortality": "Low risk"}
"""

import json

CVD_TASKS = ["CVD_diagnosis", "CVD_mortality"]

# question: lấy nguyên văn 1 trong các paraphrase đã xuất hiện lúc train
# (xem questions[] trong CVD_diagnosis_trainval.json / CVD_mortality_trainval.json)
# allowed : list giá trị hợp lệ — PHẢI khớp answer_dict / quy ước đã dùng lúc train
TASK_INFO: dict[str, dict] = {
    "CVD_diagnosis": {"q": "Can you identify any notable abnormalities in the cardiovascular system?",
                      "allowed": ["No", "Yes"]},
    "CVD_mortality": {"q": "Predict the risk of cardiovascular disease mortality.",
                      "allowed": ["Low risk", "High risk"]},
}

# int label → text — CVD_diagnosis khớp answer_dict trong CVD_diagnosis_trainval.json;
# CVD_mortality không có answer_dict riêng trong data_json, map theo đúng giá trị
# "Low risk"/"High risk" thấy trong response thật của cardiology_ready_finetune.
LABEL_MAP: dict[str, dict[int, str]] = {
    "CVD_diagnosis": {0: "No",        1: "Yes"},
    "CVD_mortality": {0: "Low risk",  1: "High risk"},
}

ROLE = "You are a helpful medical assistant."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_clinical(clinical_data) -> dict:
    if isinstance(clinical_data, str):
        return json.loads(clinical_data)
    return clinical_data or {}


def _questions_block(tasks: list[str]) -> str:
    lines = []
    for t in tasks:
        allowed = " | ".join(TASK_INFO[t]["allowed"])
        lines.append(f'- "{t}": {TASK_INFO[t]["q"]} (allowed: {allowed})')
    return "\n".join(lines)


def _keys_str(tasks: list[str]) -> str:
    return ", ".join(f'"{t}"' for t in tasks)


def _decode_labels(labels: dict, tasks: list[str]) -> dict[str, str]:
    return {t: LABEL_MAP[t][int(labels[t])] for t in tasks if t in labels}


def _training_prompt(clinical_text: str, tasks: list[str]) -> str:
    """Khớp ĐÚNG template trong cardiology_ready_finetune (verify từ datasets-server)."""
    return (
        f"{ROLE} Analyze the provided chest CT scan slices together with "
        "the patient's clinical record.\n\n"
        f"[PATIENT CLINICAL RECORD]\n{clinical_text}\n\n"
        "[QUESTIONS]\nFor each item below, choose exactly one of its allowed values:\n"
        f"{_questions_block(tasks)}\n\n"
        "[OUTPUT FORMAT]\nRespond ONLY with a single JSON object (no extra text) "
        f"using exactly these keys: {{{_keys_str(tasks)}}}."
    )


# ===========================================================================
# TRAINING formatter — dùng khi build dataset ready (giống build_radiology_ready.py)
# ===========================================================================

class CardiologyFormatter:
    """Training — multi-task CVD_diagnosis + CVD_mortality, hỏi gộp 1 lượt."""
    TASKS = CVD_TASKS

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
# INFERENCE formatter — dùng cho backend
# ===========================================================================

class CardiologyInferenceFormatter:
    """Inference — dùng cho backend. Prompt khớp training format → JSON-only output."""

    def format_prompt(self, clinical_data) -> str:
        from agents.data_prep.clinical_text import clinical_to_text
        clinical_text = clinical_to_text(_parse_clinical(clinical_data))
        return _training_prompt(clinical_text, CVD_TASKS)


# ---------------------------------------------------------------------------
# Parse output inference (backend dùng)
# ---------------------------------------------------------------------------

def parse_inference_output(text: str) -> dict:
    """
    Parse JSON-only output từ model (training format).
    Thử ```json fence trước, fallback {} object.
    """
    import re

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

    answer = {t: answer[t] for t in CVD_TASKS if t in answer}
    return {"answer": answer, "raw": text}
