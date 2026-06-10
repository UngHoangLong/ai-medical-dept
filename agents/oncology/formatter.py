"""
Oncology formatters — build prompt cho Oncology agent (MedGemma 1.5 LoRA).

1 task duy nhất:
  - lung_cancer_risk : nguy cơ ung thư phổi trong 6 năm tới
                       (No cancer within follow-up | Cancer within follow-up)

Prompt template + allowed values khớp ĐÚNG dataset đã dùng để fine-tune
(HF: Phiphi216/oncology) — verify trực tiếp từ datasets-server API,
ví dụ response thật: {"lung_cancer_risk": "No cancer within follow-up"}
"""

import json

ONCOLOGY_TASKS = ["lung_cancer_risk"]

# question: lấy nguyên văn 1 trong các paraphrase đã xuất hiện lúc train
# (xem questions[] trong lung_cancer_risk_trainval.json — câu hỏi dưới đây
# trùng khớp prompt thật lấy từ Phiphi216/oncology)
# allowed : list giá trị hợp lệ — khớp DEFAULT_ANSWER_DICT trong build_labels_subset.py
TASK_INFO: dict[str, dict] = {
    "lung_cancer_risk": {
        "q": "Given the current indicators, what's the six-year outlook for developing lung cancer?",
        "allowed": ["No cancer within follow-up", "Cancer within follow-up"],
    },
}

# int label (labels["y"]: False/True → 0/1) → text — khớp DEFAULT_ANSWER_DICT
# trong build_labels_subset.py (lung_cancer_risk không có answer_dict riêng trong data_json)
LABEL_MAP: dict[str, dict[int, str]] = {
    "lung_cancer_risk": {0: "No cancer within follow-up", 1: "Cancer within follow-up"},
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
    """labels["y"] là bool (False/True) — convert sang int rồi map sang text."""
    return {t: LABEL_MAP[t][int(bool(labels["y"]))] for t in tasks}


def _training_prompt(clinical_text: str, tasks: list[str]) -> str:
    """Khớp ĐÚNG template trong Phiphi216/oncology (verify từ datasets-server)."""
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

class OncologyFormatter:
    """Training — single-task lung_cancer_risk (binary y → 6-year risk)."""
    TASKS = ONCOLOGY_TASKS

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

class OncologyInferenceFormatter:
    """Inference — dùng cho backend. Prompt khớp training format → JSON-only output."""

    def format_prompt(self, clinical_data) -> str:
        from agents.data_prep.clinical_text import clinical_to_text
        clinical_text = clinical_to_text(_parse_clinical(clinical_data))
        return _training_prompt(clinical_text, ONCOLOGY_TASKS)


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

    answer = {t: answer[t] for t in ONCOLOGY_TASKS if t in answer}
    return {"answer": answer, "raw": text}
