"""
Verification formatter — builds the Part 1 prompt for MedGemma base (no LoRA).

Nhận parsed output của 4 agent specialist, render thành ngôn ngữ tự nhiên
dùng câu hỏi gốc từ TASK_INFO của từng agent. Model được phép sinh free-text
narrative — không ép JSON. Thought block (<unused94>thought) được giữ lại
làm phần analysis vì chứa reasoning thực sự (image grounding + cross-check).
"""

import re

from agents.cardiology.formatter import CVD_TASKS
from agents.cardiology.formatter import TASK_INFO as CARDIO_TASK_INFO
from agents.oncology.formatter import ONCOLOGY_TASKS
from agents.oncology.formatter import TASK_INFO as ONCO_TASK_INFO
from agents.radiology.formatter import DETAIL_TASKS, SCREENING_TASKS
from agents.radiology.formatter import TASK_INFO as RADIO_TASK_INFO


# ---------------------------------------------------------------------------
# Render helpers — JSON key + value → "Câu hỏi gốc  →  Kết quả"
# ---------------------------------------------------------------------------

def _render_block(answer: dict, task_list: list[str], task_info: dict) -> str:
    lines = []
    for t in task_list:
        val = answer.get(t)
        if val is not None:
            q = task_info[t]["q"]
            lines.append(f"- {q}  →  {val}")
    return "\n".join(lines) if lines else "(no data)"


# ===========================================================================
# INFERENCE formatter
# ===========================================================================

class VerificationInferenceFormatter:
    """
    Builds the verification prompt from 4 parsed agent results.

    Input từ pipeline.py sau khi parse:
        radiology_screening : {"answer": {"nodule_presence": ..., "chest_abn_54": ..., ...}}
        radiology_detail    : {"answer": {"nodule_location": ..., ...}} | None
        cardiology          : {"answer": {"CVD_diagnosis": ..., "CVD_mortality": ...}}
        oncology            : {"answer": {"lung_cancer_risk": ...}}
        finding_impression  : {"findings": "...", "impression": "..."}
    """

    def format_prompt(
        self,
        radiology_screening: dict,
        radiology_detail: dict | None,
        cardiology: dict,
        oncology: dict,
        finding_impression: dict,
    ) -> str:
        screen_ans = radiology_screening.get("answer", {})
        detail_ans = (radiology_detail or {}).get("answer", {})
        cardio_ans = cardiology.get("answer", {})
        onco_ans   = oncology.get("answer", {})
        fi_findings   = (finding_impression.get("findings") or "").strip() or "Not available"
        fi_impression = (finding_impression.get("impression") or "").strip() or "Not available"

        nodule_present = screen_ans.get("nodule_presence") == "Yes"

        screen_block = _render_block(screen_ans, SCREENING_TASKS, RADIO_TASK_INFO)
        cardio_block = _render_block(cardio_ans, CVD_TASKS, CARDIO_TASK_INFO)
        onco_block   = _render_block(onco_ans, ONCOLOGY_TASKS, ONCO_TASK_INFO)

        detail_section = ""
        if nodule_present and detail_ans:
            detail_block = _render_block(detail_ans, DETAIL_TASKS, RADIO_TASK_INFO)
            detail_section = f"\n\n[RADIOLOGY — Nodule Characteristics]\n{detail_block}"

        nodule_instruction = (
            "2. Nodule characteristics (location, size, attenuation, margin) "
            "— check whether the reported details match what you observe"
            if nodule_present
            else
            "2. Nodule characteristics — N/A (no nodule was reported)"
        )

        return (
            "You are an experienced physician conducting an independent multidisciplinary review of a chest CT scan.\n\n"
            "The following conclusions were reported by specialist AI agents for this scan:\n\n"
            f"[RADIOLOGY — Chest Findings]\n{screen_block}"
            f"{detail_section}\n\n"
            f"[CARDIOLOGY]\n{cardio_block}\n\n"
            f"[ONCOLOGY]\n{onco_block}\n\n"
            "[IMAGING REPORT (Agent 4)]\n"
            f"Findings:\n{fi_findings}\n\n"
            f"Impression:\n{fi_impression}\n\n"
            "---\n"
            "Carefully examine ALL provided CT scan images.\n\n"
            "Provide your independent clinical assessment. You MUST address ALL 6 "
            "points below, IN ORDER, as a numbered list (1. through 6.) — do not "
            "skip, merge, or reorder any point, and do not write a single free-flowing "
            "paragraph instead:\n"
            "1. Chest findings — do the CT images support the reported nodule presence, "
            "atelectasis, effusion, consolidation, emphysema, fibrosis, masses?\n"
            f"{nodule_instruction}\n"
            "3. Cardiovascular — do you see abnormalities consistent with the reported findings?\n"
            "4. Oncology risk — does the visual evidence support the reported cancer risk level?\n"
            "5. Imaging report accuracy — does the Findings/Impression text match what you observe?\n"
            "6. Unreported findings — note anything significant visible in the images "
            "that was NOT mentioned by any agent above. If none, state \"None.\"\n\n"
            "Example of the REQUIRED format (content is illustrative only):\n"
            "1. Chest findings: - Atelectasis: None reported, no evidence seen. "
            "- Pleural effusion: ...\n"
            "2. Nodule characteristics: ...\n"
            "3. Cardiovascular: ...\n"
            "4. Oncology risk: ...\n"
            "5. Imaging report accuracy: ...\n"
            "6. Unreported findings: ...\n"
        )


# ---------------------------------------------------------------------------
# Parse output inference (backend dùng)
# ---------------------------------------------------------------------------

# MedGemma-IT chain-of-thought token — model reasoning nằm trong block này
_THOUGHT_RE = re.compile(r"<unused94>thought\s*(.*?)(?:</unused94>|$)", re.DOTALL)


def parse_inference_output(text: str) -> dict:
    """Extract analysis từ MedGemma base output.

    Ưu tiên lấy nội dung <unused94>thought block (chain-of-thought reasoning
    thực sự, có image grounding + cross-check). Nếu có thêm text sau block
    thì nối vào sau. Fallback về raw text nếu không có thought block.
    """
    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group(1).strip()
        after = text[m.end():].strip()
        analysis = (thought + "\n\n" + after).strip() if after else thought
    else:
        analysis = text.strip() or None

    return {"analysis": analysis, "raw": text}
