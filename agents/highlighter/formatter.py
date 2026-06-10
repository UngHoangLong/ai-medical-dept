"""
Highlighter formatter — build prompt (text-only) để model chọn ra các cụm từ
quan trọng trong free-text report (Findings/Impression/Verification) giúp
bác sĩ dễ nắm thông tin nổi bật trên front-end.

Khác với cách cũ (bắt model chép lại toàn bộ text + chèn tag — dễ fail vì
model "viết lại" vài chữ), cách này chỉ yêu cầu model trả về DANH SÁCH các
cụm từ + màu (JSON). Backend tự tìm các cụm đó trong text gốc (exact match)
và chèn <hl c="...">...</hl> — text gốc không bao giờ bị model động vào,
cụm nào model paraphrase sai/không tìm thấy thì bị bỏ qua (không làm fail
toàn bộ đoạn).

SYSTEM_PROMPT dùng chung cho mọi provider (Anthropic, OpenAI, Groq, Gemini...).
Model KHÔNG nhận ảnh CT — chỉ nhận text đã có sẵn từ pipeline, an toàn để
gọi external API.
"""

import json
import re

SYSTEM_PROMPT = (
    "You are a clinical text annotation assistant. You will be given a "
    "radiology/clinical report written for a doctor.\n\n"
    "Your task: identify clinically significant phrases in the text and "
    "classify each one into one of these categories:\n"
    '  "critical" - malignancy, cancer, metastasis, high-risk findings\n'
    '  "warning"  - abnormal findings needing attention (nodules, masses, '
    "effusion, fibrosis, emphysema, consolidation, etc.)\n"
    '  "normal"   - explicitly normal/negative findings worth confirming to '
    "the doctor\n\n"
    "Rules:\n"
    "1. Each phrase must be an EXACT, VERBATIM substring of the input text "
    "(same wording, casing, punctuation, spacing) - it will be searched for "
    "in the original text and must match exactly.\n"
    "2. Pick short, meaningful phrases (a few words), not whole sentences "
    "or paragraphs.\n"
    "3. If nothing is significant, return an empty list.\n\n"
    'Return ONLY a JSON object of the form: {"highlights": '
    '[{"phrase": "...", "category": "critical"|"warning"|"normal"}, ...]}'
    " - no other text."
)

RETRY_NOTE = (
    "\n\n[REMINDER: your previous response was not a valid JSON object in "
    'the required format. Return ONLY {"highlights": [{"phrase": "...", '
    '"category": "critical"|"warning"|"normal"}, ...]} - no other text.]'
)


class HighlighterFormatter:
    """Build messages cho OpenAI chat completion — text-only."""

    def format_messages(self, text: str, retry_note: str = "") -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text + retry_note},
        ]


_VALID_CATEGORIES = {"critical", "warning", "normal"}
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_highlights(raw: str) -> list[dict]:
    """
    Parse JSON {"highlights": [{"phrase": ..., "category": ...}, ...]} từ
    model output. Raise nếu không parse được (caller sẽ retry/fallback).
    """
    match = _JSON_OBJ_RE.search(raw)
    data = json.loads(match.group() if match else raw)
    items = data["highlights"]
    return [
        h for h in items
        if isinstance(h, dict)
        and isinstance(h.get("phrase"), str) and h["phrase"]
        and h.get("category") in _VALID_CATEGORIES
    ]


def apply_highlights(text: str, highlights: list[dict]) -> str:
    """
    Tìm tất cả vị trí xuất hiện (exact match) của mỗi phrase trong text gốc
    và chèn <hl c="category">...</hl>. Phrase không tìm thấy trong text bị
    bỏ qua. Các match chồng lấn nhau: ưu tiên match đến trước / dài hơn.
    """
    spans = []
    for h in highlights:
        phrase, category = h["phrase"], h["category"]
        for m in re.finditer(re.escape(phrase), text):
            spans.append((m.start(), m.end(), category))

    if not spans:
        return text

    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))

    selected = []
    last_end = 0
    for start, end, category in spans:
        if start < last_end:
            continue
        selected.append((start, end, category))
        last_end = end

    parts = []
    cursor = 0
    for start, end, category in selected:
        parts.append(text[cursor:start])
        parts.append(f'<hl c="{category}">{text[start:end]}</hl>')
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)
