"""
Finding & Impression agent — serving runtime: gọi Modal → parse output.

Khác với 3 agent kia: prompt CỐ ĐỊNH (không cần clinical_data, không có
JSON Q&A) — model sinh report narrative (Findings + Impression) trực tiếp
từ ảnh CT. Output dài hơn nhiều so với JSON ngắn của các agent chẩn đoán
nên cần max_new_tokens lớn hơn default.
"""

import httpx

from serving.backend.modal_client import call_modal

from .formatter import FindingImpressionInferenceFormatter, parse_inference_output

_formatter = FindingImpressionInferenceFormatter()

MAX_NEW_TOKENS = 1024


async def run(
    client: httpx.AsyncClient,
    modal_url: str,
    cache_key: str,
    image_urls: list[str],
) -> dict:
    prompt = _formatter.format_prompt()
    raw = await call_modal(
        client, modal_url, "finding_impression", prompt, cache_key, image_urls,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return parse_inference_output(raw["output"])
