"""
Oncology agent — serving runtime: build prompt → gọi Modal → parse output.

1 task duy nhất (lung_cancer_risk — nguy cơ ung thư phổi trong 6 năm tới),
nên chỉ cần 1 lượt gọi, không sequential như radiology screening/detail.
"""

import httpx

from serving.backend.modal_client import call_modal

from .formatter import OncologyInferenceFormatter, parse_inference_output

_formatter = OncologyInferenceFormatter()


async def run(
    client: httpx.AsyncClient,
    modal_url: str,
    clinical_data: dict,
    cache_key: str,
    image_urls: list[str],
) -> dict:
    prompt = _formatter.format_prompt(clinical_data)
    raw = await call_modal(client, modal_url, "oncology", prompt, cache_key, image_urls)
    return parse_inference_output(raw["output"])
