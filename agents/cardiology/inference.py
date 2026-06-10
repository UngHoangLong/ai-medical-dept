"""
Cardiology agent — serving runtime: build prompt → gọi Modal → parse output.

1 bước duy nhất (khớp cách fine-tune — CVD_diagnosis + CVD_mortality
hỏi gộp trong cùng 1 lượt, không sequential như radiology screening/detail).
"""

import httpx

from serving.backend.modal_client import call_modal

from .formatter import CardiologyInferenceFormatter, parse_inference_output

_formatter = CardiologyInferenceFormatter()


async def run(
    client: httpx.AsyncClient,
    modal_url: str,
    clinical_data: dict,
    cache_key: str,
    image_urls: list[str],
) -> dict:
    prompt = _formatter.format_prompt(clinical_data)
    raw = await call_modal(client, modal_url, "cardiology", prompt, cache_key, image_urls)
    return parse_inference_output(raw["output"])
