"""
Radiology agent — serving runtime: build prompt → gọi Modal → parse output.

2 bước (sequential, khớp cách fine-tune):
  - screening : luôn chạy, 8 task (chest_abn_5x + nodule_presence)
  - detail    : chỉ chạy khi screening trả nodule_presence == "Yes", 4 task
"""

import httpx

from serving.backend.modal_client import call_modal

from .formatter import RadiologyInferenceFormatter, parse_inference_output

_screening_formatter = RadiologyInferenceFormatter(step="screening")
_detail_formatter = RadiologyInferenceFormatter(step="detail")


async def run_screening(
    client: httpx.AsyncClient,
    modal_url: str,
    clinical_data: dict,
    cache_key: str,
    image_urls: list[str],
) -> dict:
    prompt = _screening_formatter.format_prompt(clinical_data)
    raw = await call_modal(client, modal_url, "radiology", prompt, cache_key, image_urls)
    return parse_inference_output(raw["output"], "screening")


async def run_detail(
    client: httpx.AsyncClient,
    modal_url: str,
    clinical_data: dict,
    cache_key: str,
    image_urls: list[str],
) -> dict:
    prompt = _detail_formatter.format_prompt(clinical_data)
    raw = await call_modal(client, modal_url, "radiology", prompt, cache_key, image_urls)
    return parse_inference_output(raw["output"], "detail")
