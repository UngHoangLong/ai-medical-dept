"""
Modal client — gọi sang Modal inference endpoints.

  call_modal()          : gọi endpoint /infer (1 agent tại 1 thời điểm)
                          — dùng cho standalone test / /ask chatbot
  call_modal_pipeline() : gọi endpoint /infer_batch (toàn bộ pipeline 1 lần)
                          — dùng cho /analyze; KV prefix cache giữ nguyên
                            suốt 5 agent, không bị evict bởi request khác
"""

import logging
import time

import httpx

logger = logging.getLogger(__name__)


async def call_modal(
    client: httpx.AsyncClient,
    modal_url: str,
    adapter: str,
    prompt: str,
    cache_key: str,
    image_urls: list[str] | None = None,
    max_new_tokens: int = 512,
) -> dict:
    n_urls = len(image_urls) if image_urls else 0
    logger.info("[Modal] → adapter=%s  cache_key=%s  image_urls=%d  max_new_tokens=%d",
                adapter, cache_key, n_urls, max_new_tokens)
    logger.debug("[Modal] prompt:\n%s", prompt)

    payload = {
        "adapter": adapter,
        "prompt": prompt,
        "cache_key": cache_key,
        "max_new_tokens": max_new_tokens,
    }
    if image_urls is not None:
        payload["image_urls"] = image_urls

    t0 = time.perf_counter()
    resp = await client.post(modal_url, json=payload, timeout=300.0)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()

    result = resp.json()
    logger.info("[Modal] ← adapter=%s  cached=%s  elapsed=%.1fs  output_len=%d chars",
                adapter, result.get("cached"), elapsed, len(result.get("output", "")))
    logger.debug("[Modal] output:\n%s", result.get("output", ""))
    return result


async def call_modal_pipeline(
    client: httpx.AsyncClient,
    pipeline_url: str,
    tasks: list[dict],
    cache_key: str,
    image_urls: list[str] | None = None,
) -> dict:
    """Gọi endpoint /infer_batch — chạy toàn bộ pipeline trong 1 HTTP request."""
    task_ids = [t["id"] for t in tasks]
    n_urls = len(image_urls) if image_urls else 0
    logger.info("[Modal Pipeline] → cache_key=%s  tasks=%s  image_urls=%d",
                cache_key, task_ids, n_urls)

    payload: dict = {"cache_key": cache_key, "tasks": tasks}
    if image_urls:
        payload["image_urls"] = image_urls

    t0 = time.perf_counter()
    resp = await client.post(pipeline_url, json=payload, timeout=900.0)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()

    result = resp.json()
    done = [k for k, v in result.get("results", {}).items() if v is not None]
    logger.info("[Modal Pipeline] ← elapsed=%.1fs  cached=%s  ran=%s",
                elapsed, result.get("cached"), done)
    return result
