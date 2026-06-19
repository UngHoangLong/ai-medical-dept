"""
Highlighter client — gọi LLM text-only (Groq/Gemini/Anthropic/OpenAI) để chọn
ra các cụm từ quan trọng trong free-text report (Findings/Impression/
Verification), backend tự chèn <hl c="critical|warning|normal"> quanh các
cụm đó trong text gốc, giúp frontend highlight cho bác sĩ.

Provider chọn qua biến env HIGHLIGHTER_PROVIDER ("groq" | "gemini" | "anthropic" | "openai"),
model chọn qua HIGHLIGHTER_MODEL — đổi nhà cung cấp không cần sửa code,
chỉ cần đổi .env (và cài package tương ứng).

Chỉ nhận text — KHÔNG liên quan ảnh CT, an toàn để gọi external API.
Model không bao giờ chép lại/sửa text gốc — chỉ trả về danh sách cụm từ +
màu, nên lỗi/timeout/model trả JSON sai → fallback trả về text gốc (không
highlight), không bao giờ làm sập pipeline chính.
"""

import asyncio
import logging
import os

from agents.highlighter.formatter import (
    RETRY_NOTE,
    SYSTEM_PROMPT,
    HighlighterFormatter,
    apply_highlights,
    parse_highlights,
)

logger = logging.getLogger(__name__)

_fmt = HighlighterFormatter()

PROVIDER = os.environ.get("HIGHLIGHTER_PROVIDER", "groq").lower()

_DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek": "deepseek-chat",
}
MODEL = os.environ.get("HIGHLIGHTER_MODEL", _DEFAULT_MODELS.get(PROVIDER, ""))

_client = None


def _get_client():
    global _client
    if _client is None:
        if PROVIDER == "groq":
            from openai import AsyncOpenAI
            _client = AsyncOpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
                max_retries=0,
            )
        elif PROVIDER == "gemini":
            from google import genai
            _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        elif PROVIDER == "anthropic":
            from anthropic import AsyncAnthropic
            _client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)
        elif PROVIDER == "openai":
            from openai import AsyncOpenAI
            _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0)
        elif PROVIDER == "openrouter":
            from openai import AsyncOpenAI
            _client = AsyncOpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
                max_retries=0,
            )
        elif PROVIDER == "deepseek":
            from openai import AsyncOpenAI
            _client = AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
                max_retries=0,
            )
        else:
            raise ValueError(f"Unknown HIGHLIGHTER_PROVIDER: {PROVIDER!r}")
    return _client


async def _call_gemini(text: str, retry_note: str = "") -> str:
    from google.genai import types

    resp = await _get_client().aio.models.generate_content(
        model=MODEL,
        contents=text + retry_note,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    return resp.text or ""


async def _call_anthropic(text: str, retry_note: str = "") -> str:
    resp = await _get_client().messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text + retry_note}],
        timeout=30.0,
    )
    return resp.content[0].text


async def _call_openai(text: str, retry_note: str = "") -> str:
    resp = await _get_client().chat.completions.create(
        model=MODEL,
        messages=_fmt.format_messages(text, retry_note),
        temperature=0.0,
        timeout=30.0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


_CALL_FNS = {
    "groq": _call_openai,  # Groq dùng API tương thích OpenAI, chỉ khác base_url
    "gemini": _call_gemini,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "openrouter": _call_openai,  # OpenRouter cũng tương thích OpenAI, chỉ khác base_url
    "deepseek": _call_openai,  # DeepSeek cũng tương thích OpenAI, chỉ khác base_url
}


async def call_llm_json(
    *,
    system_prompt: str,
    user_text: str,
    retry_note: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: float = 30.0,
    model: str | None = None,
) -> str:
    """Generic text-only LLM JSON call shared by highlighter and VoxTell auto prompt.

    This reuses the same provider/client/env configuration as the highlighter:
    HIGHLIGHTER_PROVIDER and HIGHLIGHTER_MODEL, unless `model` is passed.
    It returns the raw JSON string from the model. The caller is responsible
    for schema-specific parsing and fallback.
    """
    selected_model = model or MODEL
    text = user_text + retry_note

    if PROVIDER == "gemini":
        from google.genai import types

        resp = await _get_client().aio.models.generate_content(
            model=selected_model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return resp.text or ""

    if PROVIDER == "anthropic":
        resp = await _get_client().messages.create(
            model=selected_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": text}],
            timeout=timeout,
        )
        return resp.content[0].text

    # Groq, OpenAI, OpenRouter and DeepSeek are OpenAI-compatible clients here.
    resp = await _get_client().chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=temperature,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""


async def highlight_text(text: str | None) -> str | None:
    """Annotate 1 đoạn text. Trả về text gốc nếu lỗi hoặc model trả JSON sai format.

    Model chỉ trả về danh sách cụm từ + màu (JSON), backend tự chèn <hl> vào
    text gốc — text gốc không bao giờ bị model động vào. Nếu JSON parse lỗi,
    retry 1 lần với note nhắc mạnh hơn trước khi fallback về text gốc.
    """
    if not text or not text.strip():
        return text

    call = _CALL_FNS[PROVIDER]
    try:
        raw = await call(text)
        highlights = parse_highlights(raw)
    except Exception:
        try:
            raw = await call(text, RETRY_NOTE)
            highlights = parse_highlights(raw)
        except Exception:
            logger.exception("[Highlighter] failed — fallback to plain text")
            return text

    return apply_highlights(text, highlights)


async def highlight_many(texts: dict[str, str | None]) -> dict[str, str | None]:
    """Annotate nhiều đoạn text song song. Giữ nguyên key, value None giữ None."""
    keys = list(texts.keys())
    results = await asyncio.gather(*(highlight_text(texts[k]) for k in keys))
    return dict(zip(keys, results))
