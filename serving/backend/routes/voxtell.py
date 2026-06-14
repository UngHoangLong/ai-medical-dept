import os
from typing import Annotated
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/voxtell", tags=["voxtell"])

REQUEST_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=1800.0,
    write=300.0,
    pool=30.0,
)


def _get_modal_base_url() -> str:
    base_url = os.getenv("VOXTELL_MODAL_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise HTTPException(
            status_code=500,
            detail="VOXTELL_MODAL_BASE_URL is not configured.",
        )
    return base_url


def _modal_url(path: str) -> str:
    return f"{_get_modal_base_url()}{path}"


def _raise_modal_error(status_code: int, body: bytes) -> None:
    text = body.decode("utf-8", errors="replace")
    detail = text[:1000] if text else "VoxTell Modal request failed."
    raise HTTPException(status_code=status_code, detail=detail)


@router.get("/volume/{pid}/{series_uid:path}")
async def get_voxtell_volume(pid: str, series_uid: str):
    pid = pid.strip()
    series_uid = series_uid.strip()

    if not pid or not series_uid:
        raise HTTPException(status_code=400, detail="pid and series_uid are required.")

    upstream_path = (
        f"/voxtell/volume/{quote(pid, safe='')}/{quote(series_uid, safe='')}"
    )

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:
        try:
            upstream = await client.get(_modal_url(upstream_path))
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Cannot reach VoxTell Modal service: {exc}",
            ) from exc

    if upstream.status_code >= 400:
        _raise_modal_error(upstream.status_code, upstream.content)

    headers = {}
    content_disposition = upstream.headers.get("content-disposition")
    if content_disposition:
        headers["content-disposition"] = content_disposition

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "application/gzip",
        headers=headers,
    )


@router.post("/predict")
async def voxtell_predict(
    pid: Annotated[str, Form()],
    series_uid: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
):
    pid = pid.strip()
    series_uid = series_uid.strip()
    prompt = prompt.strip()

    if not pid or not series_uid:
        raise HTTPException(status_code=400, detail="pid and series_uid are required.")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required.")

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        follow_redirects=True,
    ) as client:
        try:
            upstream = await client.post(
                _modal_url("/voxtell/predict"),
                data={
                    "pid": pid,
                    "series_uid": series_uid,
                    "prompt": prompt,
                },
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Cannot reach VoxTell Modal service: {exc}",
            ) from exc

    if upstream.status_code >= 400:
        _raise_modal_error(upstream.status_code, upstream.content)

    headers = {}
    content_disposition = upstream.headers.get("content-disposition")
    if content_disposition:
        headers["content-disposition"] = content_disposition

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type") or "application/gzip",
        headers=headers,
    )