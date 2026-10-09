"""Loopback-only keyless bridge to IAM-protected Cloud Run."""

import asyncio
import os
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .gcp import GoogleCredentials


def create_bridge(url=None, credentials=None, transport=None):
    app = FastAPI(title="Theriad local Cloud Run bridge")
    url = url or os.environ.get("THERIAD_GATEWAY_URL", "")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("An HTTPS Cloud Run service URL is required")
    credentials = credentials or GoogleCredentials(audience=url)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse(status_code=403, content={"error": "loopback_only"})
        if path not in {
            "health",
            "ready",
            "v1/models",
            "v1/chat/completions",
            "v1/responses",
            "v1/responses/compact",
        } and not path.startswith("v1/theriad/episodes/"):
            return JSONResponse(status_code=404, content={"error": "route_not_found"})
        token = await asyncio.to_thread(credentials.token)
        headers = {
            "X-Serverless-Authorization": "Bearer " + token,
            "Authorization": request.headers.get("authorization", ""),
            "Content-Type": "application/json",
        }
        client = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(600, read=120), follow_redirects=False
        )
        try:
            upstream = await client.send(
                client.build_request(
                    request.method,
                    url.rstrip("/")
                    + "/"
                    + path
                    + ("?" + request.url.query if request.url.query else ""),
                    headers=headers,
                    content=await request.body(),
                ),
                stream=True,
            )
        except httpx.HTTPError:
            await client.aclose()
            return JSONResponse(status_code=502, content={"error": "gateway_unavailable"})

        async def body():
            try:
                async for chunk in upstream.aiter_bytes():
                    if await request.is_disconnected():
                        return
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )

    return app
