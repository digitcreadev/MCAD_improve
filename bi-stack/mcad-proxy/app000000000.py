from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

app = FastAPI(title="MCAD proxy", version="ui-fix-2026-04-24")

MCAD_API_BASE = os.getenv("MCAD_API_BASE", "http://mcad-api:8000").rstrip("/")
MCAD_EVAL_URL = os.getenv("MCAD_EVAL_URL", f"{MCAD_API_BASE}/eval")
MCAD_CKG_URL = os.getenv("MCAD_CKG_URL", f"{MCAD_API_BASE}/ckg/update")
PIVOT4J_UPSTREAM = os.getenv("PIVOT4J_UPSTREAM", "http://pivot4j:8080/pivot4j").rstrip("/")
PIVOT4J_PUBLIC_PREFIX = os.getenv("PIVOT4J_PUBLIC_PREFIX", "/pivot4j").rstrip("/")
SESSION_UI_PATH = Path(__file__).with_name("session_ui.html")

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _clean_headers(headers: httpx.Headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _HOP_BY_HOP:
            continue
        out[k] = v
    return out


def _rewrite_location(location: str) -> str:
    if not location:
        return location
    if location.startswith(PIVOT4J_UPSTREAM):
        suffix = location[len(PIVOT4J_UPSTREAM):]
        return f"{PIVOT4J_PUBLIC_PREFIX}{suffix or '/'}"
    if location.startswith("/pivot4j"):
        return location
    if location.startswith("/"):
        return f"{PIVOT4J_PUBLIC_PREFIX}{location}"
    return location


def _rewrite_html(body: str) -> str:
    # only rewrite root-absolute assets/forms, not already-prefixed paths
    body = re.sub(r'(?P<a>(?:href|src|action)=(["\']))/(?!pivot4j/)', r'\g<a>pivot4j/', body)
    body = body.replace('url(/', 'url(/pivot4j/')
    body = body.replace('"/javax.faces.resource/', '"/pivot4j/javax.faces.resource/')
    body = body.replace("'/javax.faces.resource/", "'/pivot4j/javax.faces.resource/")
    return body


async def _mcad_get(path: str) -> Any:
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f"{MCAD_API_BASE}{path}")
        r.raise_for_status()
        return r.json()


@app.get("/mcad/objectives")
async def mcad_objectives() -> JSONResponse:
    return JSONResponse(await _mcad_get("/objectives"))


@app.get("/mcad/datawarehouses")
async def mcad_datawarehouses() -> JSONResponse:
    return JSONResponse(await _mcad_get("/datawarehouses"))


@app.get("/mcad/sessions")
async def mcad_sessions() -> JSONResponse:
    return JSONResponse(await _mcad_get("/sessions"))


@app.get("/mcad/session/current")
async def mcad_session_current() -> JSONResponse:
    return JSONResponse(await _mcad_get("/session/current"))


@app.api_route("/mcad/session/create", methods=["POST"])
async def mcad_session_create(request: Request) -> JSONResponse:
    payload = await request.json()
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{MCAD_API_BASE}/session/create", json=payload)
        return JSONResponse(r.json(), status_code=r.status_code)


@app.api_route("/mcad/session/resume", methods=["POST"])
async def mcad_session_resume(request: Request) -> JSONResponse:
    payload = await request.json()
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(f"{MCAD_API_BASE}/session/resume", json=payload)
        return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/mcad/session/ui", response_class=HTMLResponse)
async def mcad_session_ui() -> HTMLResponse:
    if not SESSION_UI_PATH.exists():
        raise HTTPException(status_code=500, detail=f"session_ui.html not found: {SESSION_UI_PATH}")
    return HTMLResponse(SESSION_UI_PATH.read_text(encoding="utf-8"))


@app.api_route("/xmla", methods=["GET", "POST"])
async def xmla_proxy(request: Request) -> Response:
    # Preserve the existing XMLA path used by Pivot4J.
    body = await request.body()
    headers = _clean_headers(request.headers)
    async with httpx.AsyncClient(timeout=120.0) as client:
        upstream = await client.request(
            request.method,
            os.getenv("EMONDRIAN_XMLA_URL", "http://emondrian:8080/emondrian/xmla"),
            content=body,
            headers=headers,
        )
    out_headers = _clean_headers(upstream.headers)
    return Response(content=upstream.content, status_code=upstream.status_code, headers=out_headers, media_type=upstream.headers.get("content-type"))


@app.get("/pivot4j")
async def pivot4j_root_redirect() -> RedirectResponse:
    # Normalize once on the public side; the proxied route below must return 200, not another public redirect loop.
    return RedirectResponse(url=f"{PIVOT4J_PUBLIC_PREFIX}/", status_code=307)


@app.api_route("/pivot4j/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def pivot4j_proxy(full_path: str, request: Request) -> Response:
    query = request.url.query
    suffix = full_path or ""
    upstream_url = f"{PIVOT4J_UPSTREAM}/{suffix}" if suffix else f"{PIVOT4J_UPSTREAM}/"
    if query:
        upstream_url = f"{upstream_url}?{query}"

    body = await request.body()
    headers = _clean_headers(request.headers)
    headers["X-Forwarded-Proto"] = request.url.scheme
    headers["X-Forwarded-Host"] = request.headers.get("host", "")
    headers["X-Forwarded-Prefix"] = PIVOT4J_PUBLIC_PREFIX

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        upstream = await client.request(
            request.method,
            upstream_url,
            content=body,
            headers=headers,
            cookies=request.cookies,
        )

    out_headers = _clean_headers(upstream.headers)
    if "location" in {k.lower() for k in upstream.headers.keys()}:
        for k, v in list(out_headers.items()):
            if k.lower() == "location":
                out_headers[k] = _rewrite_location(v)

    for k, v in list(out_headers.items()):
        if k.lower() == "set-cookie":
            if "Path=/" in v and f"Path={PIVOT4J_PUBLIC_PREFIX}" not in v:
                out_headers[k] = v.replace("Path=/", f"Path={PIVOT4J_PUBLIC_PREFIX}/")

    media_type = upstream.headers.get("content-type", "")
    content = upstream.content
    if "text/html" in media_type.lower():
        text = upstream.text
        text = _rewrite_html(text)
        content = text.encode("utf-8")
        out_headers["content-length"] = str(len(content))

    return Response(content=content, status_code=upstream.status_code, headers=out_headers, media_type=media_type or None)


@app.get("/")
async def root() -> PlainTextResponse:
    return PlainTextResponse("mcad-proxy ok")
