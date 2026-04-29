from __future__ import annotations

from fastapi import FastAPI, Request, Response
import os
import time
import requests
import hashlib
import re
from lxml import etree

app = FastAPI(title="MCAD XMLA Proxy", version="1.1.0")

UPSTREAM = os.getenv("UPSTREAM_XMLA", "http://emondrian:8080/emondrian/xmla")
MCAD_EVAL_URL = os.getenv("MCAD_EVAL_URL", "http://mcad-api:8000/eval")
MCAD_CKG_URL = os.getenv("MCAD_CKG_URL", "http://mcad-api:8000/ckg/update")

SOAP_FAULT = """<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <soap:Fault>
      <faultcode>soap:Client</faultcode>
      <faultstring>{msg}</faultstring>
    </soap:Fault>
  </soap:Body>
</soap:Envelope>
"""


def mdx_fingerprint(mdx: str) -> str:
    return hashlib.sha256(mdx.encode("utf-8")).hexdigest()[:16]


_JSESSION_RE = re.compile(r"(?:^|;\s*)JSESSIONID=([^;]+)")
def extract_session_id(req: Request) -> str | None:
    # Pivot4J passe un cookie JSESSIONID : on l'utilise comme id de session MCAD (stable par navigateur).
    cookie = req.headers.get("cookie") or ""
    m = _JSESSION_RE.search(cookie)
    if not m:
        return None
    js = m.group(1).strip()
    if not js:
        return None
    # limiter la taille
    return f"J_{js[:12]}"


def classify_xmla(xml_bytes: bytes) -> tuple[str, str | None]:
    try:
        root = etree.fromstring(xml_bytes)

        stmt_nodes = root.xpath("//*[local-name()='Statement']")
        if stmt_nodes:
            mdx = (stmt_nodes[0].text or "").strip()
            return ("EXECUTE", mdx if mdx else None)

        rt_nodes = root.xpath("//*[local-name()='RequestType']")
        if rt_nodes:
            rt = (rt_nodes[0].text or "").strip()
            return ("DISCOVER", rt if rt else None)

        return ("OTHER", None)
    except Exception:
        return ("OTHER", None)


def forward_xmla(body: bytes, content_type: str, timeout_s: int = 60) -> requests.Response:
    return requests.post(
        UPSTREAM,
        data=body,
        headers={"Content-Type": content_type or "text/xml"},
        timeout=timeout_s,
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "mcad-proxy",
        "upstream": UPSTREAM,
        "mcad_eval": MCAD_EVAL_URL,
        "mcad_ckg": MCAD_CKG_URL,
    }


@app.post("/xmla")
async def xmla_proxy(req: Request):
    body = await req.body()
    content_type = req.headers.get("content-type", "text/xml")

    kind, payload = classify_xmla(body)

    print("==== MCAD-PROXY REQUEST ====")
    print("TYPE:", kind)
    if kind == "EXECUTE":
        print("MDX:", payload)
    elif kind == "DISCOVER":
        print("REQUESTTYPE:", payload)
    print("============================")

    # 1) Non-EXECUTE: forward directly (Discover/metadata)
    if kind != "EXECUTE" or not payload:
        r = forward_xmla(body, content_type, timeout_s=30)
        print("MCAD-PROXY upstream status:", r.status_code)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "text/xml"),
        )

    mdx = payload
    session_id = extract_session_id(req)

    # 2) Eval MCAD (SAT/Real/Ceval/phi) via mcad-api (/eval)
    decision: dict = {
        "decision": "ALLOW",
        "phi": 1.0,
        "threshold": 0.0,
        "sat": None,
        "real": None,
        "ceval": None,
        "explain": "default allow (eval failed-open)",
        "details": {},
    }
    try:
        eval_payload = {"mdx": mdx}
        if session_id:
            eval_payload["session_id"] = session_id
        er = requests.post(MCAD_EVAL_URL, json=eval_payload, timeout=15)
        if er.ok:
            decision = er.json()
        else:
            print("MCAD-EVAL HTTP error:", er.status_code, (er.text or "")[:300])
    except Exception as e:
        print("MCAD-EVAL exception:", e)
        # fail-open by default

    print("MCAD decision:", decision)

    # 3) Block if not contributive
    if str(decision.get("decision", "ALLOW")).upper() == "BLOCK":
        msg = f"Blocked by MCAD (phi={decision.get('phi')} < {decision.get('threshold')}): {decision.get('explain','')}"
        return Response(content=SOAP_FAULT.format(msg=msg), status_code=500, media_type="text/xml")

    # 4) Forward Execute to eMondrian
    t0 = time.time()
    r = forward_xmla(body, content_type, timeout_s=60)
    elapsed_ms = int((time.time() - t0) * 1000)
    response_bytes = len(r.content or b"")
    response_digest = hashlib.sha256(r.content or b"").hexdigest()[:16]

    print("MCAD-PROXY upstream status:", r.status_code)
    print("MCAD execution elapsed_ms:", elapsed_ms)

    # 5) Update CKG (best-effort) — inclure les infos calculées par /eval
    try:
        det = decision.get("details") if isinstance(decision.get("details"), dict) else {}
        ur = requests.post(
            MCAD_CKG_URL,
            json={
                "mdx": mdx,
                "status_code": r.status_code,
                "elapsed_ms": elapsed_ms,
                "response_bytes": response_bytes,
                "response_digest": response_digest,

                "decision": decision.get("decision"),
                "phi": decision.get("phi"),
                "sat": decision.get("sat"),
                "real": decision.get("real"),
                "ceval": decision.get("ceval"),
                "threshold": decision.get("threshold"),

                "catalog": det.get("catalog"),
                "cube": (det.get("query_spec") or {}).get("cube") if isinstance(àà(det.get("query_spec"), dict) else None,
                "session_id": det.get("session_id") or session_id,
                "objective_id": det.get("objective_id"),
                "step_index": det.get("step_index"),
                "query_spec": det.get("query_spec"),
                "calculable_constraints": det.get("calculable_constraints"),
            },
            timeout=15,
        )
        print("MCAD-CKG update status:", ur.status_code)
    except Exception as e:
        print("MCAD-CKG update exception:", e)

    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/xml"))
