from __future__ import annotations

from fastapi import FastAPI, Request, Response
import os
import time
import requests
import hashlib
from lxml import etree

app = FastAPI(title="MCAD XMLA Proxy", version="1.0.0")

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
    return {"ok": True, "service": "mcad-proxy", "upstream": UPSTREAM, "mcad_eval": MCAD_EVAL_URL, "mcad_ckg": MCAD_CKG_URL}

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
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/xml"))

    mdx = payload

    # 2) Eval MCAD (SAT/Real/Ceval/phi)
    decision = {
        "decision": "ALLOW",
        "phi": 1.0,
        "threshold": 0.0,
        "sat": None,
        "real": None,
        "ceval": None,
        "explain": "default allow",
    }
    try:
        er = requests.post(MCAD_EVAL_URL, json={"mdx": mdx}, timeout=10)
        if er.ok:
            decision = er.json()
        else:
            print("MCAD-EVAL HTTP error:", er.status_code, er.text[:300])
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
    print("MCAD-PROXY upstream status:", r.status_code)
    print("MCAD execution elapsed_ms:", elapsed_ms)

    # 5) Update CKG (best-effort)
    try:
        ur = requests.post(
            MCAD_CKG_URL,
            json={
                "mdx": mdx,
                "status_code": r.status_code,
                "elapsed_ms": elapsed_ms,
                "response_bytes": len(r.content or b""),
                "decision": decision.get("decision"),
                "phi": decision.get("phi"),
                "sat": decision.get("sat"),
                "real": decision.get("real"),
                "ceval": decision.get("ceval"),
                "threshold": decision.get("threshold"),
            },
            timeout=10,
        )
        print("MCAD-CKG update status:", ur.status_code)
    except Exception as e:
        print("MCAD-CKG update exception:", e)

    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "text/xml"))
