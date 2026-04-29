from __future__ import annotations

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import os
import time
import requests
import hashlib
import re
import json
from pathlib import Path
from lxml import etree
from xml.sax.saxutils import escape as xml_escape
from xmla_result_parser import summarize_xmla_response

app = FastAPI(title="MCAD XMLA Proxy", version="1.4.0")

UPSTREAM = os.getenv("UPSTREAM_XMLA", "http://emondrian:8080/emondrian/xmla")
MCAD_EVAL_URL = os.getenv("MCAD_EVAL_URL", "http://mcad-api:8000/eval")
MCAD_CKG_URL = os.getenv("MCAD_CKG_URL", "http://mcad-api:8000/ckg/update")
MCAD_API_BASE = os.getenv("MCAD_API_BASE", "http://mcad-api:8000")
MCAD_OBJECTIVE_ID_DEFAULT = os.getenv("MCAD_OBJECTIVE_ID_DEFAULT", "")
MCAD_DW_ID_DEFAULT = os.getenv("MCAD_DW_ID_DEFAULT", "foodmart")
PIVOT4J_URL = os.getenv("PIVOT4J_URL", "http://pivot4j:8080/pivot4j")

ACTIVE_CONTEXT: dict[str, str | None] = {
    "session_id": None,
    "objective_id": MCAD_OBJECTIVE_ID_DEFAULT or None,
    "dw_id": MCAD_DW_ID_DEFAULT,
}
LAST_DECISION: dict[str, object] = {}

SOAP_FAULT = """<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <soap:Fault>
      <faultcode>soap:Client</faultcode>
      <faultstring>{faultstring}</faultstring>
      <detail>
        <mcad:MCAD xmlns:mcad="urn:mcad">
          <mcad:decision>{decision}</mcad:decision>
          <mcad:phi>{phi}</mcad:phi>
          <mcad:threshold>{threshold}</mcad:threshold>
          <mcad:objective_id>{objective_id}</mcad:objective_id>
          <mcad:session_id>{session_id}</mcad:session_id>
          <mcad:step_index>{step_index}</mcad:step_index>
          <mcad:decision_reason_code>{decision_reason_code}</mcad:decision_reason_code>
          <mcad:decision_reason>{decision_reason}</mcad:decision_reason>
          <mcad:is_redundant>{is_redundant}</mcad:is_redundant>
          <mcad:has_marginal_gain>{has_marginal_gain}</mcad:has_marginal_gain>
          <mcad:explain>{explain}</mcad:explain>
        </mcad:MCAD>
      </detail>
    </soap:Fault>
  </soap:Body>
</soap:Envelope>
"""


def mdx_fingerprint(mdx: str) -> str:
    return hashlib.sha256((mdx or "").encode("utf-8")).hexdigest()[:16]


_JSESSION_RE = re.compile(r"(?:^|;\s*)JSESSIONID=([^;]+)")


def extract_session_cookie(req: Request) -> str | None:
    cookie = req.headers.get("cookie") or ""
    m = _JSESSION_RE.search(cookie)
    if not m:
        return None
    js = m.group(1).strip()
    return js[:64] if js else None


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


def _fault_from_decision(decision: dict, session_id: str | None) -> str:
    det = decision.get("details") if isinstance(decision.get("details"), dict) else {}
    phi = float(decision.get("phi", 0.0) or 0.0)
    theta = float(decision.get("threshold", 0.0) or 0.0)
    objective_id = str(det.get("objective_id") or decision.get("objective_id") or ACTIVE_CONTEXT.get("objective_id") or "")
    step_index = str(det.get("step_index") or decision.get("step_index") or "")
    explain = str(decision.get("explain") or "")
    code = str(decision.get("decision_reason_code") or det.get("decision_reason_code") or "BLOCK_GENERIC")
    reason = str(decision.get("decision_reason") or det.get("decision_reason") or explain or "Blocked by MCAD")
    short = f"MCAD BLOCK [{code}]"
    return SOAP_FAULT.format(
        faultstring=xml_escape(short),
        decision=xml_escape(str(decision.get("decision", "BLOCK"))),
        phi=xml_escape(f"{phi:.6f}"),
        threshold=xml_escape(f"{theta:.6f}"),
        objective_id=xml_escape(objective_id),
        session_id=xml_escape(str(session_id or det.get("session_id") or ACTIVE_CONTEXT.get("session_id") or "")),
        step_index=xml_escape(step_index),
        decision_reason_code=xml_escape(code),
        decision_reason=xml_escape(reason),
        is_redundant=xml_escape(str(bool(decision.get("is_redundant", det.get("is_redundant", False)))).lower()),
        has_marginal_gain=xml_escape(str(bool(decision.get("has_marginal_gain", det.get("has_marginal_gain", False)))).lower()),
        explain=xml_escape(explain),
    )


def _load_session_ui_html() -> str:
    p = Path(__file__).with_name("session_ui.html")
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "<html><body><h1>MCAD Session Manager</h1></body></html>"


def _relay_get(path: str) -> dict:
    r = requests.get(f"{MCAD_API_BASE}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def _relay_post(path: str, payload: dict) -> dict:
    r = requests.post(f"{MCAD_API_BASE}{path}", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "mcad-proxy",
        "upstream": UPSTREAM,
        "mcad_eval": MCAD_EVAL_URL,
        "mcad_ckg": MCAD_CKG_URL,
        "active_context": ACTIVE_CONTEXT,
        "last_decision": LAST_DECISION,
    }


@app.get("/mcad/objectives")
def mcad_objectives():
    data = _relay_get("/objectives")
    return {"ok": True, "items": data if isinstance(data, list) else data.get("items", [])}


@app.get("/mcad/sessions")
def mcad_sessions():
    data = _relay_get("/sessions")
    return {"ok": True, "items": data if isinstance(data, list) else data.get("items", [])}


@app.get("/mcad/datawarehouses")
def mcad_datawarehouses():
    return {"ok": True, "items": [
        {"id": "foodmart", "label": "FoodMart", "type": "mondrian", "catalog": "FoodMart", "cube": "Sales", "xmla_url": "http://emondrian:8080/emondrian/xmla", "enabled": True, "notes": "Default sample warehouse for FoodMart / Sales cube."},
        {"id": "adventureworks", "label": "Adventure Works", "type": "mondrian", "catalog": "AdventureWorks", "cube": "Adventure Works", "xmla_url": "http://emondrian:8080/emondrian/xmla", "enabled": True, "notes": "Default sample warehouse for Adventure Works demonstrations."},
        {"id": "external_template", "label": "External DW Template", "type": "xmla", "catalog": "YOUR_CATALOG", "cube": "YOUR_CUBE", "xmla_url": "https://example.org/xmla", "enabled": False, "notes": "Duplicate and adapt this entry to register an external warehouse."}
    ]}


@app.get("/mcad/session/current")
def mcad_session_current():
    return {"ok": True, "active": ACTIVE_CONTEXT, "last_decision": LAST_DECISION}


@app.post("/mcad/session/new")
async def mcad_session_new(req: Request):
    payload = await req.json()
    objective_id = str(payload.get("objective_id") or MCAD_OBJECTIVE_ID_DEFAULT or "")
    dw_id = str(payload.get("dw_id") or MCAD_DW_ID_DEFAULT or "foodmart")
    session_resp = _relay_post("/sessions/create", {"objective_id": objective_id, "dw_id": dw_id})
    session = session_resp.get("session", session_resp)
    ACTIVE_CONTEXT["session_id"] = str(session.get("session_id") or "") or None
    ACTIVE_CONTEXT["objective_id"] = str(session.get("objective_id") or objective_id) or None
    ACTIVE_CONTEXT["dw_id"] = str(session.get("dw_id") or dw_id) or None
    LAST_DECISION.clear()
    return {"ok": True, "active": ACTIVE_CONTEXT, "session": session}


@app.post("/mcad/session/resume")
async def mcad_session_resume(req: Request):
    payload = await req.json()
    session_id = str(payload.get("session_id") or "")
    resp = _relay_get(f"/sessions/{session_id}")
    session = resp.get("session", resp)
    ACTIVE_CONTEXT["session_id"] = str(session.get("session_id") or session_id) or None
    ACTIVE_CONTEXT["objective_id"] = str(session.get("objective_id") or "") or None
    ACTIVE_CONTEXT["dw_id"] = str(session.get("dw_id") or MCAD_DW_ID_DEFAULT) or None
    LAST_DECISION.clear()
    return {"ok": True, "active": ACTIVE_CONTEXT, "session": session}


@app.get("/mcad/session/ui", response_class=HTMLResponse)
def mcad_session_ui():
    return HTMLResponse(_load_session_ui_html())




@app.get("/mcad/history/current")
def mcad_history_current():
    sid = ACTIVE_CONTEXT.get("session_id")
    oid = ACTIVE_CONTEXT.get("objective_id")
    if not sid:
        return {"ok": True, "session_id": None, "objective_id": oid, "items": []}
    data = _relay_get(f"/sessions/{sid}/history")
    return {"ok": True, "session_id": sid, "objective_id": oid, "items": data.get("items", [])}


@app.get("/mcad/graph/current")
def mcad_graph_current():
    sid = ACTIVE_CONTEXT.get("session_id")
    oid = ACTIVE_CONTEXT.get("objective_id")
    if not sid:
        return {
            "ok": True,
            "session_id": None,
            "objective_id": oid,
            "graph": {"nodes": [], "edges": []},
            "metrics": {
                "completion_rate": 0.0,
                "calculability_rate_total": 0.0,
                "calculability_rate_partial": 0.0,
                "analytic_alignment_score": 0.0,
                "allow_rate": 0.0,
                "allow_count": 0,
                "block_count": 0,
            },
        }
    data = _relay_get(f"/sessions/{sid}/graph")
    return {
        "ok": True,
        "session_id": sid,
        "objective_id": data.get("objective_id", oid),
        "dw_id": data.get("dw_id"),
        "graph": data.get("graph", {"nodes": [], "edges": []}),
        "metrics": data.get("metrics", {}),
    }


def _build_public_pivot4j_url(req: Request, subpath: str = "") -> str:
    url = str(req.url)
    if ".app.github.dev" in url:
        url = re.sub(r"-9000(\.app\.github\.dev)", r"-8090\1", url)
        base = url.split("/", 3)[:3]
        origin = "/".join(base)
        return origin.rstrip("/") + "/pivot4j/" + subpath.lstrip("/")
    host = req.headers.get("host", "localhost:9000")
    if ":9000" in host:
        host = host.replace(":9000", ":8090")
    return f"{req.url.scheme}://{host}/pivot4j/" + subpath.lstrip("/")


@app.get("/pivot4j")
@app.get("/pivot4j/")
def open_pivot4j(req: Request):
    return RedirectResponse(_build_public_pivot4j_url(req))


@app.get("/pivot4j/{path:path}")
def open_pivot4j_subpath(req: Request, path: str):
    return RedirectResponse(_build_public_pivot4j_url(req, path))


@app.post("/xmla")
async def xmla_proxy(req: Request):
    body = await req.body()
    content_type = req.headers.get("content-type", "text/xml")
    kind, payload = classify_xmla(body)

    if kind != "EXECUTE" or not payload:
        r = forward_xmla(body, content_type, timeout_s=30)
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "text/xml"),
            headers={"X-MCAD-Decision": "PASS"},
        )

    mdx = payload
    browser_key = extract_session_cookie(req)
    session_id = str(ACTIVE_CONTEXT.get("session_id") or "") or None
    objective_id = str(ACTIVE_CONTEXT.get("objective_id") or MCAD_OBJECTIVE_ID_DEFAULT or "") or None

    decision: dict = {
        "decision": "BLOCK",
        "phi": 0.0,
        "threshold": 0.0,
        "sat": 0.0,
        "real": 0.0,
        "ceval": 0.0,
        "decision_reason_code": "EVAL_UNREACHABLE",
        "decision_reason": "MCAD /eval unavailable or failed.",
        "explain": "BLOCK because MCAD /eval failed; fail-open disabled.",
        "details": {},
    }
    try:
        eval_payload: dict = {"mdx": mdx}
        if session_id:
            eval_payload["session_id"] = session_id
        if objective_id:
            eval_payload["objective_id"] = objective_id
        if browser_key:
            eval_payload.setdefault("context", {})["browser_key"] = browser_key
        er = requests.post(MCAD_EVAL_URL, json=eval_payload, timeout=15)
        if er.ok:
            decision = er.json()
        else:
            decision["decision_reason"] = f"HTTP {er.status_code}: {(er.text or '')[:300]}"
            print("MCAD-EVAL HTTP error:", er.status_code, (er.text or "")[:300])
    except Exception as e:
        decision["decision_reason"] = str(e)
        print("MCAD-EVAL exception:", e)

    det = decision.get("details") if isinstance(decision.get("details"), dict) else {}
    ACTIVE_CONTEXT["session_id"] = str(det.get("session_id") or session_id or ACTIVE_CONTEXT.get("session_id") or "") or None
    ACTIVE_CONTEXT["objective_id"] = str(det.get("objective_id") or objective_id or ACTIVE_CONTEXT.get("objective_id") or "") or None
    LAST_DECISION.clear()
    LAST_DECISION.update({
        "decision": decision.get("decision"),
        "decision_reason_code": decision.get("decision_reason_code") or det.get("decision_reason_code"),
        "decision_reason": decision.get("decision_reason") or det.get("decision_reason"),
        "is_redundant": decision.get("is_redundant", det.get("is_redundant", False)),
        "has_marginal_gain": decision.get("has_marginal_gain", det.get("has_marginal_gain", False)),
        "objective_id": det.get("objective_id") or objective_id,
        "session_id": det.get("session_id") or session_id,
        "phi": decision.get("phi"),
        "threshold": decision.get("threshold"),
        "sat": decision.get("sat"),
        "real": decision.get("real"),
        "ceval": decision.get("ceval"),
        "explain": decision.get("explain"),
        "gained_resource_ids_count": det.get("gained_resource_ids_count", len(det.get("gained_resource_ids") or [])),
        "newly_contributed_constraints_total": det.get("newly_contributed_constraints_total", []),
        "newly_contributed_constraints_partial": det.get("newly_contributed_constraints_partial", []),
        "query_fingerprint": mdx_fingerprint(mdx),
        "ts_ms": int(time.time() * 1000),
    })

    if str(decision.get("decision", "ALLOW")).upper() == "BLOCK":
        fault = _fault_from_decision(decision, session_id)
        return Response(
            content=fault,
            status_code=200,
            media_type="text/xml; charset=utf-8",
            headers={
                "X-MCAD-Decision": "BLOCK",
                "X-MCAD-Phi": str(decision.get("phi", "")),
                "X-MCAD-Threshold": str(decision.get("threshold", "")),
                "X-MCAD-Decision-Reason": str(LAST_DECISION.get("decision_reason_code") or ""),
            },
        )

    t0 = time.time()
    r = forward_xmla(body, content_type, timeout_s=60)
    elapsed_ms = int((time.time() - t0) * 1000)
    response_bytes = len(r.content or b"")
    response_digest = hashlib.sha256(r.content or b"").hexdigest()[:16]
    raw_result_summary = summarize_xmla_response(r.content or b"")

    try:
        qspec = det.get("query_spec") if isinstance(det.get("query_spec"), dict) else {}
        requests.post(
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
                "cube": qspec.get("cube") or None,
                "session_id": det.get("session_id") or session_id,
                "objective_id": det.get("objective_id") or objective_id,
                "step_index": det.get("step_index"),
                "query_spec": qspec or None,
                "calculable_constraints": det.get("calculable_constraints"),
                "covered_constraints": det.get("covered_constraints"),
                "raw_result_summary": raw_result_summary,
            },
            timeout=15,
        )
    except Exception as e:
        print("MCAD-CKG update exception:", e)

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "text/xml"),
        headers={
            "X-MCAD-Decision": "ALLOW",
            "X-MCAD-Phi": str(decision.get("phi", "")),
            "X-MCAD-Threshold": str(decision.get("threshold", "")),
            "X-MCAD-Decision-Reason": str(LAST_DECISION.get("decision_reason_code") or ""),
        },
    )


