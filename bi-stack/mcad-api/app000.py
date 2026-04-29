from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import time
import json
import hashlib
import re
from pathlib import Path

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None  # optional

app = FastAPI(title="MCAD API Adapter", version="1.1.0")

DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_DEFAULT = float(os.getenv("MCAD_THRESHOLD_DEFAULT", "0.60"))

# Neo4j (optionnel)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Prefer the richer backend MDX parser when available.
try:
    from mcad.mdx_parser import parse_mdx as backend_parse_mdx  # type: ignore
except Exception:
    backend_parse_mdx = None
try:
    from execution.useful_result_extractor import extract_useful_result_summary  # type: ignore
except Exception:
    extract_useful_result_summary = None


# -------------------------
# Models
# -------------------------

class EvalRequest(BaseModel):
    mdx: str
    session_id: Optional[str] = None
    objective_id: Optional[str] = None
    user_id: Optional[str] = None
    catalog: Optional[str] = None
    cube: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class EvalResponse(BaseModel):
    decision: str                 # "ALLOW" | "BLOCK"
    phi: float
    threshold: float
    sat: float
    real: float
    ceval: float
    explain: str
    details: Dict[str, Any] = {}


class CkgUpdateRequest(BaseModel):
    mdx: str
    status_code: int
    elapsed_ms: int
    response_bytes: Optional[int] = None
    response_digest: Optional[str] = None

    # provenant de /eval (via mcad-proxy)
    objective_id: Optional[str] = None
    step_index: Optional[int] = None
    query_spec: Optional[Dict[str, Any]] = None
    calculable_constraints: Optional[List[str]] = None
    covered_constraints: Optional[List[str]] = None
    raw_result_summary: Optional[Dict[str, Any]] = None
    useful_result_summary: Optional[Dict[str, Any]] = None

    decision: Optional[str] = None
    phi: Optional[float] = None
    sat: Optional[float] = None
    real: Optional[float] = None
    ceval: Optional[float] = None
    threshold: Optional[float] = None

    catalog: Optional[str] = None
    cube: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


# -------------------------
# MDX helpers
# -------------------------

_CUBE_RE = re.compile(r"FROM\s+\[([^\]]+)\]", re.IGNORECASE)
_MEASURE_RE = re.compile(r"\[Measures\]\.\[([^\]]+)\]", re.IGNORECASE)

def parse_cube(mdx: str) -> Optional[str]:
    m = _CUBE_RE.search(mdx or "")
    return m.group(1) if m else None

def parse_measures(mdx: str) -> List[str]:
    return _MEASURE_RE.findall(mdx or "")

def mdx_fingerprint(mdx: str) -> str:
    return hashlib.sha256((mdx or "").encode("utf-8")).hexdigest()[:16]


def build_query_spec(mdx: str, cube_override: Optional[str] = None) -> Dict[str, Any]:
    """Build a MCAD-friendly query_spec from raw MDX.

    We prefer the richer backend parser so the BI-real path uses roughly the
    same analytical structure as the offline benchmark path. We keep a small
    regex fallback for resilience.
    """
    if backend_parse_mdx is not None:
        try:
            parsed = backend_parse_mdx(mdx) or {}
        except Exception:
            parsed = {}
    else:
        parsed = {}

    cube = cube_override or parsed.get("cube") or parse_cube(mdx)
    measures = sorted({m for m in (parsed.get("measures") or parse_measures(mdx) or []) if m})
    analytics = [str(a).upper() for a in (parsed.get("analytics") or parsed.get("aggregators") or []) if a]
    group_by = [str(g) for g in (parsed.get("group_by") or []) if g]
    slicers = parsed.get("slicers") if isinstance(parsed.get("slicers"), dict) else {}
    time_members = [str(t) for t in (parsed.get("time_members") or []) if t]
    calculated_members = [str(t) for t in (parsed.get("calculated_members") or []) if t]
    named_sets = [str(t) for t in (parsed.get("named_sets") or []) if t]

    return {
        "mdx": mdx,
        "cube": cube,
        "measures": measures,
        "group_by": group_by,
        "slicers": slicers,
        "analytics": analytics,
        "axes": parsed.get("axes") or [],
        "time_members": time_members,
        "window_start": parsed.get("window_start"),
        "window_end": parsed.get("window_end"),
        "calculated_members": calculated_members,
        "named_sets": named_sets,
        "language": parsed.get("language") or "mdx",
        "fingerprint": parsed.get("fingerprint") or mdx_fingerprint(mdx),
    }


# -------------------------
# Adapter persistence (JSONL / Neo4j)
# -------------------------

def update_ckg_file(payload: CkgUpdateRequest) -> Dict[str, Any]:
    """Append-only JSONL event log in /app/data/ckg_events.jsonl."""
    # pydantic v1/v2 compatibility
    event = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    event["ts"] = int(time.time() * 1000)
    event["fingerprint"] = mdx_fingerprint(payload.mdx)

    out = DATA_DIR / "ckg_events.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"mode": "file", "path": str(out), "fingerprint": event["fingerprint"]}


def update_ckg_neo4j(payload: CkgUpdateRequest) -> Dict[str, Any]:
    """Optional Neo4j persistence."""
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver not available")
    if not (NEO4J_URI and NEO4J_PASSWORD):
        raise RuntimeError("NEO4J_URI/NEO4J_PASSWORD not configured")

    fp = mdx_fingerprint(payload.mdx)
    session_id = payload.session_id or "default-session"
    user_id = payload.user_id or "default-user"

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    cypher = """
    MERGE (u:User {id:$user_id})
    MERGE (s:Session {id:$session_id})
    MERGE (q:Query {fp:$fp})
      ON CREATE SET q.mdx=$mdx
    CREATE (e:Execution {
      ts:$ts,
      status_code:$status_code,
      elapsed_ms:$elapsed_ms,
      response_bytes:$response_bytes,
      decision:$decision,
      phi:$phi,
      sat:$sat,
      real:$real,
      ceval:$ceval,
      threshold:$threshold
    })
    MERGE (u)-[:HAS_SESSION]->(s)
    MERGE (s)-[:ISSUED]->(q)
    CREATE (q)-[:EXECUTED_AS]->(e)
    RETURN e.ts as ts
    """
    params = {
        "user_id": user_id,
        "session_id": session_id,
        "fp": fp,
        "mdx": payload.mdx,
        "ts": int(time.time() * 1000),
        "status_code": payload.status_code,
        "elapsed_ms": payload.elapsed_ms,
        "response_bytes": payload.response_bytes,
        "decision": payload.decision,
        "phi": payload.phi,
        "sat": payload.sat,
        "real": payload.real,
        "ceval": payload.ceval,
        "threshold": payload.threshold,
    }
    with driver.session() as sess:
        rec = sess.run(cypher, params).single()
    driver.close()
    return {"mode": "neo4j", "fp": fp, "ts": rec["ts"] if rec else None}


# -------------------------
# Backend CKGGraph update (normalized IDs)
# -------------------------

def _prefix_if_missing(raw: Optional[str], prefix: str) -> Optional[str]:
    if not raw:
        return None
    s = str(raw)
    return s if s.startswith(prefix) else f"{prefix}{s}"


def update_ckg_backend(payload: CkgUpdateRequest) -> Dict[str, Any]:
    """
    Best-effort: update the REAL backend CKGGraph (networkx) after ALLOW execution.

    Normalized identifiers:
      - session::S_0001
      - objective::O_...
      - qp::S_0001::t001
      - exec::S_0001::t001::<digest>::<ts>
    """
    try:
        import mcad.engine000 as engine000  # type: ignore

        # Try to obtain a CKGGraph instance from engine (robust)
        ckg = None
        if hasattr(engine000, "get_ckg") and callable(getattr(engine000, "get_ckg")):
            ckg = engine000.get_ckg()
        else:
            for attr in ("CKG", "CKG_GRAPH", "GLOBAL_CKG", "ckg", "CKG_INSTANCE"):
                if hasattr(engine000, attr):
                    ckg = getattr(engine000, attr)
                    break

        if ckg is None:
            return {"mode": "backend", "ok": False, "reason": "no_ckg_instance_exposed"}

        if not hasattr(ckg, "G"):
            return {"mode": "backend", "ok": False, "reason": "ckg_instance_has_no_graph"}

        G = getattr(ckg, "G")

        sid_raw = payload.session_id or os.getenv("MCAD_SESSION_ID_DEFAULT", "S_0001")
        sid = _prefix_if_missing(sid_raw, "session::")  # type: ignore
        oid_raw = payload.objective_id or os.getenv("MCAD_OBJECTIVE_ID_DEFAULT")
        oid = _prefix_if_missing(oid_raw, "objective::")

        step_idx = int(payload.step_index or 0)
        digest = payload.response_digest or mdx_fingerprint(payload.mdx)
        ts = int(time.time() * 1000)

        exec_id = f"exec::{sid_raw}::t{step_idx:03d}::{digest}::{ts}" if step_idx > 0 else f"exec::{sid_raw}::{digest}::{ts}"
        qpid = f"qp::{sid_raw}::t{step_idx:03d}" if step_idx > 0 else None

        # Ensure session/objective nodes
        if sid and not G.has_node(sid):
            G.add_node(sid, type="session", session_id=sid_raw)
        if oid and not G.has_node(oid):
            G.add_node(oid, type="objective", objective_id=oid_raw)

        # Execution node
        G.add_node(
            exec_id,
            type="execution",
            mdx=payload.mdx,
            fingerprint=mdx_fingerprint(payload.mdx),
            response_digest=payload.response_digest,
            status_code=payload.status_code,
            elapsed_ms=payload.elapsed_ms,
            response_bytes=payload.response_bytes,
            decision=payload.decision,
            phi=payload.phi,
            threshold=payload.threshold,
            sat=payload.sat,
            real=payload.real,
            ceval=payload.ceval,
            step_index=step_idx,
            query_spec=json.dumps(payload.query_spec or {}, ensure_ascii=False),
            calculable_constraints=json.dumps(payload.calculable_constraints or [], ensure_ascii=False),
            covered_constraints=json.dumps(payload.covered_constraints or [], ensure_ascii=False),
            raw_result_summary=json.dumps(payload.raw_result_summary or {}, ensure_ascii=False),
            useful_result_summary=json.dumps(payload.useful_result_summary or {}, ensure_ascii=False),
            ts=ts,
        )

        # Edges
        if sid:
            G.add_edge(sid, exec_id, type="HAS_EXECUTION")
        if oid:
            G.add_edge(exec_id, oid, type="FOR_OBJECTIVE")
        if qpid and G.has_node(qpid):
            G.add_edge(exec_id, qpid, type="EXECUTED_QP")

        # IMPORTANT: stop creating legacy raw session nodes ("S_0001")
        # Optional safe cleanup: remove raw node only if isolated
        if G.has_node(sid_raw) and sid_raw != sid:
            try:
                # if it has no incident edges, remove it
                deg = 0
                if hasattr(G, "degree"):
                    deg = int(G.degree(sid_raw))  # type: ignore
                if deg == 0:
                    G.remove_node(sid_raw)
            except Exception:
                pass

        # Persist snapshot (try both call signatures)
        snap = str(DATA_DIR / "ckg_state.json")
        try:
            if hasattr(ckg, "save_state_json") and callable(getattr(ckg, "save_state_json")):
                try:
                    ckg.save_state_json(snap)
                except TypeError:
                    ckg.save_state_json(path=snap)  # type: ignore
        except Exception:
            pass

        return {"mode": "backend", "ok": True, "exec_id": exec_id, "qpid": qpid, "session": sid, "objective": oid, "snapshot": snap}
    except Exception as e:
        return {"mode": "backend", "ok": False, "error": str(e)}


# -------------------------
# API endpoints
# -------------------------

@app.get("/health")
def health():
    # objective loading status from backend
    try:
        from mcad.objectives import list_objectives
        objs = list_objectives()
        obj_loaded = bool(objs)
        obj_count = len(objs)
        obj_err = None
    except Exception as e:
        obj_loaded = False
        obj_count = 0
        obj_err = str(e)

    return {
        "ok": True,
        "service": "mcad-api",
        "threshold_default": THRESHOLD_DEFAULT,
        "objectives": {
            "loaded": obj_loaded,
            "count": obj_count,
            "yaml_primary": os.getenv("MCAD_OBJECTIVES_YAML"),
            "yaml_fallback": os.getenv("MCAD_OBJECTIVES_FALLBACK_YAML"),
            "error": obj_err,
        },
    }


@app.post("/eval", response_model=EvalResponse)
def eval_query(payload: EvalRequest):
    t0 = time.time()

    from fastapi import HTTPException
    import mcad.engine000 as engine000  # type: ignore
    from mcad.models000 import EvaluateWithObjectiveAndSessionRequest  # type: ignore
    from mcad.objectives import list_objectives, get_objective  # type: ignore
    from mcad.session_store0 import SESSION_STORE  # type: ignore

    # 1) deterministic mdx -> richer query_spec
    query_spec = build_query_spec(payload.mdx, cube_override=payload.cube)
    cube = query_spec.get("cube")
    measures = list(query_spec.get("measures") or [])

    qp = {
        "mdx": payload.mdx,
        "query_spec": query_spec,
        "catalog": payload.catalog,
        "cube": cube,
        "measures": measures,
        "user_id": payload.user_id,
    }

    # 2) objective_id resolution
    objective_id = payload.objective_id
    if payload.context and isinstance(payload.context, dict):
        objective_id = objective_id or payload.context.get("objective_id")
    if not objective_id:
        objective_id = os.getenv("MCAD_OBJECTIVE_ID_DEFAULT")
    if not objective_id:
        objs = list_objectives()
        if not objs:
            raise HTTPException(status_code=500, detail="No objectives loaded (check MCAD_OBJECTIVES_YAML / MCAD_OBJECTIVES_FALLBACK_YAML).")
        objective_id = objs[0].id

    try:
        obj = get_objective(str(objective_id))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unknown objective_id={objective_id}: {e}")

    objective_id = obj.id

    # 3) session_id (provided or stable default)
    session_id = payload.session_id or os.getenv("MCAD_SESSION_ID_DEFAULT", "S_0001")
    try:
        SESSION_STORE.get_session(session_id)
    except KeyError:
        dw_id = os.getenv("MCAD_DW_ID_DEFAULT", "FoodMart")
        SESSION_STORE.create_session(objective_id=objective_id, dw_id=dw_id)

    # 4) real engine call
    req = EvaluateWithObjectiveAndSessionRequest(session_id=session_id, objective_id=objective_id, qp=qp)
    out = engine000.evaluate_with_objective_and_session(req)

    # 5) ratios for compatibility
    constraints = getattr(obj, "constraints", []) or []
    total_constraints = max(1, len(constraints))
    total_nv = sum(len(getattr(c, "virtual_nodes", []) or []) for c in constraints)

    real_ratio = (len(out.real_node_ids) / max(1, total_nv)) if total_nv > 0 else (1.0 if len(out.real_node_ids) > 0 else 0.0)
    ceval_ratio = len(out.calculable_constraints) / total_constraints

    # 6) decision rule
    threshold = float(getattr(obj, "threshold", getattr(obj, "theta", THRESHOLD_DEFAULT)) or THRESHOLD_DEFAULT)
    phi = float(out.phi)

    decision = "ALLOW"
    if (not bool(out.sat)) or (phi < threshold) or (len(out.calculable_constraints) == 0):
        decision = "BLOCK"

    print(f"ENGINE=real phi={phi:.6f} threshold={threshold:.6f} decision={decision}")

    explain = (
        f"ENGINE=real sat={bool(out.sat)} "
        f"phi={phi:.3f} theta={threshold:.3f} "
        f"Ceval={len(out.calculable_constraints)}/{total_constraints} "
        f"RealNV={len(out.real_node_ids)}/{max(1,total_nv)} => {decision}"
    )

    def _dump(x):
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "dict"):
            return x.dict()
        return x

    return EvalResponse(
        decision=decision,
        phi=phi,
        threshold=threshold,
        sat=1.0 if out.sat else 0.0,
        real=float(real_ratio),
        ceval=float(ceval_ratio),
        explain=explain,
        details={
            "engine": "evaluate_with_objective_and_session(payload)",
            "objective_id": objective_id,
            "session_id": session_id,
            "step_index": getattr(out, "step_index", None),
            "phi_weighted": float(getattr(out, "phi_weighted", 0.0)),
            "phi_leq_t": getattr(out, "phi_leq_t", None),
            "delta_phi_t": getattr(out, "delta_phi_t", None),
            "real_node_ids": list(out.real_node_ids),
            "calculable_constraints": list(out.calculable_constraints),
            "covered_constraints": list(getattr(out, "covered_constraints", []) or []),
            "clauses": [_dump(c) for c in (getattr(out, "clauses", []) or [])],
            "query_spec": query_spec,
            "eval_ms": int((time.time() - t0) * 1000),
        },
    )


@app.post("/ckg/update")
def ckg_update(payload: CkgUpdateRequest):
    """Called by mcad-proxy AFTER execution (ALLOW)."""
    t0 = time.time()

    if extract_useful_result_summary is not None and not payload.useful_result_summary:
        try:
            payload.useful_result_summary = extract_useful_result_summary(
                payload.raw_result_summary or {},
                payload.query_spec or {},
                payload.decision,
                payload.calculable_constraints or [],
                payload.covered_constraints or [],
            )
        except Exception as e:
            payload.useful_result_summary = {
                'kind': 'useful_result_summary',
                'error': str(e),
                'materialization_level': 'summary_only_v1',
            }

    backend_res = update_ckg_backend(payload)

    try:
        if NEO4J_URI and NEO4J_PASSWORD:
            adapter_res = update_ckg_neo4j(payload)
        else:
            adapter_res = update_ckg_file(payload)
        ok = True
    except Exception as e:
        ok = False
        adapter_res = {"error": str(e)}

    return {
        "ok": ok,
        "adapter_ms": int((time.time() - t0) * 1000),
        "backend": backend_res,
        "result": adapter_res,
    }


@app.get("/ckg/events", response_class=PlainTextResponse)
def ckg_events(tail: int = Query(50, ge=1, le=500)):
    """Return last N JSONL events."""
    p = Path("/app/data/ckg_events.jsonl")
    if not p.exists():
        return "ckg_events.jsonl ABSENT"
    lines = p.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-tail:]) + "\n"


@app.get("/ckg/state/summary")
def ckg_state_summary():
    """Quick summary of /app/data/ckg_state.json snapshot."""
    p = Path("/app/data/ckg_state.json")
    if not p.exists():
        return {"ok": False, "error": "ckg_state.json ABSENT"}
    d = json.loads(p.read_text(encoding="utf-8"))
    nodes = d.get("nodes", {}) or {}
    edges = d.get("edges", {}) or {}

    def _is_exec(k: str) -> bool:
        return k.startswith("exec:") or k.startswith("exec::")
    def _is_qp(k: str) -> bool:
        return k.startswith("qp:") or k.startswith("qp::")

    exec_count = sum(1 for k in nodes.keys() if _is_exec(str(k)))
    qp_count = sum(1 for k in nodes.keys() if _is_qp(str(k)))
    return {"ok": True, "nodes": len(nodes), "edges": len(edges), "exec_count": exec_count, "qp_count": qp_count}
