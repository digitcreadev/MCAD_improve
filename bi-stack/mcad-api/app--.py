from __future__ import annotations

from fastapi import FastAPI
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

app = FastAPI(title="MCAD API Adapter", version="1.0.0")

DATA_DIR = Path("/app/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Threshold par dÃ©faut (si ton modÃ¨le n'en fournit pas via objectives)
THRESHOLD_DEFAULT = float(os.getenv("MCAD_THRESHOLD_DEFAULT", "0.60"))

# Neo4j (optionnel)
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

class EvalRequest(BaseModel):
    mdx: str
    session_id: Optional[str] = None
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

_CUBE_RE = re.compile(r"FROM\s+\[([^\]]+)\]", re.IGNORECASE)
_MEASURE_RE = re.compile(r"\[Measures\]\.\[([^\]]+)\]", re.IGNORECASE)

def parse_cube(mdx: str) -> Optional[str]:
    m = _CUBE_RE.search(mdx)
    return m.group(1) if m else None

def parse_measures(mdx: str) -> List[str]:
    return _MEASURE_RE.findall(mdx)

def mdx_fingerprint(mdx: str) -> str:
    return hashlib.sha256(mdx.encode("utf-8")).hexdigest()[:16]



_MCAD_HINT_RE = re.compile(r"/\*MCAD:(.*?)\*/", re.IGNORECASE | re.DOTALL)

def build_query_spec(mdx: str) -> Dict[str, Any]:
    cube = parse_cube(mdx)
    measures = parse_measures(mdx)

    qspec: Dict[str, Any] = {
        "mdx": mdx,
        "cube": cube,
        "measures": measures,
        "group_by": [],
        "slicers": {},
        "analytics": [],
        "fingerprint": mdx_fingerprint(mdx),
    }

    m = _MCAD_HINT_RE.search(mdx or "")
    if not m:
        return qspec

    raw = m.group(1).strip()

    # Parse "k=v" séparés par ";" ou sauts de ligne
    kv: Dict[str, str] = {}
    for part in re.split(r"[;\n]+", raw):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        kv[k.strip().lower()] = v.strip()

    # measures=Marge %,Rupture %
    if "measures" in kv:
        hinted = [x.strip() for x in kv["measures"].split(",") if x.strip()]
        # On remplace carrément, pour coller au modèle objectif (même si le cube n'a pas ces measures)
        qspec["measures"] = hinted

    # slicers=Region.Nord,Year.1998,YearRange.1998
    if "slicers" in kv:
        slicers: Dict[str, str] = {}
        for tok in [t.strip() for t in kv["slicers"].split(",") if t.strip()]:
            if "." in tok:
                d, val = tok.split(".", 1)
                slicers[d.strip()] = val.strip()
        qspec["slicers"] = slicers

    # group_by=Produit.Catégorie,Temps.Mois,Région
    if "group_by" in kv:
        qspec["group_by"] = [g.strip() for g in kv["group_by"].split(",") if g.strip()]

    # analytics=corr (optionnel)
    if "analytics" in kv:
        qspec["analytics"] = [a.strip() for a in kv["analytics"].split(",") if a.strip()]

    return qspec
def compute_mcad_scores(query_spec: Dict[str, Any], payload: EvalRequest) -> Dict[str, Any]:
    """
    Version opÃ©rationnelle (sans TODO) :
    - Fonctionne immÃ©diatement (heuristique).
    - Tu peux la remplacer plus tard par ton moteur rÃ©el (engine_ckg_upgraded, objectives_upgraded, etc.)
    """
    measures = query_spec.get("measures", [])
    cube = query_spec.get("cube")

    # SAT: plus tu as de mesures pertinentes, plus SAT augmente
    sat = 0.2
    if cube:
        sat += 0.2
    sat += min(0.6, 0.2 * len(measures))  # 0,2 par mesure, max +0,6
    sat = max(0.0, min(1.0, sat))

    # REAL: proxy de "rÃ©alisme" (structure exploitable)
    real = 0.1
    if cube:
        real += 0.4
    if measures:
        real += 0.4
    real = max(0.0, min(1.0, real))

    ceval = min(sat, real)
    phi = ceval

    threshold = THRESHOLD_DEFAULT
    decision = "ALLOW" if phi >= threshold else "BLOCK"

    explain = (
        f"heuristic: sat={sat:.2f}, real={real:.2f}, ceval={ceval:.2f}, "
        f"phi={phi:.2f}, theta={threshold:.2f} => {decision}"
    )

    return {
        "sat": sat,
        "real": real,
        "ceval": ceval,
        "phi": phi,
        "threshold": threshold,
        "decision": decision,
        "explain": explain,
    }

def update_ckg_file(payload: CkgUpdateRequest) -> Dict[str, Any]:
    """Fallback: Ã©crit un Ã©vÃ¨nement JSONL dans /app/data/ckg_events.jsonl."""
    event = payload.model_dump()
    event["ts"] = int(time.time() * 1000)
    event["fingerprint"] = mdx_fingerprint(payload.mdx)

    out = DATA_DIR / "ckg_events.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"mode": "file", "path": str(out), "fingerprint": event["fingerprint"]}

def update_ckg_neo4j(payload: CkgUpdateRequest) -> Dict[str, Any]:
    """Optionnel: persist dans Neo4j si NEO4J_URI est configurÃ©."""
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

@app.get("/health")
def health():
    # objectif : dire si objectives.yaml est bien chargé par le backend
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

    # --- imports locaux (pas besoin de modifier le haut du fichier) ---
    from fastapi import HTTPException
    import mcad.engine000 as engine000
    from mcad.models000 import EvaluateWithObjectiveAndSessionRequest
    from mcad.objectives import list_objectives, get_objective
    from mcad.session_store0 import SESSION_STORE

    # --- 1) Parse minimal MDX -> query_spec (utilisé par le CKG) ---
    cube = payload.cube or parse_cube(payload.mdx)
    measures = parse_measures(payload.mdx)

    query_spec = {
        "cube": cube,
        "measures": measures,
        "group_by": [],
        "slicers": {},
        "analytics": [],
    }

    qp = {
        "mdx": payload.mdx,
        "query_spec": query_spec,
        "catalog": payload.catalog,
        "cube": cube,
        "measures": measures,
        "user_id": payload.user_id,
    }

    # --- 2) Choisir objective_id (proxy n'en envoie pas) ---
    objective_id = None
    if payload.context and isinstance(payload.context, dict):
        objective_id = payload.context.get("objective_id")

    if not objective_id:
        objective_id = os.getenv("MCAD_OBJECTIVE_ID_DEFAULT")

    if not objective_id:
        # mémoriser un objectif par défaut au 1er appel
        objective_id = getattr(eval_query, "_default_objective_id", None)

    if not objective_id:
        objs = list_objectives()
        if not objs:
            raise HTTPException(status_code=500, detail="No objectives loaded (check MCAD_OBJECTIVES_YAML).")
        objective_id = objs[0].id
        setattr(eval_query, "_default_objective_id", objective_id)

    # Valider + récupérer l'objectif (sert aussi à calculer des ratios real/ceval)
    try:
        obj = get_objective(str(objective_id))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unknown objective_id={objective_id}: {e}")

    objective_id = obj.id
    setattr(eval_query, "_default_objective_id", objective_id)

    # --- 3) Choisir/Créer une session (proxy n'en envoie pas) ---
    session_id = payload.session_id or getattr(eval_query, "_default_session_id", None)

    if not session_id:
        dw_id = os.getenv("MCAD_DW_ID_DEFAULT", "FoodMart")
        st = SESSION_STORE.create_session(objective_id=objective_id, dw_id=dw_id)
        session_id = st.session_id
        setattr(eval_query, "_default_session_id", session_id)

    # Si la session n'existe pas (redémarrage), on recrée une nouvelle
    try:
        SESSION_STORE.get_session(session_id)
    except KeyError:
        dw_id = os.getenv("MCAD_DW_ID_DEFAULT", "FoodMart")
        st = SESSION_STORE.create_session(objective_id=objective_id, dw_id=dw_id)
        session_id = st.session_id
        setattr(eval_query, "_default_session_id", session_id)

    # --- 4) Appel du moteur réel (CKG-first) ---
    req = EvaluateWithObjectiveAndSessionRequest(
        session_id=session_id,
        objective_id=objective_id,
        qp=qp,
    )
    out = engine000.evaluate_with_objective_and_session(req)

    # --- 5) Construire real/ceval sous forme de ratios (pour rester compatible avec EvalResponse) ---
    constraints = getattr(obj, "constraints", []) or []
    total_constraints = max(1, len(constraints))
    total_nv = sum(len(getattr(c, "virtual_nodes", []) or []) for c in constraints)

    real_ratio = (len(out.real_node_ids) / max(1, total_nv)) if total_nv > 0 else (1.0 if len(out.real_node_ids) > 0 else 0.0)
    ceval_ratio = len(out.calculable_constraints) / total_constraints

    # --- 6) Décision (gating) ---
    threshold = float(THRESHOLD_DEFAULT)
    phi = float(out.phi)
    decision = "ALLOW" if (bool(out.sat) and phi >= threshold and len(out.calculable_constraints) > 0) else "BLOCK"

    explain = (
        f"ENGINE=real sat={bool(out.sat)} "
        f"phi={phi:.3f} theta={threshold:.3f} "
        f"Ceval={len(out.calculable_constraints)}/{total_constraints} "
        f"RealNV={len(out.real_node_ids)}/{max(1,total_nv)} => {decision}"
    )

    # helper pydantic v1/v2
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
    """AppelÃ© par mcad-proxy APRÃˆS exÃ©cution (si ALLOW)."""
    t0 = time.time()
    try:
        if NEO4J_URI and NEO4J_PASSWORD:
            res = update_ckg_neo4j(payload)
        else:
            res = update_ckg_file(payload)
        ok = True
    except Exception as e:
        ok = False
        res = {"error": str(e)}

    return {
        "ok": ok,
        "adapter_ms": int((time.time() - t0) * 1000),
        "result": res,
    }
