# backend/routers/mdx_api.py
from __future__ import annotations

from typing import Any, Dict, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mcad.engine000 import evaluate_with_objective_and_session
from mcad.models000 import (
    EvaluateWithObjectiveAndSessionRequest,
    EvaluateWithObjectiveAndSessionResponse,
)
from mcad.objectives import get_objective
from mcad.mdx_parser import parse_mdx
from mcad.session_store0 import SESSION_STORE

router = APIRouter(prefix="/sessions", tags=["mdx"])


class EvaluateVisualMdxPayload(BaseModel):
    objective_id: str
    qp: Dict[str, Any]


def _parse_filter_token(token: str) -> Tuple[str, str]:
    """
    Parse a filter token like:
      - "Region=Nord"
      - "Year:1998"
      - "Product.Family=Bio"
    Returns (dim, value) or ("","") if not parseable.
    """
    tok = (token or "").strip()
    if not tok:
        return "", ""
    if "=" in tok:
        a, b = tok.split("=", 1)
        return a.strip(), b.strip()
    if ":" in tok:
        a, b = tok.split(":", 1)
        return a.strip(), b.strip()
    return "", ""


def _looks_like_corr(measures: Any, hint: str) -> bool:
    m = " ".join([str(x) for x in (measures or [])]).lower()
    h = (hint or "").lower()
    if "corr" in h or "correl" in h:
        return True
    # Heuristic: if both "marge" and "rupture/stockout" are present, treat as correlation case
    if ("marge" in m or "margin" in m) and ("rupture" in m or "stockout" in m or "outofstock" in m):
        return True
    return False


def normalize_visual_mdx_qp(qp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Frontend VisualMdxQP -> CKG-first canonical query_spec.

    Frontend sends:
      - cube, measures
      - rows, columns (dimensions)
      - filters (slicers)
      - mdx_hint, step_name, step_description

    CKG expects (directly or inside query_spec):
      - cube
      - measures
      - group_by
      - slicers (dict)
      - analytics (list), e.g., ["corr"] for correlation NVs
    """
    qp = dict(qp or {})
    qspec = dict(qp.get("query_spec") or {})

    cube = qp.get("cube") or qspec.get("cube") or qspec.get("fact")
    measures = qp.get("measures") or qspec.get("measures") or []

    rows = qp.get("rows") or []
    cols = qp.get("columns") or []
    group_by = qspec.get("group_by") or qp.get("group_by") or []
    if not group_by:
        # preserve order, remove duplicates
        seen = set()
        gb = []
        for x in list(rows) + list(cols):
            xs = str(x)
            if xs and xs not in seen:
                gb.append(xs)
                seen.add(xs)
        group_by = gb

    # slicers
    slicers = qspec.get("slicers") or qp.get("slicers") or {}
    if isinstance(slicers, list):
        # convert list tokens to dict
        sdict: Dict[str, str] = {}
        for tok in slicers:
            d, v = _parse_filter_token(str(tok))
            if d:
                sdict[d] = v
        slicers = sdict

    if not slicers:
        filters = qp.get("filters") or []
        sdict: Dict[str, str] = {}
        for tok in filters:
            d, v = _parse_filter_token(str(tok))
            if d:
                sdict[d] = v
        slicers = sdict

    # analytics
    analytics = qspec.get("analytics") or qp.get("analytics") or []
    if not isinstance(analytics, list):
        analytics = [str(analytics)]

    hint = " ".join(
        [
            str(qp.get("mdx_hint") or ""),
            str(qp.get("step_name") or ""),
            str(qp.get("step_description") or ""),
        ]
    ).strip()

    if _looks_like_corr(measures, hint) and "corr" not in [str(a).lower() for a in analytics]:
        analytics.append("corr")

    # Build canonical query_spec
    qspec.update(
        {
            "cube": cube,
            "measures": measures,
            "group_by": group_by,
            "slicers": slicers,
            "analytics": analytics,
        }
    )
    qp["query_spec"] = qspec

    # Keep original fields too (debugging, compatibility)
    qp.setdefault("cube", cube)
    qp.setdefault("measures", measures)
    qp.setdefault("group_by", group_by)
    qp.setdefault("slicers", slicers)
    qp.setdefault("analytics", analytics)

    return qp


@router.post("/{session_id}/evaluate_visual_mdx", response_model=EvaluateWithObjectiveAndSessionResponse)
def api_evaluate_visual_mdx(
    session_id: str,
    payload: EvaluateVisualMdxPayload,
) -> EvaluateWithObjectiveAndSessionResponse:
    """
    Évalue une requête (plan visuel) dans une session et pour un objectif.

    IMPORTANT (CKG-first):
    - Normalise le payload UI (rows/columns/filters) en query_spec (cube/measures/group_by/slicers/analytics)
      pour que le calcul Real/Ceval/φ repose effectivement sur le CKG.
    """
    try:
        SESSION_STORE.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        get_objective(payload.objective_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective not found")

    qp_norm = normalize_visual_mdx_qp(payload.qp or {})

    req = EvaluateWithObjectiveAndSessionRequest(
        session_id=session_id,
        objective_id=payload.objective_id,
        qp=qp_norm,
    )
    return evaluate_with_objective_and_session(req)

@router.post('/parse_mdx')
def api_parse_mdx(payload: Dict[str, Any]) -> Dict[str, Any]:
    mdx = str((payload or {}).get('mdx') or '')
    if not mdx:
        raise HTTPException(status_code=400, detail='mdx is required')
    return parse_mdx(mdx)
