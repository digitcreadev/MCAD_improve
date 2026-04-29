# backend/mcad/engine.py
# (UPGRADED: CKG-first as source of truth for SAT/Real/Ceval/φ and cumulative coverage)

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models000 import (
    EvaluateWithObjectiveAndSessionRequest,
    EvaluateWithObjectiveAndSessionResponse,
    SATClauseResult,
    SessionEvaluationLogEntry,
)
from .objectives import get_objective
from .session_store0 import SESSION_STORE
from .log_store import LOG_STORE
from .evidence_store import EvidenceStore
from .decision_audit_store import DecisionAuditStore


# --- Import CKGGraph robustly (supports different execution modes) ---
try:
    # When running from repo root: namespace package "backend"
    from backend.ckg.ckg_updater import CKGGraph  # type: ignore
except Exception:  # pragma: no cover
    # When running with cwd=backend
    from ckg.ckg_updater import CKGGraph  # type: ignore


ENGINE_VERSION = str(os.environ.get("MCAD_ENGINE_VERSION", "mcad-phase9"))
DECISION_CONTRACT_VERSION = str(os.environ.get("MCAD_DECISION_CONTRACT_VERSION", "mcad.decision.contract.v1"))


def _env_flag(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _results_dir() -> str:
    return str(os.environ.get("MCAD_RESULTS_DIR", "results"))


def _qp_digest(qp: Dict[str, Any]) -> str:
    payload = json.dumps(qp or {}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _objective_version(objective_id: str) -> str:
    obj = get_objective(objective_id)
    payload = json.dumps(obj.model_dump(mode="python"), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _to_clause_results(raw: List[dict]) -> List[SATClauseResult]:
    out: List[SATClauseResult] = []
    for c in raw or []:
        details = c.get("details") or {}
        norm_details = {str(k): str(v) for k, v in (details or {}).items()}
        out.append(SATClauseResult(name=str(c.get("name")), ok=bool(c.get("ok")), details=norm_details))
    return out


def _retained_payload_from_eval(qp: Dict[str, Any], out: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
    qspec = dict(qp.get("query_spec") or qp or {})
    payload: Dict[str, Any] = {
        "step_index": int(step_idx),
        "phi": float(out.get("phi") or 0.0),
        "phi_weighted": float(out.get("phi_weighted") or 0.0),
        "phi_leq_t": float(out.get("phi_leq_t") or 0.0),
        "delta_phi_t": float(out.get("delta_phi_t") or 0.0),
        "phi_weighted_leq_t": float(out.get("phi_weighted_leq_t") or 0.0),
        "delta_phi_weighted_t": float(out.get("delta_phi_weighted_t") or 0.0),
        "cube": qspec.get("cube") or qp.get("cube"),
        "measures": list(qspec.get("measures") or qp.get("measures") or []),
        "group_by": list(qspec.get("group_by") or qp.get("group_by") or []),
        "slicers": dict(qspec.get("slicers") or qp.get("slicers") or {}),
        "time_members": list(qspec.get("time_members") or qp.get("time_members") or []),
        "window_start": qspec.get("window_start") or qp.get("window_start"),
        "window_end": qspec.get("window_end") or qp.get("window_end"),
        "real_node_ids": list(out.get("real_node_ids") or []),
        "calculable_constraints": list(out.get("calculable_constraints") or []),
        "induced_mask_node_ids": list(out.get("induced_mask_node_ids") or []),
        "induced_mask_constraints": dict(out.get("induced_mask_constraints") or {}),
        "missing_requirements": dict(out.get("missing_requirements") or {}),
        "qp_canonical": dict(qspec or {}),
    }
    execution_excerpt = (
        qp.get("execution_result_excerpt")
        or qspec.get("execution_result_excerpt")
        or qp.get("retained_result_excerpt")
        or qspec.get("retained_result_excerpt")
    )
    if execution_excerpt is not None:
        payload["execution_result_excerpt"] = execution_excerpt
    return payload


# Singletons shared by API calls
_CKG: Optional[CKGGraph] = None
_EVIDENCE_STORE: Optional[EvidenceStore] = None
_DECISION_AUDIT_STORE: Optional[DecisionAuditStore] = None


def reset_runtime_state() -> None:
    global _CKG, _EVIDENCE_STORE, _DECISION_AUDIT_STORE
    _CKG = None
    _EVIDENCE_STORE = None
    _DECISION_AUDIT_STORE = None


def get_ckg() -> CKGGraph:
    global _CKG
    if _CKG is None:
        _CKG = CKGGraph(output_dir=_results_dir())
    return _CKG


def get_evidence_store() -> EvidenceStore:
    global _EVIDENCE_STORE
    if _EVIDENCE_STORE is None:
        _EVIDENCE_STORE = EvidenceStore(base_dir=_results_dir())
    return _EVIDENCE_STORE


def get_decision_audit_store() -> DecisionAuditStore:
    global _DECISION_AUDIT_STORE
    if _DECISION_AUDIT_STORE is None:
        _DECISION_AUDIT_STORE = DecisionAuditStore(base_dir=_results_dir())
    return _DECISION_AUDIT_STORE


def bootstrap_session_from_persisted_evidence(
    session_id: str,
    objective_id: str,
    *,
    statuses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    store = get_evidence_store()
    ckg = get_ckg()
    state = SESSION_STORE.get_session(session_id)
    payload = store.bootstrap_payload_for_objective(objective_id, statuses=statuses)
    seeded = ckg.seed_session_coverage_from_evidence(
        session_id=session_id,
        objective_id=objective_id,
        constraint_ids=list(payload.get("constraint_ids") or []),
        evidence_ids=list(payload.get("evidence_ids") or []),
        source="persisted_evidence",
    )
    SESSION_STORE.register_bootstrap(
        session_id,
        bootstrap_constraints=list(seeded.get("seeded_constraint_ids") or []),
        bootstrap_evidence_ids=list(payload.get("evidence_ids") or []),
        bootstrap_phi_leq_t=float(seeded.get("phi_leq_t") or 0.0),
        bootstrap_phi_weighted_leq_t=float(seeded.get("phi_weighted_leq_t") or 0.0),
    )
    return {
        "session_id": str(session_id),
        "objective_id": str(objective_id),
        "seeded_constraint_ids": list(seeded.get("seeded_constraint_ids") or []),
        "seeded_evidence_ids": list(payload.get("evidence_ids") or []),
        "phi_leq_t": float(seeded.get("phi_leq_t") or 0.0),
        "phi_weighted_leq_t": float(seeded.get("phi_weighted_leq_t") or 0.0),
        "dw_id": getattr(state, "dw_id", None),
    }


def _decision_from_eval(sat: bool, ceval_constraints: List[str], clauses: List[SATClauseResult]) -> tuple[str, str, List[str]]:
    failed = [str(c.name) for c in clauses if not bool(c.ok)]
    if sat and ceval_constraints:
        return "ALLOW", "strategically_contributive", failed
    if failed:
        return "BLOCK", f"failed_{failed[0]}", failed
    return "BLOCK", "no_calculable_constraint", failed


def evaluate_with_objective_and_session(
    payload: EvaluateWithObjectiveAndSessionRequest,
) -> EvaluateWithObjectiveAndSessionResponse:
    """
    CKG-first evaluation for a QP inside (session, objective).

    Source of truth:
      - SAT/Real/Ceval/φ are computed by the CKG engine (CKGGraph.evaluate_step)
      - Session cumulative coverage is maintained by the CKG engine (monotone)

    Palier 1 extension:
      - contributive useful evidence is persisted in a dedicated store,
      - minimally linked back to the graph and to a snapshot,
      - without claiming full lifecycle-aware graph governance.
    """
    _ = get_objective(payload.objective_id)
    state = SESSION_STORE.get_session(payload.session_id)
    step_idx = int(state.step_index) + 1

    ckg = get_ckg()
    qp = dict(payload.qp or {})
    qp.setdefault("objective_id", payload.objective_id)
    query_digest = _qp_digest(qp)
    objective_version = _objective_version(payload.objective_id)
    pre_snapshot_id = f"SNAP_PRE_{payload.session_id}_t{step_idx:03d}_{query_digest[:8]}"
    pre_snapshot_path = ckg.snapshot_path(payload.session_id, pre_snapshot_id)
    ckg.save_snapshot(payload.session_id, snapshot_id=pre_snapshot_id)

    out = ckg.evaluate_step(
        session_id=payload.session_id,
        objective_id=payload.objective_id,
        step_idx=step_idx,
        qp=qp,
    )

    clauses = _to_clause_results(out.get("clauses") or [])

    sat = bool(out.get("sat"))
    phi = float(out.get("phi") or 0.0)
    phi_w = float(out.get("phi_weighted") or 0.0)
    real_node_ids = list(out.get("real_node_ids") or [])
    ceval_constraints = list(out.get("calculable_constraints") or [])
    covered_constraints = list(out.get("covered_constraints") or [])

    phi_leq_t = out.get("phi_leq_t")
    delta_phi_t = out.get("delta_phi_t")
    phi_weighted_leq_t = out.get("phi_weighted_leq_t")
    delta_phi_weighted_t = out.get("delta_phi_weighted_t")
    if phi_leq_t is None:
        phi_leq_t = float(state.phi_leq_t)
    if delta_phi_t is None:
        delta_phi_t = 0.0
    if phi_weighted_leq_t is None:
        phi_weighted_leq_t = float(getattr(state, "phi_weighted_leq_t", 0.0))
    if delta_phi_weighted_t is None:
        delta_phi_weighted_t = 0.0

    SESSION_STORE.update_from_ckg(
        session_id=payload.session_id,
        step_index=step_idx,
        phi_leq_t=float(phi_leq_t),
        covered_constraints=covered_constraints,
        phi_weighted_leq_t=float(phi_weighted_leq_t),
    )

    retained_evidence_id: Optional[str] = None
    retained_snapshot_id: Optional[str] = None
    retained_snapshot_path: Optional[str] = None
    if sat and ceval_constraints:
        evidence_store = get_evidence_store()
        retained_evidence_id = evidence_store.make_evidence_id(payload.session_id, step_idx, query_digest)
        retained_snapshot_id = f"SNAP_{payload.session_id}_t{step_idx:03d}_{query_digest[:8]}"
        retained_snapshot_path = ckg.snapshot_path(payload.session_id, retained_snapshot_id)
        qspec = dict(qp.get("query_spec") or qp or {})
        linked_virtual_nodes = list(out.get("induced_mask_node_ids") or out.get("real_node_ids") or [])
        linked_requirement_sets = dict(out.get("induced_mask_constraints") or {})
        retained_payload = _retained_payload_from_eval(qp, out, step_idx)
        record = {
            "evidence_id": retained_evidence_id,
            "session_id": payload.session_id,
            "objective_id": payload.objective_id,
            "step_index": int(step_idx),
            "query_digest": query_digest,
            "query_language": str(qspec.get("language") or qp.get("language") or "mdx-or-canonical"),
            "constraint_ids": sorted([str(x) for x in ceval_constraints]),
            "linked_virtual_nodes": sorted([str(x) for x in linked_virtual_nodes]),
            "linked_requirement_sets": {
                str(k): [str(x) for x in (v or [])] for k, v in linked_requirement_sets.items()
            },
            "retained_payload": retained_payload,
            "snapshot_id": retained_snapshot_id,
            "snapshot_path": retained_snapshot_path,
            "pre_snapshot_id": pre_snapshot_id,
            "pre_snapshot_path": pre_snapshot_path,
            "post_snapshot_id": retained_snapshot_id,
            "post_snapshot_path": retained_snapshot_path,
            "objective_version": objective_version,
            "engine_version": ENGINE_VERSION,
            "status": "active",
            "evidence_type": "contributive_query_useful_part",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        ckg.attach_evidence(record=record, qp_node=str(out.get("qp_node_id") or ""))
        ckg.save_snapshot(payload.session_id, snapshot_id=retained_snapshot_id)
        evidence_store.persist_contributive_evidence(
            evidence_id=retained_evidence_id,
            session_id=payload.session_id,
            objective_id=payload.objective_id,
            step_index=step_idx,
            query_digest=query_digest,
            query_language=str(qspec.get("language") or qp.get("language") or "mdx-or-canonical"),
            constraint_ids=sorted([str(x) for x in ceval_constraints]),
            linked_virtual_nodes=sorted([str(x) for x in linked_virtual_nodes]),
            linked_requirement_sets={str(k): [str(x) for x in (v or [])] for k, v in linked_requirement_sets.items()},
            retained_payload=retained_payload,
            snapshot_id=retained_snapshot_id,
            snapshot_path=retained_snapshot_path,
            pre_snapshot_id=pre_snapshot_id,
            pre_snapshot_path=pre_snapshot_path,
            post_snapshot_id=retained_snapshot_id,
            post_snapshot_path=retained_snapshot_path,
            objective_version=objective_version,
            engine_version=ENGINE_VERSION,
            status="active",
        )
        SESSION_STORE.register_evidence(payload.session_id, retained_evidence_id, snapshot_id=retained_snapshot_id)

    decision, decision_reason, failed_predicates = _decision_from_eval(sat, ceval_constraints, clauses)
    explanation_id = f"EXPL_{payload.session_id}_t{step_idx:03d}_{query_digest[:8]}"
    ckg_snapshot_id = retained_snapshot_id or pre_snapshot_id
    ckg_snapshot_path = retained_snapshot_path or pre_snapshot_path
    get_decision_audit_store().persist_record({
        "explanation_id": explanation_id,
        "session_id": str(payload.session_id),
        "objective_id": str(payload.objective_id),
        "step_index": int(step_idx),
        "decision": str(decision),
        "decision_reason": str(decision_reason),
        "sat": bool(sat),
        "failed_predicates": [str(x) for x in failed_predicates],
        "matched_virtual_nodes": sorted([str(x) for x in real_node_ids]),
        "calculable_constraints": sorted([str(x) for x in ceval_constraints]),
        "missing_requirements": {str(k): [str(x) for x in (v or [])] for k, v in (out.get("missing_requirements") or {}).items()},
        "induced_mask_constraints": {str(k): [str(x) for x in (v or [])] for k, v in (out.get("induced_mask_constraints") or {}).items()},
        "query_digest": str(query_digest),
        "objective_version": str(objective_version),
        "engine_version": str(ENGINE_VERSION),
        "contract_version": str(DECISION_CONTRACT_VERSION),
        "ckg_snapshot_id": str(ckg_snapshot_id) if ckg_snapshot_id else None,
        "ckg_snapshot_path": str(ckg_snapshot_path) if ckg_snapshot_path else None,
        "pre_snapshot_id": str(pre_snapshot_id),
        "pre_snapshot_path": str(pre_snapshot_path),
        "post_snapshot_id": str(retained_snapshot_id) if retained_snapshot_id else None,
        "post_snapshot_path": str(retained_snapshot_path) if retained_snapshot_path else None,
        "retained_evidence_id": str(retained_evidence_id) if retained_evidence_id else None,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })

    LOG_STORE.append(
        SessionEvaluationLogEntry(
            session_id=payload.session_id,
            objective_id=payload.objective_id,
            t=step_idx,
            timestamp=datetime.now(timezone.utc),
            phi=phi,
            phi_weighted=phi_w,
            phi_leq_t=float(phi_leq_t),
            delta_phi_t=float(delta_phi_t),
            phi_weighted_leq_t=float(phi_weighted_leq_t),
            delta_phi_weighted_t=float(delta_phi_weighted_t),
            sat=sat,
            calculable_constraints=sorted(ceval_constraints),
            clauses=clauses,
        )
    )

    if _env_flag("MCAD_CKG_PERSIST_EACH_STEP", "0"):
        try:
            ckg.save_snapshot(payload.session_id)
        except Exception:
            pass

    state_after = SESSION_STORE.get_session(payload.session_id)
    return EvaluateWithObjectiveAndSessionResponse(
        sat=sat,
        clauses=clauses,
        phi=phi,
        phi_weighted=phi_w,
        real_node_ids=sorted([str(x) for x in real_node_ids]),
        calculable_constraints=sorted([str(x) for x in ceval_constraints]),
        phi_leq_t=float(phi_leq_t),
        delta_phi_t=float(delta_phi_t),
        step_index=step_idx,
        covered_constraints=sorted([str(x) for x in covered_constraints]),
        phi_weighted_leq_t=float(phi_weighted_leq_t),
        delta_phi_weighted_t=float(delta_phi_weighted_t),
        induced_mask_node_ids=sorted([str(x) for x in (out.get("induced_mask_node_ids") or [])]),
        induced_mask_constraints={str(k): [str(x) for x in (v or [])] for k, v in (out.get("induced_mask_constraints") or {}).items()},
        missing_requirements={str(k): [str(x) for x in (v or [])] for k, v in (out.get("missing_requirements") or {}).items()},
        retained_evidence_id=retained_evidence_id,
        retained_snapshot_id=retained_snapshot_id,
        retained_snapshot_path=retained_snapshot_path,
        explanation_id=explanation_id,
        query_digest=query_digest,
        objective_version=objective_version,
        contract_version=DECISION_CONTRACT_VERSION,
        decision=decision,
        decision_reason=decision_reason,
        failed_predicates=list(failed_predicates),
        bootstrap_constraints=list(getattr(state_after, "bootstrap_constraints", []) or []),
        bootstrap_evidence_ids=list(getattr(state_after, "bootstrap_evidence_ids", []) or []),
        bootstrap_phi_leq_t=float(getattr(state_after, "bootstrap_phi_leq_t", 0.0) or 0.0),
    )



def replay_retained_evidence(evidence_id: str) -> Dict[str, Any]:
    store = get_evidence_store()
    rec = store.get(evidence_id)
    if rec is None:
        raise KeyError(str(evidence_id))
    pre_snapshot_path = rec.get("pre_snapshot_path")
    if not pre_snapshot_path:
        return {
            "evidence_id": str(evidence_id),
            "replay_supported": False,
            "exact_match": False,
            "objective_id": str(rec.get("objective_id") or ""),
            "objective_version": rec.get("objective_version"),
            "engine_version": rec.get("engine_version"),
            "pre_snapshot_path": None,
            "post_snapshot_path": rec.get("post_snapshot_path") or rec.get("snapshot_path"),
            "compared_fields": [],
            "mismatches": {"reason": "missing pre_snapshot_path"},
            "replay_output": {},
        }
    ckg = CKGGraph.load_snapshot(str(pre_snapshot_path))
    retained = dict(rec.get("retained_payload") or {})
    qp = dict(retained.get("qp_canonical") or {})
    if not qp:
        qp = {
            "cube": retained.get("cube"),
            "measures": list(retained.get("measures") or []),
            "group_by": list(retained.get("group_by") or []),
            "slicers": dict(retained.get("slicers") or {}),
            "time_members": list(retained.get("time_members") or []),
            "window_start": retained.get("window_start"),
            "window_end": retained.get("window_end"),
        }
    qp["objective_id"] = str(rec.get("objective_id") or "")
    session_id = f"REPLAY_{str(rec.get('session_id') or 'S')}_{str(evidence_id)}"
    step_idx = int(rec.get("step_index") or 0)
    out = ckg.evaluate_step(session_id=session_id, objective_id=str(rec.get("objective_id") or ""), step_idx=step_idx, qp=qp)
    compared = ["sat", "phi", "phi_weighted", "calculable_constraints", "induced_mask_node_ids", "induced_mask_constraints", "missing_requirements"]
    expected = {
        "sat": True,
        "phi": round(float(retained.get("phi") or 0.0), 6),
        "phi_weighted": round(float(retained.get("phi_weighted") or 0.0), 6),
        "calculable_constraints": sorted([str(x) for x in (rec.get("constraint_ids") or [])]),
        "induced_mask_node_ids": sorted([str(x) for x in (retained.get("induced_mask_node_ids") or rec.get("linked_virtual_nodes") or [])]),
        "induced_mask_constraints": {str(k): sorted([str(x) for x in (v or [])]) for k, v in (retained.get("induced_mask_constraints") or rec.get("linked_requirement_sets") or {}).items()},
        "missing_requirements": {str(k): sorted([str(x) for x in (v or [])]) for k, v in (retained.get("missing_requirements") or {}).items()},
    }
    actual = {
        "sat": bool(out.get("sat")),
        "phi": round(float(out.get("phi") or 0.0), 6),
        "phi_weighted": round(float(out.get("phi_weighted") or 0.0), 6),
        "calculable_constraints": sorted([str(x) for x in (out.get("calculable_constraints") or [])]),
        "induced_mask_node_ids": sorted([str(x) for x in (out.get("induced_mask_node_ids") or [])]),
        "induced_mask_constraints": {str(k): sorted([str(x) for x in (v or [])]) for k, v in (out.get("induced_mask_constraints") or {}).items()},
        "missing_requirements": {str(k): sorted([str(x) for x in (v or [])]) for k, v in (out.get("missing_requirements") or {}).items()},
    }
    mismatches = {}
    for key in compared:
        if expected.get(key) != actual.get(key):
            mismatches[key] = {"expected": expected.get(key), "actual": actual.get(key)}
    return {
        "evidence_id": str(evidence_id),
        "replay_supported": True,
        "exact_match": not mismatches,
        "objective_id": str(rec.get("objective_id") or ""),
        "objective_version": rec.get("objective_version"),
        "engine_version": rec.get("engine_version"),
        "pre_snapshot_path": str(pre_snapshot_path),
        "post_snapshot_path": rec.get("post_snapshot_path") or rec.get("snapshot_path"),
        "compared_fields": compared,
        "mismatches": mismatches,
        "replay_output": actual,
    }
