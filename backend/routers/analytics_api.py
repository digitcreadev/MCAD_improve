# backend/routers/analytics_api.py
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from mcad.models000 import (
    EvidenceBootstrapResponse,
    EvidenceRecord,
    EvidenceStoreStats,
    EvidenceUsefulnessReport,
    GovernanceReplayResponse,
    GovernanceReportResponse,
    ObjectivePerformanceResponse,
    DecisionAuditRecord,
    DecisionAuditStats,
    SessionTimelineResponse,
    SessionTimelineStep,
)
from mcad.objectives import get_objective
from mcad.session_store0 import SESSION_STORE
from mcad.log_store import LOG_STORE
from mcad.engine000 import bootstrap_session_from_persisted_evidence, get_evidence_store, get_ckg, replay_retained_evidence, get_decision_audit_store

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/sessions/{session_id}/timeline", response_model=SessionTimelineResponse)
def api_session_timeline(session_id: str) -> SessionTimelineResponse:
    try:
        state = SESSION_STORE.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    logs = LOG_STORE.list_for_session(session_id)
    steps: List[SessionTimelineStep] = []
    for e in sorted(logs, key=lambda x: x.t):
        steps.append(
            SessionTimelineStep(
                t=e.t,
                timestamp=e.timestamp,
                phi=e.phi,
                phi_weighted=e.phi_weighted,
                phi_leq_t=e.phi_leq_t,
                delta_phi_t=e.delta_phi_t,
                sat=e.sat,
                calculable_constraints=e.calculable_constraints,
                clauses=e.clauses,
            )
        )

    return SessionTimelineResponse(
        session_id=session_id,
        objective_id=state.objective_id,
        steps=steps,
    )


@router.get("/objectives/{objective_id}/performance", response_model=ObjectivePerformanceResponse)
def api_objective_performance(objective_id: str) -> ObjectivePerformanceResponse:
    try:
        objective = get_objective(objective_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective not found")

    sessions = [s for s in SESSION_STORE.list_sessions() if s.objective_id == objective_id]
    session_ids = [s.session_id for s in sessions]

    if not sessions:
        return ObjectivePerformanceResponse(
            objective_id=objective_id,
            sessionIds=[],
            phiCurves=[],
            aucBySession=[],
            constraintIds=[c.id for c in objective.constraints],
            constraintCoverageHeatmap=[],
        )

    logs_by_session: Dict[str, List] = {}
    max_t = 0
    for sid in session_ids:
        logs = sorted(LOG_STORE.list_for_session(sid), key=lambda x: x.t)
        logs_by_session[sid] = logs
        if logs:
            max_t = max(max_t, logs[-1].t)

    phiCurves: List[Dict[str, float]] = []
    last_phi: Dict[str, float] = {sid: 0.0 for sid in session_ids}
    for t in range(1, max_t + 1):
        row: Dict[str, float] = {"t": float(t)}
        for sid in session_ids:
            logs = logs_by_session[sid]
            value = last_phi[sid]
            for e in logs:
                if e.t == t and e.phi_leq_t is not None:
                    value = float(e.phi_leq_t)
                    break
            last_phi[sid] = value
            row[sid] = value
        phiCurves.append(row)

    aucBySession: List[Dict[str, float]] = []
    for sid in session_ids:
        logs = logs_by_session[sid]
        if not logs:
            aucBySession.append({"sessionId": sid, "auc": 0.0})
            continue
        vals = [float(e.phi_leq_t or 0.0) for e in logs]
        auc = sum(vals) / float(len(vals))
        aucBySession.append({"sessionId": sid, "auc": auc})

    constraint_ids = [c.id for c in objective.constraints]
    heat_cells = []
    for cid in constraint_ids:
        for sid in session_ids:
            logs = logs_by_session[sid]
            first_t = None
            for e in logs:
                if cid in e.calculable_constraints:
                    first_t = e.t
                    break
            if first_t is None or max_t == 0:
                value = 0.0
            else:
                value = 1.0 - float(first_t - 1) / float(max_t)
            heat_cells.append({"constraintId": cid, "sessionId": sid, "value": value})

    return ObjectivePerformanceResponse(
        objective_id=objective_id,
        sessionIds=session_ids,
        phiCurves=phiCurves,
        aucBySession=aucBySession,
        constraintIds=constraint_ids,
        constraintCoverageHeatmap=heat_cells,
    )


@router.get("/evidence/stats", response_model=EvidenceStoreStats)
def api_evidence_stats() -> EvidenceStoreStats:
    return EvidenceStoreStats(**get_evidence_store().stats())


@router.get("/evidence/{evidence_id}", response_model=EvidenceRecord)
def api_get_evidence(evidence_id: str) -> EvidenceRecord:
    rec = get_evidence_store().get(evidence_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return EvidenceRecord(**rec)


@router.get("/evidence", response_model=List[EvidenceRecord])
def api_list_evidence(status: Optional[str] = Query(default=None)) -> List[EvidenceRecord]:
    store = get_evidence_store()
    rows = store.list_by_status(status) if status else store.list_all()
    return [EvidenceRecord(**r) for r in rows]


@router.get("/sessions/{session_id}/evidence", response_model=List[EvidenceRecord])
def api_list_session_evidence(session_id: str, status: Optional[str] = Query(default=None)) -> List[EvidenceRecord]:
    try:
        SESSION_STORE.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    statuses = [status] if status else None
    return [EvidenceRecord(**r) for r in get_evidence_store().list_for_session(session_id, statuses=statuses)]


@router.get("/objectives/{objective_id}/evidence", response_model=List[EvidenceRecord])
def api_list_objective_evidence(objective_id: str, status: Optional[str] = Query(default=None)) -> List[EvidenceRecord]:
    try:
        get_objective(objective_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective not found")
    statuses = [status] if status else None
    return [EvidenceRecord(**r) for r in get_evidence_store().list_for_objective(objective_id, statuses=statuses)]


@router.post("/sessions/{session_id}/evidence/compact")
def api_compact_session_evidence(session_id: str, keep_last_n: int = Query(default=8, ge=0)) -> Dict[str, object]:
    try:
        SESSION_STORE.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    store = get_evidence_store()
    ckg = get_ckg()
    result = store.compact_session(session_id, keep_last_n=keep_last_n)
    graph_result = ckg.compact_session_evidence_nodes(session_id, keep_last_n_steps=keep_last_n)
    active = [r["evidence_id"] for r in store.list_for_session(session_id, statuses=["active", "temporary"])]
    archived = [r["evidence_id"] for r in store.list_for_session(session_id, statuses=["archived"])]
    SESSION_STORE.register_evidence_lifecycle(
        session_id,
        active_evidence_ids=active,
        archived_evidence_ids=archived,
        session_summary_path=result.get("summary_path"),
    )
    summary = {
        "summary_id": f"SUM_{session_id}_compact",
        "summary_path": result.get("summary_path"),
        "generated_at": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        "status": "archived",
        "keep_last_n": int(keep_last_n),
        "evidence_ids": active + archived,
    }
    ckg.attach_session_summary(session_id, summary)
    return {"store": result, "graph": graph_result}


@router.post("/sessions/{session_id}/evidence/archive")
def api_archive_session_evidence(session_id: str) -> Dict[str, object]:
    try:
        SESSION_STORE.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    store = get_evidence_store()
    ckg = get_ckg()
    result = store.archive_session(session_id)
    active = [r["evidence_id"] for r in store.list_for_session(session_id, statuses=["active", "temporary"])]
    archived = [r["evidence_id"] for r in store.list_for_session(session_id, statuses=["archived"])]
    for eid in archived:
        ckg.update_evidence_status(eid, 'archived')
    SESSION_STORE.register_evidence_lifecycle(
        session_id,
        active_evidence_ids=active,
        archived_evidence_ids=archived,
        session_summary_path=result.get("summary_path"),
    )
    summary = {
        "summary_id": f"SUM_{session_id}_archive",
        "summary_path": result.get("summary_path"),
        "generated_at": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        "status": "archived",
        "keep_last_n": 0,
        "evidence_ids": active + archived,
    }
    ckg.attach_session_summary(session_id, summary)
    return result


@router.post("/evidence/expire")
def api_expire_evidence(max_age_days: int = Query(default=0, ge=0)) -> Dict[str, object]:
    if max_age_days <= 0:
        raise HTTPException(status_code=400, detail="max_age_days must be > 0")
    store = get_evidence_store()
    ckg = get_ckg()
    result = store.expire_before(max_age_days=max_age_days)
    for eid in result.get("expired_ids", []):
        ckg.update_evidence_status(str(eid), 'expired')
    return result


@router.get("/evidence/usefulness/report", response_model=EvidenceUsefulnessReport)
def api_evidence_usefulness_report() -> EvidenceUsefulnessReport:
    ckg = get_ckg()
    objective_totals = {str(oid): int(len((obj.get("constraints") or {}).keys())) for oid, obj in (ckg.objectives or {}).items()}
    payload = get_evidence_store().usefulness_report(objective_constraint_totals=objective_totals)
    return EvidenceUsefulnessReport(**payload)


@router.post("/objectives/{objective_id}/bootstrap-session", response_model=EvidenceBootstrapResponse)
def api_bootstrap_session_from_evidence(objective_id: str, session_id: str = Query(...), statuses: Optional[List[str]] = Query(default=None)) -> EvidenceBootstrapResponse:
    try:
        get_objective(objective_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective not found")
    try:
        SESSION_STORE.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    payload = bootstrap_session_from_persisted_evidence(session_id=session_id, objective_id=objective_id, statuses=statuses)
    return EvidenceBootstrapResponse(**payload)


@router.get("/governance/report", response_model=GovernanceReportResponse)
def api_governance_report() -> GovernanceReportResponse:
    store = get_evidence_store()
    ckg = get_ckg()
    return GovernanceReportResponse(**store.governance_report(ckg_stats=ckg.graph_stats()))


@router.get("/evidence/{evidence_id}/replay", response_model=GovernanceReplayResponse)
def api_replay_evidence(evidence_id: str) -> GovernanceReplayResponse:
    try:
        payload = replay_retained_evidence(evidence_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return GovernanceReplayResponse(**payload)


@router.get("/decision-audit/stats", response_model=DecisionAuditStats)
def api_decision_audit_stats() -> DecisionAuditStats:
    return DecisionAuditStats(**get_decision_audit_store().stats())


@router.get("/decision-audit", response_model=List[DecisionAuditRecord])
def api_list_decision_audit(session_id: Optional[str] = Query(default=None)) -> List[DecisionAuditRecord]:
    store = get_decision_audit_store()
    rows = store.list_for_session(session_id) if session_id else store.list_all()
    return [DecisionAuditRecord(**r) for r in rows]


@router.get("/decision-audit/{explanation_id}", response_model=DecisionAuditRecord)
def api_get_decision_audit(explanation_id: str) -> DecisionAuditRecord:
    rec = get_decision_audit_store().get(explanation_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Decision audit record not found")
    return DecisionAuditRecord(**rec)
