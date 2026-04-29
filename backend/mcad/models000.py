# backend/mcad/models.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VirtualNode(BaseModel):
    """
    Nœud virtuel du CKG associé à une contrainte et à un KPI.
    Représente un point de calcul potentiel dans l'entrepôt.
    """
    id: str
    fact: str
    grain: List[str]
    measure: str
    aggregator: str
    unit: str
    slicers: Dict[str, str] = Field(default_factory=dict)
    window_start: Optional[str] = None
    window_end: Optional[str] = None


class Constraint(BaseModel):
    """
    Contrainte d'un objectif stratégique/métier.

    requirement_sets permet d'exprimer explicitement les couvertures suffisantes
    utilisées par Ceval(QP, O). Si absent, le prototype retombe sur l'ensemble
    complet des virtual_nodes de la contrainte.
    """
    id: str
    kpi_id: str
    description: str
    weight: float = 1.0
    requirement_sets: List[List[str]] = Field(default_factory=list)
    virtual_nodes: List[VirtualNode] = Field(default_factory=list)


class Objective(BaseModel):
    """
    Objectif stratégique/métier (global) dans le MCAD.
    """
    id: str
    name: str
    description: str
    kpis: List[str] = Field(default_factory=list)
    constraints: List[Constraint] = Field(default_factory=list)


class SATClauseResult(BaseModel):
    """
    Résultat d'une clause de SAT(QP) (grain, corrélation, etc.).
    """
    name: str
    ok: bool
    details: Dict[str, str] = Field(default_factory=dict)


class EvaluateWithObjectiveAndSessionRequest(BaseModel):
    """
    Requête d'évaluation MCAD : (session, objectif, plan QP abstrait).
    """
    session_id: str
    objective_id: str
    qp: Dict[str, Any]


class EvaluateWithObjectiveAndSessionResponse(BaseModel):
    """
    Réponse d'évaluation MCAD pour une requête QP donnée.
    """
    sat: bool
    clauses: List[SATClauseResult]
    phi: float
    phi_weighted: float
    real_node_ids: List[str]
    calculable_constraints: List[str]
    calculable_constraints_total: List[str] = Field(default_factory=list)
    calculable_constraints_partial: List[str] = Field(default_factory=list)
    non_calculable_constraints: List[str] = Field(default_factory=list)
    newly_contributed_constraints_total: List[str] = Field(default_factory=list)
    newly_contributed_constraints_partial: List[str] = Field(default_factory=list)
    gained_resource_ids: List[str] = Field(default_factory=list)
    support_progress_by_constraint: Dict[str, Any] = Field(default_factory=dict)
    is_session_contributive: bool = False
    phi_leq_t: Optional[float] = None
    delta_phi_t: Optional[float] = None
    phi_weighted_leq_t: Optional[float] = None
    delta_phi_weighted_t: Optional[float] = None
    step_index: Optional[int] = None
    covered_constraints: List[str] = Field(default_factory=list)
    induced_mask_node_ids: List[str] = Field(default_factory=list)
    induced_mask_constraints: Dict[str, List[str]] = Field(default_factory=dict)
    missing_requirements: Dict[str, List[str]] = Field(default_factory=dict)
    retained_evidence_id: Optional[str] = None
    retained_snapshot_id: Optional[str] = None
    retained_snapshot_path: Optional[str] = None
    explanation_id: Optional[str] = None
    query_digest: Optional[str] = None
    objective_version: Optional[str] = None
    contract_version: Optional[str] = None
    decision: Optional[str] = None
    decision_reason_code: Optional[str] = None
    decision_reason: Optional[str] = None
    is_redundant: bool = False
    has_marginal_gain: bool = False
    gained_resource_ids_count: int = 0
    failed_predicates: List[str] = Field(default_factory=list)
    bootstrap_constraints: List[str] = Field(default_factory=list)
    bootstrap_evidence_ids: List[str] = Field(default_factory=list)
    bootstrap_phi_leq_t: float = 0.0


class SessionState(BaseModel):
    """
    État courant d'une session d'analyse.
    """
    session_id: str
    objective_id: str
    dw_id: str
    created_at: datetime
    status: str = "open"
    step_index: int = 0
    phi_leq_t: float = 0.0
    phi_weighted_leq_t: float = 0.0
    covered_constraints: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    archived_evidence_ids: List[str] = Field(default_factory=list)
    latest_snapshot_id: Optional[str] = None
    latest_session_summary_path: Optional[str] = None
    bootstrap_constraints: List[str] = Field(default_factory=list)
    bootstrap_evidence_ids: List[str] = Field(default_factory=list)
    bootstrap_phi_leq_t: float = 0.0


class SessionEvaluationLogEntry(BaseModel):
    """
    Log d'une évaluation MCAD dans une session à l'instant t.
    """
    session_id: str
    objective_id: str
    t: int
    timestamp: datetime
    phi: float
    phi_weighted: float
    phi_leq_t: Optional[float]
    delta_phi_t: Optional[float]
    phi_weighted_leq_t: Optional[float] = None
    delta_phi_weighted_t: Optional[float] = None
    sat: bool
    calculable_constraints: List[str]
    clauses: List[SATClauseResult]


class SessionTimelineStep(BaseModel):
    """
    Étape de timeline pour une session.
    """
    t: int
    timestamp: datetime
    phi: float
    phi_weighted: Optional[float]
    phi_leq_t: Optional[float]
    delta_phi_t: Optional[float]
    phi_weighted_leq_t: Optional[float] = None
    delta_phi_weighted_t: Optional[float] = None
    sat: bool
    calculable_constraints: List[str]
    clauses: List[SATClauseResult]


class SessionTimelineResponse(BaseModel):
    """
    Timeline complète d'une session (vue analytics).
    """
    session_id: str
    objective_id: str
    steps: List[SessionTimelineStep]


class ObjectivePerformanceResponse(BaseModel):
    """
    Agrégation des performances MCAD pour un objectif :
    - courbes φ≤t(O)
    - AUC par session
    - heatmap de couverture des contraintes
    """
    objective_id: str
    sessionIds: List[str]
    phiCurves: List[Dict[str, float]]
    aucBySession: List[Dict[str, float]]
    constraintIds: List[str]
    constraintCoverageHeatmap: List[Dict[str, Any]]


class EvidenceRecord(BaseModel):
    evidence_id: str
    session_id: str
    objective_id: str
    step_index: int
    query_digest: str
    query_language: str
    constraint_ids: List[str] = Field(default_factory=list)
    linked_virtual_nodes: List[str] = Field(default_factory=list)
    linked_requirement_sets: Dict[str, List[str]] = Field(default_factory=dict)
    retained_payload: Dict[str, Any] = Field(default_factory=dict)
    snapshot_id: Optional[str] = None
    snapshot_path: Optional[str] = None
    pre_snapshot_id: Optional[str] = None
    pre_snapshot_path: Optional[str] = None
    post_snapshot_id: Optional[str] = None
    post_snapshot_path: Optional[str] = None
    objective_version: Optional[str] = None
    engine_version: str = "mcad-palier1"
    evidence_type: str = "contributive_query_useful_part"
    status: str = "active"
    created_at: str
    archived_at: Optional[str] = None
    expired_at: Optional[str] = None
    archive_path: Optional[str] = None
    session_summary_path: Optional[str] = None
    lifecycle_reason: Optional[str] = None


class EvidenceStoreStats(BaseModel):
    n_records: int
    n_sessions: int
    n_objectives: int
    n_constraints: int
    records_path: str
    by_status: Dict[str, int] = Field(default_factory=dict)
    n_archived_with_payload: int = 0


class EvidenceUsefulnessRecord(BaseModel):
    evidence_id: str
    objective_id: str
    session_id: str
    status: str
    realized_nv_count: int = 0
    retained_nv_count: int = 0
    retained_ratio: float = 0.0
    calculable_constraint_count: int = 0
    requirement_link_count: int = 0
    execution_excerpt_present: bool = False
    execution_excerpt_items: int = 0
    usefulness_score: float = 0.0


class EvidenceUsefulnessReport(BaseModel):
    generated_at: str
    n_records: int
    useful_evidence_ratio: float = 0.0
    mean_realized_nv_count: float = 0.0
    mean_retained_nv_count: float = 0.0
    mean_retained_ratio: float = 0.0
    mean_calculable_constraint_count: float = 0.0
    mean_requirement_link_count: float = 0.0
    execution_excerpt_coverage_ratio: float = 0.0
    mean_execution_excerpt_items: float = 0.0
    mean_usefulness_score: float = 0.0
    objective_bootstrap_coverage: Dict[str, float] = Field(default_factory=dict)
    objective_bootstrap_constraints: Dict[str, List[str]] = Field(default_factory=dict)
    per_record: List[EvidenceUsefulnessRecord] = Field(default_factory=list)


class EvidenceBootstrapResponse(BaseModel):
    session_id: str
    objective_id: str
    seeded_constraint_ids: List[str] = Field(default_factory=list)
    seeded_evidence_ids: List[str] = Field(default_factory=list)
    phi_leq_t: float = 0.0
    phi_weighted_leq_t: float = 0.0


class GovernanceReplayResponse(BaseModel):
    evidence_id: str
    replay_supported: bool
    exact_match: bool
    objective_id: str
    objective_version: Optional[str] = None
    engine_version: Optional[str] = None
    pre_snapshot_path: Optional[str] = None
    post_snapshot_path: Optional[str] = None
    compared_fields: List[str] = Field(default_factory=list)
    mismatches: Dict[str, Any] = Field(default_factory=dict)
    replay_output: Dict[str, Any] = Field(default_factory=dict)


class GovernanceReportResponse(BaseModel):
    generated_at: str
    n_records: int
    by_status: Dict[str, int] = Field(default_factory=dict)
    n_sessions: int = 0
    n_objectives: int = 0
    n_constraints: int = 0
    n_active_with_snapshots: int = 0
    n_replay_ready: int = 0
    n_duplicate_groups: int = 0
    n_duplicate_records: int = 0
    evidence_per_session: Dict[str, int] = Field(default_factory=dict)
    by_objective: Dict[str, int] = Field(default_factory=dict)
    latest_engine_versions: List[str] = Field(default_factory=list)
    ckg_stats: Dict[str, Any] = Field(default_factory=dict)


class DecisionAuditRecord(BaseModel):
    explanation_id: str
    session_id: str
    objective_id: str
    step_index: int
    decision: str
    decision_reason: Optional[str] = None
    sat: bool
    failed_predicates: List[str] = Field(default_factory=list)
    matched_virtual_nodes: List[str] = Field(default_factory=list)
    calculable_constraints: List[str] = Field(default_factory=list)
    missing_requirements: Dict[str, List[str]] = Field(default_factory=dict)
    induced_mask_constraints: Dict[str, List[str]] = Field(default_factory=dict)
    query_digest: str
    objective_version: Optional[str] = None
    engine_version: Optional[str] = None
    contract_version: Optional[str] = None
    ckg_snapshot_id: Optional[str] = None
    ckg_snapshot_path: Optional[str] = None
    pre_snapshot_id: Optional[str] = None
    pre_snapshot_path: Optional[str] = None
    post_snapshot_id: Optional[str] = None
    post_snapshot_path: Optional[str] = None
    retained_evidence_id: Optional[str] = None
    created_at: str


class DecisionAuditStats(BaseModel):
    n_records: int
    by_decision: Dict[str, int] = Field(default_factory=dict)
    contract_versions: Dict[str, int] = Field(default_factory=dict)
    engine_versions: Dict[str, int] = Field(default_factory=dict)
    with_retained_evidence: int = 0
    with_decision_reasons: int = 0
    records_path: str
