from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .models import SessionState, SessionHistoryEntry


class SessionStore:
    """
    In-memory store for analysis sessions.

    Notes:
    - update_contribution() is kept for compatibility with older code paths.
    - update_from_ckg() is the preferred entry point when the CKG is the source of truth.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}

    def create_session(self, objective_id: str, dw_id: str) -> SessionState:
        session_id = f"S_{len(self._sessions) + 1:04d}"
        state = SessionState(
            session_id=session_id,
            objective_id=objective_id,
            dw_id=dw_id,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = state
        return state


    def create_session_with_id(self, session_id: str, objective_id: str, dw_id: str) -> SessionState:
        state = SessionState(
            session_id=session_id,
            objective_id=objective_id,
            dw_id=dw_id,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = state
        return state

    def next_session_id(self) -> str:
        return f"S_{len(self._sessions) + 1:04d}"

    def ensure_session(self, session_id: str, objective_id: str, dw_id: str = "foodmart") -> SessionState:
        if session_id in self._sessions:
            state = self._sessions[session_id]
            if objective_id and state.objective_id != objective_id:
                state.objective_id = objective_id
            if dw_id and state.dw_id != dw_id:
                state.dw_id = dw_id
            return state
        state = SessionState(
            session_id=session_id,
            objective_id=objective_id,
            dw_id=dw_id,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        return self._sessions[session_id]

    def list_sessions(self, objective_id: Optional[str] = None) -> List[SessionState]:
        if objective_id is None:
            return list(self._sessions.values())
        return [s for s in self._sessions.values() if s.objective_id == objective_id]

    def close_session(self, session_id: str) -> SessionState:
        state = self.get_session(session_id)
        state.status = "closed"
        return state

    def update_contribution(
        self,
        session_id: str,
        newly_ceval_constraints: List[str],
        all_ceval_constraints: List[str],
        total_constraints: int,
    ) -> Tuple[SessionState, float, float]:
        state = self.get_session(session_id)
        prev_phi_leq_t = state.phi_leq_t
        state.step_index += 1

        covered = set(state.covered_constraints)
        covered.update(all_ceval_constraints)
        state.covered_constraints = sorted(covered)

        coverage = (len(covered) / total_constraints) if total_constraints > 0 else 0.0
        state.phi_leq_t = max(prev_phi_leq_t, coverage)
        delta_phi_t = state.phi_leq_t - prev_phi_leq_t
        return state, state.phi_leq_t, delta_phi_t

    def update_from_ckg(
        self,
        session_id: str,
        step_index: int,
        phi_leq_t: float,
        covered_constraints: Optional[List[str]] = None,
        phi_weighted_leq_t: Optional[float] = None,
    ) -> SessionState:
        state = self.get_session(session_id)
        state.step_index = int(step_index)
        state.phi_leq_t = float(phi_leq_t)
        if phi_weighted_leq_t is not None:
            state.phi_weighted_leq_t = float(phi_weighted_leq_t)
        if covered_constraints is not None:
            state.covered_constraints = sorted(set(str(x) for x in covered_constraints))
        return state

    def register_evidence(self, session_id: str, evidence_id: str, snapshot_id: Optional[str] = None) -> SessionState:
        state = self.get_session(session_id)
        evids = list(state.evidence_ids or [])
        if evidence_id not in evids:
            evids.append(str(evidence_id))
        state.evidence_ids = evids
        if snapshot_id:
            state.latest_snapshot_id = str(snapshot_id)
        return state

    def register_evidence_lifecycle(
        self,
        session_id: str,
        *,
        active_evidence_ids: Optional[List[str]] = None,
        archived_evidence_ids: Optional[List[str]] = None,
        session_summary_path: Optional[str] = None,
    ) -> SessionState:
        state = self.get_session(session_id)
        if active_evidence_ids is not None:
            state.evidence_ids = [str(x) for x in active_evidence_ids]
        if archived_evidence_ids is not None:
            state.archived_evidence_ids = [str(x) for x in archived_evidence_ids]
        if session_summary_path is not None:
            state.latest_session_summary_path = str(session_summary_path)
        return state

    def register_bootstrap(
        self,
        session_id: str,
        *,
        bootstrap_constraints: Optional[List[str]] = None,
        bootstrap_evidence_ids: Optional[List[str]] = None,
        bootstrap_phi_leq_t: Optional[float] = None,
        bootstrap_phi_weighted_leq_t: Optional[float] = None,
    ) -> SessionState:
        state = self.get_session(session_id)
        if bootstrap_constraints is not None:
            state.bootstrap_constraints = sorted(set(str(x) for x in bootstrap_constraints))
            state.covered_constraints = sorted(set(state.covered_constraints) | set(state.bootstrap_constraints))
        if bootstrap_evidence_ids is not None:
            state.bootstrap_evidence_ids = [str(x) for x in bootstrap_evidence_ids]
        if bootstrap_phi_leq_t is not None:
            state.bootstrap_phi_leq_t = float(bootstrap_phi_leq_t)
            state.phi_leq_t = max(float(state.phi_leq_t), float(bootstrap_phi_leq_t))
        if bootstrap_phi_weighted_leq_t is not None:
            state.phi_weighted_leq_t = max(float(state.phi_weighted_leq_t), float(bootstrap_phi_weighted_leq_t))
        return state


    def append_history(self, session_id: str, entry: SessionHistoryEntry) -> SessionState:
        state = self.get_session(session_id)
        hist = list(getattr(state, "history", []) or [])
        hist.append(entry)
        state.history = hist
        return state

    def get_history(self, session_id: str) -> List[SessionHistoryEntry]:
        state = self.get_session(session_id)
        return list(getattr(state, "history", []) or [])


SESSION_STORE = SessionStore()
