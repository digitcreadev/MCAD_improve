# backend/mcad/log_store.py
from __future__ import annotations

from typing import List

from .models000 import SessionEvaluationLogEntry


class LogStore:
    """
    Stockage in-memory des logs d'évaluation MCAD.
    Utilisé pour reconstruire timelines, φ(t), AUC, heatmaps.
    """

    def __init__(self) -> None:
        self._logs: List[SessionEvaluationLogEntry] = []

    def append(self, entry: SessionEvaluationLogEntry) -> None:
        self._logs.append(entry)

    def list_for_session(self, session_id: str) -> List[SessionEvaluationLogEntry]:
        return [e for e in self._logs if e.session_id == session_id]


LOG_STORE = LogStore()
