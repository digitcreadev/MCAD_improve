# backend/routers/sessions_api.py
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from mcad.models000 import SessionState
from mcad.session_store0 import SESSION_STORE
from mcad.objectives import get_objective

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=List[SessionState])
def api_list_sessions(
    objective_id: Optional[str] = Query(default=None),
) -> List[SessionState]:
    """
    Liste les sessions existantes, éventuellement filtrées par objective_id.
    """
    return SESSION_STORE.list_sessions(objective_id=objective_id)


@router.post("", response_model=SessionState)
def api_create_session(payload: dict) -> SessionState:
    """
    Crée une nouvelle session d'analyse pour un objectif donné (dw_id optionnel).
    """
    objective_id = payload.get("objective_id")
    dw_id = payload.get("dw_id", "FOODMART")
    if not objective_id:
        raise HTTPException(status_code=400, detail="objective_id is required")

    try:
        get_objective(objective_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective not found")

    return SESSION_STORE.create_session(objective_id=objective_id, dw_id=dw_id)


@router.post("/{session_id}/close", response_model=SessionState)
def api_close_session(session_id: str) -> SessionState:
    """
    Ferme une session (status = 'closed') sans supprimer les données.
    """
    try:
        return SESSION_STORE.close_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
