# backend/routers/objectives_api.py
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from mcad.models import Objective
from mcad.objectives import (
    list_objectives,
    get_objective,
    save_objective,
    clone_objective,
)

router = APIRouter(prefix="/objectives", tags=["objectives"])


@router.get("", response_model=List[Objective])
def api_list_objectives() -> List[Objective]:
    """
    Liste tous les objectifs connus (chargés depuis objectives.yaml
    + créés / clonés en mémoire).
    """
    return list_objectives()


@router.get("/{objective_id}", response_model=Objective)
def api_get_objective(objective_id: str) -> Objective:
    try:
        return get_objective(objective_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective not found")


@router.post("", response_model=Objective)
def api_create_objective(obj: Objective) -> Objective:
    """
    Création / enregistrement d'un objectif (in-memory).
    """
    return save_objective(obj)


@router.put("/{objective_id}", response_model=Objective)
def api_update_objective(objective_id: str, obj: Objective) -> Objective:
    """
    Mise à jour d'un objectif existant (in-memory).
    """
    if objective_id != obj.id:
        raise HTTPException(status_code=400, detail="Objective ID mismatch")
    return save_objective(obj)


@router.post("/{objective_id}/clone", response_model=Objective)
def api_clone_objective(objective_id: str) -> Objective:
    """
    Duplique un objectif pour créer une variante (pour scénarios, tests).
    """
    try:
        return clone_objective(objective_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Objective not found")
