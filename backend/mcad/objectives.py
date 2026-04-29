# backend/mcad/objectives.py
from __future__ import annotations

import os
from typing import Dict, List, Optional

import yaml

from .models import Objective

_OBJECTIVES: Dict[str, Objective] = {}
_OBJECTIVES_YAML_PATH: Optional[str] = None


def get_objectives_yaml_path() -> str:
    """
    Resolve objectives.yaml path deterministically.

    Priority:
      1) env MCAD_OBJECTIVES_YAML
      2) <repo>/backend/config/objectives.yaml (canonical source)
      3) <repo>/backend/objectives.yaml
      4) ./backend/config/objectives.yaml
      5) ./backend/objectives.yaml
      6) ./objectives.yaml (legacy)
    """
    env_path = os.environ.get("MCAD_OBJECTIVES_YAML")
    if env_path:
        return os.path.abspath(env_path)

    # objectives.py is in backend/mcad/ -> parent is backend/
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..", "config", "objectives.yaml")),
        os.path.abspath(os.path.join(here, "..", "objectives.yaml")),
        os.path.abspath(os.path.join(os.getcwd(), "backend", "config", "objectives.yaml")),
        os.path.abspath(os.path.join(os.getcwd(), "backend", "objectives.yaml")),
        os.path.abspath(os.path.join(os.getcwd(), "objectives.yaml")),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # legacy fallback
    return os.path.abspath(os.path.join(os.getcwd(), "objectives.yaml"))


def _load_from_yaml_once(force_reload: bool = False) -> None:
    """
    Charge les objectifs depuis objectives.yaml une seule fois et les garde en mémoire.
    Aligné avec le CKG : même résolution de chemin (get_objectives_yaml_path()).

    Notes:
    - In-memory uniquement (prototype), pas de persistance automatique vers YAML.
    """
    global _OBJECTIVES, _OBJECTIVES_YAML_PATH

    if _OBJECTIVES and not force_reload:
        return

    path = get_objectives_yaml_path()
    _OBJECTIVES_YAML_PATH = path

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}
    except Exception:
        # If YAML is malformed, we keep empty objectives rather than crashing the API.
        data = {}

    _OBJECTIVES = {}
    for obj_data in data.get("objectives", []) or []:
        try:
            obj = Objective(**obj_data)
            _OBJECTIVES[obj.id] = obj
        except Exception:
            # Skip invalid objective entries (best-effort robustness)
            continue


def list_objectives() -> List[Objective]:
    _load_from_yaml_once()
    return list(_OBJECTIVES.values())


def get_objective(objective_id: str) -> Objective:
    _load_from_yaml_once()
    if objective_id not in _OBJECTIVES:
        raise KeyError(objective_id)
    return _OBJECTIVES[objective_id]


def save_objective(obj: Objective) -> Objective:
    """
    Sauvegarde en mémoire (prototype in-memory, pas de persistance YAML ici).
    """
    _load_from_yaml_once()
    _OBJECTIVES[obj.id] = obj
    return obj


def clone_objective(objective_id: str) -> Objective:
    _load_from_yaml_once()
    base = get_objective(objective_id)
    new_id = base.id + "_CLONE"
    cloned = Objective(
        id=new_id,
        name=base.name + " (clone)",
        description=base.description,
        kpis=list(base.kpis),
        constraints=list(base.constraints),
    )
    _OBJECTIVES[new_id] = cloned
    return cloned


def reload_objectives() -> None:
    """
    Recharge explicitement les objectifs depuis YAML.
    Utile si objectives.yaml est modifié et que l'API tourne en long-running.
    """
    _load_from_yaml_once(force_reload=True)
