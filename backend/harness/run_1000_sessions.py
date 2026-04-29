from __future__ import annotations

"""backend/harness/run_1000_sessions.py

Upgrade (CKG-first)
------------------
This harness now computes MCAD contribution metrics *from the CKG* (SAT/Real/Ceval/φ)
via :class:`backend.ckg.ckg_updater.CKGGraph`.

Why this change matters:
  - The legacy backend endpoint /evaluate_visual_mdx uses scenario "oracles" (target_constraints)
    for Real/Ceval, which is useful for baselines but not for a true CKG-first validation.
  - This upgraded harness evaluates each step locally using CKGGraph.evaluate_step(...), while
    keeping an optional legacy mode for regression/ablation.

Outputs are compatible with downstream scripts (aggregate_metrics_from_timelines.py, plots, etc.):
  - results_dir/timelines_1000.json
  - results_dir/sessions_index_1000.json
  - results_dir/ckg_state.json
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Allow running as a script: `python backend/harness/run_1000_sessions.py ...`
# (when executed this way, the project root is not automatically on sys.path).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.ckg.ckg_updater import CKGGraph


def load_scenarios(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    # Keep ISO-8601 without timezone for compatibility with older parsing.
    return datetime.utcnow().isoformat()


def _normalize_qp(step: Dict[str, Any], objective_id: str) -> Dict[str, Any]:
    """Make qp self-contained and CKG-ready (without mutating the input step)."""
    qp = dict(step.get("qp") or {})
    qp.setdefault("objective_id", objective_id)
    qp.setdefault("step_name", step.get("name"))
    qp.setdefault("step_description", step.get("description", ""))

    # If scenarios.yaml provides ckg_tags at step-level, mirror into qp for compatibility.
    if "ckg_tags" in step and "ckg_tags" not in qp:
        qp["ckg_tags"] = step.get("ckg_tags") or []

    return qp


def play_scenario_once_ckg(
    objective_id: str,
    dw_id: str,
    scenario: Dict[str, Any],
    ckg: CKGGraph,
    session_id: str,
    save_snapshot: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Play one scenario using ONLY the CKG evaluation pipeline."""
    scen = copy.deepcopy(scenario)
    steps: List[Dict[str, Any]] = scen.get("steps", []) or []

    timeline_steps: List[Dict[str, Any]] = []

    for idx, step in enumerate(steps):
        t = idx + 1  # align with backend SessionStore.step_index (starts at 1)
        qp = _normalize_qp(step, objective_id=objective_id)

        t0 = time.perf_counter()
        out = ckg.evaluate_step(session_id=session_id, objective_id=objective_id, step_idx=t, qp=qp)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # A timeline step mirrors backend.models.SessionTimelineStep
        timeline_steps.append(
            {
                "t": t,
                "timestamp": _now_iso(),
                "phi": float(out.get("phi", 0.0) or 0.0),
                "phi_weighted": float(out.get("phi_weighted", 0.0) or 0.0),
                "phi_leq_t": out.get("phi_leq_t"),
                "delta_phi_t": out.get("delta_phi_t"),
                "sat": bool(out.get("sat", False)),
                "calculable_constraints": out.get("calculable_constraints", []) or [],
                "clauses": out.get("clauses", []) or [],
                # Extra fields (do not break downstream scripts)
                "latency_ms": round(latency_ms, 3),
                "step_name": step.get("name"),
                "step_description": step.get("description", ""),
                "kpi_id": qp.get("kpi_id"),
            }
        )

        # Lightweight history entry (compatible with legacy update hooks).
        step_for_history = {
            "name": step.get("name"),
            "qp": qp,
            "calculable_constraints": out.get("calculable_constraints", []) or [],
        }
        ckg.update_from_step(
            step_for_history,
            scenario_id=scen.get("id"),
            step_idx=idx,
            session_id=session_id,
        )

    timeline = {
        "session_id": session_id,
        "objective_id": objective_id,
        "dw_id": dw_id,
        "scenario_id": scen.get("id"),
        "scenario_type": scen.get("type"),
        "scenario_label": scen.get("label"),
        "steps": timeline_steps,
    }

    if save_snapshot:
        ckg.save_snapshot(session_id=session_id)

    print(f"[MCAD/1000] Session {session_id} (CKG-first) jouée pour scénario '{scen.get('id')}'.")
    return session_id, timeline


def build_pools(scenarios: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    guided_pool: List[Dict[str, Any]] = []
    naive_pool: List[Dict[str, Any]] = []

    for s in scenarios:
        scen_type = (s.get("type") or "").lower()
        if scen_type == "happy":
            guided_pool.append(s)
        else:
            naive_pool.append(s)

    if not guided_pool:
        raise RuntimeError("Aucun scénario 'happy' trouvé dans scenarios.yaml")
    if not naive_pool:
        raise RuntimeError("Aucun scénario non-'happy' trouvé dans scenarios.yaml")

    return guided_pool, naive_pool


def run_1000_sessions(
    config_path: str,
    results_dir: str = "results_1000",
    n_guided: int = 500,
    n_naive: int = 500,
    seed: int = 42,
    save_snapshots: bool = False,
) -> None:
    print(f"[MCAD/1000] Chargement de la configuration depuis {config_path}...")
    cfg = load_scenarios(config_path)
    objective_id = cfg["objective_id"]
    dw_id = cfg.get("dw_id", "FOODMART")
    scenarios = cfg.get("scenarios", []) or []

    guided_pool, naive_pool = build_pools(scenarios)
    ensure_dir(results_dir)

    random.seed(seed)

    timelines_all: Dict[str, Dict[str, Any]] = {}
    sessions_index: Dict[str, Dict[str, Any]] = {}

    # CKG is the single source of truth
    ckg = CKGGraph(output_dir=results_dir)

    def _run_one(synth_id: str, scenario: Dict[str, Any], scen_type: str) -> None:
        session_id = f"S_{synth_id}"  # stable + unique
        _, timeline = play_scenario_once_ckg(
            objective_id=objective_id,
            dw_id=dw_id,
            scenario=scenario,
            ckg=ckg,
            session_id=session_id,
            save_snapshot=save_snapshots,
        )
        timelines_all[synth_id] = timeline
        sessions_index[synth_id] = {
            "session_id": session_id,
            "label": (scenario.get("label") or scenario.get("id") or synth_id),
            "type": scen_type,
            "source_scenario_id": scenario.get("id"),
            "source_scenario_type": scenario.get("type"),
        }

    for i in range(1, n_guided + 1):
        scenario = random.choice(guided_pool)
        synth_id = f"guided_{i:04d}"
        print(f"[MCAD/1000] (Guided {i}/{n_guided}) - {scenario['id']}")
        _run_one(synth_id, scenario, scen_type="guided")

    for i in range(1, n_naive + 1):
        scenario = random.choice(naive_pool)
        synth_id = f"naive_{i:04d}"
        print(f"[MCAD/1000] (Naive {i}/{n_naive}) - {scenario['id']}")
        _run_one(synth_id, scenario, scen_type="naive")

    timelines_path = os.path.join(results_dir, "timelines_1000.json")
    sessions_index_path = os.path.join(results_dir, "sessions_index_1000.json")
    ckg_state_path = os.path.join(results_dir, "ckg_state.json")

    with open(timelines_path, "w", encoding="utf-8") as f:
        json.dump(timelines_all, f, indent=2, ensure_ascii=False)
    with open(sessions_index_path, "w", encoding="utf-8") as f:
        json.dump(sessions_index, f, indent=2, ensure_ascii=False)

    ckg.save_global_graph(path=ckg_state_path)
    print("[MCAD/1000] Sessions générées et CKG sauvegardé.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="harness/scenarios.yaml")
    # Backward compatibility: legacy script used the backend API.
    # The upgraded version is local CKG-first, so base-url is ignored.
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--results-dir", type=str, default="results_1000")
    parser.add_argument("--n-guided", type=int, default=500)
    parser.add_argument("--n-naive", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Écrit un snapshot JSON du CKG après chaque session (peut être volumineux).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_1000_sessions(
        config_path=args.config,
        results_dir=args.results_dir,
        n_guided=args.n_guided,
        n_naive=args.n_naive,
        seed=args.seed,
        save_snapshots=args.save_snapshots,
    )
