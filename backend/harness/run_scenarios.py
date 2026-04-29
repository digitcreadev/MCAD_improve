# backend/harness/run_scenarios.py
"""
CKG-first scenarios runner (local harness)

This upgraded script executes scenarios WITHOUT calling the backend API as the source of truth.
Each step is evaluated locally via the CKG engine (backend.ckg.ckg_updater.CKGGraph.evaluate_step),
producing:
  - results/performance.json
  - results/sessions_index.json
  - results/timelines.json
  - CSV exports (phi.csv, auc.csv, heatmap.csv)

Compatibility:
- Keeps the original CLI flags (--config, --base-url, --results-dir).
- Adds --mode {local,api}. Default is local (CKG-first).
- Preserves qp.force_sat / qp.target_constraints in scenarios.yaml as baseline/oracle fields,
  but the CKG-first evaluation ignores target_constraints and uses the CKG for Real/Ceval/φ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Optional (API mode only)
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# Ensure repo root is importable (namespace package "backend")
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ckg.ckg_updater import CKGGraph  # noqa: E402


def load_scenarios(config_path: str) -> Dict[str, Any]:
    """Charge le fichier YAML de scénarios."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _compute_performance_from_timelines(
    objective_id: str,
    session_ids: List[str],
    timelines_by_session: Dict[str, List[Dict[str, Any]]],
    constraint_ids: List[str],
) -> Dict[str, Any]:
    """
    Mirror backend /analytics/objectives/{objective_id}/performance logic:
      - phiCurves (propagation of phi_leq_t)
      - aucBySession (mean phi_leq_t)
      - constraintCoverageHeatmap (earliness)
    """
    max_t = 0
    for sid in session_ids:
        steps = sorted(timelines_by_session.get(sid, []), key=lambda x: int(x.get("t", 0)))
        if steps:
            max_t = max(max_t, int(steps[-1].get("t", 0)))

    # Courbes φ≤t(O) propagées
    phiCurves: List[Dict[str, float]] = []
    last_phi: Dict[str, float] = {sid: 0.0 for sid in session_ids}

    for t in range(1, max_t + 1):
        row: Dict[str, float] = {"t": float(t)}
        for sid in session_ids:
            steps = timelines_by_session.get(sid, [])
            value = last_phi[sid]
            for e in steps:
                if int(e.get("t", 0)) == t and e.get("phi_leq_t") is not None:
                    value = float(e.get("phi_leq_t") or 0.0)
                    break
            last_phi[sid] = value
            row[sid] = value
        phiCurves.append(row)

    # AUC = moyenne des φ≤t observés dans la session
    aucBySession: List[Dict[str, float]] = []
    for sid in session_ids:
        steps = timelines_by_session.get(sid, [])
        if not steps:
            aucBySession.append({"sessionId": sid, "auc": 0.0})
            continue
        vals = [float(e.get("phi_leq_t") or 0.0) for e in steps]
        auc = sum(vals) / float(len(vals)) if vals else 0.0
        aucBySession.append({"sessionId": sid, "auc": auc})

    # Heatmap : précocité de la première couverture (1=très tôt, 0=jamais)
    heat_cells: List[Dict[str, Any]] = []
    for cid in constraint_ids:
        for sid in session_ids:
            steps = sorted(timelines_by_session.get(sid, []), key=lambda x: int(x.get("t", 0)))
            first_t = None
            for e in steps:
                if cid in (e.get("calculable_constraints") or []):
                    first_t = int(e.get("t", 0))
                    break
            if first_t is None or max_t == 0:
                value = 0.0
            else:
                value = 1.0 - float(first_t - 1) / float(max_t)
            heat_cells.append({"constraintId": cid, "sessionId": sid, "value": value})

    return {
        "objective_id": objective_id,
        "sessionIds": session_ids,
        "phiCurves": phiCurves,
        "aucBySession": aucBySession,
        "constraintIds": constraint_ids,
        "constraintCoverageHeatmap": heat_cells,
    }


def export_csv(
    performance: Dict[str, Any],
    session_index: Dict[str, Dict[str, Any]],
    results_dir: str,
) -> None:
    """
    Exporte quelques CSV (φ(t), AUC, heatmap) pour post-traitement ou import.
    Format identique à la version API.
    """
    ensure_dir(results_dir)

    # φ(t) par session (phiCurves)
    phi_curves = performance.get("phiCurves", [])
    phi_csv_path = os.path.join(results_dir, "phi.csv")
    with open(phi_csv_path, "w", encoding="utf-8") as f:
        if phi_curves:
            # header: t + sessionIds
            keys = [k for k in phi_curves[0].keys() if k != "t"]
            f.write("t;" + ";".join(keys) + "\n")
            for row in phi_curves:
                t = int(float(row.get("t", 0)))
                values = [str(row.get(k, 0.0)) for k in keys]
                f.write(str(t) + ";" + ";".join(values) + "\n")
        else:
            f.write("t\n")
    print(f"[MCAD] φ(t) exporté dans {phi_csv_path}")

    # AUC par session
    auc_entries = performance.get("aucBySession", [])
    auc_csv_path = os.path.join(results_dir, "auc.csv")
    with open(auc_csv_path, "w", encoding="utf-8") as f:
        f.write("sessionId;auc;scenarioId;scenarioLabel;scenarioType\n")
        if auc_entries:
            rev_index: Dict[str, Dict[str, Any]] = {}
            for scen_id, info in session_index.items():
                sid = info["session_id"]
                rev_index[sid] = {
                    "scenarioId": scen_id,
                    "scenarioLabel": info.get("label", ""),
                    "scenarioType": info.get("type", ""),
                }
            for entry in auc_entries:
                sid = entry.get("sessionId", "")
                auc = entry.get("auc", 0.0)
                scen_info = rev_index.get(
                    sid, {"scenarioId": "", "scenarioLabel": "", "scenarioType": ""}
                )
                f.write(
                    f"{sid};{auc};{scen_info['scenarioId']};"
                    f"{scen_info['scenarioLabel']};{scen_info['scenarioType']}\n"
                )
    print(f"[MCAD] AUC exporté dans {auc_csv_path}")

    # Heatmap contraintes × sessions
    heat_cells = performance.get("constraintCoverageHeatmap", [])
    heat_csv_path = os.path.join(results_dir, "heatmap.csv")
    with open(heat_csv_path, "w", encoding="utf-8") as f:
        f.write("constraintId;sessionId;value\n")
        for cell in heat_cells:
            cid = cell.get("constraintId", "")
            sid = cell.get("sessionId", "")
            val = cell.get("value", 0.0)
            f.write(f"{cid};{sid};{val}\n")
    print(f"[MCAD] heatmap exportée dans {heat_csv_path}")


def run_scenarios_local(config_path: str, results_dir: str, objectives_yaml: str | None = None) -> None:
    """Execute scenarios locally (CKG-first) and export analytics artifacts."""
    print(f"[MCAD] Chargement des scénarios depuis {config_path}...")
    cfg = load_scenarios(config_path)

    objective_id = cfg.get("objective_id")
    if not objective_id:
        raise ValueError("Le fichier de config doit contenir 'objective_id'.")

    dw_id = cfg.get("dw_id", "FOODMART")
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        raise ValueError("Aucun scénario trouvé dans 'scenarios'.")

    ensure_dir(results_dir)

    # Initialize CKG (auto-bootstrap from backend/objectives.yaml if present)
    ckg = CKGGraph(output_dir=results_dir)
    if objectives_yaml:
        ckg.bootstrap_objectives(objectives_yaml)

    if objective_id not in ckg.objectives:
        raise RuntimeError(
            f"Objective '{objective_id}' introuvable dans le CKG. "
            f"Vérifiez backend/objectives.yaml ou fournissez --objectives-yaml."
        )

    # Prepare indices
    session_index: Dict[str, Dict[str, Any]] = {}
    created_session_ids: List[str] = []
    timelines: Dict[str, Any] = {}

    # For performance aggregation
    timelines_by_session: Dict[str, List[Dict[str, Any]]] = {}

    # Execute each scenario as one session
    for i, scen in enumerate(scenarios, start=1):
        scen_id = scen["id"]
        label = scen.get("label", "")
        scen_type = scen.get("type", "unknown")
        steps = scen.get("steps", []) or []

        # Deterministic-ish local session id
        session_id = f"S_LOCAL_{i:04d}_{scen_id}"
        created_session_ids.append(session_id)
        session_index[scen_id] = {"session_id": session_id, "label": label, "type": scen_type}

        print(f"[MCAD] Scénario '{scen_id}' ({scen_type}) : {label}")
        print(f"        SessionId: {session_id} | Nombre d'étapes : {len(steps)}")

        timeline_steps: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps):
            t = idx + 1
            qp = _normalize_qp(step, objective_id=objective_id)

            t0 = time.perf_counter()
            out = ckg.evaluate_step(session_id=session_id, objective_id=objective_id, step_idx=t, qp=qp)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            timeline_steps.append(
                {
                    "t": int(t),
                    "timestamp": _now_iso(),
                    "phi": float(out.get("phi") or 0.0),
                    "phi_weighted": float(out.get("phi_weighted") or 0.0),
                    "phi_leq_t": out.get("phi_leq_t"),
                    "delta_phi_t": out.get("delta_phi_t"),
                    "sat": bool(out.get("sat")),
                    "calculable_constraints": out.get("calculable_constraints") or [],
                    "clauses": out.get("clauses") or [],
                    # extra (non-breaking)
                    "latency_ms": round(latency_ms, 3),
                    "step_name": step.get("name"),
                    "step_description": step.get("description", ""),
                    "kpi_id": qp.get("kpi_id"),
                    "real_node_ids": out.get("real_node_ids") or [],
                    "covered_constraints": out.get("covered_constraints") or [],
                }
            )

            # Keep history updated (legacy hook)
            ckg.update_from_step(
                {
                    "name": step.get("name"),
                    "qp": qp,
                    "calculable_constraints": out.get("calculable_constraints") or [],
                },
                scenario_id=scen_id,
                step_idx=idx,
                session_id=session_id,
            )

        # Session timeline response (same structure as /analytics/sessions/{sid}/timeline)
        timelines[scen_id] = {"session_id": session_id, "objective_id": objective_id, "steps": timeline_steps}
        timelines_by_session[session_id] = timeline_steps

    # Compute performance summary (same structure as /analytics/objectives/{objective_id}/performance)
    constraint_ids = sorted(list((ckg.objectives.get(objective_id) or {}).get("constraints", {}).keys()))
    performance = _compute_performance_from_timelines(
        objective_id=objective_id,
        session_ids=created_session_ids,
        timelines_by_session=timelines_by_session,
        constraint_ids=constraint_ids,
    )

    # Save raw JSON
    perf_path = os.path.join(results_dir, "performance.json")
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(performance, f, indent=2, ensure_ascii=False)
    print(f"[MCAD] performance.json écrit dans {perf_path}")

    idx_path = os.path.join(results_dir, "sessions_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(session_index, f, indent=2, ensure_ascii=False)
    print(f"[MCAD] sessions_index.json écrit dans {idx_path}")

    tl_path = os.path.join(results_dir, "timelines.json")
    with open(tl_path, "w", encoding="utf-8") as f:
        json.dump(timelines, f, indent=2, ensure_ascii=False)
    print(f"[MCAD] timelines.json écrit dans {tl_path}")

    # Export CSV
    export_csv(performance, session_index, results_dir)

    # Persist CKG (audit)
    try:
        ckg.save_global_graph(os.path.join(results_dir, "ckg_state.json"))
        ckg.save_global_graph(os.path.join(results_dir, "ckg_global.graphml"))
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible d'exporter l'état CKG : {e}")


def run_scenarios_api(config_path: str, base_url: str, results_dir: str) -> None:
    """Legacy mode: run scenarios against the backend API (original behavior)."""
    if requests is None:
        raise RuntimeError("requests n'est pas disponible, impossible d'utiliser --mode api")

    print(f"[MCAD] Chargement des scénarios depuis {config_path}...")
    cfg = load_scenarios(config_path)

    objective_id = cfg.get("objective_id")
    if not objective_id:
        raise ValueError("Le fichier de config doit contenir 'objective_id'.")

    dw_id = cfg.get("dw_id", "FOODMART")
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        raise ValueError("Aucun scénario trouvé dans 'scenarios'.")

    ensure_dir(results_dir)

    session_index: Dict[str, Dict[str, Any]] = {}
    created_session_ids: List[str] = []

    # 1) Play scenarios via API
    for scen in scenarios:
        scen_id = scen["id"]
        label = scen.get("label", "")
        scen_type = scen.get("type", "unknown")
        steps = scen.get("steps", [])

        print(f"[MCAD] Scénario '{scen_id}' ({scen_type}) : {label}")
        print(f"        Nombre d'étapes : {len(steps)}")

        # create session
        r = requests.post(f"{base_url}/sessions", json={"objective_id": objective_id, "dw_id": dw_id})
        r.raise_for_status()
        session_id = r.json().get("session_id")
        created_session_ids.append(session_id)

        session_index[scen_id] = {"session_id": session_id, "label": label, "type": scen_type}

        for step in steps:
            qp = dict(step.get("qp") or {})
            qp.setdefault("step_name", step.get("name"))
            qp.setdefault("step_description", step.get("description", ""))
            payload = {"objective_id": objective_id, "qp": qp}
            print(f"   - Step '{step.get('name')}' → evaluate_visual_mdx")
            r_eval = requests.post(f"{base_url}/sessions/{session_id}/evaluate_visual_mdx", json=payload)
            r_eval.raise_for_status()

        try:
            requests.post(f"{base_url}/sessions/{session_id}/close")
        except Exception as e:
            print(f"[AVERTISSEMENT] Impossible de fermer la session {session_id} : {e}")

    # 2) Fetch performance
    performance: Dict[str, Any] = {}
    try:
        print(f"[MCAD] Récupération de la synthèse globale pour l'objectif {objective_id} ...")
        perf_resp = requests.get(f"{base_url}/analytics/objectives/{objective_id}/performance", timeout=30)
        if perf_resp.status_code == 200:
            performance = perf_resp.json()
        else:
            print(f"[AVERTISSEMENT] /performance status={perf_resp.status_code} ; pas d'export.")
    except Exception as e:
        print(f"[AVERTISSEMENT] Erreur /performance : {e}")

    # 3) Fetch timelines
    timelines: Dict[str, Any] = {}
    for scen_id, info in session_index.items():
        sid = info["session_id"]
        try:
            r_tl = requests.get(f"{base_url}/analytics/sessions/{sid}/timeline", timeout=30)
            if r_tl.status_code == 200:
                timelines[scen_id] = r_tl.json()
            else:
                print(f"[AVERTISSEMENT] timeline status={r_tl.status_code} pour session {sid}")
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur timeline {sid} : {e}")

    # 4) Save
    ensure_dir(results_dir)
    if performance:
        perf_path = os.path.join(results_dir, "performance.json")
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(performance, f, indent=2, ensure_ascii=False)
        print(f"[MCAD] performance.json écrit dans {perf_path}")
        export_csv(performance, session_index, results_dir)

    idx_path = os.path.join(results_dir, "sessions_index.json")
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(session_index, f, indent=2, ensure_ascii=False)
    print(f"[MCAD] sessions_index.json écrit dans {idx_path}")

    tl_path = os.path.join(results_dir, "timelines.json")
    with open(tl_path, "w", encoding="utf-8") as f:
        json.dump(timelines, f, indent=2, ensure_ascii=False)
    print(f"[MCAD] timelines.json écrit dans {tl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCAD experimental scenarios (CKG-first by default).")
    parser.add_argument(
        "--config",
        type=str,
        default="backend/harness/scenarios.yaml",
        help="Path to scenarios YAML file",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL of the MCAD backend (API mode only)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "api"],
        help="Execution mode: local (CKG-first) or api (legacy backend)",
    )
    parser.add_argument(
        "--objectives-yaml",
        type=str,
        default="",
        help="Optional path to objectives.yaml (overrides backend/objectives.yaml in local mode)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "api":
        run_scenarios_api(args.config, args.base_url, args.results_dir)
    else:
        obj = args.objectives_yaml.strip() or None
        run_scenarios_local(args.config, args.results_dir, objectives_yaml=obj)
