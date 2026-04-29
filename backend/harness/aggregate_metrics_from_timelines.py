# backend/harness/aggregate_metrics_from_timelines.py
# (UPGRADED v2: CKG-first compatible + merges explainability metrics + exports IEEE LaTeX tables)

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # backend sans interface graphique
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.ckg.ckg_updater import CKGGraph


@dataclass
class SessionMetrics:
    scenario_id: str
    scenario_label: str
    scenario_type: str
    session_id: str
    objective_id: str

    n_steps: int
    n_contrib_steps: int
    non_contrib_ratio: Optional[float]

    phi_leq_T: float
    phi_weighted_leq_T: float
    mean_phi_leq_t: Optional[float]

    mean_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]

    # CKG-side contributive steps (robustly computed per step)
    ckg_n_contrib_steps: Optional[int] = None

    # Optional extra diagnostics (non-breaking for CSV consumers)
    sat_fail_ratio: Optional[float] = None
    mean_real_size: Optional[float] = None
    mean_constraints_per_step: Optional[float] = None


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.mean(vals)


def _get_phi_cum(step: Dict[str, Any]) -> Optional[float]:
    for k in ("phi_leq_t", "phi_leq", "phi_cum", "phi_cumulative", "phi_session"):
        if k in step and step.get(k) is not None:
            try:
                return float(step.get(k))
            except Exception:
                return 0.0
    if "phi" in step and step.get("phi") is not None:
        try:
            return float(step.get("phi"))
        except Exception:
            return 0.0
    return None


def _get_phi_weighted_cum(step: Dict[str, Any]) -> Optional[float]:
    for k in ("phi_weighted_leq_t", "phi_weighted_leq", "phi_weighted_cum"):
        if k in step and step.get(k) is not None:
            try:
                return float(step.get(k))
            except Exception:
                return 0.0
    if step.get("phi_weighted") is not None:
        try:
            return float(step.get("phi_weighted"))
        except Exception:
            return 0.0
    return None


def compute_session_metrics(
    scen_id: str,
    scen_payload: Dict[str, Any],
    sessions_index: Optional[Dict[str, Any]] = None,
) -> SessionMetrics:
    session_id = scen_payload.get("session_id", "")
    objective_id = scen_payload.get("objective_id", "")
    steps: List[Dict[str, Any]] = scen_payload.get("steps", []) or []

    scen_label = ""
    scen_type = "unknown"
    if sessions_index is not None and scen_id in sessions_index:
        meta = sessions_index[scen_id]
        scen_label = meta.get("label", "") or ""
        scen_type = meta.get("type", "unknown") or "unknown"

    n_steps = len(steps)

    n_contrib_steps = 0
    ckg_n_contrib_steps = 0

    phi_leq_T = 0.0
    phi_weighted_leq_T = 0.0
    phi_leq_values: List[float] = []

    latencies_ms: List[float] = []

    sat_fails = 0
    real_sizes: List[float] = []
    constraints_per_step: List[float] = []

    for s in steps:
        phi = float(s.get("phi", 0.0) or 0.0)
        calculable_constraints = s.get("calculable_constraints", []) or []
        constraints_per_step.append(float(len(calculable_constraints)))

        if phi > 0.0 or len(calculable_constraints) > 0:
            n_contrib_steps += 1

        sat = s.get("sat", None)
        if sat is False:
            sat_fails += 1

        if "real_node_ids" in s and s.get("real_node_ids") is not None:
            try:
                real_sizes.append(float(len(s.get("real_node_ids") or [])))
            except Exception:
                pass

        phi_cum = _get_phi_cum(s)
        if phi_cum is not None:
            phi_leq_T = float(phi_cum)
            phi_leq_values.append(float(phi_cum))

        phiw_cum = _get_phi_weighted_cum(s)
        if phiw_cum is not None:
            phi_weighted_leq_T = float(phiw_cum)

        if "latency_ms" in s:
            try:
                latencies_ms.append(float(s["latency_ms"]))
            except Exception:
                pass
        elif "elapsed_ms" in s:
            try:
                latencies_ms.append(float(s["elapsed_ms"]))
            except Exception:
                pass

        # CKG contributive steps (robust) – derive from delta / constraints / phi
        if "ckg_contributive" in s:
            if bool(s.get("ckg_contributive")):
                ckg_n_contrib_steps += 1
        else:
            delta = s.get("delta_phi_t", None)
            try:
                delta_val = float(delta) if delta is not None else 0.0
            except Exception:
                delta_val = 0.0
            if bool(s.get("sat", True)) and (len(calculable_constraints) > 0 or delta_val > 0.0 or phi > 0.0):
                ckg_n_contrib_steps += 1

    non_contrib_ratio: Optional[float] = None
    if n_steps > 0:
        non_contrib_ratio = (n_steps - n_contrib_steps) / n_steps

    mean_phi_leq_t = safe_mean(phi_leq_values)

    mean_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    if latencies_ms:
        mean_latency_ms = statistics.mean(latencies_ms)
        sorted_lat = sorted(latencies_ms)
        idx = int(0.95 * (len(sorted_lat) - 1))
        p95_latency_ms = sorted_lat[idx]

    sat_fail_ratio: Optional[float] = None
    if n_steps > 0:
        sat_fail_ratio = sat_fails / n_steps

    mean_real_size = safe_mean(real_sizes) if real_sizes else None
    mean_constraints_per_step = safe_mean(constraints_per_step) if constraints_per_step else None

    return SessionMetrics(
        scenario_id=scen_id,
        scenario_label=scen_label,
        scenario_type=scen_type,
        session_id=session_id,
        objective_id=objective_id,
        n_steps=n_steps,
        n_contrib_steps=n_contrib_steps,
        non_contrib_ratio=non_contrib_ratio,
        phi_leq_T=phi_leq_T,
        phi_weighted_leq_T=phi_weighted_leq_T,
        mean_phi_leq_t=mean_phi_leq_t,
        mean_latency_ms=mean_latency_ms,
        p95_latency_ms=p95_latency_ms,
        ckg_n_contrib_steps=ckg_n_contrib_steps,
        sat_fail_ratio=sat_fail_ratio,
        mean_real_size=mean_real_size,
        mean_constraints_per_step=mean_constraints_per_step,
    )


def write_csv_dicts(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        print("[MCAD/METRICS] Aucun contenu à écrire.")
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[MCAD/METRICS] CSV écrit : {out_path}")


def write_csv_by_session(metrics: List[SessionMetrics], out_path: str) -> None:
    rows = [asdict(m) for m in metrics]
    write_csv_dicts(rows, out_path)


def aggregate_by_type(metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in metrics:
        by_type.setdefault(str(r.get("scenario_type") or "unknown"), []).append(r)

    out: List[Dict[str, Any]] = []
    for t, group in sorted(by_type.items()):
        def col(name: str) -> List[float]:
            xs: List[float] = []
            for g in group:
                v = g.get(name)
                if v is None or v == "":
                    continue
                try:
                    xs.append(float(v))
                except Exception:
                    pass
            return xs

        row = {
            "scenario_type": t,
            "n_sessions": len(group),
            "mean_phi_final": safe_mean(col("phi_leq_T")),
            "mean_phi_weighted_final": safe_mean(col("phi_weighted_leq_T")),
            "mean_auc": safe_mean(col("mean_phi_leq_t")),
            "mean_non_contrib_ratio": safe_mean(col("non_contrib_ratio")),
            "mean_latency_ms": safe_mean(col("mean_latency_ms")),
            "p95_latency_ms": safe_mean(col("p95_latency_ms")),
            "mean_sat_fail_ratio": safe_mean(col("sat_fail_ratio")),
            "mean_real_size": safe_mean(col("mean_real_size")),
            "mean_constraints_per_step": safe_mean(col("mean_constraints_per_step")),
            # explainability extras may be present after merge
            "mean_real_empty_ratio": safe_mean(col("real_empty_ratio")),
            "mean_ceval_empty_ratio": safe_mean(col("ceval_empty_ratio")),
            "mean_earliness_score": safe_mean(col("earliness_score")),
        }
        out.append(row)
    return out


def plot_phi_final_by_type(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    types = [r["scenario_type"] for r in rows]
    vals = [float(r.get("mean_phi_final") or 0.0) for r in rows]
    plt.figure(figsize=(8, 4))
    plt.bar(types, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean φ^{≤T}(O)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[MCAD/METRICS] Plot écrit : {out_path}")


def compute_ckg_metrics(ckg_path: str, output_txt: str) -> None:
    print(f"[MCAD/CKG] Chargement du graphe CKG depuis {ckg_path}...")
    if not os.path.exists(ckg_path):
        print("[MCAD/CKG] Aucune structure CKG trouvée.")
        return

    ckg = CKGGraph.load_from_file(ckg_path)
    if hasattr(ckg, "G"):
        n_nodes = len(ckg.G.nodes())
        n_edges = len(ckg.G.edges())
    else:
        n_nodes = len(getattr(ckg, "nodes", {}) or {})
        n_edges = sum(len(v) for v in (getattr(ckg, "edges", {}) or {}).values())

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=== Statistiques du graphe CKG ===\n")
        f.write(f"Nombre de nœuds : {n_nodes}\n")
        f.write(f"Nombre d'arêtes : {n_edges}\n")
        f.write(f"Historique des mises à jour : {len(getattr(ckg, 'history', []) or [])}\n")

    print(f"[MCAD/CKG] Statistiques CKG écrites dans {output_txt}")


def read_csv_semicolon(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def merge_explainability(
    base_rows: List[Dict[str, Any]],
    explain_csv_path: str,
) -> List[Dict[str, Any]]:
    """
    Merge explainability_metrics_by_session.csv onto base metrics by (scenario_id, session_id).
    Does not overwrite existing keys unless empty.
    """
    if not explain_csv_path or not os.path.exists(explain_csv_path):
        return base_rows

    exp_rows = read_csv_semicolon(explain_csv_path)
    exp_by_sid: Dict[str, Dict[str, str]] = {}
    exp_by_scen: Dict[str, Dict[str, str]] = {}
    for r in exp_rows:
        sid = r.get("session_id") or r.get("sessionId") or ""
        scen = r.get("scenario_id") or r.get("scenarioId") or ""
        if sid:
            exp_by_sid[sid] = r
        if scen:
            exp_by_scen[scen] = r

    merged: List[Dict[str, Any]] = []
    for r in base_rows:
        sid = str(r.get("session_id") or "")
        scen = str(r.get("scenario_id") or "")
        exp = exp_by_sid.get(sid) or exp_by_scen.get(scen)
        if not exp:
            merged.append(r)
            continue

        out = dict(r)
        # inject selected explainability fields
        for k in ("sat_fail_ratio", "real_empty_ratio", "ceval_empty_ratio", "earliness_score", "t_first_contrib"):
            if k in exp:
                if out.get(k) in (None, "",):
                    out[k] = exp.get(k)
                else:
                    # keep base (but allow explainability to complement if base is missing)
                    pass
        merged.append(out)

    return merged


def to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def ieee_table_by_type(rows: List[Dict[str, Any]], out_tex: str, caption: str, label: str) -> None:
    os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)

    # Choose a compact set of columns that reviewers expect
    cols = [
        ("Type", "scenario_type"),
        ("N", "n_sessions"),
        ("φ≤T", "mean_phi_final"),
        ("AUC", "mean_auc"),
        ("Early", "mean_earliness_score"),
        ("SATfail", "mean_sat_fail_ratio"),
        ("Real∅", "mean_real_empty_ratio"),
        ("Ceval∅", "mean_ceval_empty_ratio"),
        ("Lat(ms)", "mean_latency_ms"),
    ]

    def fmt(v: Any, key: str) -> str:
        if key == "scenario_type":
            return str(v)
        if key == "n_sessions":
            try:
                return str(int(float(v)))
            except Exception:
                return str(v)
        f = to_float(v)
        if f is None:
            return "-"
        # ratios and scores in [0,1]
        if "ratio" in key or "score" in key or key in ("mean_phi_final", "mean_auc"):
            return f"{f:.3f}"
        # latency
        if "latency" in key:
            return f"{f:.1f}"
        return f"{f:.3f}"

    # IEEE-style table (single column). For many types, use table* in your paper.
    # We produce a regular table; user can change to table* easily.
    align = "l" + "c" * (len(cols) - 1)

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join([c[0] for c in cols]) + r" \\")
    lines.append(r"\hline")

    for row in rows:
        vals = [fmt(row.get(k), k) for _, k in cols]
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[MCAD/TEX] Table IEEE écrite : {out_tex}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MCAD metrics from timelines.json (CKG-first) + explainability merge + IEEE tables.")
    parser.add_argument("--timelines", type=str, default="results/timelines.json")
    parser.add_argument("--sessions-index", type=str, default="results/sessions_index.json")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--ckg-path", type=str, default="results/ckg_state.json")
    parser.add_argument("--explain-csv", type=str, default="", help="Path to explain_metrics_by_session.csv (optional). If omitted, auto-detect under <out-dir>/explain/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[MCAD/METRICS] Chargement {args.timelines} ...")
    timelines = load_json(args.timelines)

    sessions_index = None
    if os.path.exists(args.sessions_index):
        sessions_index = load_json(args.sessions_index)

    print("[MCAD/METRICS] Calcul des métriques par session...")
    metrics: List[SessionMetrics] = []
    for scen_id, scen_payload in timelines.items():
        metrics.append(compute_session_metrics(scen_id, scen_payload, sessions_index=sessions_index))

    # base CSV
    out_csv_sessions = os.path.join(args.out_dir, "metrics_by_session.csv")
    write_csv_by_session(metrics, out_csv_sessions)

    # Merge explainability metrics if available
    explain_csv = args.explain_csv.strip()
    if not explain_csv:
        auto = os.path.join(args.out_dir, "explain", "explain_metrics_by_session.csv")
        if os.path.exists(auto):
            explain_csv = auto

    base_rows = [asdict(m) for m in metrics]
    merged_rows = merge_explainability(base_rows, explain_csv)
    master_csv_sessions = os.path.join(args.out_dir, "master_metrics_by_session.csv")
    write_csv_dicts(merged_rows, master_csv_sessions)

    # Aggregate by type
    rows_by_type = aggregate_by_type(merged_rows)
    out_csv_types = os.path.join(args.out_dir, "metrics_by_type.csv")
    write_csv_dicts(rows_by_type, out_csv_types)

    # Master by type (same as above; kept for naming clarity)
    master_csv_types = os.path.join(args.out_dir, "master_metrics_by_type.csv")
    write_csv_dicts(rows_by_type, master_csv_types)

    plot_phi_final_by_type(rows_by_type, os.path.join(args.out_dir, "phi_final_by_type.png"))

    compute_ckg_metrics(args.ckg_path, os.path.join(args.out_dir, "ckg_stats.txt"))

    # IEEE table
    tables_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    ieee_table_by_type(
        rows_by_type,
        out_tex=os.path.join(tables_dir, "table_metrics_by_type.tex"),
        caption="Aggregated experimental metrics by scenario type.",
        label="tab:mcad-metrics-by-type",
    )

    print("[MCAD/METRICS] Agrégation terminée.")


if __name__ == "__main__":
    main()
