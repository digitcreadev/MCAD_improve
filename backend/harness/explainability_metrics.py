# backend/harness/explainability_metrics.py
# (NEW: CKG-first explainability metrics + plots for academic reporting)

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


MANDATORY_CLAUSES = ["measures_ok", "cube_ok", "slicers_ok", "time_window_ok"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_paths(results_dir: str, timelines: str = "", sessions_index: str = "") -> Tuple[str, str]:
    if timelines and sessions_index:
        return timelines, sessions_index

    candidates = [
        ("timelines.json", "sessions_index.json"),
        ("timelines_1000.json", "sessions_index_1000.json"),
    ]
    for tl, idx in candidates:
        tlp = os.path.join(results_dir, tl)
        idxp = os.path.join(results_dir, idx)
        if os.path.exists(tlp) and os.path.exists(idxp):
            return tlp, idxp

    return (
        timelines or os.path.join(results_dir, "timelines.json"),
        sessions_index or os.path.join(results_dir, "sessions_index.json"),
    )


def extract_steps(tl_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return (
        tl_payload.get("steps")
        or tl_payload.get("events")
        or tl_payload.get("timeline")
        or tl_payload.get("points")
        or []
    )


def clauses_map(step: Dict[str, Any]) -> Dict[str, bool]:
    """
    Normalize clauses into a dict name->ok.
    Accepts:
      - list of dict: {"name":..., "ok":..., "details":...}
    """
    cm: Dict[str, bool] = {}
    raw = step.get("clauses") or []
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, dict):
                name = str(c.get("name", "")).strip()
                if not name:
                    continue
                cm[name] = bool(c.get("ok", False))
    return cm


def safe_mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return statistics.mean(xs)


@dataclass
class ExplainSessionMetrics:
    scenario_id: str
    scenario_type: str
    label: str
    session_id: str
    objective_id: str

    n_steps: int

    # SAT-level
    sat_fail_count: int
    sat_fail_ratio: float
    fail_measures_ratio: float
    fail_cube_ratio: float
    fail_slicers_ratio: float
    fail_time_ratio: float

    # Real/Ceval-level (only meaningful when SAT ok)
    real_empty_count: int
    real_empty_ratio: float
    ceval_empty_count: int
    ceval_empty_ratio: float

    # Earliness (progress)
    t_first_contrib: Optional[int]
    earliness_score: float  # 1 early, 0 never

    # Quality / performance
    mean_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]


def compute_one_session(
    scenario_id: str,
    meta: Dict[str, Any],
    timelines: Dict[str, Any],
) -> ExplainSessionMetrics:
    session_id = str(meta.get("session_id") or scenario_id)
    scen_type = str(meta.get("type") or "unknown")
    label = str(meta.get("label") or "")
    tl = timelines.get(scenario_id) or timelines.get(session_id) or {}
    objective_id = str(tl.get("objective_id") or meta.get("objective_id") or "")

    steps = extract_steps(tl)
    n_steps = len(steps)

    sat_fail = 0
    fail_counts = {k: 0 for k in MANDATORY_CLAUSES}

    real_empty = 0
    ceval_empty = 0
    sat_ok_steps = 0
    real_nonempty_steps = 0

    latencies: List[float] = []

    t_first_contrib: Optional[int] = None

    for i, s in enumerate(steps, start=1):
        sat = bool(s.get("sat", True))
        cm = clauses_map(s)

        # Latency
        if "latency_ms" in s:
            try:
                latencies.append(float(s["latency_ms"]))
            except Exception:
                pass

        # Contrib earliness (progress)
        delta = s.get("delta_phi_t", None)
        try:
            delta_val = float(delta) if delta is not None else 0.0
        except Exception:
            delta_val = 0.0
        ceval = s.get("calculable_constraints") or []
        if t_first_contrib is None and (delta_val > 0.0 or (isinstance(ceval, list) and len(ceval) > 0)):
            t_first_contrib = i

        # SAT fail reasons
        if not sat:
            sat_fail += 1
            for k in MANDATORY_CLAUSES:
                if k in cm and not cm[k]:
                    fail_counts[k] += 1
            # If clauses missing, keep fail reason unknown (not counted)
            continue

        sat_ok_steps += 1

        # Real empty when SAT ok
        if "real_nonempty" in cm and not cm["real_nonempty"]:
            real_empty += 1
        else:
            real_nonempty_steps += 1

        # Ceval empty when SAT ok & Real non-empty
        if ("real_nonempty" in cm and cm["real_nonempty"]) and ("ceval_nonempty" in cm and not cm["ceval_nonempty"]):
            ceval_empty += 1

    # ratios
    sat_fail_ratio = float(sat_fail) / float(n_steps) if n_steps > 0 else 0.0
    fail_measures_ratio = float(fail_counts["measures_ok"]) / float(n_steps) if n_steps > 0 else 0.0
    fail_cube_ratio = float(fail_counts["cube_ok"]) / float(n_steps) if n_steps > 0 else 0.0
    fail_slicers_ratio = float(fail_counts["slicers_ok"]) / float(n_steps) if n_steps > 0 else 0.0
    fail_time_ratio = float(fail_counts["time_window_ok"]) / float(n_steps) if n_steps > 0 else 0.0

    real_empty_ratio = float(real_empty) / float(sat_ok_steps) if sat_ok_steps > 0 else 0.0
    ceval_empty_ratio = float(ceval_empty) / float(real_nonempty_steps) if real_nonempty_steps > 0 else 0.0

    # earliness score
    if n_steps <= 0 or t_first_contrib is None:
        earliness_score = 0.0
    else:
        earliness_score = 1.0 - float(t_first_contrib - 1) / float(n_steps)

    # latency summary
    mean_lat = safe_mean(latencies) if latencies else None
    p95_lat = None
    if latencies:
        srt = sorted(latencies)
        idx = int(0.95 * (len(srt) - 1))
        p95_lat = srt[idx]

    return ExplainSessionMetrics(
        scenario_id=scenario_id,
        scenario_type=scen_type,
        label=label,
        session_id=session_id,
        objective_id=objective_id,
        n_steps=n_steps,
        sat_fail_count=sat_fail,
        sat_fail_ratio=sat_fail_ratio,
        fail_measures_ratio=fail_measures_ratio,
        fail_cube_ratio=fail_cube_ratio,
        fail_slicers_ratio=fail_slicers_ratio,
        fail_time_ratio=fail_time_ratio,
        real_empty_count=real_empty,
        real_empty_ratio=real_empty_ratio,
        ceval_empty_count=ceval_empty,
        ceval_empty_ratio=ceval_empty_ratio,
        t_first_contrib=t_first_contrib,
        earliness_score=earliness_score,
        mean_latency_ms=mean_lat,
        p95_latency_ms=p95_lat,
    )


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        print("[MCAD/EXPLAIN] Nothing to write.")
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(";".join(headers) + "\n")
        for r in rows:
            f.write(";".join("" if r.get(h) is None else str(r.get(h)) for h in headers) + "\n")
    print(f"[MCAD/EXPLAIN] {out_path}")


def aggregate_by_type(metrics: List[ExplainSessionMetrics]) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[ExplainSessionMetrics]] = {}
    for m in metrics:
        by_type.setdefault(m.scenario_type or "unknown", []).append(m)

    out: List[Dict[str, Any]] = []
    for t, group in sorted(by_type.items()):
        out.append(
            {
                "scenario_type": t,
                "n_sessions": len(group),
                "mean_sat_fail_ratio": safe_mean([g.sat_fail_ratio for g in group]),
                "mean_real_empty_ratio": safe_mean([g.real_empty_ratio for g in group]),
                "mean_ceval_empty_ratio": safe_mean([g.ceval_empty_ratio for g in group]),
                "mean_earliness_score": safe_mean([g.earliness_score for g in group]),
                "mean_latency_ms": safe_mean([g.mean_latency_ms for g in group if g.mean_latency_ms is not None]),
                "p95_latency_ms": safe_mean([g.p95_latency_ms for g in group if g.p95_latency_ms is not None]),
                "mean_fail_measures_ratio": safe_mean([g.fail_measures_ratio for g in group]),
                "mean_fail_cube_ratio": safe_mean([g.fail_cube_ratio for g in group]),
                "mean_fail_slicers_ratio": safe_mean([g.fail_slicers_ratio for g in group]),
                "mean_fail_time_ratio": safe_mean([g.fail_time_ratio for g in group]),
            }
        )
    return out


def plot_bar(rows: List[Dict[str, Any]], key: str, out_path: str, title: str, ylabel: str) -> None:
    if not rows:
        return
    types = [r["scenario_type"] for r in rows]
    vals = [float(r.get(key) or 0.0) for r in rows]

    plt.figure(figsize=(9, 4))
    xs = list(range(len(types)))
    plt.bar(xs, vals)
    plt.xticks(xs, types, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0.0, 1.05 if "ratio" in key or "score" in key else max(vals + [1.0]) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[MCAD/EXPLAIN] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explainability metrics for MCAD (CKG-first).")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--timelines", type=str, default="")
    p.add_argument("--sessions-index", type=str, default="")
    p.add_argument("--out-dir", type=str, default="", help="default: <results-dir>/explain")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tl_path, idx_path = detect_paths(args.results_dir, timelines=args.timelines, sessions_index=args.sessions_index)
    out_dir = args.out_dir.strip() or os.path.join(args.results_dir, "explain")
    ensure_dir(out_dir)

    sessions_index = load_json(idx_path)
    timelines = load_json(tl_path)

    metrics: List[ExplainSessionMetrics] = []
    for scen_id, meta in sessions_index.items():
        metrics.append(compute_one_session(scen_id, meta, timelines))

    # CSV by session
    rows_session = [asdict(m) for m in metrics]
    write_csv(rows_session, os.path.join(out_dir, "explain_metrics_by_session.csv"))

    # CSV by type
    rows_type = aggregate_by_type(metrics)
    write_csv(rows_type, os.path.join(out_dir, "explain_metrics_by_type.csv"))

    # Plots
    plot_bar(
        rows_type,
        key="mean_sat_fail_ratio",
        out_path=os.path.join(out_dir, "sat_fail_ratio_by_type.png"),
        title="Mean SAT-fail ratio by scenario type",
        ylabel="Mean SAT-fail ratio",
    )
    plot_bar(
        rows_type,
        key="mean_real_empty_ratio",
        out_path=os.path.join(out_dir, "real_empty_ratio_by_type.png"),
        title="Mean Real-empty ratio (given SAT ok) by scenario type",
        ylabel="Mean Real-empty ratio",
    )
    plot_bar(
        rows_type,
        key="mean_ceval_empty_ratio",
        out_path=os.path.join(out_dir, "ceval_empty_ratio_by_type.png"),
        title="Mean Ceval-empty ratio (given Real non-empty) by scenario type",
        ylabel="Mean Ceval-empty ratio",
    )
    plot_bar(
        rows_type,
        key="mean_earliness_score",
        out_path=os.path.join(out_dir, "earliness_score_by_type.png"),
        title="Mean earliness score of first contribution by scenario type",
        ylabel="Mean earliness score",
    )

    # Optional: diagnose SAT-fail causes
    plot_bar(
        rows_type,
        key="mean_fail_measures_ratio",
        out_path=os.path.join(out_dir, "sat_fail_measures_by_type.png"),
        title="Mean ratio of measures_ok failures by scenario type",
        ylabel="Mean ratio",
    )
    plot_bar(
        rows_type,
        key="mean_fail_cube_ratio",
        out_path=os.path.join(out_dir, "sat_fail_cube_by_type.png"),
        title="Mean ratio of cube_ok failures by scenario type",
        ylabel="Mean ratio",
    )
    plot_bar(
        rows_type,
        key="mean_fail_slicers_ratio",
        out_path=os.path.join(out_dir, "sat_fail_slicers_by_type.png"),
        title="Mean ratio of slicers_ok failures by scenario type",
        ylabel="Mean ratio",
    )
    plot_bar(
        rows_type,
        key="mean_fail_time_ratio",
        out_path=os.path.join(out_dir, "sat_fail_timewindow_by_type.png"),
        title="Mean ratio of time_window_ok failures by scenario type",
        ylabel="Mean ratio",
    )

    print("[MCAD/EXPLAIN] Done.")


if __name__ == "__main__":
    main()
