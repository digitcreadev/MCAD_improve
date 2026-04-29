# backend/harness/plot_results.py
# (UPGRADED v2: works for scenarios + 1000 sessions; produces guided-vs-naive summaries)

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_paths(results_dir: str, timelines: str = "", sessions_index: str = "") -> Tuple[str, str]:
    """
    Detect timelines / sessions_index paths for either:
      - run_scenarios.py  -> timelines.json + sessions_index.json
      - run_1000_sessions -> timelines_1000.json + sessions_index_1000.json
    """
    if timelines and sessions_index:
        return timelines, sessions_index

    # auto-detect
    candidates = [
        ("timelines.json", "sessions_index.json"),
        ("timelines_1000.json", "sessions_index_1000.json"),
    ]
    for tl, idx in candidates:
        tl_path = os.path.join(results_dir, tl)
        idx_path = os.path.join(results_dir, idx)
        if os.path.exists(tl_path) and os.path.exists(idx_path):
            return tl_path, idx_path

    # fallback: explicit or best-guess
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


def extract_phi_curve(events: List[Dict[str, Any]]) -> List[float]:
    phi: List[float] = []
    for step in events:
        # prefer cumulative keys
        v = (
            step.get("phi_leq_t")
            or step.get("phi_leq")
            or step.get("phi_cum")
            or step.get("phi_cumulative")
            or step.get("phi_session")
            or step.get("phi")
        )
        if v is None:
            continue
        try:
            phi.append(float(v))
        except Exception:
            phi.append(0.0)
    return phi


def auc(phi: List[float]) -> float:
    if not phi:
        return 0.0
    return float(sum(phi)) / float(len(phi))


def build_session_curves(
    sessions_index: Dict[str, Any],
    timelines: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build per-session curve objects:
      {scenario_id, session_id, type, label, phi_curve, auc}
    """
    curves: List[Dict[str, Any]] = []
    for scen_id, meta in sessions_index.items():
        sid = meta.get("session_id") or scen_id
        scen_type = meta.get("type", "unknown")
        label = meta.get("label", "")

        tl = timelines.get(scen_id) or timelines.get(sid) or {}
        events = extract_steps(tl)
        phi_curve = extract_phi_curve(events)

        curves.append(
            {
                "scenario_id": scen_id,
                "session_id": sid,
                "type": scen_type,
                "label": label,
                "phi": phi_curve,
                "auc": auc(phi_curve),
            }
        )
    return curves


def plot_auc_distribution(curves: List[Dict[str, Any]], out_dir: str) -> None:
    """
    AUC distribution by scenario type (guided/naive/happy/border/adversarial/stress...)
    Uses a simple boxplot (robust for 1000 sessions).
    """
    ensure_dir(out_dir)

    # group
    by_type: Dict[str, List[float]] = {}
    for c in curves:
        by_type.setdefault(c["type"], []).append(float(c["auc"]))

    types = sorted(by_type.keys())
    data = [by_type[t] for t in types]

    if not types:
        print("[MCAD/PLOTS] Aucun AUC à tracer.")
        return

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, labels=types, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("AUC = mean φ^{≤t}(O)")
    plt.title("Distribution AUC par type de scénario")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "auc_boxplot_by_type.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[MCAD/PLOTS] {out_path}")


def _pad_curves(phi_curves: List[List[float]], T: int) -> List[List[float]]:
    padded: List[List[float]] = []
    for c in phi_curves:
        if not c:
            padded.append([0.0] * T)
            continue
        last = c[-1]
        row = c[:T] + [last] * max(0, T - len(c))
        padded.append(row)
    return padded


def plot_mean_phi_by_type(curves: List[Dict[str, Any]], out_dir: str, max_T: int = 25) -> None:
    """
    Mean φ^{≤t}(O) by scenario type.
    For each type, compute mean curve and plot as a line (no CI to keep it simple).
    """
    ensure_dir(out_dir)
    by_type: Dict[str, List[List[float]]] = {}
    for c in curves:
        by_type.setdefault(c["type"], []).append(c["phi"])

    if not by_type:
        print("[MCAD/PLOTS] Aucun curve à tracer.")
        return

    plt.figure(figsize=(10, 6))
    for tname, phi_curves in sorted(by_type.items()):
        # set T based on max length bounded by max_T
        T = min(max_T, max((len(x) for x in phi_curves), default=0))
        if T <= 0:
            continue
        padded = _pad_curves(phi_curves, T)
        # mean across sessions
        mean_curve = [sum(p[i] for p in padded) / float(len(padded)) for i in range(T)]
        xs = list(range(1, T + 1))
        plt.plot(xs, mean_curve, marker="o", label=tname)

    plt.xlabel("t (étape)")
    plt.ylabel("Mean φ^{≤t}(O)")
    plt.ylim(0.0, 1.05)
    plt.title("Courbes moyennes φ^{≤t}(O) par type")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phi_mean_curves_by_type.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[MCAD/PLOTS] {out_path}")


def plot_overlay_some_curves(curves: List[Dict[str, Any]], out_dir: str, max_curves: int = 20) -> None:
    """
    Overlay a small number of φ curves for qualitative inspection.
    For 1000 sessions, max_curves keeps it readable.
    """
    ensure_dir(out_dir)

    # choose deterministic subset: top max_curves by AUC
    sorted_curves = sorted(curves, key=lambda x: float(x.get("auc", 0.0)), reverse=True)[:max_curves]

    plt.figure(figsize=(10, 6))
    for c in sorted_curves:
        phi = c["phi"]
        xs = list(range(1, len(phi) + 1))
        label = f"{c['type']}::{c['scenario_id']}"
        plt.plot(xs, phi, marker="o", label=label)

    plt.xlabel("t (étape)")
    plt.ylabel("φ^{≤t}(O)")
    plt.ylim(0.0, 1.05)
    plt.title(f"Overlay de {len(sorted_curves)} courbes (top AUC)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "phi_overlay_top_auc.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[MCAD/PLOTS] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MCAD results (CKG-first aware).")
    p.add_argument("--results-dir", type=str, default="results", help="Directory containing JSON outputs")
    p.add_argument("--timelines", type=str, default="", help="Explicit timelines path (optional)")
    p.add_argument("--sessions-index", type=str, default="", help="Explicit sessions_index path (optional)")
    p.add_argument("--out-dir", type=str, default="", help="Output directory for plots (default: <results-dir>/plots)")
    p.add_argument("--max-overlay", type=int, default=20, help="Max curves to overlay in qualitative plot")
    p.add_argument("--max-T", type=int, default=25, help="Max steps for mean-curve plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    out_dir = args.out_dir.strip() or os.path.join(results_dir, "plots")

    tl_path, idx_path = detect_paths(results_dir, timelines=args.timelines, sessions_index=args.sessions_index)

    print(f"[MCAD/PLOTS] Loading sessions_index: {idx_path}")
    sessions_index = load_json(idx_path)

    print(f"[MCAD/PLOTS] Loading timelines: {tl_path}")
    timelines = load_json(tl_path)

    curves = build_session_curves(sessions_index, timelines)

    ensure_dir(out_dir)
    plot_auc_distribution(curves, out_dir)
    plot_mean_phi_by_type(curves, out_dir, max_T=args.max_T)
    plot_overlay_some_curves(curves, out_dir, max_curves=args.max_overlay)

    print("[MCAD/PLOTS] Done.")


if __name__ == "__main__":
    main()
