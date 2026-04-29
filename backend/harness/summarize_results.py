# backend/harness/summarize_results.py
# (UPGRADED v2: works for scenarios + 1000 sessions; CLI paths)

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_paths(results_dir: str, timelines: str = "", sessions_index: str = "") -> Tuple[str, str]:
    if timelines and sessions_index:
        return timelines, sessions_index

    candidates = [
        ("timelines.json", "sessions_index.json"),
        ("timelines_1000.json", "sessions_index_1000.json"),
    ]
    for tl, idx in candidates:
        tl_path = os.path.join(results_dir, tl)
        idx_path = os.path.join(results_dir, idx)
        if os.path.exists(tl_path) and os.path.exists(idx_path):
            return tl_path, idx_path

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


def extract_phi(events: List[Dict[str, Any]]) -> List[float]:
    phi: List[float] = []
    for step in events:
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


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        print("[MCAD/SUMMARY] Nothing to write.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(";".join(header) + "\n")
        for r in rows:
            f.write(";".join(str(r.get(h, "")) for h in header) + "\n")
    print(f"[MCAD/SUMMARY] {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize MCAD results to CSV (CKG-first aware).")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--timelines", type=str, default="")
    p.add_argument("--sessions-index", type=str, default="")
    p.add_argument("--out", type=str, default="", help="Output CSV (default: <results-dir>/summary_metrics.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    out_csv = args.out.strip() or os.path.join(results_dir, "summary_metrics.csv")

    tl_path, idx_path = detect_paths(results_dir, timelines=args.timelines, sessions_index=args.sessions_index)

    sessions_index = load_json(idx_path)
    timelines = load_json(tl_path)

    rows: List[Dict[str, Any]] = []

    for scen_id, meta in sessions_index.items():
        sid = meta.get("session_id") or scen_id
        scen_type = meta.get("type", "unknown")
        label = (meta.get("label") or "").replace(";", ",")

        tl = timelines.get(scen_id) or timelines.get(sid) or {}
        events = extract_steps(tl)
        phi = extract_phi(events)

        rows.append(
            {
                "scenarioId": scen_id,
                "sessionId": sid,
                "scenarioType": scen_type,
                "label": label,
                "numSteps": len(phi),
                "phiFinal": f"{(phi[-1] if phi else 0.0):.4f}",
                "auc": f"{auc(phi):.4f}",
            }
        )

    write_csv(rows, out_csv)
    print("[MCAD/SUMMARY] Done.")


if __name__ == "__main__":
    main()
