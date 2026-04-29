# backend/harness/generate_ieee_tables.py
# (NEW: generate IEEE LaTeX tables from CSV outputs)

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional


def read_csv_semicolon(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def ieee_table_by_type(rows: List[Dict[str, Any]], out_tex: str, caption: str, label: str) -> None:
    os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)

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
        if "ratio" in key or "score" in key or key in ("mean_phi_final", "mean_auc"):
            return f"{f:.3f}"
        if "latency" in key:
            return f"{f:.1f}"
        return f"{f:.3f}"

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
        vals = [fmt(row.get(k, ""), k) for _, k in cols]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[MCAD/TEX] {out_tex}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IEEE LaTeX tables from MCAD CSV outputs.")
    p.add_argument("--csv", type=str, default="results/master_metrics_by_type.csv", help="Input CSV (semicolon delimiter)")
    p.add_argument("--out-tex", type=str, default="results/tables/table_metrics_by_type.tex")
    p.add_argument("--caption", type=str, default="Aggregated experimental metrics by scenario type.")
    p.add_argument("--label", type=str, default="tab:mcad-metrics-by-type")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv_semicolon(args.csv)
    if not rows:
        print("[MCAD/TEX] No rows found.")
        return
    ieee_table_by_type(rows, args.out_tex, args.caption, args.label)


if __name__ == "__main__":
    main()
