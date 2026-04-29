# backend/harness/paper_artifacts.py
# (NEW: Generate two IEEE LaTeX tables + a composite explainability figure for publication-grade reporting)
#
# Inputs (auto-detected under --results-dir):
#   - master_metrics_by_type.csv (from aggregate_metrics_from_timelines.py)
#   - explain/explain_metrics_by_type.csv (from explainability_metrics.py) [optional but recommended]
#
# Outputs (written under <results-dir>/tables and <results-dir>/figures):
#   - tables/table_contribution_by_type.tex
#   - tables/table_explainability_by_type.tex
#   - figures/fig_explainability_composite_by_type.png
#
# Usage:
#   python backend/harness/paper_artifacts.py --results-dir results_ckg
#
from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_semicolon(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
    return rows


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def fmt_ratio(x: Any) -> str:
    v = to_float(x)
    return "-" if v is None else f"{v:.3f}"


def fmt_latency(x: Any) -> str:
    v = to_float(x)
    return "-" if v is None else f"{v:.1f}"


def fmt_int(x: Any) -> str:
    v = to_float(x)
    if v is None:
        return "-"
    try:
        return str(int(round(v)))
    except Exception:
        return str(v)


def detect_inputs(results_dir: str) -> Tuple[str, Optional[str]]:
    master = os.path.join(results_dir, "master_metrics_by_type.csv")
    explain = os.path.join(results_dir, "explain", "explain_metrics_by_type.csv")
    if not os.path.exists(master):
        raise FileNotFoundError(f"master_metrics_by_type.csv introuvable dans {results_dir}")
    if not os.path.exists(explain):
        explain = None
    return master, explain


def ieee_table(
    rows: List[Dict[str, str]],
    out_tex: str,
    caption: str,
    label: str,
    columns: List[Tuple[str, str, str]],
) -> None:
    """
    columns: list of (Header, key, kind) where kind in {"text","int","ratio","lat"}
    """
    ensure_dir(os.path.dirname(out_tex) or ".")

    align = "l" + "c" * (len(columns) - 1)
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join([h for h, _, _ in columns]) + r" \\")
    lines.append(r"\hline")

    for r in rows:
        vals: List[str] = []
        for _, key, kind in columns:
            if kind == "text":
                vals.append(str(r.get(key, "")))
            elif kind == "int":
                vals.append(fmt_int(r.get(key, "")))
            elif kind == "lat":
                vals.append(fmt_latency(r.get(key, "")))
            else:
                vals.append(fmt_ratio(r.get(key, "")))
        lines.append(" & ".join(vals) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[MCAD/PAPER] {out_tex}")


def merge_explain_by_type(
    master_rows: List[Dict[str, str]],
    explain_rows: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    if not explain_rows:
        return master_rows
    exp_by_type: Dict[str, Dict[str, str]] = {r.get("scenario_type", ""): r for r in explain_rows}
    out: List[Dict[str, str]] = []
    for r in master_rows:
        t = r.get("scenario_type", "")
        exp = exp_by_type.get(t, {})
        merged = dict(r)
        # add per-cause SAT fail ratios if present
        for k in (
            "mean_fail_measures_ratio",
            "mean_fail_cube_ratio",
            "mean_fail_slicers_ratio",
            "mean_fail_time_ratio",
        ):
            if k in exp and k not in merged:
                merged[k] = exp.get(k, "")
        out.append(merged)
    return out


def plot_explainability_composite(
    rows: List[Dict[str, str]],
    out_png: str,
) -> None:
    """
    Composite figure with 2x2 panels:
      - SAT-fail
      - Real-empty
      - Ceval-empty
      - Earliness
    """
    ensure_dir(os.path.dirname(out_png) or ".")

    types = [r.get("scenario_type", "") for r in rows]
    sat_fail = [to_float(r.get("mean_sat_fail_ratio")) or 0.0 for r in rows]
    real_empty = [to_float(r.get("mean_real_empty_ratio")) or 0.0 for r in rows]
    ceval_empty = [to_float(r.get("mean_ceval_empty_ratio")) or 0.0 for r in rows]
    early = [to_float(r.get("mean_earliness_score")) or 0.0 for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    def bar(ax, vals, title, ylabel):
        xs = list(range(len(types)))
        ax.bar(xs, vals)
        ax.set_xticks(xs)
        ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    bar(axes[0, 0], sat_fail, "SAT-fail ratio", "Mean ratio")
    bar(axes[0, 1], real_empty, "Real-empty ratio (given SAT ok)", "Mean ratio")
    bar(axes[1, 0], ceval_empty, "Ceval-empty ratio (given Real non-empty)", "Mean ratio")
    bar(axes[1, 1], early, "Earliness score (first contribution)", "Mean score")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[MCAD/PAPER] {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready LaTeX tables + composite figure (MCAD).")
    p.add_argument("--results-dir", type=str, default="results", help="Directory containing master_metrics_by_type.csv")
    p.add_argument("--caption-prefix", type=str, default="MCAD", help="Prefix used in table captions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    master_path, explain_path = detect_inputs(results_dir)
    master_rows = read_csv_semicolon(master_path)
    explain_rows = read_csv_semicolon(explain_path) if explain_path else None
    rows = merge_explain_by_type(master_rows, explain_rows)

    tables_dir = os.path.join(results_dir, "tables")
    figs_dir = os.path.join(results_dir, "figures")
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    # Table A: contribution & performance (compact)
    contrib_cols = [
        ("Type", "scenario_type", "text"),
        ("N", "n_sessions", "int"),
        ("φ≤T", "mean_phi_final", "ratio"),
        ("AUC", "mean_auc", "ratio"),
        ("NonContrib", "mean_non_contrib_ratio", "ratio"),
        ("Lat(ms)", "mean_latency_ms", "lat"),
        ("p95(ms)", "p95_latency_ms", "lat"),
    ]
    ieee_table(
        rows,
        out_tex=os.path.join(tables_dir, "table_contribution_by_type.tex"),
        caption=f"{args.caption_prefix}: contribution and performance metrics aggregated by scenario type.",
        label="tab:mcad-contrib-by-type",
        columns=contrib_cols,
    )

    # Table B: explainability & failures
    explain_cols = [
        ("Type", "scenario_type", "text"),
        ("N", "n_sessions", "int"),
        ("SATfail", "mean_sat_fail_ratio", "ratio"),
        ("Real∅", "mean_real_empty_ratio", "ratio"),
        ("Ceval∅", "mean_ceval_empty_ratio", "ratio"),
        ("Early", "mean_earliness_score", "ratio"),
    ]
    # If per-cause SAT diagnostics are available, append them
    has_causes = rows and ("mean_fail_measures_ratio" in rows[0])
    if has_causes:
        explain_cols += [
            ("FailM", "mean_fail_measures_ratio", "ratio"),
            ("FailC", "mean_fail_cube_ratio", "ratio"),
            ("FailS", "mean_fail_slicers_ratio", "ratio"),
            ("FailT", "mean_fail_time_ratio", "ratio"),
        ]

    ieee_table(
        rows,
        out_tex=os.path.join(tables_dir, "table_explainability_by_type.tex"),
        caption=f"{args.caption_prefix}: explainability and failure diagnostics aggregated by scenario type.",
        label="tab:mcad-explain-by-type",
        columns=explain_cols,
    )

    # Composite figure
    plot_explainability_composite(
        rows,
        out_png=os.path.join(figs_dir, "fig_explainability_composite_by_type.png"),
    )

    print("[MCAD/PAPER] Done.")


if __name__ == "__main__":
    main()
