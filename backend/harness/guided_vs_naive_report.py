# backend/harness/guided_vs_naive_report.py
# (NEW: Guided vs Naive statistical report + paper-ready plots)
#
# Goal:
#   Provide quantitative evidence that CKG-guidance improves contribution dynamics (φ≤t),
#   earliness of first contribution, and reduces failure modes (SAT-fail / Real∅ / Ceval∅),
#   with simple, reproducible statistics suitable for an academic submission.
#
# Inputs (auto-detected under --results-dir):
#   - master_metrics_by_session.csv   (from aggregate_metrics_from_timelines.py v2)
#     If missing, falls back to metrics_by_session.csv and explain/explain_metrics_by_session.csv where possible.
#
# Outputs:
#   - <results-dir>/reports/guided_vs_naive_summary.csv
#   - <results-dir>/reports/guided_vs_naive_tests.csv
#   - <results-dir>/reports/guided_vs_naive_report.txt
#   - <results-dir>/figures/guided_vs_naive_boxplot_<metric>.png
#   - <results-dir>/figures/guided_vs_naive_ecdf_<metric>.png
#
# Usage:
#   python backend/harness/guided_vs_naive_report.py --results-dir results_1000
#   python backend/harness/guided_vs_naive_report.py --results-dir results_ckg --group-a guided --group-b naive
#
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    ("phi_final", "phi_leq_T", "Higher is better (final cumulative contribution)"),
    ("auc_phi", "mean_phi_leq_t", "Higher is better (AUC / mean φ≤t)"),
    ("earliness", "earliness_score", "Higher is better (earliness of first contribution)"),
    ("sat_fail", "sat_fail_ratio", "Lower is better (SAT failure ratio)"),
    ("real_empty", "real_empty_ratio", "Lower is better (Real empty given SAT ok)"),
    ("ceval_empty", "ceval_empty_ratio", "Lower is better (Ceval empty given Real non-empty)"),
    ("lat_mean", "mean_latency_ms", "Lower is better (mean latency)"),
]


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


def detect_master_by_session(results_dir: str) -> Tuple[str, bool]:
    """
    Returns (path, is_master).
    """
    master = os.path.join(results_dir, "master_metrics_by_session.csv")
    if os.path.exists(master):
        return master, True
    base = os.path.join(results_dir, "metrics_by_session.csv")
    if os.path.exists(base):
        return base, False
    raise FileNotFoundError("Aucun fichier metrics_by_session.csv / master_metrics_by_session.csv trouvé.")


def group_values(rows: List[Dict[str, str]], group: str, metric_key: str) -> List[float]:
    vals: List[float] = []
    for r in rows:
        t = str(r.get("scenario_type") or r.get("scenarioType") or "unknown")
        if t != group:
            continue
        v = to_float(r.get(metric_key))
        if v is None:
            # sometimes camelCase from other scripts
            v = to_float(r.get(metric_key[0].lower() + metric_key[1:]))
        if v is None:
            continue
        vals.append(v)
    return vals


def quantiles(xs: List[float]) -> Tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    s = sorted(xs)
    n = len(s)
    def q(p: float) -> float:
        if n == 1:
            return s[0]
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(n - 1, lo + 1)
        w = idx - lo
        return (1 - w) * s[lo] + w * s[hi]
    return (q(0.25), q(0.5), q(0.75))


def cohen_d(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    # pooled (population) variance weighted
    pooled = ((len(a) * va) + (len(b) * vb)) / float(len(a) + len(b))
    if pooled <= 0:
        return None
    return (ma - mb) / (pooled ** 0.5)


def mann_whitney_u(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    Return (U, cliff_delta) using rank method with average ties.
    Delta > 0 => A tends to be larger than B.
    """
    if not a or not b:
        return 0.0, 0.0

    combined = [(x, 0) for x in a] + [(x, 1) for x in b]
    combined.sort(key=lambda t: t[0])

    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        # average rank for ties (1-indexed ranks)
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    # sum ranks for group A (label 0)
    R_a = 0.0
    n1 = len(a)
    n2 = len(b)
    for (val, g), r in zip(combined, ranks):
        if g == 0:
            R_a += r

    U1 = R_a - (n1 * (n1 + 1) / 2.0)
    # cliff delta derived from U for A>B
    delta = (2.0 * U1) / (n1 * n2) - 1.0
    return float(U1), float(delta)


def permutation_p_value(a: List[float], b: List[float], n_perm: int = 2000, seed: int = 7) -> Optional[float]:
    """
    Two-sided permutation test on difference in means.
    Returns p-value (approx). Deterministic with seed.
    """
    if not a or not b:
        return None
    rng = random.Random(seed)
    obs = abs(statistics.mean(a) - statistics.mean(b))
    combined = a + b
    n1 = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        a2 = combined[:n1]
        b2 = combined[n1:]
        diff = abs(statistics.mean(a2) - statistics.mean(b2))
        if diff >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def bootstrap_ci_mean_diff(a: List[float], b: List[float], n_boot: int = 2000, seed: int = 11) -> Optional[Tuple[float, float]]:
    """
    95% bootstrap CI for mean(a)-mean(b).
    """
    if not a or not b:
        return None
    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.mean(sa) - statistics.mean(sb))
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return (lo, hi)


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[MCAD/GvN] {out_path}")


def plot_box(a: List[float], b: List[float], labels: Tuple[str, str], out_png: str, title: str, ylabel: str) -> None:
    ensure_dir(os.path.dirname(out_png) or ".")
    plt.figure(figsize=(6.5, 4))
    plt.boxplot([a, b], labels=list(labels), showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[MCAD/GvN] {out_png}")


def plot_ecdf(a: List[float], b: List[float], labels: Tuple[str, str], out_png: str, title: str, xlabel: str) -> None:
    ensure_dir(os.path.dirname(out_png) or ".")
    def ecdf(xs: List[float]) -> Tuple[List[float], List[float]]:
        s = sorted(xs)
        n = len(s)
        ys = [(i + 1) / n for i in range(n)]
        return s, ys

    plt.figure(figsize=(6.5, 4))
    if a:
        xa, ya = ecdf(a)
        plt.step(xa, ya, where="post", label=labels[0])
    if b:
        xb, yb = ecdf(b)
        plt.step(xb, yb, where="post", label=labels[1])

    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[MCAD/GvN] {out_png}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Guided vs Naive report (MCAD, CKG-first).")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--group-a", type=str, default="guided", help="Scenario type A (e.g., guided)")
    p.add_argument("--group-b", type=str, default="naive", help="Scenario type B (e.g., naive)")
    p.add_argument("--n-perm", type=int, default=2000, help="Permutations for p-value (mean diff)")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstraps for CI (mean diff)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    a_name, b_name = args.group_a, args.group_b

    path, is_master = detect_master_by_session(results_dir)
    rows = read_csv_semicolon(path)

    reports_dir = os.path.join(results_dir, "reports")
    figs_dir = os.path.join(results_dir, "figures")
    ensure_dir(reports_dir)
    ensure_dir(figs_dir)

    summary_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    report_lines: List[str] = []
    report_lines.append("=== MCAD Guided vs Naive Report ===")
    report_lines.append(f"Results dir: {results_dir}")
    report_lines.append(f"Input: {os.path.basename(path)} (master={is_master})")
    report_lines.append(f"Group A: {a_name}")
    report_lines.append(f"Group B: {b_name}")
    report_lines.append("")

    for short, key, desc in DEFAULT_METRICS:
        a = group_values(rows, a_name, key)
        b = group_values(rows, b_name, key)

        if not a or not b:
            report_lines.append(f"[SKIP] {short} ({key}) missing values for one group.")
            continue

        qa25, qa50, qa75 = quantiles(a)
        qb25, qb50, qb75 = quantiles(b)

        ma, mb = statistics.mean(a), statistics.mean(b)
        sda = statistics.pstdev(a) if len(a) > 1 else 0.0
        sdb = statistics.pstdev(b) if len(b) > 1 else 0.0

        diff = ma - mb
        rel = (diff / abs(mb)) if mb != 0 else None

        d = cohen_d(a, b)
        U, cliff = mann_whitney_u(a, b)
        p_perm = permutation_p_value(a, b, n_perm=args.n_perm, seed=7)
        ci = bootstrap_ci_mean_diff(a, b, n_boot=args.n_boot, seed=11)

        summary_rows.append(
            {
                "metric": short,
                "metric_key": key,
                "description": desc,
                "n_A": len(a),
                "mean_A": f"{ma:.6f}",
                "median_A": f"{qa50:.6f}",
                "std_A": f"{sda:.6f}",
                "n_B": len(b),
                "mean_B": f"{mb:.6f}",
                "median_B": f"{qb50:.6f}",
                "std_B": f"{sdb:.6f}",
                "diff_mean_A_minus_B": f"{diff:.6f}",
                "rel_improvement_vs_B": "" if rel is None else f"{rel:.6f}",
            }
        )

        test_rows.append(
            {
                "metric": short,
                "metric_key": key,
                "cohen_d": "" if d is None else f"{d:.6f}",
                "mann_whitney_U": f"{U:.6f}",
                "cliffs_delta": f"{cliff:.6f}",
                "p_perm_mean_diff": "" if p_perm is None else f"{p_perm:.6f}",
                "ci95_mean_diff_lo": "" if ci is None else f"{ci[0]:.6f}",
                "ci95_mean_diff_hi": "" if ci is None else f"{ci[1]:.6f}",
            }
        )

        # Plots per metric
        plot_box(
            a, b,
            labels=(a_name, b_name),
            out_png=os.path.join(figs_dir, f"guided_vs_naive_boxplot_{short}.png"),
            title=f"{short}: {a_name} vs {b_name}",
            ylabel=key,
        )
        plot_ecdf(
            a, b,
            labels=(a_name, b_name),
            out_png=os.path.join(figs_dir, f"guided_vs_naive_ecdf_{short}.png"),
            title=f"ECDF – {short}: {a_name} vs {b_name}",
            xlabel=key,
        )

        # Text report section
        report_lines.append(f"--- {short} ({key}) ---")
        report_lines.append(desc)
        report_lines.append(f"A: n={len(a)}, mean={ma:.4f}, median={qa50:.4f}, IQR=[{qa25:.4f},{qa75:.4f}]")
        report_lines.append(f"B: n={len(b)}, mean={mb:.4f}, median={qb50:.4f}, IQR=[{qb25:.4f},{qb75:.4f}]")
        report_lines.append(f"Δmean(A−B)={diff:.4f}" + (f"  (rel={rel:.2%})" if rel is not None else ""))
        if d is not None:
            report_lines.append(f"Cohen's d={d:.3f} | Cliff's δ={cliff:.3f} | p_perm={p_perm:.4f}")
        else:
            report_lines.append(f"Cliff's δ={cliff:.3f} | p_perm={p_perm:.4f}")
        if ci is not None:
            report_lines.append(f"95% CI (bootstrap) for Δmean: [{ci[0]:.4f}, {ci[1]:.4f}]")
        report_lines.append("")

    # Write outputs
    write_csv(summary_rows, os.path.join(reports_dir, "guided_vs_naive_summary.csv"))
    write_csv(test_rows, os.path.join(reports_dir, "guided_vs_naive_tests.csv"))

    txt_path = os.path.join(reports_dir, "guided_vs_naive_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"[MCAD/GvN] {txt_path}")

    print("[MCAD/GvN] Done.")


if __name__ == "__main__":
    main()
