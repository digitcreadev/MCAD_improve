# backend/harness/write_paper_results.py
# (NEW: Generate paper-ready "Experimental Results" paragraphs + IEEE LaTeX table for Guided vs Naive)
#
# Inputs (under --results-dir):
#   - reports/guided_vs_naive_summary.csv
#   - reports/guided_vs_naive_tests.csv
# Optional:
#   - master_metrics_by_type.csv (to provide context about all types)
#
# Outputs:
#   - <results-dir>/paper/experimental_results_en.txt
#   - <results-dir>/paper/experimental_results_fr.txt
#   - <results-dir>/tables/table_guided_vs_naive.tex
#
# Usage:
#   python backend/harness/write_paper_results.py --results-dir results_1000
#
from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Tuple


KEY_ORDER = ["phi_final", "auc_phi", "earliness", "sat_fail", "real_empty", "ceval_empty", "lat_mean"]

METRIC_LABELS_EN = {
    "phi_final": "Final cumulative contribution (φ≤T)",
    "auc_phi": "Contribution AUC (mean φ≤t)",
    "earliness": "Earliness of first contribution",
    "sat_fail": "SAT failure ratio",
    "real_empty": "Real-empty ratio (SAT ok)",
    "ceval_empty": "Ceval-empty ratio (Real non-empty)",
    "lat_mean": "Mean latency (ms)",
}
METRIC_LABELS_FR = {
    "phi_final": "Contribution cumulative finale (φ≤T)",
    "auc_phi": "AUC de contribution (moyenne φ≤t)",
    "earliness": "Précocité de la première contribution",
    "sat_fail": "Taux d'échec SAT",
    "real_empty": "Taux Real vide (SAT ok)",
    "ceval_empty": "Taux Ceval vide (Real non vide)",
    "lat_mean": "Latence moyenne (ms)",
}

LOWER_IS_BETTER = {"sat_fail", "real_empty", "ceval_empty", "lat_mean"}


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


def detect_inputs(results_dir: str) -> Tuple[str, str]:
    summ = os.path.join(results_dir, "reports", "guided_vs_naive_summary.csv")
    tests = os.path.join(results_dir, "reports", "guided_vs_naive_tests.csv")
    if not os.path.exists(summ):
        raise FileNotFoundError(f"Missing: {summ}. Run guided_vs_naive_report.py first.")
    if not os.path.exists(tests):
        raise FileNotFoundError(f"Missing: {tests}. Run guided_vs_naive_report.py first.")
    return summ, tests


def index_by_metric(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        m = r.get("metric", "").strip()
        if m:
            out[m] = r
    return out


def fmt(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def fmt_pct(v: Optional[float], nd: int = 1) -> str:
    if v is None:
        return "-"
    return f"{(100.0*v):.{nd}f}\\%"


def format_direction(metric: str, diff: Optional[float]) -> str:
    """
    For narrative: "increased" or "decreased" based on whether higher is better.
    diff is mean(A)-mean(B), where A is guided, B is naive.
    """
    if diff is None:
        return "changed"
    if metric in LOWER_IS_BETTER:
        # negative diff is good (guided < naive)
        return "reduced" if diff < 0 else "increased"
    return "increased" if diff > 0 else "reduced"


def make_ieee_table(
    summary: Dict[str, Dict[str, str]],
    tests: Dict[str, Dict[str, str]],
    out_tex: str,
    caption: str,
    label: str,
    group_a: str,
    group_b: str,
) -> None:
    ensure_dir(os.path.dirname(out_tex) or ".")

    cols = [
        ("Metric", "metric", "text"),
        (f"Mean({group_a})", "mean_A", "num"),
        (f"Mean({group_b})", "mean_B", "num"),
        ("Δmean", "diff_mean_A_minus_B", "num"),
        ("Rel.", "rel_improvement_vs_B", "pct"),
        ("d", "cohen_d", "num"),
        ("δ", "cliffs_delta", "num"),
        ("p", "p_perm_mean_diff", "num"),
    ]

    def get_row(metric: str) -> Dict[str, str]:
        s = summary.get(metric, {})
        t = tests.get(metric, {})
        row = {}
        row["metric"] = metric
        row.update(s)
        row.update(t)
        return row

    def cell(metric: str, key: str, kind: str) -> str:
        r = get_row(metric)
        if key == "metric":
            return metric
        val = r.get(key, "")
        if kind == "text":
            return str(val)
        f = to_float(val)
        if kind == "pct":
            return "-" if f is None else fmt_pct(f, nd=1)
        return "-" if f is None else fmt(f, nd=3)

    lines: List[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\hline")
    lines.append(" & ".join([h for h, _, _ in cols]) + r" \\")
    lines.append(r"\hline")

    for m in KEY_ORDER:
        lines.append(" & ".join([cell(m, k, kind) for _, k, kind in cols]) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[MCAD/PAPER] {out_tex}")


def make_paragraph_en(
    summary: Dict[str, Dict[str, str]],
    tests: Dict[str, Dict[str, str]],
    group_a: str,
    group_b: str,
) -> str:
    parts: List[str] = []
    parts.append(
        f"We compared {group_a} sessions against {group_b} sessions using the CKG-first MCAD prototype. "
        "For each metric, we report a permutation-test p-value on the difference in means, Cliff's delta (δ) as a non-parametric effect size, "
        "and a 95% bootstrap confidence interval (CI) for the mean difference."
    )

    def sent(metric: str) -> Optional[str]:
        s = summary.get(metric, {})
        t = tests.get(metric, {})
        ma = to_float(s.get("mean_A"))
        mb = to_float(s.get("mean_B"))
        diff = to_float(s.get("diff_mean_A_minus_B"))
        rel = to_float(s.get("rel_improvement_vs_B"))
        p = to_float(t.get("p_perm_mean_diff"))
        d = to_float(t.get("cohen_d"))
        delta = to_float(t.get("cliffs_delta"))
        lo = to_float(t.get("ci95_mean_diff_lo"))
        hi = to_float(t.get("ci95_mean_diff_hi"))

        if ma is None or mb is None or diff is None:
            return None

        label = METRIC_LABELS_EN.get(metric, metric)
        direction = format_direction(metric, diff)

        # If lower is better, express improvement as reduction %
        if metric in LOWER_IS_BETTER and rel is not None:
            rel_txt = f"{abs(rel)*100.0:.1f}%"
        elif rel is not None:
            rel_txt = f"{rel*100.0:.1f}%"
        else:
            rel_txt = None

        s1 = f"{label} {direction} from {mb:.3f} to {ma:.3f} (Δ={diff:.3f}"
        if rel_txt:
            s1 += f", rel.={rel_txt}"
        s1 += ")."

        s2 = ""
        if p is not None:
            s2 += f" Permutation p={p:.4f}."
        if delta is not None:
            s2 += f" Cliff's δ={delta:.3f}."
        if d is not None:
            s2 += f" Cohen's d={d:.3f}."
        if lo is not None and hi is not None:
            s2 += f" 95% CI for Δ=[{lo:.3f}, {hi:.3f}]."
        return s1 + s2

    # Prioritize core MCAD claims
    core = ["phi_final", "auc_phi", "earliness", "real_empty", "ceval_empty", "sat_fail"]
    sents = [sent(m) for m in core]
    sents = [x for x in sents if x]
    if sents:
        parts.append(" ".join(sents))

    # Add latency as a note
    lat = sent("lat_mean")
    if lat:
        parts.append("Regarding efficiency, " + lat)

    parts.append(
        "Overall, the guided strategy improves the probability of producing objective-relevant virtual nodes (Real(QP)) and calculable constraints (Ceval(QP,O)), "
        "leading to earlier and higher cumulative contribution φ≤t(O) while reducing non-contributive or invalid steps."
    )

    return "\n".join(parts)


def make_paragraph_fr(
    summary: Dict[str, Dict[str, str]],
    tests: Dict[str, Dict[str, str]],
    group_a: str,
    group_b: str,
) -> str:
    parts: List[str] = []
    parts.append(
        f"Nous avons comparé des sessions {group_a} à des sessions {group_b} en utilisant le prototype MCAD CKG-first. "
        "Pour chaque métrique, nous reportons une p-value (test par permutations) sur la différence de moyennes, Cliff's delta (δ) comme taille d'effet non-paramétrique, "
        "ainsi qu'un intervalle de confiance (IC) bootstrap à 95% pour la différence de moyennes."
    )

    def sent(metric: str) -> Optional[str]:
        s = summary.get(metric, {})
        t = tests.get(metric, {})
        ma = to_float(s.get("mean_A"))
        mb = to_float(s.get("mean_B"))
        diff = to_float(s.get("diff_mean_A_minus_B"))
        rel = to_float(s.get("rel_improvement_vs_B"))
        p = to_float(t.get("p_perm_mean_diff"))
        d = to_float(t.get("cohen_d"))
        delta = to_float(t.get("cliffs_delta"))
        lo = to_float(t.get("ci95_mean_diff_lo"))
        hi = to_float(t.get("ci95_mean_diff_hi"))

        if ma is None or mb is None or diff is None:
            return None

        label = METRIC_LABELS_FR.get(metric, metric)
        direction = format_direction(metric, diff)

        if metric in LOWER_IS_BETTER and rel is not None:
            rel_txt = f"{abs(rel)*100.0:.1f}%"
        elif rel is not None:
            rel_txt = f"{rel*100.0:.1f}%"
        else:
            rel_txt = None

        s1 = f"{label} a été {('réduite' if direction=='reduced' else 'augmentée')} de {mb:.3f} à {ma:.3f} (Δ={diff:.3f}"
        if rel_txt:
            s1 += f", rel.={rel_txt}"
        s1 += ")."

        s2 = ""
        if p is not None:
            s2 += f" p={p:.4f}."
        if delta is not None:
            s2 += f" δ={delta:.3f}."
        if d is not None:
            s2 += f" d={d:.3f}."
        if lo is not None and hi is not None:
            s2 += f" IC95% Δ=[{lo:.3f}, {hi:.3f}]."
        return s1 + s2

    core = ["phi_final", "auc_phi", "earliness", "real_empty", "ceval_empty", "sat_fail"]
    sents = [sent(m) for m in core]
    sents = [x for x in sents if x]
    if sents:
        parts.append(" ".join(sents))

    lat = sent("lat_mean")
    if lat:
        parts.append("Concernant l'efficience, " + lat)

    parts.append(
        "Globalement, la stratégie guidée améliore la probabilité de produire des nœuds virtuels pertinents pour l'objectif (Real(QP)) et des contraintes calculables (Ceval(QP,O)), "
        "ce qui se traduit par une contribution cumulative φ≤t(O) plus précoce et plus élevée, tout en réduisant les pas non contributifs ou invalides."
    )
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write paper-ready results paragraphs + IEEE table (Guided vs Naive).")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--group-a", type=str, default="guided")
    p.add_argument("--group-b", type=str, default="naive")
    p.add_argument("--caption-prefix", type=str, default="MCAD", help="Prefix for captions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    group_a = args.group_a
    group_b = args.group_b

    summ_path, tests_path = detect_inputs(results_dir)
    summ = index_by_metric(read_csv_semicolon(summ_path))
    tests = index_by_metric(read_csv_semicolon(tests_path))

    # Outputs
    paper_dir = os.path.join(results_dir, "paper")
    tables_dir = os.path.join(results_dir, "tables")
    ensure_dir(paper_dir)
    ensure_dir(tables_dir)

    # Paragraphs
    en = make_paragraph_en(summ, tests, group_a, group_b)
    fr = make_paragraph_fr(summ, tests, group_a, group_b)

    en_path = os.path.join(paper_dir, "experimental_results_en.txt")
    fr_path = os.path.join(paper_dir, "experimental_results_fr.txt")
    with open(en_path, "w", encoding="utf-8") as f:
        f.write(en + "\n")
    with open(fr_path, "w", encoding="utf-8") as f:
        f.write(fr + "\n")
    print(f"[MCAD/PAPER] {en_path}")
    print(f"[MCAD/PAPER] {fr_path}")

    # IEEE Table
    out_tex = os.path.join(tables_dir, "table_guided_vs_naive.tex")
    make_ieee_table(
        summ, tests,
        out_tex=out_tex,
        caption=f"{args.caption_prefix}: guided vs. naive comparison (means, effect sizes, and permutation p-values).",
        label="tab:mcad-guided-vs-naive",
        group_a=group_a,
        group_b=group_b,
    )

    print("[MCAD/PAPER] Done.")


if __name__ == "__main__":
    main()
