# backend/harness/generate_ieee_results_section.py
# (NEW: Generate a complete IEEEtran-ready LaTeX Results+Discussion subsection that references
#  the produced tables/figures and injects the generated narrative text.)
#
# Inputs (under --results-dir):
#   - paper/experimental_results_en.txt (recommended)
#   - paper/experimental_results_fr.txt (optional)
#   - tables/table_guided_vs_naive.tex (optional but recommended)
#   - tables/table_contribution_by_type.tex (optional)
#   - tables/table_explainability_by_type.tex (optional)
#   - figures/fig_explainability_composite_by_type.png (optional)
#   - figures/guided_vs_naive_boxplot_*.png and guided_vs_naive_ecdf_*.png (optional)
#
# Outputs:
#   - <results-dir>/latex/ieee_results_section_en.tex
#   - <results-dir>/latex/ieee_results_section_fr.tex
#
# Usage:
#   python backend/harness/generate_ieee_results_section.py --results-dir results_1000
#
from __future__ import annotations

import argparse
import os
from typing import List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def latex_escape(s: str) -> str:
    # minimal safe escaping for IEEEtran
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def exists(results_dir: str, rel: str) -> bool:
    return os.path.exists(os.path.join(results_dir, rel))


def rel_from_latex_dir(rel_path: str) -> str:
    # output .tex is in <results-dir>/latex, so tables/figures are one level up
    return os.path.join("..", rel_path).replace("\\", "/")


def include_figure(fig_rel: str, caption: str, label: str, width: str = "0.98\\linewidth") -> str:
    p = rel_from_latex_dir(fig_rel)
    lines: List[str] = []
    lines.append(r"\begin{figure}[t]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width={width}]{{{p}}}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def input_table(tex_rel: str) -> str:
    p = rel_from_latex_dir(tex_rel)
    return rf"\input{{{p}}}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IEEEtran-ready LaTeX Results+Discussion section from MCAD outputs.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--section-title-en", type=str, default="Experimental Results")
    p.add_argument("--section-title-fr", type=str, default="Résultats expérimentaux")
    p.add_argument("--include-discussion", action="store_true", help="Append a short discussion paragraph")
    return p.parse_args()


def build_en(results_dir: str, args: argparse.Namespace) -> str:
    narrative = read_text(os.path.join(results_dir, "paper", "experimental_results_en.txt")) or ""
    narrative_tex = latex_escape(narrative)

    blocks: List[str] = []
    blocks.append(rf"\subsection{{{args.section_title_en}}}")

    if narrative_tex:
        blocks.append(narrative_tex)
        blocks.append("")

    # Guided vs Naive table
    if exists(results_dir, "tables/table_guided_vs_naive.tex"):
        blocks.append(r"\paragraph{Guided vs. Naive comparison.}")
        blocks.append(input_table("tables/table_guided_vs_naive.tex"))
        blocks.append("")

    # Composite explainability figure
    if exists(results_dir, "figures/fig_explainability_composite_by_type.png"):
        blocks.append(
            include_figure(
                "figures/fig_explainability_composite_by_type.png",
                caption="Explainability diagnostics aggregated by scenario type (SAT failures, Real-empty, Ceval-empty, and earliness).",
                label="fig:mcad-explainability-composite",
            )
        )
        blocks.append("")

    # Two by-type tables (if present)
    if exists(results_dir, "tables/table_contribution_by_type.tex"):
        blocks.append(r"\paragraph{Aggregated contribution metrics.}")
        blocks.append(input_table("tables/table_contribution_by_type.tex"))
        blocks.append("")

    if exists(results_dir, "tables/table_explainability_by_type.tex"):
        blocks.append(r"\paragraph{Aggregated explainability diagnostics.}")
        blocks.append(input_table("tables/table_explainability_by_type.tex"))
        blocks.append("")

    # Optional: include a small set of example plots if present
    example_figs = [
        ("figures/guided_vs_naive_boxplot_phi_final.png", "Boxplot of final cumulative contribution (φ≤T).", "fig:gvn-box-phi"),
        ("figures/guided_vs_naive_ecdf_phi_final.png", "ECDF of final cumulative contribution (φ≤T).", "fig:gvn-ecdf-phi"),
        ("figures/guided_vs_naive_boxplot_earliness.png", "Boxplot of earliness score.", "fig:gvn-box-early"),
        ("figures/guided_vs_naive_ecdf_earliness.png", "ECDF of earliness score.", "fig:gvn-ecdf-early"),
    ]
    any_examples = any(exists(results_dir, p) for p, _, _ in example_figs)
    if any_examples:
        blocks.append(r"\paragraph{Distributional comparison.}")
        for p, cap, lab in example_figs:
            if exists(results_dir, p):
                blocks.append(include_figure(p, cap, lab, width="0.95\\linewidth"))
                blocks.append("")

    if args.include_discussion:
        blocks.append(r"\subsection{Discussion}")
        blocks.append(
            latex_escape(
                "The results support the MCAD claim that reasoning over the contextual knowledge graph (CKG) improves the "
                "alignment between the query plan and the strategic objective. In particular, guided exploration reduces "
                "invalid or non-productive steps (SAT failures and Real/Ceval empty cases) and accelerates the acquisition "
                "of calculable constraints, leading to earlier and larger cumulative contribution."
            )
        )
        blocks.append("")

    return "\n".join(blocks).strip() + "\n"


def build_fr(results_dir: str, args: argparse.Namespace) -> str:
    narrative = read_text(os.path.join(results_dir, "paper", "experimental_results_fr.txt")) or ""
    narrative_tex = latex_escape(narrative)

    blocks: List[str] = []
    blocks.append(rf"\subsection{{{args.section_title_fr}}}")

    if narrative_tex:
        blocks.append(narrative_tex)
        blocks.append("")

    if exists(results_dir, "tables/table_guided_vs_naive.tex"):
        blocks.append(r"\paragraph{Comparaison guidé vs. naïf.}")
        blocks.append(input_table("tables/table_guided_vs_naive.tex"))
        blocks.append("")

    if exists(results_dir, "figures/fig_explainability_composite_by_type.png"):
        blocks.append(
            include_figure(
                "figures/fig_explainability_composite_by_type.png",
                caption="Diagnostics d'explicabilité agrégés par type (échecs SAT, Real vide, Ceval vide, et précocité).",
                label="fig:mcad-explainability-composite",
            )
        )
        blocks.append("")

    if exists(results_dir, "tables/table_contribution_by_type.tex"):
        blocks.append(r"\paragraph{Métriques de contribution agrégées.}")
        blocks.append(input_table("tables/table_contribution_by_type.tex"))
        blocks.append("")

    if exists(results_dir, "tables/table_explainability_by_type.tex"):
        blocks.append(r"\paragraph{Diagnostics d'explicabilité agrégés.}")
        blocks.append(input_table("tables/table_explainability_by_type.tex"))
        blocks.append("")

    example_figs = [
        ("figures/guided_vs_naive_boxplot_phi_final.png", "Boxplot de la contribution cumulative finale (φ≤T).", "fig:gvn-box-phi"),
        ("figures/guided_vs_naive_ecdf_phi_final.png", "ECDF de la contribution cumulative finale (φ≤T).", "fig:gvn-ecdf-phi"),
        ("figures/guided_vs_naive_boxplot_earliness.png", "Boxplot du score de précocité.", "fig:gvn-box-early"),
        ("figures/guided_vs_naive_ecdf_earliness.png", "ECDF du score de précocité.", "fig:gvn-ecdf-early"),
    ]
    any_examples = any(exists(results_dir, p) for p, _, _ in example_figs)
    if any_examples:
        blocks.append(r"\paragraph{Comparaison distributionnelle.}")
        for p, cap, lab in example_figs:
            if exists(results_dir, p):
                blocks.append(include_figure(p, cap, lab, width="0.95\\linewidth"))
                blocks.append("")

    if args.include_discussion:
        blocks.append(r"\subsection{Discussion}")
        blocks.append(
            latex_escape(
                "Les résultats confortent l'hypothèse MCAD selon laquelle le raisonnement sur le graphe de connaissances contextuelles (CKG) "
                "améliore l'alignement entre le plan de requête et l'objectif stratégique. En particulier, le guidage réduit les pas invalides "
                "(échecs SAT et cas Real/Ceval vides) et accélère l'acquisition des contraintes calculables, ce qui se traduit par une contribution "
                "cumulative plus précoce et plus élevée."
            )
        )
        blocks.append("")

    return "\n".join(blocks).strip() + "\n"


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    out_dir = os.path.join(results_dir, "latex")
    ensure_dir(out_dir)

    en_tex = build_en(results_dir, args)
    fr_tex = build_fr(results_dir, args)

    en_path = os.path.join(out_dir, "ieee_results_section_en.tex")
    fr_path = os.path.join(out_dir, "ieee_results_section_fr.tex")

    with open(en_path, "w", encoding="utf-8") as f:
        f.write(en_tex)
    with open(fr_path, "w", encoding="utf-8") as f:
        f.write(fr_tex)

    print(f"[MCAD/LATEX] {en_path}")
    print(f"[MCAD/LATEX] {fr_path}")
    print("[MCAD/LATEX] Done.")


if __name__ == "__main__":
    main()
