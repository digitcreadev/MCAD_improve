# backend/harness/sanity_checks.py
# (NEW: Sanity checks + consistency validation for MCAD CKG-first outputs)
#
# Purpose:
#   - Detect anomalies/inconsistencies in the generated timelines (φ dynamics, SAT/Real/Ceval coherence, ranges, duplicates)
#   - Produce reviewer-friendly validation artifacts:
#       * reports/sanity_checks_report.txt
#       * reports/sanity_checks_summary.csv
#       * reports/sanity_checks_issues.csv (issue-level log)
#       * paper/threats_to_validity_en.txt (optional narrative starter)
#       * latex/ieee_validity_section_en.tex (optional LaTeX snippet)
#
# Inputs:
#   - <results-dir>/timelines.json OR timelines_1000.json
#   - <results-dir>/sessions_index.json (optional)
#
# Usage:
#   python backend/harness/sanity_checks.py --results-dir results_1000
#   python backend/harness/sanity_checks.py --results-dir results_ckg
#
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ---------- Helpers ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def get_phi_cum(step: Dict[str, Any]) -> Optional[float]:
    for k in ("phi_leq_t", "phi_leq", "phi_cum", "phi_cumulative", "phi_session"):
        if k in step and step.get(k) is not None:
            v = to_float(step.get(k))
            return 0.0 if v is None else v
    # If only per-step φ exists, treat it as not cumulative
    return None


def get_phi_weighted_cum(step: Dict[str, Any]) -> Optional[float]:
    for k in ("phi_weighted_leq_t", "phi_weighted_leq", "phi_weighted_cum"):
        if k in step and step.get(k) is not None:
            v = to_float(step.get(k))
            return 0.0 if v is None else v
    return None


def detect_timelines(results_dir: str) -> str:
    cand = [
        os.path.join(results_dir, "timelines_1000.json"),
        os.path.join(results_dir, "timelines.json"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Aucun timelines*.json trouvé dans {results_dir}")


def detect_sessions_index(results_dir: str) -> Optional[str]:
    cand = [
        os.path.join(results_dir, "sessions_index_1000.json"),
        os.path.join(results_dir, "sessions_index.json"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None


# ---------- Issue taxonomy ----------

ISSUE_TYPES = [
    "phi_negative",
    "phi_out_of_range",
    "phi_non_monotone",
    "phi_weighted_out_of_range",
    "delta_negative",
    "sat_false_but_real_nonempty",
    "sat_false_but_ceval_nonempty",
    "real_empty_but_ceval_nonempty",
    "duplicate_constraints_in_step",
    "missing_required_fields",
]


@dataclass
class Issue:
    issue_type: str
    scenario_id: str
    session_id: str
    step_index: int
    field: str
    value: str
    detail: str


# ---------- Core checks ----------

def check_session(scenario_id: str, payload: Dict[str, Any]) -> Tuple[List[Issue], Dict[str, Any]]:
    issues: List[Issue] = []

    session_id = str(payload.get("session_id") or "")
    steps: List[Dict[str, Any]] = payload.get("steps", []) or []

    # Stats
    n_steps = len(steps)
    n_sat_false = 0
    n_real_empty = 0
    n_ceval_empty = 0

    # Monotonicity (if cumulative φ is present)
    last_phi_cum: Optional[float] = None
    last_phi_w_cum: Optional[float] = None

    for i, s in enumerate(steps):
        # Required fields
        if "sat" not in s:
            issues.append(Issue("missing_required_fields", scenario_id, session_id, i, "sat", "", "Missing 'sat' in step"))
        if "phi" not in s and get_phi_cum(s) is None:
            issues.append(Issue("missing_required_fields", scenario_id, session_id, i, "phi/phi_cum", "", "Missing 'phi' and cumulative φ fields"))

        # φ per step (bounded)
        phi = to_float(s.get("phi"))
        if phi is not None:
            if phi < -1e-12:
                issues.append(Issue("phi_negative", scenario_id, session_id, i, "phi", str(phi), "Per-step φ is negative"))
            if phi < -1e-6 or phi > 1.0 + 1e-6:
                issues.append(Issue("phi_out_of_range", scenario_id, session_id, i, "phi", str(phi), "Per-step φ outside [0,1]"))

        # cumulative φ monotonic (if present)
        phi_cum = get_phi_cum(s)
        if phi_cum is not None:
            if phi_cum < -1e-6 or phi_cum > 1.0 + 1e-6:
                issues.append(Issue("phi_out_of_range", scenario_id, session_id, i, "phi_cum", str(phi_cum), "Cumulative φ outside [0,1]"))
            if last_phi_cum is not None and phi_cum + 1e-9 < last_phi_cum:
                issues.append(Issue("phi_non_monotone", scenario_id, session_id, i, "phi_cum", str(phi_cum), f"Non-monotone cumulative φ: prev={last_phi_cum} curr={phi_cum}"))
            last_phi_cum = phi_cum

        # weighted cumulative φ
        phi_w = get_phi_weighted_cum(s)
        if phi_w is not None:
            if phi_w < -1e-6 or phi_w > 1.0 + 1e-6:
                issues.append(Issue("phi_weighted_out_of_range", scenario_id, session_id, i, "phi_weighted_cum", str(phi_w), "Weighted cumulative φ outside [0,1]"))
            if last_phi_w_cum is not None and phi_w + 1e-9 < last_phi_w_cum:
                issues.append(Issue("phi_non_monotone", scenario_id, session_id, i, "phi_weighted_cum", str(phi_w), f"Non-monotone weighted φ: prev={last_phi_w_cum} curr={phi_w}"))
            last_phi_w_cum = phi_w

        # delta φ should not be negative if present
        if "delta_phi_t" in s and s.get("delta_phi_t") is not None:
            d = to_float(s.get("delta_phi_t"))
            if d is not None and d < -1e-9:
                issues.append(Issue("delta_negative", scenario_id, session_id, i, "delta_phi_t", str(d), "Negative delta_phi_t"))

        sat = s.get("sat", True)
        if sat is False:
            n_sat_false += 1

        real_ids = s.get("real_node_ids", None)
        ceval = s.get("calculable_constraints", None) or []

        # Real emptiness
        real_nonempty = False
        if isinstance(real_ids, list):
            real_nonempty = len(real_ids) > 0
            if not real_nonempty:
                n_real_empty += 1
        elif real_ids is None:
            # if missing, we don't count emptiness but may still flag if ceval exists
            pass

        # Ceval emptiness
        if isinstance(ceval, list):
            if len(ceval) == 0:
                n_ceval_empty += 1

        # Consistency: SAT false should imply Real empty and Ceval empty (in typical MCAD semantics)
        if sat is False:
            if real_nonempty:
                issues.append(Issue("sat_false_but_real_nonempty", scenario_id, session_id, i, "real_node_ids", str(len(real_ids)), "SAT=false but Real(QP) non-empty"))
            if isinstance(ceval, list) and len(ceval) > 0:
                issues.append(Issue("sat_false_but_ceval_nonempty", scenario_id, session_id, i, "calculable_constraints", str(len(ceval)), "SAT=false but Ceval(QP,O) non-empty"))

        # Consistency: Real empty but Ceval non-empty is suspicious
        if isinstance(real_ids, list) and len(real_ids) == 0 and isinstance(ceval, list) and len(ceval) > 0:
            issues.append(Issue("real_empty_but_ceval_nonempty", scenario_id, session_id, i, "calculable_constraints", str(len(ceval)), "Real empty but Ceval non-empty"))

        # Duplicate constraints in step (by id if dicts, else by str)
        if isinstance(ceval, list) and len(ceval) > 1:
            seen = set()
            dups = 0
            for c in ceval:
                if isinstance(c, dict):
                    key = str(c.get("id") or c.get("name") or c)
                else:
                    key = str(c)
                if key in seen:
                    dups += 1
                seen.add(key)
            if dups > 0:
                issues.append(Issue("duplicate_constraints_in_step", scenario_id, session_id, i, "calculable_constraints", str(dups), "Duplicate constraints detected within the same step"))

    # Summary stats per session
    summary = {
        "scenario_id": scenario_id,
        "session_id": session_id,
        "n_steps": n_steps,
        "n_issues": len(issues),
        "sat_false_ratio": (n_sat_false / n_steps) if n_steps else 0.0,
        "real_empty_ratio_raw": (n_real_empty / n_steps) if n_steps else 0.0,   # raw, includes SAT-false steps
        "ceval_empty_ratio_raw": (n_ceval_empty / n_steps) if n_steps else 0.0, # raw, includes SAT-false steps
        "has_phi_cum": 1 if last_phi_cum is not None else 0,
        "has_phi_weighted_cum": 1 if last_phi_w_cum is not None else 0,
    }
    return issues, summary


# ---------- Narrative helpers ----------

def threats_to_validity_text_en(summary_rows: List[Dict[str, Any]], issues_by_type: Dict[str, int]) -> str:
    # Aggregate headline rates
    n_sessions = len(summary_rows)
    n_issues = sum(int(r.get("n_issues", 0)) for r in summary_rows)
    frac_sessions_with_issues = (sum(1 for r in summary_rows if int(r.get("n_issues", 0)) > 0) / n_sessions) if n_sessions else 0.0

    sat_false = [float(r.get("sat_false_ratio", 0.0)) for r in summary_rows]
    sat_false_mean = statistics.mean(sat_false) if sat_false else 0.0

    return (
        "Threats to validity. "
        "We performed automated sanity checks over the produced timelines to validate internal consistency of the MCAD execution trace "
        "(monotonicity and range of cumulative contribution, coherence between SAT/Real/Ceval, and duplicate constraint detection). "
        f"Across {n_sessions} sessions, the checker reported {n_issues} issues in total; "
        f"{frac_sessions_with_issues*100.0:.1f}% of sessions contained at least one flagged anomaly. "
        f"The mean SAT-failure ratio was {sat_false_mean:.3f}. "
        "Flagged anomalies primarily indicate trace-level inconsistencies (e.g., SAT=false with non-empty Real/Ceval, or non-monotone φ≤t), "
        "which can stem from instrumentation, schema mismatches, or scenario mis-specification rather than conceptual flaws in MCAD itself. "
        "We therefore interpret these checks as construct-validity safeguards and report the issue taxonomy and counts in the companion artifact files."
    )


def validity_section_latex_en(threats_text: str) -> str:
    # Minimal LaTeX-safe escaping for %, _, &
    t = threats_text.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    return "\\section{Threats to Validity}\n" + t + "\n"


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCAD sanity checks over timelines outputs.")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--max-issues", type=int, default=2000, help="Cap the number of issue rows written to issues.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    timelines_path = detect_timelines(results_dir)
    sessions_index_path = detect_sessions_index(results_dir)

    timelines = load_json(timelines_path)
    sessions_index = load_json(sessions_index_path) if sessions_index_path else None

    issues: List[Issue] = []
    summaries: List[Dict[str, Any]] = []

    # Keep label/type if available
    for scen_id, payload in timelines.items():
        iss, summ = check_session(str(scen_id), payload)
        # enrich summary with type/label
        if sessions_index and str(scen_id) in sessions_index:
            meta = sessions_index[str(scen_id)]
            summ["scenario_type"] = meta.get("type", "unknown")
            summ["scenario_label"] = meta.get("label", "")
        else:
            summ["scenario_type"] = "unknown"
            summ["scenario_label"] = ""
        issues.extend(iss)
        summaries.append(summ)

    # Issue counts by type
    issues_by_type: Dict[str, int] = {k: 0 for k in ISSUE_TYPES}
    for it in issues:
        issues_by_type[it.issue_type] = issues_by_type.get(it.issue_type, 0) + 1

    # Write CSV outputs
    reports_dir = os.path.join(results_dir, "reports")
    ensure_dir(reports_dir)

    summary_csv = os.path.join(reports_dir, "sanity_checks_summary.csv")
    issues_csv = os.path.join(reports_dir, "sanity_checks_issues.csv")

    write_csv(summaries, summary_csv)

    issue_rows: List[Dict[str, Any]] = []
    for it in issues[: args.max_issues]:
        issue_rows.append({
            "issue_type": it.issue_type,
            "scenario_id": it.scenario_id,
            "session_id": it.session_id,
            "step_index": it.step_index,
            "field": it.field,
            "value": it.value,
            "detail": it.detail,
        })
    write_csv(issue_rows, issues_csv)

    # Write text report
    report_lines: List[str] = []
    report_lines.append("=== MCAD Sanity Checks Report ===")
    report_lines.append(f"Results dir: {results_dir}")
    report_lines.append(f"Timelines: {os.path.basename(timelines_path)}")
    report_lines.append(f"Sessions index: {os.path.basename(sessions_index_path) if sessions_index_path else '-'}")
    report_lines.append("")
    report_lines.append(f"Total sessions: {len(summaries)}")
    report_lines.append(f"Total issues: {len(issues)}")
    report_lines.append("")
    report_lines.append("Issue counts by type:")
    for k in sorted(issues_by_type.keys()):
        report_lines.append(f"- {k}: {issues_by_type.get(k, 0)}")
    report_lines.append("")
    report_lines.append("Top example issues (first 25):")
    for it in issues[:25]:
        report_lines.append(
            f"* [{it.issue_type}] scen={it.scenario_id} sess={it.session_id} step={it.step_index} "
            f"field={it.field} value={it.value} :: {it.detail}"
        )

    report_path = os.path.join(reports_dir, "sanity_checks_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"[MCAD/SANITY] {summary_csv}")
    print(f"[MCAD/SANITY] {issues_csv}")
    print(f"[MCAD/SANITY] {report_path}")

    # Paper-friendly validity text
    paper_dir = os.path.join(results_dir, "paper")
    latex_dir = os.path.join(results_dir, "latex")
    ensure_dir(paper_dir)
    ensure_dir(latex_dir)

    threats = threats_to_validity_text_en(summaries, issues_by_type)
    threats_path = os.path.join(paper_dir, "threats_to_validity_en.txt")
    with open(threats_path, "w", encoding="utf-8") as f:
        f.write(threats + "\n")
    print(f"[MCAD/SANITY] {threats_path}")

    validity_tex = validity_section_latex_en(threats)
    validity_tex_path = os.path.join(latex_dir, "ieee_threats_to_validity_en.tex")
    with open(validity_tex_path, "w", encoding="utf-8") as f:
        f.write(validity_tex + "\n")
    print(f"[MCAD/SANITY] {validity_tex_path}")

    print("[MCAD/SANITY] Done.")


if __name__ == "__main__":
    main()
