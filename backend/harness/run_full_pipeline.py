# backend/harness/run_full_pipeline.py
# One-command runner for the complete MCAD CKG-first evaluation + paper artifacts pipeline
#
# Stages:
#   1) Generate traces:
#        - mode=scenarios: run_scenarios.py (local CKG-first) -> timelines.json + sessions_index.json + performance.json
#        - mode=sessions : run_1000_sessions.py              -> timelines_1000.json + sessions_index_1000.json + ckg_state.json
#   2) explainability_metrics.py
#   3) aggregate_metrics_from_timelines.py
#   4) paper_artifacts.py
#   5) guided_vs_naive_report.py
#   6) write_paper_results.py
#   7) generate_ieee_results_section.py
#   8) generate_ieee_minipaper.py
#   9) generate_camera_ready_pack.py
#  10) sanity_checks.py
#
# Usage:
#   python backend/harness/run_full_pipeline.py --mode scenarios --results-dir results_ckg
#   python backend/harness/run_full_pipeline.py --mode sessions  --results-dir results_1000 --n-guided 500 --n-naive 500
#
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional, Tuple


def here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def exists(path: str) -> bool:
    return os.path.exists(path)


def python_exe() -> str:
    return sys.executable or "python"


def run(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("[MCAD/PIPE] $ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def opt_script(rel: str) -> Optional[str]:
    p = os.path.join(here(), rel)
    return p if exists(p) else None


def detect_outputs(results_dir: str, mode: str) -> Tuple[str, str]:
    if mode == "sessions":
        tl = os.path.join(results_dir, "timelines_1000.json")
        idx = os.path.join(results_dir, "sessions_index_1000.json")
        if exists(tl) and exists(idx):
            return tl, idx
    return os.path.join(results_dir, "timelines.json"), os.path.join(results_dir, "sessions_index.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full MCAD CKG-first pipeline and generate paper artifacts.")
    p.add_argument("--mode", choices=["scenarios", "sessions"], default="scenarios",
                   help="scenarios: run_scenarios.py ; sessions: run_1000_sessions.py")
    p.add_argument("--run-root", type=str, default="", help="Optional run root under which results-dir is resolved")
    p.add_argument("--results-dir", type=str, default="results_ckg")
    p.add_argument("--config", type=str, default="", help="Path to harness/scenarios.yaml (optional override)")

    # sessions-mode controls (mirrors run_1000_sessions.py)
    p.add_argument("--n-guided", type=int, default=500)
    p.add_argument("--n-naive", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-snapshots", action="store_true")

    # downstream grouping labels
    p.add_argument("--group-a", type=str, default="guided")
    p.add_argument("--group-b", type=str, default="naive")

    # stage controls
    p.add_argument("--start", type=int, default=1, help="Start stage (1..10)")
    p.add_argument("--stop", type=int, default=10, help="Stop stage (1..10)")

    # paper options
    p.add_argument("--include-discussion", action="store_true", help="Include Discussion in LaTeX results section")
    p.add_argument("--overleaf-root", type=str, default="", help="Root folder name inside camera-ready ZIP")
    p.add_argument("--with-policy-benchmark", action="store_true", help="Run baselines + ablations benchmark after the standard pipeline")
    p.add_argument("--with-multidataset-benchmark", action="store_true", help="Run the multi-dataset benchmark across several configs")
    p.add_argument("--benchmark-repeats", type=int, default=50, help="Number of benchmark repeats for policy benchmark")
    p.add_argument("--benchmark-config", action="append", default=[], help="Repeatable benchmark config path; used by the multi-dataset benchmark")
    p.add_argument("--with-human-validation-dry-run", action="store_true", help="Generate an expert annotation pack and score systems against the scenario oracle (dry run).")
    p.add_argument("--with-robustness-benchmark", action="store_true", help="Run the dedicated robustness / adversarial benchmark across robustness configs.")
    p.add_argument("--with-scalability-benchmark", action="store_true", help="Run the scalability and CKG growth-control benchmark.")
    p.add_argument("--with-statistical-analysis", action="store_true", help="Run the Phase 7 statistical analysis layer when benchmark outputs are available.")

    # optional objectives override for scenarios mode
    p.add_argument("--objectives-yaml", type=str, default="", help="Optional path to objectives.yaml for run_scenarios.py (local mode)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if args.run_root and not os.path.isabs(results_dir):
        results_dir = os.path.join(args.run_root, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    config = args.config.strip() or os.path.join(here(), "scenarios.yaml")

    # Stage 1: Generate traces (timelines + sessions_index [+ ckg_state])
    if args.start <= 1 <= args.stop:
        if args.mode == "scenarios":
            script = opt_script("run_scenarios.py")
            if not script:
                raise FileNotFoundError("backend/harness/run_scenarios.py not found.")
            cmd = [python_exe(), script, "--config", config, "--results-dir", results_dir, "--mode", "local"]
            if args.objectives_yaml.strip():
                cmd += ["--objectives-yaml", args.objectives_yaml.strip()]
            run(cmd)
        else:
            script = opt_script("run_1000_sessions.py")
            if not script:
                raise FileNotFoundError("backend/harness/run_1000_sessions.py not found.")
            cmd = [
                python_exe(), script,
                "--config", config,
                "--results-dir", results_dir,
                "--n-guided", str(args.n_guided),
                "--n-naive", str(args.n_naive),
                "--seed", str(args.seed),
            ]
            if args.save_snapshots:
                cmd.append("--save-snapshots")
            run(cmd)

    timelines_path, sessions_index_path = detect_outputs(results_dir, args.mode)

    # Stage 2: Explainability metrics
    if args.start <= 2 <= args.stop:
        script = opt_script("explainability_metrics.py")
        if script:
            run([python_exe(), script, "--results-dir", results_dir])
        else:
            print("[MCAD/PIPE] (skip) explainability_metrics.py not found")

    # Stage 3: Aggregate metrics
    if args.start <= 3 <= args.stop:
        script = opt_script("aggregate_metrics_from_timelines.py")
        if script:
            run([
                python_exe(), script,
                "--timelines", timelines_path,
                "--sessions-index", sessions_index_path,
                "--out-dir", results_dir,
                "--ckg-path", os.path.join(results_dir, "ckg_state.json"),
            ])
        else:
            print("[MCAD/PIPE] (skip) aggregate_metrics_from_timelines.py not found")

    # Stage 4: Paper artifacts
    if args.start <= 4 <= args.stop:
        script = opt_script("paper_artifacts.py")
        if script:
            run([python_exe(), script, "--results-dir", results_dir])
        else:
            print("[MCAD/PIPE] (skip) paper_artifacts.py not found")

    # Stage 5: Stats report
    if args.start <= 5 <= args.stop:
        script = opt_script("guided_vs_naive_report.py")
        if script:
            run([python_exe(), script, "--results-dir", results_dir, "--group-a", args.group_a, "--group-b", args.group_b])
        else:
            print("[MCAD/PIPE] (skip) guided_vs_naive_report.py not found")

    # Stage 6: Paper-ready paragraphs
    if args.start <= 6 <= args.stop:
        script = opt_script("write_paper_results.py")
        if script:
            run([python_exe(), script, "--results-dir", results_dir, "--group-a", args.group_a, "--group-b", args.group_b])
        else:
            print("[MCAD/PIPE] (skip) write_paper_results.py not found")

    # Stage 7: LaTeX Results section
    if args.start <= 7 <= args.stop:
        script = opt_script("generate_ieee_results_section.py")
        if script:
            cmd = [python_exe(), script, "--results-dir", results_dir]
            if args.include_discussion:
                cmd.append("--include-discussion")
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) generate_ieee_results_section.py not found")

    # Stage 8: IEEE minipaper
    if args.start <= 8 <= args.stop:
        script = opt_script("generate_ieee_minipaper.py")
        if script:
            run([python_exe(), script, "--results-dir", results_dir])
        else:
            print("[MCAD/PIPE] (skip) generate_ieee_minipaper.py not found")

    # Stage 9: Camera-ready pack
    if args.start <= 9 <= args.stop:
        script = opt_script("generate_camera_ready_pack.py")
        if script:
            cmd = [python_exe(), script, "--results-dir", results_dir]
            if args.overleaf_root.strip():
                cmd += ["--overleaf-root", args.overleaf_root.strip()]
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) generate_camera_ready_pack.py not found")

    # Stage 10: Sanity checks
    if args.start <= 10 <= args.stop:
        script = opt_script("sanity_checks.py")
        if script:
            run([python_exe(), script, "--results-dir", results_dir])
        else:
            print("[MCAD/PIPE] (skip) sanity_checks.py not found")


    # Optional policy benchmark (baselines + ablations)
    if args.with_policy_benchmark:
        script = opt_script("run_baselines_and_ablations.py")
        if script:
            run([python_exe(), script, "--config", config, "--results-dir", os.path.join(results_dir, "policy_benchmark"), "--repeats", str(args.benchmark_repeats), "--seed", str(args.seed)])
        else:
            print("[MCAD/PIPE] (skip) run_baselines_and_ablations.py not found")

    if args.with_multidataset_benchmark:
        script = opt_script("run_multidataset_policy_benchmark.py")
        if script:
            cmd = [python_exe(), script, "--results-dir", os.path.join(results_dir, "multidataset_benchmark"), "--repeats", str(args.benchmark_repeats), "--seed", str(args.seed)]
            cfgs = args.benchmark_config or []
            if not cfgs:
                cfgs = [config, os.path.join(here(), "scenarios_adventureworks.yaml")]
            for cfg in cfgs:
                cmd += ["--config", cfg]
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) run_multidataset_policy_benchmark.py not found")

    if args.with_human_validation_dry_run:
        gen = opt_script("generate_expert_annotation_pack.py")
        score = opt_script("score_human_validation.py")
        hv_dir = os.path.join(results_dir, "human_validation")
        cfgs = args.benchmark_config or [config, os.path.join(here(), "scenarios_adventureworks.yaml")]
        if gen:
            cmd = [python_exe(), gen, "--out-dir", os.path.join(hv_dir, "pack"), "--include-oracle"]
            for cfg in cfgs:
                cmd += ["--config", cfg]
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) generate_expert_annotation_pack.py not found")
        if score:
            cmd = [python_exe(), score, "--out-dir", os.path.join(hv_dir, "report")]
            for cfg in cfgs:
                cmd += ["--config", cfg]
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) score_human_validation.py not found")

    if args.with_robustness_benchmark:
        script = opt_script("run_robustness_benchmark.py")
        if script:
            rb_dir = os.path.join(results_dir, "robustness_benchmark")
            cmd = [python_exe(), script, "--results-dir", rb_dir, "--repeats", str(args.benchmark_repeats), "--seed", str(args.seed)]
            cfgs = args.benchmark_config or [os.path.join(here(), "scenarios_robustness_foodmart.yaml"), os.path.join(here(), "scenarios_robustness_adventureworks.yaml")]
            for cfg in cfgs:
                cmd += ["--config", cfg]
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) run_robustness_benchmark.py not found")

    if args.with_scalability_benchmark:
        script = opt_script("run_scalability_benchmark.py")
        if script:
            sb_dir = os.path.join(results_dir, "scalability_benchmark")
            cmd = [python_exe(), script, "--results-dir", sb_dir]
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) run_scalability_benchmark.py not found")

    if args.with_statistical_analysis:
        script = opt_script("run_statistical_analysis.py")
        if script:
            stats_dir = os.path.join(results_dir, "phase7_stats")
            cmd = [python_exe(), script, "--out-dir", stats_dir]
            multi_csv = os.path.join(results_dir, "multidataset_benchmark", "aggregate_policy_session_metrics.csv")
            robust_csv = os.path.join(results_dir, "robustness_benchmark", "robustness_policy_session_metrics.csv")
            evidence_json = os.path.join(results_dir, "phase6_evidence", "evidence_usefulness_report.json")
            evidence_boot = os.path.join(results_dir, "phase6_evidence", "bootstrap_benefit_summary.json")
            if exists(multi_csv):
                cmd += ["--multidataset-session-csv", multi_csv]
            if exists(robust_csv):
                cmd += ["--robustness-session-csv", robust_csv]
            if exists(evidence_json):
                cmd += ["--evidence-report-json", evidence_json]
            if exists(evidence_boot):
                cmd += ["--evidence-bootstrap-json", evidence_boot]
            run(cmd)
        else:
            print("[MCAD/PIPE] (skip) run_statistical_analysis.py not found")

    print("[MCAD/PIPE] Done.")


if __name__ == "__main__":
    main()
