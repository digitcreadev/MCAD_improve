from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BENEFIT_METRICS = ['phi_final', 'auc_phi']
COST_METRICS = ['false_allow_rate', 'non_contrib_exec_rate', 'false_block_rate']
DEFAULT_COMPARATORS = [
    'baseline_naive',
    'baseline_measure_overlap',
    'baseline_random_matched',
    'ablation_no_sat',
    'ablation_no_real',
    'ablation_ceval_any_intersection',
]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_csv_auto(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine='python')


def bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 4000, seed: int = 42) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_advantage_ci(mcad: np.ndarray, other: np.ndarray, metric: str, *, n_boot: int = 4000, seed: int = 42) -> Tuple[float, float, float, np.ndarray]:
    mcad = np.asarray(mcad, dtype=float)
    other = np.asarray(other, dtype=float)
    if metric in BENEFIT_METRICS:
        diff = mcad - other
    else:
        diff = other - mcad
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), diff


def sign_flip_pvalue(diff: np.ndarray, *, n_perm: int = 5000, seed: int = 42) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return 1.0
    obs = abs(float(diff.mean()))
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))
    p = (np.sum(perm_means >= obs) + 1) / (n_perm + 1)
    return float(p)


def dominance_rate(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if diff.size == 0:
        return 0.0
    wins = np.sum(diff > 0)
    ties = np.sum(diff == 0)
    return float((wins + 0.5 * ties) / diff.size)


def paired_frame(df: pd.DataFrame, policy_a: str, policy_b: str, key_cols: List[str], metric: str) -> Tuple[np.ndarray, np.ndarray]:
    cols = key_cols + [metric]
    a = df[df['policy'] == policy_a][cols].copy()
    b = df[df['policy'] == policy_b][cols].copy()
    merged = a.merge(b, on=key_cols, suffixes=('_a', '_b'))
    return merged[f'{metric}_a'].to_numpy(dtype=float), merged[f'{metric}_b'].to_numpy(dtype=float)


def compute_policy_ci_table(df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for policy, sub in sorted(df.groupby('policy')):
        row: Dict[str, object] = {'policy': policy, 'n_sessions': int(len(sub))}
        for metric in metrics:
            mean_v, lo, hi = bootstrap_mean_ci(sub[metric].to_numpy(dtype=float))
            row[f'{metric}_mean'] = round(mean_v, 6)
            row[f'{metric}_ci_low'] = round(lo, 6)
            row[f'{metric}_ci_high'] = round(hi, 6)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pairwise_advantages(df: pd.DataFrame, comparators: Iterable[str], metrics: Iterable[str], key_cols: List[str]) -> pd.DataFrame:
    rows = []
    for comp in comparators:
        for metric in metrics:
            mcad, other = paired_frame(df, 'mcad', comp, key_cols, metric)
            if mcad.size == 0:
                continue
            adv, lo, hi, diff = paired_bootstrap_advantage_ci(mcad, other, metric)
            p = sign_flip_pvalue(diff)
            rows.append({
                'comparator': comp,
                'metric': metric,
                'n_pairs': int(diff.size),
                'mcad_advantage_mean': round(adv, 6),
                'ci_low': round(lo, 6),
                'ci_high': round(hi, 6),
                'p_value_sign_flip': round(p, 6),
                'dominance_rate': round(dominance_rate(diff), 6),
            })
    return pd.DataFrame(rows)


def compute_ablation_sensitivity(df: pd.DataFrame, metrics: Iterable[str], key_cols: List[str]) -> pd.DataFrame:
    ablations = ['ablation_no_sat', 'ablation_no_real', 'ablation_ceval_any_intersection']
    rows = []
    for abl in ablations:
        for metric in metrics:
            mcad, other = paired_frame(df, 'mcad', abl, key_cols, metric)
            if mcad.size == 0:
                continue
            adv, lo, hi, diff = paired_bootstrap_advantage_ci(mcad, other, metric)
            rows.append({
                'ablation': abl,
                'metric': metric,
                'mcad_advantage_mean': round(adv, 6),
                'ci_low': round(lo, 6),
                'ci_high': round(hi, 6),
                'dominance_rate': round(dominance_rate(diff), 6),
            })
    return pd.DataFrame(rows)


def plot_policy_ci(policy_ci: pd.DataFrame, metric: str, title: str, out_path: str) -> None:
    sub = policy_ci[['policy', f'{metric}_mean', f'{metric}_ci_low', f'{metric}_ci_high']].copy()
    sub = sub.sort_values(f'{metric}_mean', ascending=(metric in COST_METRICS))
    y = np.arange(len(sub))
    mean_vals = sub[f'{metric}_mean'].to_numpy(dtype=float)
    lo = sub[f'{metric}_ci_low'].to_numpy(dtype=float)
    hi = sub[f'{metric}_ci_high'].to_numpy(dtype=float)
    plt.figure(figsize=(8.2, 3.8))
    plt.errorbar(mean_vals, y, xerr=[mean_vals - lo, hi - mean_vals], fmt='o', capsize=3)
    plt.yticks(y, sub['policy'])
    plt.xlabel(metric)
    plt.title(title)
    plt.grid(axis='x', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_advantage_ci(pairwise: pd.DataFrame, metric: str, title: str, out_path: str) -> None:
    sub = pairwise[pairwise['metric'] == metric].copy()
    if sub.empty:
        return
    sub = sub.sort_values('mcad_advantage_mean', ascending=True)
    y = np.arange(len(sub))
    mean_vals = sub['mcad_advantage_mean'].to_numpy(dtype=float)
    lo = sub['ci_low'].to_numpy(dtype=float)
    hi = sub['ci_high'].to_numpy(dtype=float)
    plt.figure(figsize=(8.4, 4.2))
    plt.axvline(0.0, linewidth=1)
    plt.errorbar(mean_vals, y, xerr=[mean_vals - lo, hi - mean_vals], fmt='o', capsize=3)
    plt.yticks(y, sub['comparator'])
    xlabel = f'MCAD advantage on {metric} (positive favors MCAD)'
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis='x', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_ablation_sensitivity(sens: pd.DataFrame, metric: str, title: str, out_path: str) -> None:
    sub = sens[sens['metric'] == metric].copy()
    if sub.empty:
        return
    sub = sub.sort_values('mcad_advantage_mean', ascending=False)
    y = np.arange(len(sub))
    mean_vals = sub['mcad_advantage_mean'].to_numpy(dtype=float)
    plt.figure(figsize=(7.6, 3.5))
    plt.barh(y, mean_vals)
    plt.yticks(y, sub['ablation'])
    plt.xlabel(f'MCAD advantage on {metric}')
    plt.title(title)
    plt.grid(axis='x', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def policy_ci_json(policy_ci: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for _, row in policy_ci.iterrows():
        policy = str(row['policy'])
        entry: Dict[str, float] = {'n_sessions': int(row['n_sessions'])}
        for col in row.index:
            if col in {'policy', 'n_sessions'}:
                continue
            entry[col] = float(row[col])
        out[policy] = entry
    return out


def build_markdown_report(policy_ci: pd.DataFrame, pairwise: pd.DataFrame, sensitivity: pd.DataFrame, out_path: str) -> None:
    lines: List[str] = []
    lines.append('# Phase 7 statistical analysis report')
    lines.append('')
    lines.append('This report provides bootstrap confidence intervals, paired sign-flip tests, and ablation sensitivity summaries based on the current MCAD benchmark outputs.')
    lines.append('')
    lines.append('## Policy-level confidence intervals')
    lines.append('')
    headers = ['policy', 'n_sessions', 'phi_final_mean', 'phi_final_ci_low', 'phi_final_ci_high', 'auc_phi_mean', 'auc_phi_ci_low', 'auc_phi_ci_high', 'false_allow_rate_mean', 'false_allow_rate_ci_low', 'false_allow_rate_ci_high']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for _, row in policy_ci.iterrows():
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    lines.append('')
    lines.append('## Paired policy comparisons (MCAD advantage)')
    lines.append('')
    headers = ['comparator', 'metric', 'n_pairs', 'mcad_advantage_mean', 'ci_low', 'ci_high', 'p_value_sign_flip', 'dominance_rate']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for _, row in pairwise.iterrows():
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    lines.append('')
    lines.append('## Ablation sensitivity (robustness campaign)')
    lines.append('')
    headers = ['ablation', 'metric', 'mcad_advantage_mean', 'ci_low', 'ci_high', 'dominance_rate']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for _, row in sensitivity.iterrows():
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run Phase 7 statistical analysis on MCAD benchmark outputs.')
    p.add_argument('--run-root', default='', help='Optional run root containing multidataset/ robustness/ evidence/ folders.')
    p.add_argument('--multidataset-session-csv', default='results_phase7_stats/multidataset/aggregate_policy_session_metrics.csv')
    p.add_argument('--robustness-session-csv', default='results_robustness_final/robustness_policy_session_metrics.csv')
    p.add_argument('--evidence-report-json', default='results_phase6_evidence/evidence_usefulness_report.json')
    p.add_argument('--evidence-bootstrap-json', default='results_phase6_evidence/bootstrap_benefit_summary.json')
    p.add_argument('--out-dir', default='results_phase7_stats')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve() if args.run_root else None
    out_dir = Path(args.out_dir)
    if run_root is not None and not out_dir.is_absolute():
        out_dir = run_root / out_dir
    fig_dir = out_dir / 'figures'
    ensure_dir(fig_dir)

    multidataset_csv = Path(args.multidataset_session_csv)
    robustness_csv = Path(args.robustness_session_csv)
    evidence_report_path = Path(args.evidence_report_json)
    evidence_boot_path = Path(args.evidence_bootstrap_json)
    if run_root is not None:
        if not multidataset_csv.exists():
            multidataset_csv = run_root / 'multidataset' / 'aggregate_policy_session_metrics.csv'
        if not robustness_csv.exists():
            robustness_csv = run_root / 'robustness' / 'robustness_policy_session_metrics.csv'
        if not evidence_report_path.exists():
            evidence_report_path = run_root / 'evidence' / 'evidence_usefulness_report.json'
        if not evidence_boot_path.exists():
            evidence_boot_path = run_root / 'evidence' / 'bootstrap_benefit_summary.json'
    print(f'[MCAD/PHASE7] using {multidataset_csv}')
    print(f'[MCAD/PHASE7] using {robustness_csv}')
    print(f'[MCAD/PHASE7] using {evidence_report_path}')
    print(f'[MCAD/PHASE7] using {evidence_boot_path}')
    multi = load_csv_auto(multidataset_csv)
    robust = load_csv_auto(robustness_csv)

    for df in [multi, robust]:
        for c in ['repeat_id', 'step_idx']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        for c in ['phi_final', 'auc_phi', 'false_allow_rate', 'false_block_rate', 'non_contrib_exec_rate']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    multi_key = ['config_path', 'dw_id', 'objective_id', 'scenario_id', 'repeat_id']
    robust_key = ['config_path', 'dw_id', 'objective_id', 'scenario_id', 'repeat_id']

    metrics_main = ['phi_final', 'auc_phi', 'false_allow_rate', 'non_contrib_exec_rate']
    metrics_robust = ['false_allow_rate', 'auc_phi', 'false_block_rate']

    policy_ci = compute_policy_ci_table(multi, metrics_main)
    pairwise = compute_pairwise_advantages(multi, DEFAULT_COMPARATORS, metrics_main, multi_key)
    sensitivity = compute_ablation_sensitivity(robust, metrics_robust, robust_key)

    policy_ci.to_csv(out_dir / 'policy_confidence_intervals.csv', index=False)
    pairwise.to_csv(out_dir / 'pairwise_policy_statistics.csv', index=False)
    sensitivity.to_csv(out_dir / 'ablation_sensitivity_summary.csv', index=False)

    plot_policy_ci(policy_ci, 'phi_final', 'Policy means with 95% bootstrap CI - phi_final', str(fig_dir / 'policy_ci_phi_final.png'))
    plot_policy_ci(policy_ci, 'false_allow_rate', 'Policy means with 95% bootstrap CI - false_allow_rate', str(fig_dir / 'policy_ci_false_allow.png'))
    plot_advantage_ci(pairwise, 'auc_phi', 'MCAD advantage over comparators - AUC(phi)', str(fig_dir / 'mcad_advantage_auc_phi.png'))
    plot_advantage_ci(pairwise, 'false_allow_rate', 'MCAD advantage over comparators - false_allow_rate', str(fig_dir / 'mcad_advantage_false_allow.png'))
    plot_ablation_sensitivity(sensitivity, 'false_allow_rate', 'Ablation sensitivity on robustness - false_allow_rate', str(fig_dir / 'ablation_sensitivity_false_allow.png'))
    plot_ablation_sensitivity(sensitivity, 'auc_phi', 'Ablation sensitivity on robustness - AUC(phi)', str(fig_dir / 'ablation_sensitivity_auc_phi.png'))

    evidence_report = json.loads(Path(evidence_report_path).read_text(encoding='utf-8')) if Path(evidence_report_path).exists() else {}
    bootstrap_summary = json.loads(Path(evidence_boot_path).read_text(encoding='utf-8')) if Path(evidence_boot_path).exists() else []

    summary = {
        'run_root': str(run_root) if run_root else '',
        'policy_confidence_intervals': policy_ci_json(policy_ci),
        'pairwise_highlights': pairwise.to_dict(orient='records'),
        'ablation_sensitivity': sensitivity.to_dict(orient='records'),
        'evidence_usefulness_report': evidence_report,
        'evidence_bootstrap_summary': bootstrap_summary,
    }
    Path(out_dir / 'phase7_statistical_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    build_markdown_report(policy_ci, pairwise, sensitivity, out_dir / 'phase7_statistical_report.md')
    print(f'[MCAD/PHASE7] done -> {out_dir}')


if __name__ == '__main__':
    main()
