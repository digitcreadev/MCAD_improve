from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.ckg.ckg_updater import CKGGraph  # noqa: E402
from backend.harness.run_baselines_and_ablations import (  # noqa: E402
    _normalize_qp,
    build_report,
    ensure_dir,
    load_yaml,
    play_policy_on_scenario,
    policy_decision,
    summarize_by_policy,
    write_csv,
)

POLICIES = [
    'mcad',
    'baseline_naive',
    'baseline_measure_overlap',
    'ablation_no_sat',
    'ablation_no_real',
    'ablation_ceval_any_intersection',
    'baseline_random_matched',
]

FAILURE_PRIORITY = [
    'measures_present',
    'cube_present',
    'objective_known',
    'slc_ok',
    'time_ok',
    'grain_ok',
    'agg_ok',
    'unit_ok',
    'nvac_ok',
]


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    idx = p * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    w = idx - lo
    return float((1 - w) * s[lo] + w * s[hi])


def _slug(text: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(text)).strip('_')


def _estimate_mcad_allow_probability(objective_id: str, scenarios: List[Dict[str, Any]], repeats: int, seed: int) -> float:
    mcad_allows = 0
    total_steps = 0
    for rep in range(repeats):
        for scenario in scenarios:
            ckg = CKGGraph(output_dir=os.environ.get('MCAD_TMP_DIR', 'results_policy_tmp'))
            for idx, step in enumerate(scenario.get('steps') or [], start=1):
                qp = _normalize_qp(step, objective_id)
                probe = policy_decision('mcad', ckg, objective_id, qp, rng=random.Random(seed + rep), matched_random_allow_prob=0.0, step_idx=idx)
                mcad_allows += int(bool(probe['allow']))
                total_steps += 1
    return (mcad_allows / total_steps) if total_steps else 0.5


def _summarize_by_scenario_type(session_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[(str(row.get('scenario_type') or ''), str(row.get('policy') or ''))].append(row)

    out: List[Dict[str, Any]] = []
    for (stype, policy), rows in sorted(grouped.items()):
        def avg(key: str) -> float:
            vals = [float(r.get(key) or 0.0) for r in rows if r.get(key) not in ('', None)]
            return round(sum(vals) / len(vals), 6) if vals else 0.0
        out.append({
            'scenario_type': stype,
            'policy': policy,
            'n_sessions': len(rows),
            'mean_phi_final': avg('phi_final'),
            'mean_auc_phi': avg('auc_phi'),
            'mean_false_allow_rate': avg('false_allow_rate'),
            'mean_false_block_rate': avg('false_block_rate'),
            'mean_non_contrib_exec_rate': avg('non_contrib_exec_rate'),
            'mean_time_to_0_8': avg('time_to_0_8'),
        })
    return out


def _plot_grouped_by_type(rows: List[Dict[str, Any]], metric_key: str, out_png: str, title: str, policies: List[str] | None = None) -> None:
    ensure_dir(os.path.dirname(out_png) or '.')
    stypes = sorted({str(r['scenario_type']) for r in rows})
    policy_order = policies or sorted({str(r['policy']) for r in rows})
    width = 0.12 if len(policy_order) > 4 else 0.18
    xs = list(range(len(stypes)))
    plt.figure(figsize=(10.5, 4.8))
    for j, pol in enumerate(policy_order):
        vals = []
        for st in stypes:
            row = next((r for r in rows if r['scenario_type'] == st and r['policy'] == pol), None)
            vals.append(float(row.get(metric_key) or 0.0) if row else 0.0)
        offs = [x + (j - (len(policy_order) - 1) / 2.0) * width for x in xs]
        plt.bar(offs, vals, width=width, label=pol)
    plt.xticks(xs, stypes, rotation=20, ha='right')
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _collect_mcad_explainability(config_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    explain_rows: List[Dict[str, Any]] = []
    for config_path in config_paths:
        cfg = load_yaml(config_path)
        objective_id = str(cfg['objective_id'])
        dw_id = str(cfg.get('dw_id') or 'UNKNOWN')
        for scenario in list(cfg.get('scenarios') or []):
            ckg = CKGGraph(output_dir=os.environ.get('MCAD_TMP_DIR', 'results_robustness_explain_tmp'))
            session_id = f"exp::{scenario.get('id')}"
            for idx, step in enumerate(scenario.get('steps') or [], start=1):
                qp = _normalize_qp(step, objective_id)
                result = ckg.evaluate_step(session_id=session_id, objective_id=objective_id, step_idx=idx, qp=qp)
                allow = bool(result.get('sat') and result.get('calculable_constraints'))
                failed = [c['name'] for c in (result.get('clauses') or []) if not bool(c.get('ok', False))]
                missing = result.get('missing_requirements') or {}
                primary = 'missing_requirement_set'
                for clause in FAILURE_PRIORITY:
                    if clause in failed:
                        primary = clause
                        break
                if not failed and not missing and not allow:
                    primary = 'unclassified_block'
                explain_rows.append({
                    'dw_id': dw_id,
                    'objective_id': objective_id,
                    'scenario_id': scenario.get('id'),
                    'scenario_type': scenario.get('type'),
                    'step_idx': idx,
                    'step_name': step.get('name'),
                    'oracle_allow': bool(step.get('oracle_allow', False)),
                    'mcad_allow': allow,
                    'sat': bool(result.get('sat')),
                    'failed_clauses': ','.join(failed),
                    'primary_reason': primary if not allow else 'allowed',
                    'has_missing_requirements': bool(missing),
                    'missing_requirement_constraints': ','.join(sorted(missing.keys())),
                    'n_missing_requirements': int(sum(len(v or []) for v in missing.values())),
                    'n_calculable_constraints': len(result.get('calculable_constraints') or []),
                    'induced_mask_size': len(result.get('induced_mask_node_ids') or []),
                    'explainable_block': int((not allow) and (bool(failed) or bool(missing))),
                })
    summary_map: Dict[str, Dict[str, Any]] = {}
    blocked_reason_counter: Counter[str] = Counter()
    for row in explain_rows:
        stype = str(row['scenario_type'])
        blocked = not bool(row['mcad_allow'])
        bucket = summary_map.setdefault(stype, {
            'scenario_type': stype,
            'n_steps': 0,
            'n_blocked': 0,
            'n_explainable_blocks': 0,
            'mean_missing_requirements_when_blocked': 0.0,
            'dominant_block_reason': '',
        })
        bucket['n_steps'] += 1
        if blocked:
            bucket['n_blocked'] += 1
            bucket['n_explainable_blocks'] += int(row['explainable_block'])
            bucket['mean_missing_requirements_when_blocked'] += float(row['n_missing_requirements'])
            blocked_reason_counter[str(row['primary_reason'])] += 1
            local = bucket.setdefault('_reason_counter', Counter())
            local[str(row['primary_reason'])] += 1
    summaries: List[Dict[str, Any]] = []
    for stype, bucket in sorted(summary_map.items()):
        n_blocked = max(1, int(bucket['n_blocked']))
        local_counter = bucket.pop('_reason_counter', Counter())
        bucket['explainable_block_rate'] = round(float(bucket['n_explainable_blocks']) / float(n_blocked), 6) if bucket['n_blocked'] else 1.0
        bucket['mean_missing_requirements_when_blocked'] = round(float(bucket['mean_missing_requirements_when_blocked']) / float(n_blocked), 6) if bucket['n_blocked'] else 0.0
        bucket['dominant_block_reason'] = local_counter.most_common(1)[0][0] if local_counter else ''
        summaries.append(bucket)
    reason_rows = [{'primary_reason': reason, 'count': count} for reason, count in blocked_reason_counter.most_common()]
    return explain_rows, summaries + reason_rows


def _plot_reason_distribution(reason_rows: List[Dict[str, Any]], out_png: str) -> None:
    ensure_dir(os.path.dirname(out_png) or '.')
    labels = [r['primary_reason'] for r in reason_rows]
    vals = [int(r['count']) for r in reason_rows]
    plt.figure(figsize=(10, 4.6))
    plt.bar(labels, vals)
    plt.xticks(rotation=25, ha='right')
    plt.title('MCAD primary block reasons on robustness workloads')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _plot_explainable_block_rate(summary_rows: List[Dict[str, Any]], out_png: str) -> None:
    ensure_dir(os.path.dirname(out_png) or '.')
    labels = [r['scenario_type'] for r in summary_rows]
    vals = [float(r['explainable_block_rate']) for r in summary_rows]
    plt.figure(figsize=(8.5, 4.4))
    plt.bar(labels, vals)
    plt.ylim(0.0, 1.05)
    plt.xticks(rotation=20, ha='right')
    plt.title('Explainable block rate by scenario type (MCAD)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _build_robustness_report(aggregate_summary: List[Dict[str, Any]], by_type_rows: List[Dict[str, Any]], explain_summary: List[Dict[str, Any]], out_path: str, config_paths: List[str]) -> None:
    explain_type_rows = [r for r in explain_summary if 'scenario_type' in r]
    reason_rows = [r for r in explain_summary if 'primary_reason' in r]
    lines: List[str] = []
    lines.append('# MCAD robustness and adversarial workload report')
    lines.append('')
    lines.append('## Configurations')
    lines.append('')
    for p in config_paths:
        lines.append(f'- `{p}`')
    lines.append('')
    lines.append('## Aggregate policy summary')
    lines.append('')
    headers = ['policy', 'mean_phi_final', 'mean_auc_phi', 'mean_false_allow_rate', 'mean_false_block_rate', 'mean_non_contrib_exec_rate']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in aggregate_summary:
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    lines.append('')
    lines.append('## Breakdown by scenario type and policy')
    lines.append('')
    headers = ['scenario_type', 'policy', 'mean_phi_final', 'mean_auc_phi', 'mean_false_allow_rate', 'mean_false_block_rate', 'mean_non_contrib_exec_rate']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in by_type_rows:
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    lines.append('')
    lines.append('## MCAD explainability on blocked steps')
    lines.append('')
    headers = ['scenario_type', 'n_steps', 'n_blocked', 'n_explainable_blocks', 'explainable_block_rate', 'mean_missing_requirements_when_blocked', 'dominant_block_reason']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in explain_type_rows:
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    lines.append('')
    lines.append('### Dominant block reasons overall')
    lines.append('')
    for row in reason_rows:
        lines.append(f"- **{row['primary_reason']}**: {row['count']} blocked steps")
    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run robustness / adversarial benchmark for MCAD.')
    p.add_argument('--config', action='append', default=[], help='Repeatable path to a robustness scenario YAML config.')
    p.add_argument('--run-root', type=str, default='', help='Optional run root under which results-dir is resolved')
    p.add_argument('--results-dir', type=str, default='results_robustness_benchmark')
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = list(args.config) or [
        'backend/harness/scenarios_robustness_foodmart.yaml',
        'backend/harness/scenarios_robustness_adventureworks.yaml',
    ]
    results_dir = args.results_dir
    if args.run_root and not os.path.isabs(results_dir):
        results_dir = os.path.join(args.run_root, results_dir)
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, 'figures'))

    all_sessions: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    for config_path in config_paths:
        cfg = load_yaml(config_path)
        objective_id = str(cfg['objective_id'])
        dw_id = str(cfg.get('dw_id') or 'UNKNOWN')
        scenarios = list(cfg.get('scenarios') or [])
        if not scenarios:
            raise RuntimeError(f'No scenarios found in config: {config_path}')
        matched_random_allow_prob = _estimate_mcad_allow_probability(objective_id, scenarios, args.repeats, args.seed)
        tag = _slug(f'{dw_id}_{objective_id}_robustness')
        out_dir = os.path.join(results_dir, tag)
        ensure_dir(out_dir)
        for rep in range(args.repeats):
            for scen_idx, scenario in enumerate(scenarios):
                for policy in POLICIES:
                    session_row, local_steps = play_policy_on_scenario(
                        policy=policy,
                        scenario=scenario,
                        objective_id=objective_id,
                        seed=args.seed + rep * 1000 + scen_idx,
                        matched_random_allow_prob=matched_random_allow_prob,
                    )
                    session_row['repeat_id'] = rep
                    session_row['objective_id'] = objective_id
                    session_row['dw_id'] = dw_id
                    session_row['config_path'] = config_path
                    all_sessions.append(session_row)
                    for row in local_steps:
                        row['repeat_id'] = rep
                        row['objective_id'] = objective_id
                        row['dw_id'] = dw_id
                        row['config_path'] = config_path
                        all_steps.append(row)
        metas.append({
            'objective_id': objective_id,
            'dw_id': dw_id,
            'n_scenarios': len(scenarios),
            'repeats': args.repeats,
            'matched_random_allow_prob': matched_random_allow_prob,
            'config_path': config_path,
        })

    aggregate_summary = summarize_by_policy(all_sessions)
    by_type_rows = _summarize_by_scenario_type(all_sessions)
    explain_rows, explain_summary_mixed = _collect_mcad_explainability(config_paths)
    explain_type_rows = [r for r in explain_summary_mixed if 'scenario_type' in r]
    reason_rows = [r for r in explain_summary_mixed if 'primary_reason' in r]

    write_csv(os.path.join(results_dir, 'robustness_policy_session_metrics.csv'), all_sessions)
    write_csv(os.path.join(results_dir, 'robustness_policy_step_metrics.csv'), all_steps)
    write_csv(os.path.join(results_dir, 'robustness_policy_summary.csv'), aggregate_summary)
    write_csv(os.path.join(results_dir, 'robustness_summary_by_scenario_type_and_policy.csv'), by_type_rows)
    write_csv(os.path.join(results_dir, 'mcad_block_explainability_steps.csv'), explain_rows)
    write_csv(os.path.join(results_dir, 'mcad_block_explainability_summary.csv'), explain_type_rows)
    write_csv(os.path.join(results_dir, 'mcad_block_reason_distribution.csv'), reason_rows)
    Path(os.path.join(results_dir, 'robustness_meta.json')).write_text(json.dumps(metas, indent=2, ensure_ascii=False), encoding='utf-8')

    from backend.harness.run_baselines_and_ablations import plot_metric
    plot_metric(aggregate_summary, 'mean_false_allow_rate', 'Robustness workload — false allow rate by policy', os.path.join(results_dir, 'figures', 'robustness_false_allow_by_policy.png'))
    plot_metric(aggregate_summary, 'mean_false_block_rate', 'Robustness workload — false block rate by policy', os.path.join(results_dir, 'figures', 'robustness_false_block_by_policy.png'))
    plot_metric(aggregate_summary, 'mean_auc_phi', 'Robustness workload — AUC phi(t) by policy', os.path.join(results_dir, 'figures', 'robustness_auc_phi_by_policy.png'))
    _plot_grouped_by_type(by_type_rows, 'mean_false_allow_rate', os.path.join(results_dir, 'figures', 'robustness_false_allow_by_type.png'), 'False allow rate by scenario type and policy', policies=['mcad', 'baseline_naive', 'baseline_measure_overlap', 'baseline_random_matched'])
    _plot_reason_distribution(reason_rows, os.path.join(results_dir, 'figures', 'robustness_block_reason_distribution_mcad.png'))
    _plot_explainable_block_rate(explain_type_rows, os.path.join(results_dir, 'figures', 'robustness_explainable_block_rate_by_type.png'))
    _build_robustness_report(aggregate_summary, by_type_rows, explain_summary_mixed, os.path.join(results_dir, 'robustness_report.md'), config_paths)

    print(f'[MCAD/ROBUSTNESS] done -> {results_dir}')


if __name__ == '__main__':
    main()
