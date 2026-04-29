from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.harness.run_baselines_and_ablations import (  # noqa: E402
    build_report,
    ensure_dir,
    load_yaml,
    play_policy_on_scenario,
    plot_metric,
    policy_decision,
    summarize_by_policy,
    write_csv,
    _normalize_qp,
)
from backend.ckg.ckg_updater import CKGGraph  # noqa: E402


POLICIES = [
    'mcad',
    'baseline_naive',
    'baseline_measure_overlap',
    'ablation_no_sat',
    'ablation_no_real',
    'ablation_ceval_any_intersection',
    'baseline_random_matched',
]


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


def _run_single_config(config_path: str, results_dir: str, repeats: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, 'figures'))
    cfg = load_yaml(config_path)
    objective_id = str(cfg['objective_id'])
    dw_id = str(cfg.get('dw_id') or 'UNKNOWN')
    scenarios = list(cfg.get('scenarios') or [])
    if not scenarios:
        raise RuntimeError(f'No scenarios found in config: {config_path}')

    matched_random_allow_prob = _estimate_mcad_allow_probability(objective_id, scenarios, repeats, seed)

    session_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for rep in range(repeats):
        for scen_idx, scenario in enumerate(scenarios):
            for policy in POLICIES:
                session_row, local_steps = play_policy_on_scenario(
                    policy=policy,
                    scenario=scenario,
                    objective_id=objective_id,
                    seed=seed + rep * 1000 + scen_idx,
                    matched_random_allow_prob=matched_random_allow_prob,
                )
                session_row['repeat_id'] = rep
                session_row['objective_id'] = objective_id
                session_row['dw_id'] = dw_id
                session_row['config_path'] = config_path
                session_rows.append(session_row)
                for row in local_steps:
                    row['repeat_id'] = rep
                    row['objective_id'] = objective_id
                    row['dw_id'] = dw_id
                    row['config_path'] = config_path
                    step_rows.append(row)

    summary_rows = summarize_by_policy(session_rows)
    for row in summary_rows:
        row['objective_id'] = objective_id
        row['dw_id'] = dw_id
        row['config_path'] = config_path

    write_csv(os.path.join(results_dir, 'policy_session_metrics.csv'), session_rows)
    write_csv(os.path.join(results_dir, 'policy_step_metrics.csv'), step_rows)
    write_csv(os.path.join(results_dir, 'policy_summary.csv'), summary_rows)
    build_report(summary_rows, os.path.join(results_dir, 'policy_benchmark_report.md'), config_path)
    plot_metric(summary_rows, 'mean_phi_final', f'Mean final coverage by policy — {dw_id}', os.path.join(results_dir, 'figures', 'mean_phi_final_by_policy.png'))
    plot_metric(summary_rows, 'mean_false_allow_rate', f'False allow rate by policy — {dw_id}', os.path.join(results_dir, 'figures', 'false_allow_by_policy.png'))

    meta = {
        'objective_id': objective_id,
        'dw_id': dw_id,
        'n_scenarios': len(scenarios),
        'repeats': repeats,
        'matched_random_allow_prob': matched_random_allow_prob,
        'config_path': config_path,
    }
    Path(os.path.join(results_dir, 'policy_benchmark_meta.json')).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return session_rows, step_rows, summary_rows, meta


def _summarize_by_dataset_and_policy(session_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[(str(row.get('dw_id') or ''), str(row.get('policy') or ''))].append(row)

    out: List[Dict[str, Any]] = []
    for (dw_id, policy), rows in sorted(grouped.items()):
        def mean_of(key: str) -> float:
            vals = [float(r.get(key) or 0.0) for r in rows if r.get(key) not in ('', None)]
            return round(sum(vals) / len(vals), 6) if vals else 0.0
        out.append({
            'dw_id': dw_id,
            'policy': policy,
            'n_sessions': len(rows),
            'mean_phi_final': mean_of('phi_final'),
            'mean_auc_phi': mean_of('auc_phi'),
            'mean_false_allow_rate': mean_of('false_allow_rate'),
            'mean_false_block_rate': mean_of('false_block_rate'),
            'mean_non_contrib_exec_rate': mean_of('non_contrib_exec_rate'),
            'mean_sat_fail_rate': mean_of('sat_fail_rate'),
            'mean_real_empty_rate': mean_of('real_empty_rate'),
            'mean_ceval_empty_rate': mean_of('ceval_empty_rate'),
        })
    return out


def _build_multidataset_report(aggregate_summary: List[Dict[str, Any]], by_dataset_rows: List[Dict[str, Any]], out_path: str, config_paths: List[str]) -> None:
    lines: List[str] = []
    lines.append('# MCAD multi-dataset benchmark report')
    lines.append('')
    lines.append('## Configurations')
    lines.append('')
    for p in config_paths:
        lines.append(f'- `{p}`')
    lines.append('')
    lines.append('## Aggregate summary across datasets')
    lines.append('')
    headers = ['policy', 'mean_phi_final', 'mean_auc_phi', 'mean_false_allow_rate', 'mean_false_block_rate', 'mean_non_contrib_exec_rate']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in aggregate_summary:
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    lines.append('')
    lines.append('## By dataset and policy')
    lines.append('')
    headers = ['dw_id', 'policy', 'mean_phi_final', 'mean_auc_phi', 'mean_false_allow_rate', 'mean_false_block_rate', 'mean_non_contrib_exec_rate']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in by_dataset_rows:
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run MCAD policy benchmark across multiple datasets/configs.')
    p.add_argument('--config', action='append', default=[], help='Repeatable path to a scenarios YAML config.')
    p.add_argument('--run-root', type=str, default='', help='Optional run root under which results-dir is resolved')
    p.add_argument('--results-dir', type=str, default='results_multidataset_benchmark')
    p.add_argument('--repeats', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = list(args.config) or ['backend/harness/scenarios.yaml', 'backend/harness/scenarios_adventureworks.yaml']
    results_dir = args.results_dir
    if args.run_root and not os.path.isabs(results_dir):
        results_dir = os.path.join(args.run_root, results_dir)
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, 'figures'))

    all_sessions: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    for config_path in config_paths:
        cfg = load_yaml(config_path)
        dw_id = str(cfg.get('dw_id') or 'UNKNOWN')
        objective_id = str(cfg.get('objective_id') or 'UNKNOWN')
        tag = _slug(f'{dw_id}_{objective_id}')
        out_dir = os.path.join(results_dir, tag)
        sessions, steps, summaries, meta = _run_single_config(config_path, out_dir, args.repeats, args.seed)
        all_sessions.extend(sessions)
        all_steps.extend(steps)
        all_summaries.extend(summaries)
        metas.append(meta)

    aggregate_summary = summarize_by_policy(all_sessions)
    by_dataset_rows = _summarize_by_dataset_and_policy(all_sessions)

    write_csv(os.path.join(results_dir, 'aggregate_policy_session_metrics.csv'), all_sessions)
    write_csv(os.path.join(results_dir, 'aggregate_policy_step_metrics.csv'), all_steps)
    write_csv(os.path.join(results_dir, 'aggregate_policy_summary.csv'), aggregate_summary)
    write_csv(os.path.join(results_dir, 'summary_by_dataset_and_policy.csv'), by_dataset_rows)
    Path(os.path.join(results_dir, 'multidataset_meta.json')).write_text(json.dumps(metas, indent=2, ensure_ascii=False), encoding='utf-8')
    plot_metric(aggregate_summary, 'mean_phi_final', 'Mean final coverage by policy — all datasets', os.path.join(results_dir, 'figures', 'aggregate_mean_phi_final_by_policy.png'))
    plot_metric(aggregate_summary, 'mean_false_allow_rate', 'False allow rate by policy — all datasets', os.path.join(results_dir, 'figures', 'aggregate_false_allow_by_policy.png'))
    _build_multidataset_report(aggregate_summary, by_dataset_rows, os.path.join(results_dir, 'multidataset_benchmark_report.md'), config_paths)
    print(f'[MCAD/MULTI] done -> {results_dir}')


if __name__ == '__main__':
    main()
