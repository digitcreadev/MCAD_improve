from __future__ import annotations

"""Run offline policy benchmarks for MCAD baselines and ablations.

This script separates two concerns:
1) the *decision policy* (ALLOW/BLOCK)
2) the *oracle contribution* committed when a query is executed

That separation is essential for a fair comparison:
- MCAD, heuristics, random, and ablations decide whether a query is executed.
- Coverage / AUC / time-to-coverage are computed from the scenario oracle
  (oracle_allow + oracle_ceval), not from each policy's internal logic.

Outputs:
  - policy_session_metrics.csv
  - policy_step_metrics.csv
  - policy_summary.csv
  - policy_latency_summary.csv
  - policy_benchmark_report.md
  - figures/*.png
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.ckg.ckg_updater import CKGGraph


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    idx = p * (len(s) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(s[lo])
    w = idx - lo
    return float((1.0 - w) * s[lo] + w * s[hi])


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _normalize_qp(step: Dict[str, Any], objective_id: str) -> Dict[str, Any]:
    qp = dict(step.get('qp') or {})
    qspec = dict(qp.get('query_spec') or qp)
    qp['query_spec'] = qspec
    qp.setdefault('objective_id', objective_id)
    return qp


def _objective_measure_set(ckg: CKGGraph, objective_id: str) -> Set[str]:
    obj = ckg.objectives.get(objective_id) or {}
    measures: Set[str] = set()
    for cinfo in (obj.get('constraints') or {}).values():
        for nvid in cinfo.get('virtual_nodes') or []:
            nvnode = f'nv::{nvid}'
            if ckg.G.has_node(nvnode):
                m = str(ckg.G.nodes[nvnode].get('measure') or '').strip()
                if m:
                    measures.add(m)
    return measures


def _probe_mcad(ckg: CKGGraph, objective_id: str, step_idx: int, qp: Dict[str, Any], ignore_sat: bool = False, ceval_any_intersection: bool = False) -> Dict[str, Any]:
    session_id = '_probe'
    qpnode = ckg.add_qp_node(session_id=session_id, step_idx=step_idx, qp=qp)
    sat_ok, clauses = ckg.sat(qp=qp, qp_node=qpnode)
    real_nv_ids: Set[str] = set()
    ceval_ids: Set[str] = set()
    if sat_ok or ignore_sat:
        real_nv_ids = ckg.real(objective_id=objective_id, qp_node=qpnode)
        if ceval_any_intersection:
            ceval_ids = set()
            if real_nv_ids:
                obj = ckg.objectives.get(objective_id) or {}
                for cid, cinfo in (obj.get('constraints') or {}).items():
                    needed = set(cinfo.get('virtual_nodes') or [])
                    if needed & set(real_nv_ids):
                        ceval_ids.add(cid)
        else:
            ceval_ids = ckg.ceval(objective_id=objective_id, real_nv_ids=real_nv_ids)
    phi, phi_w = ckg.phi(objective_id=objective_id, ceval_ids=ceval_ids)
    return {
        'sat': bool(sat_ok),
        'clauses': clauses,
        'real_nv_ids': sorted(real_nv_ids),
        'ceval_ids': sorted(ceval_ids),
        'phi': float(phi),
        'phi_weighted': float(phi_w),
    }


def policy_decision(policy: str, ckg: CKGGraph, objective_id: str, qp: Dict[str, Any], rng: random.Random, matched_random_allow_prob: float, step_idx: int) -> Dict[str, Any]:
    qspec = qp.get('query_spec') or qp
    t0 = time.perf_counter()
    decision: Dict[str, Any]

    if policy == 'mcad':
        probe = _probe_mcad(ckg, objective_id, step_idx, qp)
        allow = bool(probe['sat'] and probe['ceval_ids'])
        decision = {'allow': allow, 'mode': 'mcad', **probe}
    elif policy == 'baseline_naive':
        decision = {'allow': True, 'mode': 'always_allow', 'sat': None, 'real_nv_ids': [], 'ceval_ids': [], 'phi': 0.0, 'phi_weighted': 0.0}
    elif policy == 'baseline_measure_overlap':
        objective_measures = _objective_measure_set(ckg, objective_id)
        qp_measures = set(str(m) for m in (qspec.get('measures') or []))
        allow = bool(objective_measures & qp_measures)
        decision = {'allow': allow, 'mode': 'measure_overlap', 'sat': None, 'real_nv_ids': [], 'ceval_ids': [], 'phi': 0.0, 'phi_weighted': 0.0, 'measure_overlap': sorted(objective_measures & qp_measures)}
    elif policy == 'baseline_random_matched':
        allow = rng.random() < matched_random_allow_prob
        decision = {'allow': allow, 'mode': 'random_matched', 'sat': None, 'real_nv_ids': [], 'ceval_ids': [], 'phi': 0.0, 'phi_weighted': 0.0}
    elif policy == 'ablation_no_sat':
        probe = _probe_mcad(ckg, objective_id, step_idx, qp, ignore_sat=True)
        allow = bool(probe['ceval_ids'])
        decision = {'allow': allow, 'mode': 'ignore_sat', **probe}
    elif policy == 'ablation_ceval_any_intersection':
        probe = _probe_mcad(ckg, objective_id, step_idx, qp, ceval_any_intersection=True)
        allow = bool(probe['sat'] and probe['ceval_ids'])
        decision = {'allow': allow, 'mode': 'ceval_any_intersection', **probe}
    elif policy == 'ablation_no_real':
        objective_measures = _objective_measure_set(ckg, objective_id)
        qp_measures = set(str(m) for m in (qspec.get('measures') or []))
        probe = _probe_mcad(ckg, objective_id, step_idx, qp)
        allow = bool(probe['sat'] and (objective_measures & qp_measures))
        decision = {'allow': allow, 'mode': 'ignore_real', **probe, 'measure_overlap': sorted(objective_measures & qp_measures)}
    else:
        raise ValueError(f'Unknown policy: {policy}')

    latency_ms = (time.perf_counter() - t0) * 1000.0
    decision['latency_ms'] = round(latency_ms, 4)
    return decision


def play_policy_on_scenario(policy: str, scenario: Dict[str, Any], objective_id: str, seed: int, matched_random_allow_prob: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    ckg = CKGGraph(output_dir=os.environ.get('MCAD_TMP_DIR', 'results_policy_tmp'))
    obj = ckg.objectives.get(objective_id) or {}
    n_constraints = len(obj.get('constraints') or {})

    coverage: Set[str] = set()
    step_rows: List[Dict[str, Any]] = []
    latency_all: List[float] = []
    latency_cold: List[float] = []
    latency_warm: List[float] = []
    false_allow = 0
    false_block = 0
    executed_non_contrib = 0
    sat_fail_count = 0
    real_empty_count = 0
    ceval_empty_count = 0
    phi_curve: List[float] = []

    steps = list(scenario.get('steps') or [])
    for idx, step in enumerate(steps, start=1):
        qp = _normalize_qp(step, objective_id)
        oracle_allow = bool(step.get('oracle_allow', False))
        oracle_ceval = list(step.get('oracle_ceval') or [])
        decision = policy_decision(policy, ckg, objective_id, qp, rng=rng, matched_random_allow_prob=matched_random_allow_prob, step_idx=idx)
        allow = bool(decision['allow'])
        if allow and oracle_ceval:
            coverage |= set(oracle_ceval)
        if allow and not oracle_allow:
            false_allow += 1
        if (not allow) and oracle_allow:
            false_block += 1
        if allow and not oracle_ceval:
            executed_non_contrib += 1
        if decision.get('sat') is False:
            sat_fail_count += 1
        if decision.get('sat') is True and not decision.get('real_nv_ids'):
            real_empty_count += 1
        if decision.get('sat') is True and decision.get('real_nv_ids') and not decision.get('ceval_ids'):
            ceval_empty_count += 1

        phi_t = (len(coverage) / n_constraints) if n_constraints else 0.0
        phi_curve.append(phi_t)
        latency = float(decision['latency_ms'])
        latency_all.append(latency)
        if idx == 1:
            latency_cold.append(latency)
        else:
            latency_warm.append(latency)

        step_rows.append({
            'policy': policy,
            'scenario_id': scenario.get('id'),
            'scenario_type': scenario.get('type'),
            'step_idx': idx,
            'step_name': step.get('name'),
            'oracle_allow': oracle_allow,
            'oracle_ceval': ','.join(oracle_ceval),
            'allow': allow,
            'false_allow': int(allow and not oracle_allow),
            'false_block': int((not allow) and oracle_allow),
            'executed_non_contrib': int(allow and not oracle_ceval),
            'phi_leq_t': round(phi_t, 6),
            'latency_ms': round(latency, 4),
            'sat': decision.get('sat'),
            'real_size': len(decision.get('real_nv_ids') or []),
            'ceval_size': len(decision.get('ceval_ids') or []),
        })

    def first_time_to(threshold: float) -> Optional[int]:
        for i, val in enumerate(phi_curve, start=1):
            if val >= threshold:
                return i
        return None

    session_row = {
        'policy': policy,
        'scenario_id': scenario.get('id'),
        'scenario_label': scenario.get('label'),
        'scenario_type': scenario.get('type'),
        'n_steps': len(steps),
        'n_executed': sum(1 for r in step_rows if r['allow']),
        'phi_final': round(phi_curve[-1] if phi_curve else 0.0, 6),
        'auc_phi': round(mean(phi_curve), 6),
        'time_to_0_8': first_time_to(0.8) or '',
        'time_to_0_9': first_time_to(0.9) or '',
        'false_allow': false_allow,
        'false_block': false_block,
        'executed_non_contrib': executed_non_contrib,
        'false_allow_rate': round(false_allow / len(steps), 6) if steps else 0.0,
        'false_block_rate': round(false_block / len(steps), 6) if steps else 0.0,
        'non_contrib_exec_rate': round(executed_non_contrib / max(1, sum(1 for r in step_rows if r['allow'])), 6),
        'sat_fail_rate': round(sat_fail_count / len(steps), 6) if steps else 0.0,
        'real_empty_rate': round(real_empty_count / len(steps), 6) if steps else 0.0,
        'ceval_empty_rate': round(ceval_empty_count / len(steps), 6) if steps else 0.0,
        'latency_mean_ms': round(mean(latency_all), 6),
        'latency_p50_ms': round(percentile(latency_all, 0.50), 6),
        'latency_p95_ms': round(percentile(latency_all, 0.95), 6),
        'latency_p99_ms': round(percentile(latency_all, 0.99), 6),
        'latency_cold_p50_ms': round(percentile(latency_cold, 0.50), 6),
        'latency_warm_p50_ms': round(percentile(latency_warm, 0.50), 6),
    }
    return session_row, step_rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path) or '.')
    headers = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers, delimiter=';')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize_by_policy(session_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        grouped[str(row['policy'])].append(row)

    out: List[Dict[str, Any]] = []
    for policy, rows in grouped.items():
        def col(name: str) -> List[float]:
            vals: List[float] = []
            for r in rows:
                v = r.get(name)
                if v == '' or v is None:
                    continue
                vals.append(float(v))
            return vals
        out.append({
            'policy': policy,
            'n_sessions': len(rows),
            'mean_phi_final': round(mean(col('phi_final')), 6),
            'mean_auc_phi': round(mean(col('auc_phi')), 6),
            'mean_time_to_0_8': round(mean(col('time_to_0_8')), 6) if col('time_to_0_8') else '',
            'mean_time_to_0_9': round(mean(col('time_to_0_9')), 6) if col('time_to_0_9') else '',
            'mean_false_allow_rate': round(mean(col('false_allow_rate')), 6),
            'mean_false_block_rate': round(mean(col('false_block_rate')), 6),
            'mean_non_contrib_exec_rate': round(mean(col('non_contrib_exec_rate')), 6),
            'mean_sat_fail_rate': round(mean(col('sat_fail_rate')), 6),
            'mean_real_empty_rate': round(mean(col('real_empty_rate')), 6),
            'mean_ceval_empty_rate': round(mean(col('ceval_empty_rate')), 6),
            'latency_p50_ms': round(percentile(col('latency_p50_ms'), 0.50), 6),
            'latency_p95_ms': round(percentile(col('latency_p95_ms'), 0.95), 6),
            'latency_p99_ms': round(percentile(col('latency_p99_ms'), 0.99), 6),
            'latency_cold_p50_ms': round(percentile(col('latency_cold_p50_ms'), 0.50), 6),
            'latency_warm_p50_ms': round(percentile(col('latency_warm_p50_ms'), 0.50), 6),
        })
    return sorted(out, key=lambda r: r['policy'])


def plot_metric(summary_rows: List[Dict[str, Any]], key: str, title: str, out_png: str) -> None:
    ensure_dir(os.path.dirname(out_png) or '.')
    labels = [r['policy'] for r in summary_rows]
    vals = [float(r.get(key) or 0.0) for r in summary_rows]
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, vals)
    plt.xticks(rotation=25, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def build_report(summary_rows: List[Dict[str, Any]], out_path: str, config_path: str) -> None:
    by_policy = {r['policy']: r for r in summary_rows}
    lines = []
    lines.append('# MCAD policy benchmark report')
    lines.append('')
    lines.append(f'- Scenario config: `{config_path}`')
    lines.append(f'- Policies: {", ".join(sorted(by_policy.keys()))}')
    lines.append('')
    if 'mcad' in by_policy:
        mcad = by_policy['mcad']
        lines.append('## Key takeaways')
        lines.append('')
        lines.append(f"- MCAD mean final coverage: **{mcad['mean_phi_final']:.3f}**")
        lines.append(f"- MCAD mean AUC φ(t): **{mcad['mean_auc_phi']:.3f}**")
        lines.append(f"- MCAD false allow rate: **{mcad['mean_false_allow_rate']:.3f}**")
        lines.append(f"- MCAD false block rate: **{mcad['mean_false_block_rate']:.3f}**")
        lines.append(f"- MCAD latency p50/p95/p99 (ms): **{mcad['latency_p50_ms']:.3f} / {mcad['latency_p95_ms']:.3f} / {mcad['latency_p99_ms']:.3f}**")
        lines.append('')
    lines.append('## Policy summary')
    lines.append('')
    headers = ['policy','mean_phi_final','mean_auc_phi','mean_false_allow_rate','mean_false_block_rate','mean_non_contrib_exec_rate','latency_p50_ms','latency_p95_ms','latency_p99_ms']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in summary_rows:
        vals = [str(row.get(h, '')) for h in headers]
        lines.append('|' + '|'.join(vals) + '|')
    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run MCAD baselines and ablations on an oracle-backed scenario set.')
    p.add_argument('--config', type=str, default='backend/harness/scenarios.yaml')
    p.add_argument('--run-root', type=str, default='', help='Optional run root under which results-dir is resolved')
    p.add_argument('--results-dir', type=str, default='results_policy_benchmark')
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if args.run_root and not os.path.isabs(results_dir):
        results_dir = os.path.join(args.run_root, results_dir)
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, 'figures'))
    cfg = load_yaml(args.config)
    objective_id = str(cfg['objective_id'])
    scenarios = list(cfg.get('scenarios') or [])
    if not scenarios:
        raise RuntimeError('No scenarios found in config.')

    policies = [
        'mcad',
        'baseline_naive',
        'baseline_measure_overlap',
        'ablation_no_sat',
        'ablation_no_real',
        'ablation_ceval_any_intersection',
    ]

    # First pass: estimate MCAD allow probability over the full campaign.
    mcad_allows = 0
    total_steps = 0
    for rep in range(args.repeats):
        for scenario in scenarios:
            ckg = CKGGraph(output_dir=os.environ.get('MCAD_TMP_DIR', 'results_policy_tmp'))
            for idx, step in enumerate(scenario.get('steps') or [], start=1):
                qp = _normalize_qp(step, objective_id)
                probe = policy_decision('mcad', ckg, objective_id, qp, rng=random.Random(args.seed + rep), matched_random_allow_prob=0.0, step_idx=idx)
                mcad_allows += int(bool(probe['allow']))
                total_steps += 1
    matched_random_allow_prob = (mcad_allows / total_steps) if total_steps else 0.5

    session_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for rep in range(args.repeats):
        for scen_idx, scenario in enumerate(scenarios):
            for policy in policies + ['baseline_random_matched']:
                session_row, local_steps = play_policy_on_scenario(
                    policy=policy,
                    scenario=scenario,
                    objective_id=objective_id,
                    seed=args.seed + rep * 1000 + scen_idx,
                    matched_random_allow_prob=matched_random_allow_prob,
                )
                session_row['repeat_id'] = rep
                session_rows.append(session_row)
                for row in local_steps:
                    row['repeat_id'] = rep
                    step_rows.append(row)

    summary_rows = summarize_by_policy(session_rows)

    write_csv(os.path.join(results_dir, 'policy_session_metrics.csv'), session_rows)
    write_csv(os.path.join(results_dir, 'policy_step_metrics.csv'), step_rows)
    write_csv(os.path.join(results_dir, 'policy_summary.csv'), summary_rows)

    latency_rows = []
    for row in summary_rows:
        latency_rows.append({
            'policy': row['policy'],
            'latency_p50_ms': row['latency_p50_ms'],
            'latency_p95_ms': row['latency_p95_ms'],
            'latency_p99_ms': row['latency_p99_ms'],
            'latency_cold_p50_ms': row['latency_cold_p50_ms'],
            'latency_warm_p50_ms': row['latency_warm_p50_ms'],
        })
    write_csv(os.path.join(results_dir, 'policy_latency_summary.csv'), latency_rows)

    plot_metric(summary_rows, 'mean_phi_final', 'Mean final coverage by policy', os.path.join(results_dir, 'figures', 'mean_phi_final_by_policy.png'))
    plot_metric(summary_rows, 'mean_auc_phi', 'Mean AUC φ(t) by policy', os.path.join(results_dir, 'figures', 'mean_auc_by_policy.png'))
    plot_metric(summary_rows, 'mean_false_allow_rate', 'Mean false allow rate by policy', os.path.join(results_dir, 'figures', 'false_allow_by_policy.png'))
    plot_metric(summary_rows, 'mean_false_block_rate', 'Mean false block rate by policy', os.path.join(results_dir, 'figures', 'false_block_by_policy.png'))

    build_report(summary_rows, os.path.join(results_dir, 'policy_benchmark_report.md'), args.config)

    meta = {
        'objective_id': objective_id,
        'n_scenarios': len(scenarios),
        'repeats': args.repeats,
        'matched_random_allow_prob': matched_random_allow_prob,
    }
    Path(os.path.join(results_dir, 'policy_benchmark_meta.json')).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[MCAD/BENCH] done -> {results_dir}")


if __name__ == '__main__':
    main()
