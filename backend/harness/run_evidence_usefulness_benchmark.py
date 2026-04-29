from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mcad.engine000 import (
    bootstrap_session_from_persisted_evidence,
    evaluate_with_objective_and_session,
    get_ckg,
    get_evidence_store,
    reset_runtime_state,
)
from mcad.models000 import EvaluateWithObjectiveAndSessionRequest
from mcad.session_store0 import SESSION_STORE


def _foodmart_queries() -> Dict[str, Dict[str, Any]]:
    return {
        'c1': {'query_spec': {'cube': 'Sales', 'measures': ['Margin%'], 'aggregators': ['AVG'], 'units': ['PERCENT'], 'group_by': ['Time.Month', 'Product.Category'], 'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North', 'Time.Year': '1998'}, 'time_members': ['1998'], 'window_start': '1998-01-01', 'window_end': '1998-12-31', 'language': 'mdx', 'execution_result_excerpt': {'rows_retained': 12, 'kpi': 'Margin%'}}},
        'c2': {'query_spec': {'cube': 'Sales', 'measures': ['StockoutRate'], 'aggregators': ['AVG'], 'units': ['PERCENT'], 'group_by': ['Store.Store', 'Product.Category'], 'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North', 'Time.Year': '1998'}, 'time_members': ['1998'], 'window_start': '1998-01-01', 'window_end': '1998-12-31', 'language': 'mdx', 'execution_result_excerpt': {'rows_retained': 7, 'kpi': 'StockoutRate'}}},
        'c3': {'query_spec': {'cube': 'Sales', 'measures': ['Store Sales'], 'aggregators': ['SUM'], 'units': ['CURRENCY'], 'group_by': ['Time.Month', 'Product.Category'], 'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North'}, 'time_members': ['1997', '1998'], 'window_start': '1997-01-01', 'window_end': '1998-12-31', 'language': 'mdx', 'execution_result_excerpt': {'rows_retained': 24, 'kpi': 'Store Sales'}}},
    }


def _aw_queries() -> Dict[str, Dict[str, Any]]:
    return {
        'aw_c1': {'query_spec': {'cube': 'Adventure Works Sales', 'measures': ['Gross Margin%'], 'aggregators': ['AVG'], 'units': ['PERCENT'], 'group_by': ['Date.Month', 'Product.Category'], 'slicers': {'Product.Category': 'Bikes', 'SalesTerritory.Region': 'Europe', 'Date.Year': '2013'}, 'time_members': ['2013'], 'window_start': '2013-01-01', 'window_end': '2013-12-31', 'language': 'sql', 'execution_result_excerpt': {'rows_retained': 12, 'kpi': 'Gross Margin%'}}},
        'aw_c2': {'query_spec': {'cube': 'Adventure Works Sales', 'measures': ['ReturnRate'], 'aggregators': ['AVG'], 'units': ['PERCENT'], 'group_by': ['Reseller.Reseller', 'Product.Category'], 'slicers': {'Product.Category': 'Bikes', 'SalesTerritory.Region': 'Europe', 'Date.Year': '2013'}, 'time_members': ['2013'], 'window_start': '2013-01-01', 'window_end': '2013-12-31', 'language': 'sql', 'execution_result_excerpt': {'rows_retained': 9, 'kpi': 'ReturnRate'}}},
        'aw_c3': {'query_spec': {'cube': 'Adventure Works Sales', 'measures': ['Sales Amount'], 'aggregators': ['SUM'], 'units': ['CURRENCY'], 'group_by': ['Date.Month', 'Product.Category'], 'slicers': {'Product.Category': 'Bikes', 'SalesTerritory.Region': 'Europe'}, 'time_members': ['2012', '2013'], 'window_start': '2012-01-01', 'window_end': '2013-12-31', 'language': 'sql', 'execution_result_excerpt': {'rows_retained': 24, 'kpi': 'Sales Amount'}}},
    }


def _run_sequence(session_id: str, objective_id: str, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    phis: List[float] = []
    steps_to_full = None
    final_resp = None
    for qp in queries:
        resp = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=session_id, objective_id=objective_id, qp=qp))
        phis.append(float(resp.phi_leq_t or 0.0))
        if steps_to_full is None and float(resp.phi_leq_t or 0.0) >= 0.999999:
            steps_to_full = int(resp.step_index or 0)
        final_resp = resp
    return {'steps': len(queries), 'steps_to_full': int(steps_to_full) if steps_to_full is not None else None, 'auc_phi': round(mean(phis), 6) if phis else 0.0, 'final_phi': round(float(phis[-1] if phis else 0.0), 6), 'final_covered_constraints': list(getattr(final_resp, 'covered_constraints', []) or [])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run evidence usefulness benchmark into a configurable output root.')
    p.add_argument('--run-root', default='', help='Optional run root under which results-dir is resolved')
    p.add_argument('--results-dir', default='results_phase6_evidence', help='Output directory for evidence artefacts')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    if args.run_root and not out_dir.is_absolute():
        out_dir = Path(args.run_root) / out_dir
    fig_dir = out_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MCAD_RESULTS_DIR'] = str(out_dir)
    os.environ.setdefault('MCAD_TMP_DIR', str(out_dir / 'tmp'))
    reset_runtime_state()
    ckg = get_ckg()
    store = get_evidence_store()

    setups = [('OHF_NORD', _foodmart_queries(), ['c1', 'c2'], ['c3']), ('OAW_BIKES_EUROPE', _aw_queries(), ['aw_c1', 'aw_c2'], ['aw_c3'])]
    summary_rows: List[Dict[str, Any]] = []
    for objective_id, qmap, seed_keys, followup_keys in setups:
        seed_session = SESSION_STORE.ensure_session(session_id=f'SEED_{objective_id}', objective_id=objective_id, dw_id='benchmark')
        _run_sequence(seed_session.session_id, objective_id, [qmap[k] for k in seed_keys])
        with_boot = SESSION_STORE.ensure_session(session_id=f'FOLLOWUP_BOOT_{objective_id}', objective_id=objective_id, dw_id='benchmark')
        boot_payload = bootstrap_session_from_persisted_evidence(with_boot.session_id, objective_id)
        with_boot_metrics = _run_sequence(with_boot.session_id, objective_id, [qmap[k] for k in followup_keys])
        with_boot_metrics['bootstrap_phi'] = boot_payload['phi_leq_t']
        no_boot = SESSION_STORE.ensure_session(session_id=f'FOLLOWUP_NOBOOT_{objective_id}', objective_id=objective_id, dw_id='benchmark')
        no_boot_metrics = _run_sequence(no_boot.session_id, objective_id, [qmap[k] for k in seed_keys + followup_keys])
        summary_rows += [
            {'objective_id': objective_id, 'mode': 'no_bootstrap', 'steps_to_full': no_boot_metrics['steps_to_full'] or 999, 'auc_phi': no_boot_metrics['auc_phi'], 'final_phi': no_boot_metrics['final_phi'], 'bootstrap_phi': 0.0},
            {'objective_id': objective_id, 'mode': 'bootstrap', 'steps_to_full': with_boot_metrics['steps_to_full'] or 999, 'auc_phi': with_boot_metrics['auc_phi'], 'final_phi': with_boot_metrics['final_phi'], 'bootstrap_phi': with_boot_metrics.get('bootstrap_phi', 0.0)},
        ]

    objective_totals = {str(oid): int(len((obj.get('constraints') or {}).keys())) for oid, obj in (ckg.objectives or {}).items()}
    usefulness = store.usefulness_report(objective_constraint_totals=objective_totals)
    (out_dir / 'evidence_usefulness_report.json').write_text(json.dumps(usefulness, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'bootstrap_benefit_summary.json').write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding='utf-8')

    objectives = [r['objective_id'] for r in summary_rows if r['mode'] == 'no_bootstrap']
    no_steps = [next(r['steps_to_full'] for r in summary_rows if r['objective_id'] == oid and r['mode'] == 'no_bootstrap') for oid in objectives]
    boot_steps = [next(r['steps_to_full'] for r in summary_rows if r['objective_id'] == oid and r['mode'] == 'bootstrap') for oid in objectives]
    x = range(len(objectives)); width = 0.35
    plt.figure(figsize=(7.0, 3.5))
    plt.bar([i - width/2 for i in x], no_steps, width=width, label='No bootstrap')
    plt.bar([i + width/2 for i in x], boot_steps, width=width, label='With evidence bootstrap')
    plt.xticks(list(x), objectives, rotation=10); plt.ylabel('Steps to full coverage'); plt.legend(); plt.tight_layout(); plt.savefig(fig_dir / 'evidence_bootstrap_steps_to_full.png', dpi=200); plt.close()

    plt.figure(figsize=(7.0, 3.5))
    metrics = ['useful_evidence_ratio', 'mean_retained_ratio', 'execution_excerpt_coverage_ratio', 'mean_usefulness_score']
    vals = [float(usefulness.get(m) or 0.0) for m in metrics]
    plt.bar(range(len(metrics)), vals)
    plt.xticks(range(len(metrics)), metrics, rotation=20, ha='right'); plt.ylim(0, 1.05); plt.tight_layout(); plt.savefig(fig_dir / 'evidence_usefulness_summary.png', dpi=200); plt.close()

    lines = ['# MCAD Phase 6 — Evidence / provenance / usefulness report', '', f"- n_records = {usefulness['n_records']}", f"- useful_evidence_ratio = {usefulness['useful_evidence_ratio']}", f"- mean_retained_ratio = {usefulness['mean_retained_ratio']}", f"- mean_usefulness_score = {usefulness['mean_usefulness_score']}", '', '## Bootstrap benefit']
    for row in summary_rows:
        lines.append(f"- {row['objective_id']} / {row['mode']}: steps_to_full={row['steps_to_full']}, auc_phi={row['auc_phi']}, final_phi={row['final_phi']}, bootstrap_phi={row['bootstrap_phi']}")
    (out_dir / 'phase6_evidence_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[MCAD/EVIDENCE] done -> {out_dir}')


if __name__ == '__main__':
    main()
