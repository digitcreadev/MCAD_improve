from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.ckg.ckg_updater import CKGGraph  # noqa: E402


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def write_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    ensure_dir(str(Path(path).parent))
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    headers: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


BASE_OBJECTIVES = ['OHF_NORD', 'OAW_BIKES_EUROPE']


def _base_key(objective_id: str) -> str:
    if str(objective_id).startswith('OHF_NORD'):
        return 'OHF_NORD'
    if str(objective_id).startswith('OAW_BIKES_EUROPE'):
        return 'OAW_BIKES_EUROPE'
    return str(objective_id)


def template_cycle(objective_id: str) -> List[Dict[str, Any]]:
    base = _base_key(objective_id)
    if base == 'OHF_NORD':
        return [
            {
                'query_spec': {
                    'cube': 'Sales', 'language': 'canonical',
                    'measures': ['Margin%'], 'aggregators': ['AVG'], 'units': ['PERCENT'],
                    'group_by': ['Time.Month', 'Product.Category'],
                    'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North', 'Time.Year': '1998'},
                    'window_start': '1998-01-01', 'window_end': '1998-12-31',
                }
            },
            {
                'query_spec': {
                    'cube': 'Sales', 'language': 'canonical',
                    'measures': ['StockoutRate'], 'aggregators': ['AVG'], 'units': ['PERCENT'],
                    'group_by': ['Store.Store', 'Product.Category'],
                    'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North', 'Time.Year': '1998'},
                    'window_start': '1998-01-01', 'window_end': '1998-12-31',
                }
            },
            {
                'query_spec': {
                    'cube': 'Sales', 'language': 'canonical',
                    'measures': ['Store Sales'], 'aggregators': ['SUM'], 'units': ['CURRENCY'],
                    'group_by': ['Time.Month', 'Product.Category'],
                    'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North'},
                    'time_members': ['1997', '1998'],
                    'window_start': '1997-01-01', 'window_end': '1998-12-31',
                }
            },
        ]
    return [
        {
            'query_spec': {
                'cube': 'Adventure Works Sales', 'language': 'canonical',
                'measures': ['Gross Margin%'], 'aggregators': ['AVG'], 'units': ['PERCENT'],
                'group_by': ['Date.Month', 'Product.Category'],
                'slicers': {'Product.Category': 'Bikes', 'SalesTerritory.Region': 'Europe', 'Date.Year': '2013'},
                'window_start': '2013-01-01', 'window_end': '2013-12-31',
            }
        },
        {
            'query_spec': {
                'cube': 'Adventure Works Sales', 'language': 'canonical',
                'measures': ['ReturnRate'], 'aggregators': ['AVG'], 'units': ['PERCENT'],
                'group_by': ['Reseller.Reseller', 'Product.Category'],
                'slicers': {'Product.Category': 'Bikes', 'SalesTerritory.Region': 'Europe', 'Date.Year': '2013'},
                'window_start': '2013-01-01', 'window_end': '2013-12-31',
            }
        },
        {
            'query_spec': {
                'cube': 'Adventure Works Sales', 'language': 'canonical',
                'measures': ['Sales Amount'], 'aggregators': ['SUM'], 'units': ['CURRENCY'],
                'group_by': ['Date.Month', 'Product.Category'],
                'slicers': {'Product.Category': 'Bikes', 'SalesTerritory.Region': 'Europe'},
                'time_members': ['2012', '2013'],
                'window_start': '2012-01-01', 'window_end': '2013-12-31',
            }
        },
    ]


def build_scaled_ckg(scale: int, output_dir: str) -> Tuple[CKGGraph, List[str]]:
    ckg = CKGGraph(output_dir=output_dir)
    objective_ids = list(BASE_OBJECTIVES)
    for base in BASE_OBJECTIVES:
        for i in range(1, int(scale)):
            oid = f'{base}__SCALE_{i:03d}'
            ckg.clone_objective(base, oid, suffix=f'scale_{i:03d}')
            objective_ids.append(oid)
    return ckg, objective_ids


def run_catalog_scalability(scales: List[int], results_dir: str, steps_per_session: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scale in scales:
        out_dir = os.path.join(results_dir, f'scale_{scale:03d}')
        ensure_dir(out_dir)
        tracemalloc.start()
        ckg, objective_ids = build_scaled_ckg(scale, out_dir)
        peak_build_cur, peak_build = tracemalloc.get_traced_memory()
        latencies: List[float] = []
        cold_latencies: List[float] = []
        warm_latencies: List[float] = []
        for oi, objective_id in enumerate(objective_ids):
            cycle = template_cycle(objective_id)
            session_id = f'scale{scale}::sess{oi:03d}'
            for step_idx in range(1, steps_per_session + 1):
                qp = dict(cycle[(step_idx - 1) % len(cycle)])
                qp['objective_id'] = objective_id
                t0 = time.perf_counter_ns()
                _ = ckg.evaluate_step(session_id=session_id, objective_id=objective_id, step_idx=step_idx, qp=qp)
                dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                latencies.append(dt_ms)
                if step_idx == 1:
                    cold_latencies.append(dt_ms)
                else:
                    warm_latencies.append(dt_ms)
        t0 = time.perf_counter_ns()
        snapshot_path = ckg.save_global_graph(os.path.join(out_dir, 'ckg_state.json'))
        snapshot_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        snapshot_bytes = os.path.getsize(snapshot_path) if os.path.exists(snapshot_path) else 0
        peak_cur, peak_total = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats = ckg.graph_stats()
        row = {
            'scale_factor': int(scale),
            'n_objectives': int(stats['n_objectives']),
            'n_constraints': int(stats['n_constraints']),
            'n_virtual_nodes': int(stats['n_virtual_nodes']),
            'n_nodes': int(stats['n_nodes']),
            'n_edges': int(stats['n_edges']),
            'history_len': int(stats['history_len']),
            'steps_per_session': int(steps_per_session),
            'p50_eval_ms': round(percentile(latencies, 0.50), 6),
            'p95_eval_ms': round(percentile(latencies, 0.95), 6),
            'p99_eval_ms': round(percentile(latencies, 0.99), 6),
            'mean_eval_ms': round(sum(latencies) / len(latencies), 6),
            'cold_p50_ms': round(percentile(cold_latencies, 0.50), 6),
            'warm_p95_ms': round(percentile(warm_latencies, 0.95), 6),
            'snapshot_ms': round(snapshot_ms, 6),
            'snapshot_bytes': int(snapshot_bytes),
            'peak_build_kib': round(peak_build / 1024.0, 3),
            'peak_total_kib': round(peak_total / 1024.0, 3),
        }
        rows.append(row)
    return rows


def run_growth_control(scale: int, results_dir: str, total_steps: int, keep_last_n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    objective_id = BASE_OBJECTIVES[0]
    cycle = template_cycle(objective_id)
    for mode, compact in [('no_compaction', None), ('keep_last', keep_last_n)]:
        ckg, _ = build_scaled_ckg(scale, os.path.join(results_dir, f'growth_{mode}'))
        session_id = f'growth::{mode}'
        for step_idx in range(1, total_steps + 1):
            qp = dict(cycle[(step_idx - 1) % len(cycle)])
            qp['objective_id'] = objective_id
            t0 = time.perf_counter_ns()
            _ = ckg.evaluate_step(session_id=session_id, objective_id=objective_id, step_idx=step_idx, qp=qp)
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            removed = 0
            if compact is not None:
                removed = ckg.compact_session_query_nodes(session_id=session_id, keep_last_n_steps=compact)
            if step_idx == 1 or step_idx % 10 == 0 or step_idx == total_steps:
                stats = ckg.graph_stats()
                rows.append({
                    'mode': mode,
                    'step_idx': int(step_idx),
                    'eval_ms': round(dt_ms, 6),
                    'n_nodes': int(stats['n_nodes']),
                    'n_edges': int(stats['n_edges']),
                    'history_len': int(stats['history_len']),
                    'removed_qp_nodes': int(removed),
                })
        snap_path = ckg.save_global_graph(os.path.join(results_dir, f'growth_{mode}', 'ckg_state.json'))
        rows.append({
            'mode': mode,
            'step_idx': int(total_steps),
            'eval_ms': 0.0,
            'n_nodes': int(ckg.G.number_of_nodes()),
            'n_edges': int(ckg.G.number_of_edges()),
            'history_len': int(len(ckg.history)),
            'removed_qp_nodes': 0,
            'snapshot_bytes': int(os.path.getsize(snap_path) if os.path.exists(snap_path) else 0),
        })
    return rows


def plot_latency_vs_nvs(rows: List[Dict[str, Any]], out_png: str) -> None:
    ensure_dir(str(Path(out_png).parent))
    xs = [int(r['n_virtual_nodes']) for r in rows]
    plt.figure(figsize=(8.8, 4.6))
    plt.plot(xs, [float(r['p50_eval_ms']) for r in rows], marker='o', label='p50')
    plt.plot(xs, [float(r['p95_eval_ms']) for r in rows], marker='s', label='p95')
    plt.plot(xs, [float(r['p99_eval_ms']) for r in rows], marker='^', label='p99')
    plt.xlabel('Number of virtual nodes in the CKG')
    plt.ylabel('Evaluation latency (ms)')
    plt.title('MCAD evaluation latency as the CKG grows')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_snapshot_vs_nodes(rows: List[Dict[str, Any]], out_png: str) -> None:
    ensure_dir(str(Path(out_png).parent))
    xs = [int(r['n_nodes']) for r in rows]
    ys = [float(r['snapshot_bytes']) / 1024.0 for r in rows]
    plt.figure(figsize=(8.4, 4.4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('CKG nodes')
    plt.ylabel('Serialized graph size (KiB)')
    plt.title('Snapshot size as a function of CKG size')
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_cold_warm(rows: List[Dict[str, Any]], out_png: str) -> None:
    ensure_dir(str(Path(out_png).parent))
    labels = [str(r['scale_factor']) for r in rows]
    xs = list(range(len(labels)))
    width = 0.35
    plt.figure(figsize=(8.8, 4.6))
    plt.bar([x - width / 2 for x in xs], [float(r['cold_p50_ms']) for r in rows], width=width, label='cold p50')
    plt.bar([x + width / 2 for x in xs], [float(r['warm_p95_ms']) for r in rows], width=width, label='warm p95')
    plt.xticks(xs, labels)
    plt.xlabel('Scale factor (objective replications per base objective)')
    plt.ylabel('Latency (ms)')
    plt.title('Cold vs warm latency across CKG scales')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_growth_control(rows: List[Dict[str, Any]], out_png: str) -> None:
    ensure_dir(str(Path(out_png).parent))
    plt.figure(figsize=(8.8, 4.8))
    for mode in ['no_compaction', 'keep_last']:
        sub = [r for r in rows if r['mode'] == mode and int(r.get('step_idx', 0)) > 0]
        plt.plot([int(r['step_idx']) for r in sub], [int(r['n_nodes']) for r in sub], marker='o', label=mode.replace('_', ' '))
    plt.xlabel('Session step')
    plt.ylabel('CKG nodes')
    plt.title('Runtime CKG growth with and without compaction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def build_report(catalog_rows: List[Dict[str, Any]], growth_rows: List[Dict[str, Any]], out_path: str, keep_last_n: int) -> None:
    lines: List[str] = []
    lines.append('# MCAD scalability and CKG growth-control report')
    lines.append('')
    lines.append('## Catalog scalability summary')
    lines.append('')
    headers = ['scale_factor', 'n_objectives', 'n_constraints', 'n_virtual_nodes', 'n_nodes', 'n_edges', 'p50_eval_ms', 'p95_eval_ms', 'p99_eval_ms', 'snapshot_ms', 'snapshot_bytes', 'peak_total_kib']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in catalog_rows:
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    lines.append('')
    lines.append(f'## Runtime growth control (keep-last = {keep_last_n} query-plan nodes)')
    lines.append('')
    headers = ['mode', 'step_idx', 'n_nodes', 'n_edges', 'history_len', 'removed_qp_nodes', 'eval_ms']
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in growth_rows:
        if 'snapshot_bytes' in row:
            continue
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    Path(out_path).write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run MCAD scalability and CKG growth benchmark.')
    p.add_argument('--run-root', type=str, default='', help='Optional run root under which results-dir is resolved')
    p.add_argument('--results-dir', type=str, default='results_scalability_benchmark')
    p.add_argument('--scales', type=int, nargs='*', default=[1, 5, 10, 20, 40])
    p.add_argument('--steps-per-session', type=int, default=12)
    p.add_argument('--growth-scale', type=int, default=40)
    p.add_argument('--growth-steps', type=int, default=120)
    p.add_argument('--keep-last-n', type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    if args.run_root and not os.path.isabs(results_dir):
        results_dir = os.path.join(args.run_root, results_dir)
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, 'figures'))

    catalog_rows = run_catalog_scalability(list(args.scales), results_dir, steps_per_session=int(args.steps_per_session))
    write_csv(os.path.join(results_dir, 'catalog_scalability.csv'), catalog_rows)

    growth_rows = run_growth_control(int(args.growth_scale), results_dir, total_steps=int(args.growth_steps), keep_last_n=int(args.keep_last_n))
    write_csv(os.path.join(results_dir, 'growth_control.csv'), growth_rows)

    plot_latency_vs_nvs(catalog_rows, os.path.join(results_dir, 'figures', 'scalability_latency_vs_nvs.png'))
    plot_snapshot_vs_nodes(catalog_rows, os.path.join(results_dir, 'figures', 'snapshot_size_vs_nodes.png'))
    plot_cold_warm(catalog_rows, os.path.join(results_dir, 'figures', 'cold_vs_warm_latency.png'))
    plot_growth_control(growth_rows, os.path.join(results_dir, 'figures', 'ckg_growth_control_nodes.png'))

    build_report(catalog_rows, growth_rows, os.path.join(results_dir, 'scalability_report.md'), int(args.keep_last_n))
    summary = {
        'catalog_rows': catalog_rows,
        'growth_rows': growth_rows,
        'keep_last_n': int(args.keep_last_n),
    }
    Path(os.path.join(results_dir, 'scalability_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'results_dir': results_dir, 'n_catalog_rows': len(catalog_rows), 'n_growth_rows': len(growth_rows)}, indent=2))


if __name__ == '__main__':
    main()
