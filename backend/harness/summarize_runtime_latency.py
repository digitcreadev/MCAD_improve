from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=';')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe(row: Dict[str, Any], key: str) -> str:
    v = row.get(key, '')
    return '' if v is None else str(v)


def collect_rows(root_results_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # Single-dataset policy benchmark summary
    p1 = os.path.join(root_results_dir, 'policy_benchmark', 'policy_summary.csv')
    for row in read_csv(p1):
        rows.append({
            'scope': 'single_dataset',
            'dataset': _safe(row, 'objective_id') or 'N/A',
            'policy': _safe(row, 'policy'),
            'latency_p50_ms': _safe(row, 'latency_p50_ms'),
            'latency_p95_ms': _safe(row, 'latency_p95_ms'),
            'latency_p99_ms': _safe(row, 'latency_p99_ms'),
            'latency_cold_p50_ms': _safe(row, 'latency_cold_p50_ms'),
            'latency_warm_p50_ms': _safe(row, 'latency_warm_p50_ms'),
            'mean_phi_final': _safe(row, 'mean_phi_final'),
            'mean_auc_phi': _safe(row, 'mean_auc_phi'),
            'mean_false_allow_rate': _safe(row, 'mean_false_allow_rate'),
            'mean_false_block_rate': _safe(row, 'mean_false_block_rate'),
            'mean_non_contrib_exec_rate': _safe(row, 'mean_non_contrib_exec_rate'),
            'source': p1,
        })

    # Aggregate multi-dataset summary already includes latency columns
    p2 = os.path.join(root_results_dir, 'multidataset_benchmark', 'aggregate_policy_summary.csv')
    for row in read_csv(p2):
        rows.append({
            'scope': 'multidataset_aggregate',
            'dataset': 'ALL',
            'policy': _safe(row, 'policy'),
            'latency_p50_ms': _safe(row, 'latency_p50_ms'),
            'latency_p95_ms': _safe(row, 'latency_p95_ms'),
            'latency_p99_ms': _safe(row, 'latency_p99_ms'),
            'latency_cold_p50_ms': _safe(row, 'latency_cold_p50_ms'),
            'latency_warm_p50_ms': _safe(row, 'latency_warm_p50_ms'),
            'mean_phi_final': _safe(row, 'mean_phi_final'),
            'mean_auc_phi': _safe(row, 'mean_auc_phi'),
            'mean_false_allow_rate': _safe(row, 'mean_false_allow_rate'),
            'mean_false_block_rate': _safe(row, 'mean_false_block_rate'),
            'mean_non_contrib_exec_rate': _safe(row, 'mean_non_contrib_exec_rate'),
            'source': p2,
        })

    # Per-dataset summaries inside multidataset_benchmark/*/policy_summary.csv
    multi_root = os.path.join(root_results_dir, 'multidataset_benchmark')
    if os.path.isdir(multi_root):
        for child in sorted(Path(multi_root).iterdir()):
            p3 = child / 'policy_summary.csv'
            meta = child / 'policy_benchmark_meta.json'
            dataset_label = child.name
            if p3.exists():
                for row in read_csv(str(p3)):
                    rows.append({
                        'scope': 'multidataset_per_dataset',
                        'dataset': dataset_label,
                        'policy': _safe(row, 'policy'),
                        'latency_p50_ms': _safe(row, 'latency_p50_ms'),
                        'latency_p95_ms': _safe(row, 'latency_p95_ms'),
                        'latency_p99_ms': _safe(row, 'latency_p99_ms'),
                        'latency_cold_p50_ms': _safe(row, 'latency_cold_p50_ms'),
                        'latency_warm_p50_ms': _safe(row, 'latency_warm_p50_ms'),
                        'mean_phi_final': _safe(row, 'mean_phi_final'),
                        'mean_auc_phi': _safe(row, 'mean_auc_phi'),
                        'mean_false_allow_rate': _safe(row, 'mean_false_allow_rate'),
                        'mean_false_block_rate': _safe(row, 'mean_false_block_rate'),
                        'mean_non_contrib_exec_rate': _safe(row, 'mean_non_contrib_exec_rate'),
                        'source': str(p3),
                    })
    return rows


def write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append('# MCAD runtime latency summary')
    lines.append('')
    lines.append('This file consolidates the policy-level runtime indicators needed for the article and reproducibility pack.')
    lines.append('')

    mcad_rows = [r for r in rows if r.get('policy') == 'mcad']
    if mcad_rows:
        lines.append('## MCAD highlights')
        lines.append('')
        for row in mcad_rows:
            lines.append(
                f"- `{row['scope']}` / `{row['dataset']}`: p50/p95/p99 = "
                f"**{row['latency_p50_ms']} / {row['latency_p95_ms']} / {row['latency_p99_ms']} ms**, "
                f"cold p50 = **{row['latency_cold_p50_ms']} ms**, warm p50 = **{row['latency_warm_p50_ms']} ms**, "
                f"mean φ final = **{row['mean_phi_final']}**, false allow = **{row['mean_false_allow_rate']}**"
            )
        lines.append('')

    lines.append('## Consolidated table')
    lines.append('')
    headers = [
        'scope', 'dataset', 'policy',
        'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
        'latency_cold_p50_ms', 'latency_warm_p50_ms',
        'mean_phi_final', 'mean_auc_phi',
        'mean_false_allow_rate', 'mean_false_block_rate', 'mean_non_contrib_exec_rate'
    ]
    lines.append('|' + '|'.join(headers) + '|')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows:
        lines.append('|' + '|'.join(str(row.get(h, '')) for h in headers) + '|')
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Summarize runtime latency outputs for MCAD reproducibility/artifact packs.')
    p.add_argument('--results-dir', type=str, required=True, help='Root results directory produced by reproduce_final.py or run_full_pipeline.py')
    p.add_argument('--out-dir', type=str, default='', help='Output directory. Defaults to <results-dir>/runtime_latency')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, 'runtime_latency')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.results_dir)
    if not rows:
        raise SystemExit(f'No latency summaries found under {args.results_dir}')
    csv_path = os.path.join(out_dir, 'runtime_latency_summary.csv')
    md_path = os.path.join(out_dir, 'runtime_latency_summary.md')
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f'[MCAD/RUNTIME] wrote {csv_path}')
    print(f'[MCAD/RUNTIME] wrote {md_path}')


if __name__ == '__main__':
    main()
