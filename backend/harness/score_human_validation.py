from __future__ import annotations

"""Score human validation annotations and compare them with MCAD/baselines.

Supported inputs:
- one or more annotator CSV files based on annotation_items.csv
- optional adjudication CSV with final labels
- optional scenario configs to compute MCAD / baseline predictions on the same items

Outputs:
  - agreement_summary.json
  - system_vs_human.csv
  - human_validation_report.md
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.ckg.ckg_updater import CKGGraph
from backend.harness.run_baselines_and_ablations import policy_decision


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_bool_cell(value: str) -> Optional[bool]:
    s = str(value or '').strip().lower()
    if s in {'1', 'true', 'yes', 'y', 'allow', 'allowed'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'block', 'blocked'}:
        return False
    return None


def parse_ceval_cell(value: str) -> Set[str]:
    s = str(value or '').strip()
    if not s:
        return set()
    return {x.strip() for x in s.replace(',', '|').split('|') if x.strip()}


def read_annotation_csv(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            item_id = str(row.get('item_id') or '').strip()
            if not item_id:
                continue
            rows[item_id] = {
                'item_id': item_id,
                'dataset': row.get('dataset', ''),
                'objective_id': row.get('objective_id', ''),
                'scenario_id': row.get('scenario_id', ''),
                'step_idx': int(str(row.get('step_idx') or '0') or 0),
                'query_spec_json': row.get('query_spec_json', ''),
                'label_allow': parse_bool_cell(row.get('label_allow', '')),
                'label_ceval': parse_ceval_cell(row.get('label_ceval', '')),
            }
    return rows


def read_adjudication_csv(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            item_id = str(row.get('item_id') or '').strip()
            if not item_id:
                continue
            rows[item_id] = {
                'item_id': item_id,
                'label_allow': parse_bool_cell(row.get('adjudicated_allow', '')),
                'label_ceval': parse_ceval_cell(row.get('adjudicated_ceval', '')),
            }
    return rows


def iter_scenario_items(configs: Sequence[str]) -> Iterable[Dict[str, Any]]:
    for cfg_path in configs:
        doc = load_yaml(cfg_path)
        objective_id = str(doc.get('objective_id') or '')
        dw_id = str(doc.get('dw_id') or '')
        for scenario in (doc.get('scenarios') or []):
            sid = str(scenario.get('id') or '')
            for idx, step in enumerate((scenario.get('steps') or []), start=1):
                qp = dict(step.get('qp') or {})
                qspec = dict(qp.get('query_spec') or qp)
                yield {
                    'item_id': f'{dw_id}::{sid}::{idx:02d}',
                    'dataset': dw_id,
                    'objective_id': objective_id,
                    'scenario_id': sid,
                    'step_idx': idx,
                    'qp': {'objective_id': objective_id, 'query_spec': qspec},
                    'oracle_allow': bool(step.get('oracle_allow', False)),
                    'oracle_ceval': set(step.get('oracle_ceval') or []),
                }


def cohen_kappa(a: Sequence[Optional[bool]], b: Sequence[Optional[bool]]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return 0.0
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pa = Counter(x for x, _ in pairs)
    pb = Counter(y for _, y in pairs)
    pe = sum((pa[val] / n) * (pb[val] / n) for val in {True, False})
    if math.isclose(1.0 - pe, 0.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1.0 - pe)


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        'n': float(len(y_true)), 'tp': float(tp), 'tn': float(tn), 'fp': float(fp), 'fn': float(fn),
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc,
    }


def ceval_exact_match(ref: Sequence[Set[str]], pred: Sequence[Set[str]]) -> float:
    if not ref:
        return 0.0
    return sum(1 for r, p in zip(ref, pred) if r == p) / len(ref)


def avg_jaccard(ref: Sequence[Set[str]], pred: Sequence[Set[str]]) -> float:
    if not ref:
        return 0.0
    vals = []
    for r, p in zip(ref, pred):
        union = r | p
        vals.append((len(r & p) / len(union)) if union else 1.0)
    return sum(vals) / len(vals)


def build_reference(annotators: List[Dict[str, Dict[str, Any]]], adjudicated: Dict[str, Dict[str, Any]], fallback_to_oracle: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if adjudicated:
        return adjudicated
    if len(annotators) >= 2:
        ref: Dict[str, Dict[str, Any]] = {}
        keys = set(annotators[0]) & set(annotators[1])
        for key in keys:
            a = annotators[0][key]
            b = annotators[1][key]
            la = a.get('label_allow')
            lb = b.get('label_allow')
            if la is None and lb is None:
                continue
            if la is None:
                allow = lb
            elif lb is None:
                allow = la
            else:
                allow = (int(bool(la)) + int(bool(lb))) >= 1
            ceval = set(a.get('label_ceval') or set()) | set(b.get('label_ceval') or set()) if allow else set()
            ref[key] = {'item_id': key, 'label_allow': allow, 'label_ceval': ceval}
        return ref
    if annotators:
        return {k: {'item_id': k, 'label_allow': v.get('label_allow'), 'label_ceval': v.get('label_ceval') or set()} for k, v in annotators[0].items()}
    return fallback_to_oracle


def system_predictions(configs: Sequence[str], policies: Sequence[str], seed: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {p: {} for p in policies}
    items = list(iter_scenario_items(configs))
    for policy in policies:
        rng = random.Random(seed)
        for item in items:
            ckg = CKGGraph(output_dir='results_human_validation_tmp')
            decision = policy_decision(policy, ckg, item['objective_id'], item['qp'], rng=rng, matched_random_allow_prob=0.5, step_idx=item['step_idx'])
            out[policy][item['item_id']] = {
                'allow': bool(decision.get('allow')),
                'ceval': set(decision.get('ceval_ids') or []),
                'dataset': item['dataset'],
                'objective_id': item['objective_id'],
            }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Score MCAD human validation annotations and compare systems to the adjudicated reference.')
    p.add_argument('--annotations', action='append', default=[], help='Annotator CSV(s) generated from annotation_items.csv. Repeatable.')
    p.add_argument('--adjudication', default='', help='Optional adjudication CSV file.')
    p.add_argument('--config', action='append', default=[], help='Scenario config(s) used to build the annotation pack. Repeatable.')
    p.add_argument('--run-root', default='', help='Optional run root under which out-dir is resolved')
    p.add_argument('--out-dir', required=True, help='Output directory.')
    p.add_argument('--policy', action='append', default=['mcad', 'baseline_naive', 'baseline_measure_overlap'], help='Policies to score against the reference.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.run_root and not out_dir.is_absolute():
        out_dir = Path(args.run_root) / out_dir
    ensure_dir(out_dir)

    annotators = [read_annotation_csv(p) for p in args.annotations]
    adjudicated = read_adjudication_csv(args.adjudication) if args.adjudication else {}

    scenario_items = list(iter_scenario_items(args.config)) if args.config else []
    fallback_oracle = {
        item['item_id']: {
            'item_id': item['item_id'],
            'label_allow': item['oracle_allow'],
            'label_ceval': set(item['oracle_ceval']),
        }
        for item in scenario_items
    }

    reference = build_reference(annotators, adjudicated, fallback_oracle)

    agreement: Dict[str, Any] = {
        'n_reference_items': len(reference),
        'n_annotators': len(annotators),
    }
    if len(annotators) >= 2:
        common = sorted(set(annotators[0]) & set(annotators[1]))
        a_allow = [annotators[0][k].get('label_allow') for k in common]
        b_allow = [annotators[1][k].get('label_allow') for k in common]
        agreement['allow_cohen_kappa'] = round(cohen_kappa(a_allow, b_allow), 6)
        agreement['ceval_exact_match'] = round(ceval_exact_match([annotators[0][k].get('label_ceval') or set() for k in common], [annotators[1][k].get('label_ceval') or set() for k in common]), 6)
        agreement['ceval_avg_jaccard'] = round(avg_jaccard([annotators[0][k].get('label_ceval') or set() for k in common], [annotators[1][k].get('label_ceval') or set() for k in common]), 6)
    (out_dir / 'agreement_summary.json').write_text(json.dumps(agreement, indent=2, ensure_ascii=False), encoding='utf-8')

    predictions = system_predictions(args.config, args.policy, seed=args.seed) if args.config else {}
    rows: List[Dict[str, Any]] = []
    for policy, pred_map in predictions.items():
        keys = sorted(set(reference) & set(pred_map))
        y_true = [bool(reference[k].get('label_allow')) for k in keys]
        y_pred = [bool(pred_map[k].get('allow')) for k in keys]
        allow_metrics = binary_metrics(y_true, y_pred)
        ceval_ref = [set(reference[k].get('label_ceval') or set()) for k in keys]
        ceval_pred = [set(pred_map[k].get('ceval') or set()) for k in keys]
        rows.append({
            'policy': policy,
            **{k: round(v, 6) for k, v in allow_metrics.items()},
            'ceval_exact_match': round(ceval_exact_match(ceval_ref, ceval_pred), 6),
            'ceval_avg_jaccard': round(avg_jaccard(ceval_ref, ceval_pred), 6),
        })

    with (out_dir / 'system_vs_human.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['policy', 'n', 'tp', 'tn', 'fp', 'fn', 'precision', 'recall', 'f1', 'accuracy', 'ceval_exact_match', 'ceval_avg_jaccard']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    report_lines = [
        '# Human validation report',
        '',
        f'- Reference items: {len(reference)}',
        f'- Annotators: {len(annotators)}',
    ]
    if 'allow_cohen_kappa' in agreement:
        report_lines.append(f"- Allow Cohen's kappa: {agreement['allow_cohen_kappa']}")
        report_lines.append(f"- Ceval exact match: {agreement['ceval_exact_match']}")
        report_lines.append(f"- Ceval average Jaccard: {agreement['ceval_avg_jaccard']}")
    if not annotators:
        report_lines.append('- Note: no human annotation CSV was provided; the scenario oracle was used as a dry-run reference only.')
    report_lines.append('')
    report_lines.append('## System vs reference')
    report_lines.append('')
    for r in rows:
        report_lines.append(f"- {r['policy']}: F1={r['f1']}, accuracy={r['accuracy']}, ceval_exact_match={r['ceval_exact_match']}, ceval_avg_jaccard={r['ceval_avg_jaccard']}")
    (out_dir / 'human_validation_report.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    print(f'[MCAD/HUMAN] report written to {out_dir}')


if __name__ == '__main__':
    main()
