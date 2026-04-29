from __future__ import annotations

"""Generate a human-validation annotation pack from one or more scenario YAML files.

The pack is designed for expert reviewers who judge whether a candidate query is
contributive for a strategic objective and, if so, which constraints become calculable.

Outputs:
  - annotation_items.csv   : one row per step to label
  - annotation_items.jsonl : same information in JSONL form
  - adjudication_template.csv
  - README_annotation_pack.md

The generator can optionally include the scenario oracle columns. This is useful for
internal dry runs, but should be disabled for blind human annotation.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_items(configs: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for cfg_path in configs:
        doc = load_yaml(cfg_path)
        objective_id = str(doc.get('objective_id') or '')
        dw_id = str(doc.get('dw_id') or '')
        for scenario in (doc.get('scenarios') or []):
            sid = str(scenario.get('id') or '')
            stype = str(scenario.get('type') or '')
            slabel = str(scenario.get('label') or '')
            for idx, step in enumerate((scenario.get('steps') or []), start=1):
                qp = dict(step.get('qp') or {})
                qspec = dict(qp.get('query_spec') or qp)
                yield {
                    'dataset': dw_id,
                    'objective_id': objective_id,
                    'scenario_id': sid,
                    'scenario_type': stype,
                    'scenario_label': slabel,
                    'step_idx': idx,
                    'item_id': f'{dw_id}::{sid}::{idx:02d}',
                    'step_name': str(step.get('name') or f'step_{idx}'),
                    'description': str(step.get('description') or ''),
                    'query_spec_json': json.dumps(qspec, ensure_ascii=False, sort_keys=True),
                    'oracle_allow': bool(step.get('oracle_allow', False)),
                    'oracle_ceval': list(step.get('oracle_ceval') or []),
                }


def write_csv(rows: List[Dict[str, Any]], path: Path, include_oracle: bool) -> None:
    fieldnames = [
        'item_id', 'dataset', 'objective_id', 'scenario_id', 'scenario_type', 'scenario_label',
        'step_idx', 'step_name', 'description', 'query_spec_json',
        'annotator_id', 'label_allow', 'label_ceval', 'comment',
    ]
    if include_oracle:
        fieldnames += ['oracle_allow', 'oracle_ceval']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {
                'item_id': r['item_id'],
                'dataset': r['dataset'],
                'objective_id': r['objective_id'],
                'scenario_id': r['scenario_id'],
                'scenario_type': r['scenario_type'],
                'scenario_label': r['scenario_label'],
                'step_idx': r['step_idx'],
                'step_name': r['step_name'],
                'description': r['description'],
                'query_spec_json': r['query_spec_json'],
                'annotator_id': '',
                'label_allow': '',
                'label_ceval': '',
                'comment': '',
            }
            if include_oracle:
                out['oracle_allow'] = int(bool(r['oracle_allow']))
                out['oracle_ceval'] = '|'.join(r['oracle_ceval'])
            w.writerow(out)


def write_jsonl(rows: List[Dict[str, Any]], path: Path, include_oracle: bool) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            out = dict(r)
            if not include_oracle:
                out.pop('oracle_allow', None)
                out.pop('oracle_ceval', None)
            f.write(json.dumps(out, ensure_ascii=False) + '\n')


def write_adjudication_template(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        'item_id', 'dataset', 'objective_id', 'scenario_id', 'step_idx', 'annotator_a_allow', 'annotator_b_allow',
        'annotator_a_ceval', 'annotator_b_ceval', 'adjudicated_allow', 'adjudicated_ceval', 'adjudicator_comment'
    ]
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                'item_id': r['item_id'],
                'dataset': r['dataset'],
                'objective_id': r['objective_id'],
                'scenario_id': r['scenario_id'],
                'step_idx': r['step_idx'],
                'annotator_a_allow': '',
                'annotator_b_allow': '',
                'annotator_a_ceval': '',
                'annotator_b_ceval': '',
                'adjudicated_allow': '',
                'adjudicated_ceval': '',
                'adjudicator_comment': '',
            })


def write_readme(rows: List[Dict[str, Any]], out_dir: Path, include_oracle: bool) -> None:
    datasets = sorted({r['dataset'] for r in rows})
    objectives = sorted({r['objective_id'] for r in rows})
    text = f"""# Human validation annotation pack

This pack was generated from the MCAD scenario harness.

Contents:
- `annotation_items.csv`: file to send to annotators.
- `annotation_items.jsonl`: JSONL version with the same items.
- `adjudication_template.csv`: file to consolidate disagreements.

Summary:
- Items: {len(rows)}
- Datasets: {', '.join(datasets)}
- Objectives: {', '.join(objectives)}
- Blind oracle hidden: {'no' if include_oracle else 'yes'}

Recommended protocol:
1. Give `annotation_items.csv` to at least 2 annotators.
2. Ask them to label `label_allow` as 1 or 0.
3. If `label_allow=1`, ask them to fill `label_ceval` with the calculable constraints separated by `|`.
4. Consolidate disagreements in `adjudication_template.csv`.
5. Score agreement and system-vs-human metrics with `score_human_validation.py`.
"""
    (out_dir / 'README_annotation_pack.md').write_text(text, encoding='utf-8')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate a human-validation annotation pack from scenario YAML files.')
    p.add_argument('--config', action='append', required=True, help='Path to a scenario YAML file. Repeatable.')
    p.add_argument('--run-root', default='', help='Optional run root under which out-dir is resolved')
    p.add_argument('--out-dir', required=True, help='Output directory')
    p.add_argument('--include-oracle', action='store_true', help='Include scenario oracle columns for internal dry runs.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.run_root and not out_dir.is_absolute():
        out_dir = Path(args.run_root) / out_dir
    ensure_dir(out_dir)
    rows = list(iter_items(args.config))
    write_csv(rows, out_dir / 'annotation_items.csv', include_oracle=args.include_oracle)
    write_jsonl(rows, out_dir / 'annotation_items.jsonl', include_oracle=args.include_oracle)
    write_adjudication_template(rows, out_dir / 'adjudication_template.csv')
    write_readme(rows, out_dir, include_oracle=args.include_oracle)
    print(f'[MCAD/HUMAN] annotation pack written to {out_dir} ({len(rows)} items)')


if __name__ == '__main__':
    main()
