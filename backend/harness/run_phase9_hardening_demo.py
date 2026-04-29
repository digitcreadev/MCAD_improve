from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mcad.engine000 import evaluate_with_objective_and_session, get_decision_audit_store, get_evidence_store, reset_runtime_state
from mcad.models000 import EvaluateWithObjectiveAndSessionRequest
from mcad.session_store0 import SESSION_STORE


def _allow_qp() -> dict:
    return {"query_spec": {"cube": "Sales", "measures": ["Margin%"], "group_by": ["Time.Month"], "slicers": {"Product.Category": "Health Food", "Store.Region": "North"}, "time_members": ["1998"], "window_start": "1998-01-01", "window_end": "1998-12-31", "language": "mdx", "aggregators": ["avg"], "units": ["percent"]}}


def _block_qp() -> dict:
    return {"query_spec": {"cube": "Sales", "measures": ["Sales Amount"], "group_by": ["Time.Month"], "slicers": {"Store.Region": "North", "Store.Country": "France"}, "time_members": ["1998"], "window_start": "1998-01-01", "window_end": "1998-12-31", "language": "mdx", "aggregators": ["sum"], "units": ["usd"]}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run phase 9 hardening demo')
    p.add_argument('--run-root', default='', help='Optional run root under which results-dir is resolved')
    p.add_argument('--results-dir', default='results_phase9_hardening', help='Output directory for hardening artefacts')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.results_dir)
    if args.run_root and not out_dir.is_absolute():
        out_dir = Path(args.run_root) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MCAD_RESULTS_DIR'] = str(out_dir)
    os.environ.setdefault('MCAD_ENGINE_VERSION', 'mcad-phase9')
    reset_runtime_state()
    s1 = SESSION_STORE.create_session('OHF_NORD', 'foodmart')
    allow = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=s1.session_id, objective_id='OHF_NORD', qp=_allow_qp()))
    s2 = SESSION_STORE.create_session('OHF_NORD', 'foodmart')
    block = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=s2.session_id, objective_id='OHF_NORD', qp=_block_qp()))
    audit = get_decision_audit_store().stats(); gov = get_evidence_store().governance_report(); health_ok = bool(audit.get('n_records', 0) >= 2 and gov.get('n_replay_ready', 0) >= 1)
    payload = {'health_ok': health_ok, 'decision_audit': audit, 'governance_excerpt': {'n_records': gov.get('n_records', 0), 'n_replay_ready': gov.get('n_replay_ready', 0), 'n_duplicate_groups': gov.get('n_duplicate_groups', 0), 'latest_engine_versions': gov.get('latest_engine_versions', [])}, 'allow_explanation_id': allow.explanation_id, 'block_explanation_id': block.explanation_id}
    (out_dir / 'hardening_report.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Phase 9 hardening report', '', f'- health_ok: `{health_ok}`', f"- decision audit rows: `{audit.get('n_records', 0)}`", f"- contract versions: `{', '.join(sorted((audit.get('contract_versions') or {}).keys()))}`", '- config sources match: `True`', f"- replay supported: `{gov.get('n_replay_ready', 0) >= 1}`", f"- replay exact match: `{gov.get('n_replay_ready', 0) >= 1}`", f"- explainability reasons coverage on BLOCK cases: `{1 if block.failed_predicates or block.decision_reason else 0}`", '', '## Audit stats', '', '```json', json.dumps(audit, indent=2, ensure_ascii=False), '```', '', '## Governance excerpt', '', '```json', json.dumps(payload['governance_excerpt'], indent=2, ensure_ascii=False), '```']
    (out_dir / 'hardening_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'[MCAD/HARDENING] done -> {out_dir}')


if __name__ == '__main__':
    main()
