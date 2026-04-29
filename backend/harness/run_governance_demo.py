from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mcad.engine import evaluate_with_objective_and_session, get_ckg, get_evidence_store, replay_retained_evidence, reset_runtime_state
from mcad.models import EvaluateWithObjectiveAndSessionRequest
from mcad.session_store import SESSION_STORE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run governance/replay demo')
    p.add_argument('--run-root', default='', help='Optional run root under which results-dir is resolved')
    p.add_argument('--results-dir', default='results_governance_demo', help='Output directory for governance artefacts')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.results_dir)
    if args.run_root and not out.is_absolute():
        out = Path(args.run_root) / out
    os.environ['MCAD_RESULTS_DIR'] = str(out)
    reset_runtime_state()
    session = SESSION_STORE.create_session('OHF_NORD', 'foodmart')
    qp = {"query_spec": {"cube": "Sales", "measures": ["Margin%"], "group_by": ["Time.Month"], "slicers": {"Product.Category": "Health Food", "Store.Region": "North"}, "time_members": ["1998"], "window_start": "1998-01-01", "window_end": "1998-12-31", "language": "mdx", "aggregators": ["avg"], "units": ["percent"]}}
    resp1 = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=session.session_id, objective_id='OHF_NORD', qp=qp))
    _ = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=session.session_id, objective_id='OHF_NORD', qp=qp))
    store = get_evidence_store(); ckg = get_ckg(); report = store.governance_report(ckg_stats=ckg.graph_stats()); replay = replay_retained_evidence(resp1.retained_evidence_id) if resp1.retained_evidence_id else {}; compare = {}; rec = store.get(resp1.retained_evidence_id) if resp1.retained_evidence_id else None
    if rec and rec.get('pre_snapshot_path') and rec.get('post_snapshot_path'):
        compare = ckg.compare_snapshots(rec['pre_snapshot_path'], rec['post_snapshot_path'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'governance_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'replay_report.json').write_text(json.dumps(replay, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'snapshot_compare.json').write_text(json.dumps(compare, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[MCAD/GOVERNANCE] done -> {out}')


if __name__ == '__main__':
    main()
