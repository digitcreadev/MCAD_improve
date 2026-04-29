
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from backend.ckg.ckg_updater import CKGGraph
from mcad.engine import evaluate_with_objective_and_session, get_evidence_store, replay_retained_evidence, reset_runtime_state
from mcad.models import EvaluateWithObjectiveAndSessionRequest
from mcad.session_store import SESSION_STORE


class TestPalier3Governance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='mcad_palier3_')
        os.environ['MCAD_RESULTS_DIR'] = self.tmp
        reset_runtime_state()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _qp(self):
        return {
            'query_spec': {
                'cube': 'Sales',
                'measures': ['Margin%'],
                'group_by': ['Time.Month'],
                'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North'},
                'time_members': ['1998'],
                'window_start': '1998-01-01',
                'window_end': '1998-12-31',
                'language': 'mdx',
                'aggregators': ['avg'],
                'units': ['percent'],
            }
        }

    def test_replay_retained_evidence_exact_match(self):
        session = SESSION_STORE.create_session('OHF_NORD', 'foodmart')
        resp = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=session.session_id, objective_id='OHF_NORD', qp=self._qp()))
        self.assertTrue(resp.retained_evidence_id)
        report = replay_retained_evidence(resp.retained_evidence_id)
        self.assertTrue(report['replay_supported'])
        self.assertTrue(report['exact_match'])
        self.assertFalse(report['mismatches'])

    def test_compare_snapshots_and_duplicate_groups(self):
        session = SESSION_STORE.create_session('OHF_NORD', 'foodmart')
        resp1 = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=session.session_id, objective_id='OHF_NORD', qp=self._qp()))
        resp2 = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=session.session_id, objective_id='OHF_NORD', qp=self._qp()))
        store = get_evidence_store()
        rec1 = store.get(resp1.retained_evidence_id)
        self.assertIsNotNone(rec1)
        diff = CKGGraph.compare_snapshots(rec1['pre_snapshot_path'], rec1['post_snapshot_path'])
        self.assertGreaterEqual(diff['node_delta'], 1)
        gov = store.governance_report()
        self.assertGreaterEqual(gov['n_duplicate_groups'], 1)
        self.assertGreaterEqual(gov['n_replay_ready'], 2)


if __name__ == '__main__':
    unittest.main()
