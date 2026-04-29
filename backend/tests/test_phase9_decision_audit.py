from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from mcad.engine000 import evaluate_with_objective_and_session, get_decision_audit_store, reset_runtime_state
from mcad.models000 import EvaluateWithObjectiveAndSessionRequest
from mcad.session_store0 import SESSION_STORE


class TestPhase9DecisionAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='mcad_phase9_')
        os.environ['MCAD_RESULTS_DIR'] = self.tmp
        os.environ['MCAD_ENGINE_VERSION'] = 'mcad-phase9'
        reset_runtime_state()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _allow_qp(self):
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

    def _block_qp(self):
        return {
            'query_spec': {
                'cube': 'Sales',
                'measures': ['Sales Amount'],
                'group_by': ['Time.Month'],
                'slicers': {'Store.Region': 'North', 'Store.Country': 'France'},
                'time_members': ['1998'],
                'window_start': '1998-01-01',
                'window_end': '1998-12-31',
                'language': 'mdx',
                'aggregators': ['sum'],
                'units': ['usd'],
            }
        }

    def test_persists_stable_decision_contract_for_allow_and_block(self):
        allow_session = SESSION_STORE.create_session('OHF_NORD', 'foodmart')
        allow = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=allow_session.session_id, objective_id='OHF_NORD', qp=self._allow_qp()))
        block_session = SESSION_STORE.create_session('OHF_NORD', 'foodmart')
        block = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=block_session.session_id, objective_id='OHF_NORD', qp=self._block_qp()))

        self.assertEqual(allow.decision, 'ALLOW')
        self.assertEqual(block.decision, 'BLOCK')
        self.assertTrue(allow.explanation_id)
        self.assertTrue(block.explanation_id)
        self.assertEqual(allow.contract_version, 'mcad.decision.contract.v1')

        store = get_decision_audit_store()
        stats = store.stats()
        self.assertEqual(stats['n_records'], 2)
        self.assertEqual(stats['by_decision'].get('ALLOW'), 1)
        self.assertEqual(stats['by_decision'].get('BLOCK'), 1)

        allow_rec = store.get(allow.explanation_id)
        block_rec = store.get(block.explanation_id)
        self.assertEqual(allow_rec['contract_version'], 'mcad.decision.contract.v1')
        self.assertEqual(allow_rec['engine_version'], 'mcad-phase9')
        self.assertTrue(allow_rec['retained_evidence_id'])
        self.assertTrue(block_rec['decision_reason'])


if __name__ == '__main__':
    unittest.main()
