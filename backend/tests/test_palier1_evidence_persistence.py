import os
import tempfile
import unittest

from mcad.engine import (
    evaluate_with_objective_and_session,
    get_ckg,
    get_evidence_store,
    reset_runtime_state,
)
from mcad.models import EvaluateWithObjectiveAndSessionRequest
from mcad.session_store import SESSION_STORE


class TestPalier1EvidencePersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix='mcad_palier1_')
        self.addCleanup(self.tmpdir.cleanup)
        os.environ['MCAD_RESULTS_DIR'] = self.tmpdir.name
        reset_runtime_state()

    def tearDown(self):
        os.environ.pop('MCAD_RESULTS_DIR', None)
        reset_runtime_state()

    def test_contributive_query_persists_useful_evidence_and_snapshot(self):
        session_id = 'S_PALIER1'
        SESSION_STORE.ensure_session(session_id=session_id, objective_id='OHF_NORD', dw_id='FOODMART')
        qp = {
            'objective_id': 'OHF_NORD',
            'query_spec': {
                'cube': 'Sales',
                'measures': ['Margin%'],
                'aggregators': ['AVG'],
                'units': ['PERCENT'],
                'group_by': ['Time.Month', 'Product.Category'],
                'slicers': {
                    'Product.Category': 'Health Food',
                    'Store.Region': 'North',
                    'Time.Year': '1998',
                },
                'window_start': '1998-01-01',
                'window_end': '1998-12-31',
                'language': 'mdx',
                'execution_result_excerpt': {'rows_retained': 12, 'kpi': 'Margin%'}
            },
        }
        resp = evaluate_with_objective_and_session(
            EvaluateWithObjectiveAndSessionRequest(
                session_id=session_id,
                objective_id='OHF_NORD',
                qp=qp,
            )
        )
        self.assertTrue(resp.sat)
        self.assertEqual(resp.calculable_constraints, ['c1'])
        self.assertIsNotNone(resp.retained_evidence_id)
        self.assertIsNotNone(resp.retained_snapshot_path)
        self.assertTrue(os.path.exists(str(resp.retained_snapshot_path)))

        store = get_evidence_store()
        recs = store.list_for_session(session_id)
        self.assertEqual(len(recs), 1)
        rec = recs[0]
        self.assertEqual(rec['evidence_id'], resp.retained_evidence_id)
        self.assertEqual(rec['constraint_ids'], ['c1'])
        self.assertIn('NV_MARGIN_HF_NORTH_1998_MONTH', rec['linked_virtual_nodes'])
        self.assertIn('execution_result_excerpt', rec['retained_payload'])
        self.assertEqual(rec['snapshot_path'], resp.retained_snapshot_path)

        ckg = get_ckg()
        ev_node = f"evidence::{resp.retained_evidence_id}"
        qp_node = f"qp::{session_id}::t001"
        self.assertTrue(ckg.G.has_node(ev_node))
        self.assertTrue(ckg.G.has_edge(qp_node, ev_node))
        self.assertTrue(ckg.G.has_edge(ev_node, 'constraint::c1'))

        state = SESSION_STORE.get_session(session_id)
        self.assertIn(str(resp.retained_evidence_id), state.evidence_ids)
        self.assertEqual(state.latest_snapshot_id, resp.retained_snapshot_id)

    def test_non_contributive_query_does_not_persist_useful_evidence(self):
        session_id = 'S_PALIER1_BLOCK'
        SESSION_STORE.ensure_session(session_id=session_id, objective_id='OHF_NORD', dw_id='FOODMART')
        qp = {
            'objective_id': 'OHF_NORD',
            'query_spec': {
                'cube': 'Sales',
                'measures': ['Margin%'],
                'aggregators': ['AVG'],
                'units': ['CURRENCY'],
                'group_by': ['Time.Month', 'Product.Category'],
                'slicers': {
                    'Product.Category': 'Health Food',
                    'Store.Region': 'North',
                    'Time.Year': '1998',
                },
                'window_start': '1998-01-01',
                'window_end': '1998-12-31',
                'language': 'mdx',
            },
        }
        resp = evaluate_with_objective_and_session(
            EvaluateWithObjectiveAndSessionRequest(
                session_id=session_id,
                objective_id='OHF_NORD',
                qp=qp,
            )
        )
        self.assertFalse(resp.sat)
        self.assertIsNone(resp.retained_evidence_id)
        self.assertEqual(get_evidence_store().list_for_session(session_id), [])


if __name__ == '__main__':
    unittest.main()
