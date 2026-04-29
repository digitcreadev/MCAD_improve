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


BASE_QP = {
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


class TestPalier2Lifecycle(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix='mcad_palier2_')
        self.addCleanup(self.tmpdir.cleanup)
        os.environ['MCAD_RESULTS_DIR'] = self.tmpdir.name
        reset_runtime_state()

    def tearDown(self):
        os.environ.pop('MCAD_RESULTS_DIR', None)
        reset_runtime_state()

    def _run_step(self, session_id: str, qp_suffix: str) -> str:
        qp = dict(BASE_QP)
        qspec = dict(BASE_QP['query_spec'])
        qspec['execution_result_excerpt'] = {'rows_retained': 12, 'tag': qp_suffix}
        qp['query_spec'] = qspec
        resp = evaluate_with_objective_and_session(
            EvaluateWithObjectiveAndSessionRequest(session_id=session_id, objective_id='OHF_NORD', qp=qp)
        )
        self.assertTrue(resp.sat)
        self.assertEqual(resp.calculable_constraints, ['c1'])
        self.assertIsNotNone(resp.retained_evidence_id)
        return str(resp.retained_evidence_id)

    def test_compact_session_archives_old_evidence_but_keeps_summary(self):
        session_id = 'S_P2_COMPACT'
        SESSION_STORE.ensure_session(session_id=session_id, objective_id='OHF_NORD', dw_id='FOODMART')
        ids = [self._run_step(session_id, f's{i}') for i in range(1, 4)]

        store = get_evidence_store()
        ckg = get_ckg()
        result = store.compact_session(session_id, keep_last_n=1)
        graph_result = ckg.compact_session_evidence_nodes(session_id, keep_last_n_steps=1)
        active = [r['evidence_id'] for r in store.list_for_session(session_id, statuses=['active', 'temporary'])]
        archived = [r['evidence_id'] for r in store.list_for_session(session_id, statuses=['archived'])]
        SESSION_STORE.register_evidence_lifecycle(
            session_id,
            active_evidence_ids=active,
            archived_evidence_ids=archived,
            session_summary_path=result.get('summary_path'),
        )

        self.assertEqual(len(active), 1)
        self.assertEqual(active[0], ids[-1])
        self.assertEqual(set(archived), set(ids[:-1]))
        self.assertTrue(os.path.exists(result['summary_path']))
        self.assertEqual(result['archived_count'], 2)
        self.assertEqual(len(graph_result['archived_evidence_nodes']), 2)
        self.assertTrue(ckg.G.nodes[f'evidence::{ids[0]}']['status'] == 'archived')
        state = SESSION_STORE.get_session(session_id)
        self.assertEqual(state.evidence_ids, [ids[-1]])
        self.assertEqual(set(state.archived_evidence_ids), set(ids[:-1]))
        self.assertTrue(str(state.latest_session_summary_path).endswith('_compact_summary.json'))

    def test_expire_before_marks_old_evidence_expired(self):
        session_id = 'S_P2_EXPIRE'
        SESSION_STORE.ensure_session(session_id=session_id, objective_id='OHF_NORD', dw_id='FOODMART')
        eid = self._run_step(session_id, 'old')
        store = get_evidence_store()
        rec = store.get(eid)
        rec['created_at'] = '2000-01-01T00:00:00Z'
        store._save_all()  # test-only backdating
        result = store.expire_before(max_age_days=1)
        self.assertEqual(result['expired_count'], 1)
        expired = store.get(eid)
        self.assertEqual(expired['status'], 'expired')
        self.assertIsNotNone(expired['archive_path'])
        self.assertTrue(os.path.exists(expired['archive_path']))


if __name__ == '__main__':
    unittest.main()
