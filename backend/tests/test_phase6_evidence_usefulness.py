from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from mcad.engine import (
    bootstrap_session_from_persisted_evidence,
    evaluate_with_objective_and_session,
    get_evidence_store,
    reset_runtime_state,
)
from mcad.models import EvaluateWithObjectiveAndSessionRequest
from mcad.session_store import SESSION_STORE


Q_C1 = {
    'query_spec': {
        'cube': 'Sales', 'measures': ['Margin%'], 'aggregators': ['AVG'], 'units': ['PERCENT'],
        'group_by': ['Time.Month', 'Product.Category'],
        'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North', 'Time.Year': '1998'},
        'time_members': ['1998'], 'window_start': '1998-01-01', 'window_end': '1998-12-31',
        'language': 'mdx', 'execution_result_excerpt': {'rows_retained': 12}
    }
}
Q_C2 = {
    'query_spec': {
        'cube': 'Sales', 'measures': ['StockoutRate'], 'aggregators': ['AVG'], 'units': ['PERCENT'],
        'group_by': ['Store.Store', 'Product.Category'],
        'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North', 'Time.Year': '1998'},
        'time_members': ['1998'], 'window_start': '1998-01-01', 'window_end': '1998-12-31',
        'language': 'mdx', 'execution_result_excerpt': {'rows_retained': 7}
    }
}
Q_C3 = {
    'query_spec': {
        'cube': 'Sales', 'measures': ['Store Sales'], 'aggregators': ['SUM'], 'units': ['CURRENCY'],
        'group_by': ['Time.Month', 'Product.Category'],
        'slicers': {'Product.Category': 'Health Food', 'Store.Region': 'North'},
        'time_members': ['1997', '1998'], 'window_start': '1997-01-01', 'window_end': '1998-12-31',
        'language': 'mdx', 'execution_result_excerpt': {'rows_retained': 24}
    }
}


class TestPhase6EvidenceUsefulness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='mcad_phase6_')
        os.environ['MCAD_RESULTS_DIR'] = self.tmp
        reset_runtime_state()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop('MCAD_RESULTS_DIR', None)
        reset_runtime_state()

    def test_usefulness_report_has_nonzero_metrics(self):
        session = SESSION_STORE.ensure_session('S_EVID_1', 'OHF_NORD', 'foodmart')
        evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=session.session_id, objective_id='OHF_NORD', qp=Q_C1))
        store = get_evidence_store()
        report = store.usefulness_report(objective_constraint_totals={'OHF_NORD': 3})
        self.assertEqual(report['n_records'], 1)
        self.assertGreater(report['mean_retained_ratio'], 0.0)
        self.assertGreater(report['mean_usefulness_score'], 0.0)
        self.assertGreater(report['execution_excerpt_coverage_ratio'], 0.0)
        self.assertGreater(report['objective_bootstrap_coverage']['OHF_NORD'], 0.0)

    def test_bootstrap_from_evidence_improves_followup_coverage(self):
        seed = SESSION_STORE.ensure_session('S_SEED', 'OHF_NORD', 'foodmart')
        evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=seed.session_id, objective_id='OHF_NORD', qp=Q_C1))
        evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=seed.session_id, objective_id='OHF_NORD', qp=Q_C2))

        follow_boot = SESSION_STORE.ensure_session('S_BOOT', 'OHF_NORD', 'foodmart')
        boot = bootstrap_session_from_persisted_evidence(follow_boot.session_id, 'OHF_NORD')
        self.assertEqual(set(boot['seeded_constraint_ids']), {'c1', 'c2'})
        self.assertAlmostEqual(float(boot['phi_leq_t']), 2/3, places=4)
        boot_resp = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=follow_boot.session_id, objective_id='OHF_NORD', qp=Q_C3))
        self.assertAlmostEqual(float(boot_resp.phi_leq_t or 0.0), 1.0, places=4)
        self.assertEqual(set(boot_resp.bootstrap_constraints), {'c1', 'c2'})
        self.assertGreater(len(boot_resp.bootstrap_evidence_ids), 0)

        follow_plain = SESSION_STORE.ensure_session('S_PLAIN', 'OHF_NORD', 'foodmart')
        plain_resp = evaluate_with_objective_and_session(EvaluateWithObjectiveAndSessionRequest(session_id=follow_plain.session_id, objective_id='OHF_NORD', qp=Q_C3))
        self.assertAlmostEqual(float(plain_resp.phi_leq_t or 0.0), 1/3, places=4)


if __name__ == '__main__':
    unittest.main()
