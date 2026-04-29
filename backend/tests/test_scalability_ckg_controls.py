import unittest

from backend.ckg.ckg_updater import CKGGraph


class TestScalabilityCKGControls(unittest.TestCase):
    def test_clone_objective_preserves_calculability(self):
        ckg = CKGGraph(output_dir='results_test_scalability')
        new_oid = ckg.clone_objective('OHF_NORD', 'OHF_NORD__CLONE', suffix='clone')
        self.assertIn(new_oid, ckg.objectives)
        qp = {
            'objective_id': new_oid,
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
            },
        }
        qp_node = ckg.add_qp_node('sess-scale', 1, qp)
        real_ids = ckg.real(objective_id=new_oid, qp_node=qp_node)
        self.assertTrue(any('NV_MARGIN_HF_NORTH_1998_MONTH__clone' == rid for rid in real_ids))
        self.assertEqual(ckg.ceval(objective_id=new_oid, real_nv_ids=real_ids), {'c1__clone'})

    def test_compact_session_query_nodes_keeps_recent_steps(self):
        ckg = CKGGraph(output_dir='results_test_scalability')
        objective_id = 'OHF_NORD'
        qp = {
            'objective_id': objective_id,
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
            },
        }
        for step_idx in range(1, 6):
            ckg.evaluate_step(session_id='sess1', objective_id=objective_id, step_idx=step_idx, qp=qp)
        removed = ckg.compact_session_query_nodes('sess1', keep_last_n_steps=2)
        self.assertEqual(removed, 3)
        qp_nodes = [(n, a) for n, a in ckg.G.nodes(data=True) if a.get('type') == 'query_plan' and a.get('session_id') == 'sess1']
        kept_steps = sorted(int(a.get('step_idx')) for _, a in qp_nodes)
        self.assertEqual(kept_steps, [4, 5])
        self.assertIn('c1', ckg.session_coverage.get('sess1', {}).get(objective_id, set()))


if __name__ == '__main__':
    unittest.main()
