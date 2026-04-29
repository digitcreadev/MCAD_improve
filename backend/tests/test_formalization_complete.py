import unittest

from backend.ckg.ckg_updater import CKGGraph
from backend.mcad.query_plan import parse_sql_analytic


class TestFormalizationCoverage(unittest.TestCase):
    def test_sat_clauses_fail_on_objective_contradiction(self):
        ckg = CKGGraph(output_dir='results_test_formal_complete')
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
                    'Store.Region': 'South',
                    'Time.Year': '1998',
                },
                'window_start': '1998-01-01',
                'window_end': '1998-12-31',
            },
        }
        qp_node = ckg.add_qp_node('S_TEST', 1, qp)
        sat, clauses = ckg.sat(qp, qp_node)
        self.assertFalse(sat)
        by_name = {c.name: c for c in clauses}
        self.assertFalse(by_name['slc_ok'].ok)
        self.assertFalse(by_name['nvac_ok'].ok)

    def test_induced_mask_keeps_only_sufficient_support(self):
        ckg = CKGGraph(output_dir='results_test_formal_complete')
        real_nv = {'NV_CA_HF_NORTH_1997_MONTH', 'NV_CA_HF_NORTH_1998_MONTH'}
        ceval = {'c3'}
        mask = ckg.induced_mask('OHF_NORD', real_nv, ceval)
        self.assertEqual(set(mask['node_ids']), real_nv)
        self.assertEqual(mask['constraints']['c3'], ['NV_CA_HF_NORTH_1997_MONTH', 'NV_CA_HF_NORTH_1998_MONTH'])

    def test_sql_analytic_parser_builds_canonical_plan(self):
        sql = "SELECT SUM(store_sales) AS sales FROM sales WHERE product_category = 'Health Food' AND store_region = 'North' AND year = 1998 GROUP BY month, product_category"
        q = parse_sql_analytic(sql)
        self.assertEqual(q['cube'].lower(), 'sales')
        self.assertIn('store_sales', q['measures'][0].lower())
        self.assertIn('SUM', q['aggregators'])
        self.assertIn('month', q['group_by'][0].lower())
        self.assertEqual(q['language'], 'sql')


if __name__ == '__main__':
    unittest.main()
