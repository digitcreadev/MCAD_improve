import unittest

from backend.ckg.ckg_updater import CKGGraph


class TestSATClauseDiagnostics(unittest.TestCase):
    def test_wrong_unit_is_reported_as_unit_failure(self):
        ckg = CKGGraph(output_dir='results_test_sat_diag')
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
            },
        }
        qp_node = ckg.add_qp_node('S_DIAG', 1, qp)
        sat, clauses = ckg.sat(qp, qp_node)
        by_name = {c.name: c for c in clauses}
        self.assertFalse(sat)
        self.assertFalse(by_name['unit_ok'].ok)
        self.assertTrue(by_name['slc_ok'].ok)
        self.assertTrue(by_name['time_ok'].ok)

    def test_conflicting_region_aliases_are_reported_as_slicer_failure(self):
        ckg = CKGGraph(output_dir='results_test_sat_diag')
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
                    'region': 'South',
                    'Time.Year': '1998',
                },
                'window_start': '1998-01-01',
                'window_end': '1998-12-31',
            },
        }
        qp_node = ckg.add_qp_node('S_DIAG', 2, qp)
        sat, clauses = ckg.sat(qp, qp_node)
        by_name = {c.name: c for c in clauses}
        self.assertFalse(sat)
        self.assertFalse(by_name['slc_ok'].ok)


if __name__ == '__main__':
    unittest.main()
