import unittest

from backend.ckg.ckg_updater import CKGGraph


class TestAdventureWorksMultiDataset(unittest.TestCase):
    def test_objective_loaded_and_growth_requires_both_years(self):
        ckg = CKGGraph(output_dir='results_test_aw')
        self.assertIn('OAW_BIKES_EUROPE', ckg.objectives)

        objective_id = 'OAW_BIKES_EUROPE'
        qp = {
            'objective_id': objective_id,
            'query_spec': {
                'cube': 'Adventure Works Sales',
                'measures': ['Sales Amount'],
                'aggregators': ['SUM'],
                'units': ['CURRENCY'],
                'group_by': ['Date.Month', 'Product.Category'],
                'slicers': {
                    'Product.Category': 'Bikes',
                    'SalesTerritory.Region': 'Europe',
                },
                'time_members': ['2012', '2013'],
                'window_start': '2012-01-01',
                'window_end': '2013-12-31',
            },
        }
        qp_node = ckg.add_qp_node('sess-aw', 1, qp)
        real_ids = ckg.real(objective_id=objective_id, qp_node=qp_node)
        self.assertIn('NV_AW_SALES_BIKES_EUROPE_2012_MONTH', real_ids)
        self.assertIn('NV_AW_SALES_BIKES_EUROPE_2013_MONTH', real_ids)
        self.assertEqual(ckg.ceval(objective_id=objective_id, real_nv_ids=real_ids), {'aw_c3'})

    def test_wrong_region_does_not_realize_aw_margin_constraint(self):
        ckg = CKGGraph(output_dir='results_test_aw')
        objective_id = 'OAW_BIKES_EUROPE'
        qp = {
            'objective_id': objective_id,
            'query_spec': {
                'cube': 'Adventure Works Sales',
                'measures': ['Gross Margin%'],
                'aggregators': ['AVG'],
                'units': ['PERCENT'],
                'group_by': ['Date.Month', 'Product.Category'],
                'slicers': {
                    'Product.Category': 'Bikes',
                    'SalesTerritory.Region': 'Pacific',
                    'Date.Year': '2013',
                },
                'window_start': '2013-01-01',
                'window_end': '2013-12-31',
            },
        }
        qp_node = ckg.add_qp_node('sess-aw', 2, qp)
        real_ids = ckg.real(objective_id=objective_id, qp_node=qp_node)
        self.assertEqual(real_ids, set())
        self.assertEqual(ckg.ceval(objective_id=objective_id, real_nv_ids=real_ids), set())


if __name__ == '__main__':
    unittest.main()
