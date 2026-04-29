import unittest

from backend.ckg.ckg_updater import CKGGraph


class TestGrowthWindowRealization(unittest.TestCase):
    def test_broad_window_realizes_year_specific_growth_nodes(self):
        ckg = CKGGraph(output_dir='results_test_growth_window')
        objective_id = 'OHF_NORD'
        qp = {
            'objective_id': objective_id,
            'query_spec': {
                'cube': 'Sales',
                'measures': ['Store Sales'],
                'aggregators': ['SUM'],
                'units': ['CURRENCY'],
                'group_by': ['Time.Month', 'Product.Category'],
                'slicers': {
                    'Product.Category': 'Health Food',
                    'Store.Region': 'North',
                },
                'time_members': ['1997', '1998'],
                'window_start': '1997-01-01',
                'window_end': '1998-12-31',
            },
        }
        qp_node = ckg.add_qp_node('S_TEST', 1, qp)
        real_ids = ckg.real(objective_id=objective_id, qp_node=qp_node)
        self.assertIn('NV_CA_HF_NORTH_1997_MONTH', real_ids)
        self.assertIn('NV_CA_HF_NORTH_1998_MONTH', real_ids)
        self.assertEqual(ckg.ceval(objective_id=objective_id, real_nv_ids=real_ids), {'c3'})


if __name__ == '__main__':
    unittest.main()
