import unittest

from backend.mcad.objectives import get_objective, reload_objectives


class TestObjectiveEnrichmentCatalog(unittest.TestCase):
    def test_new_objectives_are_loadable(self):
        reload_objectives()
        for oid in [
            "OFM_HEALTHFOOD_NORTH_COVERAGE_1998",
            "OFM_BEER_WA_SALES_PROFIT_GROWTH_9798",
            "OAW_BIKES_EUROPE_MULTIRESOURCE_2013",
            "OAW_BIKES_PACIFIC_GROWTH_2012_2013",
        ]:
            obj = get_objective(oid)
            self.assertGreaterEqual(len(obj.constraints), 3)
            self.assertGreaterEqual(len(obj.kpis), 3)


if __name__ == "__main__":
    unittest.main()
