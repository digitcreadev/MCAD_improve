
import os
import tempfile
import textwrap
import unittest

from backend.ckg.ckg_updater import CKGGraph


class TestCevalRequirements(unittest.TestCase):
    def test_ceval_requires_complete_requirement_set(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = os.path.join(tmpdir.name, 'objectives.yaml')
        yaml_text = textwrap.dedent('''
        objectives:
          - id: O1
            name: O1
            description: test
            kpis: [K1]
            constraints:
              - id: c1
                kpi_id: K1
                description: growth
                weight: 1.0
                requirement_sets:
                  - [nv1997, nv1998]
                virtual_nodes:
                  - id: nv1997
                    fact: Sales
                    grain: [Time.Month]
                    measure: Store Sales
                    aggregator: SUM
                    unit: CURRENCY
                    slicers: {Time.Year: "1997"}
                  - id: nv1998
                    fact: Sales
                    grain: [Time.Month]
                    measure: Store Sales
                    aggregator: SUM
                    unit: CURRENCY
                    slicers: {Time.Year: "1998"}
        ''')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(yaml_text)
        ckg = CKGGraph(output_dir=tmpdir.name)
        ckg.bootstrap_objectives(path)
        self.assertEqual(ckg.ceval('O1', {'nv1998'}), set())
        self.assertEqual(ckg.ceval('O1', {'nv1997', 'nv1998'}), {'c1'})


if __name__ == '__main__':
    unittest.main()
