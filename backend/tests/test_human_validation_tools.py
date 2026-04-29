import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestHumanValidationTools(unittest.TestCase):
    def test_generate_and_score_dry_run(self):
        repo = Path(__file__).resolve().parents[2]
        out_dir = Path(tempfile.mkdtemp(prefix='mcad_hv_'))
        pack_dir = out_dir / 'pack'
        report_dir = out_dir / 'report'
        cfg1 = repo / 'backend' / 'harness' / 'scenarios.yaml'
        cfg2 = repo / 'backend' / 'harness' / 'scenarios_adventureworks.yaml'

        subprocess.run([
            sys.executable, str(repo / 'backend' / 'harness' / 'generate_expert_annotation_pack.py'),
            '--config', str(cfg1), '--config', str(cfg2),
            '--out-dir', str(pack_dir), '--include-oracle'
        ], check=True)
        self.assertTrue((pack_dir / 'annotation_items.csv').exists())

        subprocess.run([
            sys.executable, str(repo / 'backend' / 'harness' / 'score_human_validation.py'),
            '--config', str(cfg1), '--config', str(cfg2),
            '--out-dir', str(report_dir)
        ], check=True)
        self.assertTrue((report_dir / 'agreement_summary.json').exists())
        self.assertTrue((report_dir / 'system_vs_human.csv').exists())

        with open(report_dir / 'system_vs_human.csv', 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        policies = {r['policy'] for r in rows}
        self.assertIn('mcad', policies)
        mcad_row = next(r for r in rows if r['policy'] == 'mcad')
        self.assertGreaterEqual(float(mcad_row['accuracy']), 0.99)


if __name__ == '__main__':
    unittest.main()
