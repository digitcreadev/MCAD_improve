from __future__ import annotations

import pandas as pd

from backend.harness.run_statistical_analysis import (
    compute_pairwise_advantages,
    compute_policy_ci_table,
    compute_ablation_sensitivity,
)


def _sample_df() -> pd.DataFrame:
    rows = []
    for rep in [0, 1]:
        for scenario_id in ['s1', 's2']:
            base = {
                'config_path': 'cfg.yaml',
                'dw_id': 'DW',
                'objective_id': 'OBJ',
                'scenario_id': scenario_id,
                'repeat_id': rep,
            }
            rows.append({**base, 'policy': 'mcad', 'phi_final': 1.0, 'auc_phi': 0.8, 'false_allow_rate': 0.0, 'non_contrib_exec_rate': 0.0, 'false_block_rate': 0.0})
            rows.append({**base, 'policy': 'baseline_naive', 'phi_final': 1.0, 'auc_phi': 0.8, 'false_allow_rate': 0.5, 'non_contrib_exec_rate': 0.5, 'false_block_rate': 0.0})
            rows.append({**base, 'policy': 'ablation_no_real', 'phi_final': 1.0, 'auc_phi': 0.8, 'false_allow_rate': 0.25, 'non_contrib_exec_rate': 0.25, 'false_block_rate': 0.0})
            rows.append({**base, 'policy': 'ablation_no_sat', 'phi_final': 1.0, 'auc_phi': 0.8, 'false_allow_rate': 0.0, 'non_contrib_exec_rate': 0.0, 'false_block_rate': 0.0})
            rows.append({**base, 'policy': 'ablation_ceval_any_intersection', 'phi_final': 1.0, 'auc_phi': 0.8, 'false_allow_rate': 0.25, 'non_contrib_exec_rate': 0.25, 'false_block_rate': 0.0})
    return pd.DataFrame(rows)


def test_compute_policy_ci_table_smoke() -> None:
    df = _sample_df()
    out = compute_policy_ci_table(df, ['phi_final', 'false_allow_rate'])
    assert set(out['policy']) >= {'mcad', 'baseline_naive'}
    mcad = out[out['policy'] == 'mcad'].iloc[0]
    assert float(mcad['false_allow_rate_mean']) == 0.0


def test_compute_pairwise_advantages_positive_for_false_allow() -> None:
    df = _sample_df()
    out = compute_pairwise_advantages(df, ['baseline_naive'], ['false_allow_rate'], ['config_path', 'dw_id', 'objective_id', 'scenario_id', 'repeat_id'])
    row = out.iloc[0]
    assert row['comparator'] == 'baseline_naive'
    assert float(row['mcad_advantage_mean']) > 0.0


def test_compute_ablation_sensitivity_smoke() -> None:
    df = _sample_df()
    out = compute_ablation_sensitivity(df, ['false_allow_rate'], ['config_path', 'dw_id', 'objective_id', 'scenario_id', 'repeat_id'])
    assert 'ablation_no_real' in set(out['ablation'])
