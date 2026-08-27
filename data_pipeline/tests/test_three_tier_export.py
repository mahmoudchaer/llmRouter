import numpy as np
import pandas as pd

from src.export_three_tier_dataset import relabel_frame


def test_exact_and_censored_mapping_preserves_audit_columns():
    source = pd.DataFrame(
        {
            "prompt_id": list("abcdef"),
            "tier": [1, 2, 3, 4, np.nan, np.nan],
            "tier_min": [1, 2, 3, 4, 2, 4],
            "tier_max": [1, 2, 3, 4, 3, np.nan],
            "resolved": [True, True, True, True, False, False],
            "unresolved_reason": [None, None, None, None, "missing", "no_group"],
            "beyond_tier4_possible": [False, False, False, False, False, True],
        }
    )
    result = relabel_frame(source)
    assert result.tier.tolist()[:4] == [1, 2, 2, 3]
    assert result.loc[4, "tier_min"] == result.loc[4, "tier_max"] == 2
    assert result.loc[4, "three_tier_interval_collapsed"]
    assert not result.loc[4, "resolved"] and pd.isna(result.loc[4, "tier"])
    assert result.loc[5, "tier_min"] == 3 and pd.isna(result.loc[5, "tier_max"])
    assert result.loc[5, "beyond_tier3_possible"]
    assert "tier_4" in result and "resolved_4" in result
