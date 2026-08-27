import numpy as np

from src.run_tier_router_v1 import metrics


THREE_TIER_COST = [[0, 0.25, 0.5], [1, 0, 0.25], [4, 1, 0]]


def test_three_tier_safety_metrics_and_cost():
    y = [3, 3, 2, 1]
    p = [1, 2, 1, 2]
    prob = np.eye(3)[np.asarray(p) - 1] * 0.98 + 0.02 / 3
    result = metrics(y, p, prob, THREE_TIER_COST)
    assert result["T3_to_T1"] == 1
    assert result["T3_to_T2"] == 1
    assert result["T2_to_T1"] == 1
    assert result["severe_under_routing"] == 0.25
    assert result["routing_cost"] == (1 + 0.25 + 0.25 + 0.0625) / 4
    assert len(result["confusion_matrix"]) == 3
