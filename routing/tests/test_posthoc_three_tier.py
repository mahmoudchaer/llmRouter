import numpy as np
import pandas as pd

from routing.evaluation.posthoc_three_tier import (
    analyze_predictions,
    merge_probabilities,
    remap_tiers,
)


def sample_frame():
    return pd.DataFrame(
        {
            "prompt_id": ["a", "b", "c", "d"],
            "source_dataset": ["mmlu_pro", "mmlu_pro", "aime", "swe_bench"],
            "true_tier": [1, 2, 3, 4],
            "predicted_tier": [1, 3, 1, 2],
            "probabilities": [
                [0.8, 0.1, 0.05, 0.05],
                [0.1, 0.3, 0.5, 0.1],
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.5, 0.2, 0.2],
            ],
        }
    )


def test_tier_and_probability_mapping():
    assert remap_tiers([1, 2, 3, 4]).tolist() == [1, 2, 2, 3]
    merged = merge_probabilities([[0.1, 0.2, 0.3, 0.4]])
    np.testing.assert_allclose(merged, [[0.1, 0.5, 0.4]])


def test_direct_remap_is_primary_and_reduces_merged_boundary_errors():
    results = analyze_predictions(sample_frame())
    predictions = results["predictions"]
    assert predictions.predicted_tier_3_direct_remap.tolist() == [1, 2, 1, 2]
    metrics = results["metrics"].set_index("system")
    assert metrics.loc["four_tier_original", "exact_accuracy"] == 0.25
    assert metrics.loc["three_tier_direct_remap", "exact_accuracy"] == 0.5
    assert metrics.loc["three_tier_direct_remap", "severe_under_routing"] == 0.0
    assert results["summary"]["primary_method"] == "three_tier_direct_remap"


def test_per_source_and_class_balance_are_complete():
    results = analyze_predictions(sample_frame())
    assert set(results["per_source"].source_dataset) == {"mmlu_pro", "aime", "swe_bench"}
    direct = results["class_balance"].query("system == 'three_tier_direct_remap' and kind == 'ground_truth'")
    assert direct["count"].sum() == 4
    assert set(direct.tier) == {1, 2, 3}


def test_string_probabilities_are_supported():
    frame = sample_frame()
    frame["probabilities"] = frame.probabilities.map(str)
    results = analyze_predictions(frame)
    assert len(results["predictions"]) == 4
