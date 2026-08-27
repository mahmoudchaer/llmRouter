from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TIER_MAP_4_TO_3 = {1: 1, 2: 2, 3: 2, 4: 3}
CONFIDENCE_EDGES = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
CONFIDENCE_LABELS = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90%+"]
DEFAULT_COST_4 = np.asarray(
    [[0, 0.25, 0.50, 0.75], [1, 0, 0.25, 0.50], [4, 1, 0, 0.25], [9, 4, 1, 0]],
    dtype=float,
)
DEFAULT_COST_3 = np.asarray(
    [[0, 0.25, 0.50], [1, 0, 0.25], [4, 1, 0]], dtype=float
)


def remap_tiers(values: Iterable[int]) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    invalid = sorted(set(values) - set(TIER_MAP_4_TO_3))
    if invalid:
        raise ValueError(f"Invalid 4-tier labels: {invalid}")
    return np.asarray([TIER_MAP_4_TO_3[int(value)] for value in values], dtype=int)


def _coerce_probability_vector(value) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    result = np.asarray(value, dtype=float)
    if result.shape != (4,):
        raise ValueError(f"Expected four probabilities, got shape {result.shape}")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    total = result.sum()
    if total <= 0:
        raise ValueError("Probability vector has zero mass")
    return result / total


def merge_probabilities(values: Iterable) -> np.ndarray:
    old = np.vstack([_coerce_probability_vector(value) for value in values])
    return np.column_stack([old[:, 0], old[:, 1] + old[:, 2], old[:, 3]])


def _macro_f1(y: np.ndarray, p: np.ndarray, labels: range) -> float:
    scores = []
    for label in labels:
        tp = np.sum((y == label) & (p == label))
        fp = np.sum((y != label) & (p == label))
        fn = np.sum((y == label) & (p != label))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else float(2 * tp / denominator))
    return float(np.mean(scores))


def evaluate(
    true_tier: Iterable[int],
    predicted_tier: Iterable[int],
    probabilities: np.ndarray,
    cost_matrix: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(true_tier, dtype=int)
    p = np.asarray(predicted_tier, dtype=int)
    k = probabilities.shape[1]
    if len(y) == 0 or len(y) != len(p) or probabilities.shape != (len(y), k):
        raise ValueError("Labels, predictions, and probabilities must have matching non-zero lengths")
    if set(y) - set(range(1, k + 1)) or set(p) - set(range(1, k + 1)):
        raise ValueError("Tier values do not match probability columns")
    delta = p - y
    clipped = np.clip(probabilities, 1e-12, 1.0)
    one_hot = np.eye(k)[y - 1]
    costs = cost_matrix[y - 1, p - 1] / float(cost_matrix.max())
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "exact_accuracy": float(np.mean(p == y)),
        "macro_f1": _macro_f1(y, p, range(1, k + 1)),
        "mae": float(np.mean(np.abs(delta))),
        "within_one_tier": float(np.mean(np.abs(delta) <= 1)),
        "under_routing": float(np.mean(delta < 0)),
        "over_routing": float(np.mean(delta > 0)),
        "severe_under_routing": float(np.mean(delta <= -2)),
        "routing_cost": float(np.mean(costs)),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(y)), y - 1]))),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "mean_confidence": float(np.mean(probabilities[np.arange(len(y)), p - 1])),
    }
    for true_value in range(2, k + 1):
        for predicted_value in range(1, true_value):
            result[f"T{true_value}_to_T{predicted_value}"] = int(
                np.sum((y == true_value) & (p == predicted_value))
            )
    return result


def calibration(
    true_tier: np.ndarray, predicted_tier: np.ndarray, confidence: np.ndarray, system: str
) -> pd.DataFrame:
    frame = pd.DataFrame({"true": true_tier, "pred": predicted_tier, "confidence": confidence})
    frame["confidence_bucket"] = pd.cut(
        frame.confidence,
        CONFIDENCE_EDGES,
        labels=CONFIDENCE_LABELS,
        right=False,
        include_lowest=True,
    )
    rows = []
    for bucket, part in frame.groupby("confidence_bucket", observed=False):
        delta = part.pred.to_numpy() - part.true.to_numpy()
        rows.append(
            {
                "system": system,
                "confidence_bucket": str(bucket),
                "count": len(part),
                "mean_confidence": float(part.confidence.mean()) if len(part) else None,
                "exact_accuracy": float(np.mean(delta == 0)) if len(part) else None,
                "mae": float(np.mean(np.abs(delta))) if len(part) else None,
                "under_routing": float(np.mean(delta < 0)) if len(part) else None,
                "severe_under_routing": float(np.mean(delta <= -2)) if len(part) else None,
            }
        )
    return pd.DataFrame(rows)


def _distribution(values: np.ndarray, system: str, kind: str, n_tiers: int) -> list[dict]:
    return [
        {
            "system": system,
            "kind": kind,
            "tier": tier,
            "count": int(np.sum(values == tier)),
            "fraction": float(np.mean(values == tier)),
        }
        for tier in range(1, n_tiers + 1)
    ]


def analyze_predictions(frame: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    required = {"prompt_id", "source_dataset", "true_tier", "predicted_tier", "probabilities"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    output = frame.copy()
    old_y = output.true_tier.to_numpy(dtype=int)
    old_p = output.predicted_tier.to_numpy(dtype=int)
    old_probs = np.vstack([_coerce_probability_vector(value) for value in output.probabilities])
    new_y = remap_tiers(old_y)
    direct_p = remap_tiers(old_p)
    new_probs = merge_probabilities(output.probabilities)
    merged_argmax_p = np.argmax(new_probs, axis=1) + 1

    output["true_tier_3"] = new_y
    output["predicted_tier_3_direct_remap"] = direct_p
    output["predicted_tier_3_merged_probability_argmax"] = merged_argmax_p
    output["probabilities_3"] = list(new_probs)
    output["confidence_3_direct_remap"] = new_probs[np.arange(len(output)), direct_p - 1]
    output["confidence_3_merged_probability_argmax"] = np.max(new_probs, axis=1)
    output["posthoc_prediction_methods_disagree"] = direct_p != merged_argmax_p

    systems = [
        ("four_tier_original", old_y, old_p, old_probs, DEFAULT_COST_4),
        ("three_tier_direct_remap", new_y, direct_p, new_probs, DEFAULT_COST_3),
        ("three_tier_merged_probability_argmax_diagnostic", new_y, merged_argmax_p, new_probs, DEFAULT_COST_3),
    ]
    metric_rows, calibration_frames, balance_rows, source_rows = [], [], [], []
    for name, y, p, probs, costs in systems:
        metric_rows.append({"system": name, **evaluate(y, p, probs, costs)})
        confidence = probs[np.arange(len(y)), p - 1]
        calibration_frames.append(calibration(y, p, confidence, name))
        balance_rows.extend(_distribution(y, name, "ground_truth", probs.shape[1]))
        balance_rows.extend(_distribution(p, name, "prediction", probs.shape[1]))
        for source, indices in output.groupby("source_dataset").indices.items():
            idx = np.asarray(indices)
            source_rows.append(
                {"system": name, "source_dataset": source, **evaluate(y[idx], p[idx], probs[idx], costs)}
            )

    summary = {
        "mapping": {"new_T1": [1], "new_T2": [2, 3], "new_T3": [4]},
        "primary_method": "three_tier_direct_remap",
        "diagnostic_method": "three_tier_merged_probability_argmax_diagnostic",
        "probability_mapping": ["P(old T1)", "P(old T2)+P(old T3)", "P(old T4)"],
        "prediction_method_disagreement_count": int(np.sum(direct_p != merged_argmax_p)),
        "prediction_method_disagreement_fraction": float(np.mean(direct_p != merged_argmax_p)),
        "three_tier_cost_matrix": DEFAULT_COST_3.tolist(),
        "notes": [
            "The primary result remaps the existing four-tier prediction and does not change its decision.",
            "Merged-probability argmax is diagnostic because probability merging can change the predicted class.",
            "No model is trained or checkpoint modified by this analysis.",
        ],
    }
    return {
        "summary": summary,
        "predictions": output,
        "metrics": pd.DataFrame(metric_rows),
        "calibration": pd.concat(calibration_frames, ignore_index=True),
        "class_balance": pd.DataFrame(balance_rows),
        "per_source": pd.DataFrame(source_rows),
    }


def write_reports(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(results["summary"], indent=2) + "\n")
    for name in ["metrics", "calibration", "class_balance", "per_source"]:
        results[name].to_csv(output_dir / f"{name}.csv", index=False)
    results["predictions"].to_parquet(output_dir / "posthoc_predictions.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc comparison of the frozen 4-tier system with a 3-tier merge")
    parser.add_argument("predictions", help="Seed prediction parquet or CSV")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = Path(args.predictions)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    results = analyze_predictions(frame)
    write_reports(results, Path(args.output_dir))
    print(results["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
