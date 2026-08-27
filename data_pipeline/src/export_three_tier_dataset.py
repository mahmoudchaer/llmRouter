from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIER_MAP = {1: 1, 2: 2, 3: 2, 4: 3}


def map_tier(value):
    if pd.isna(value):
        return np.nan
    return TIER_MAP[int(value)]


def relabel_frame(source: pd.DataFrame) -> pd.DataFrame:
    required = {"prompt_id", "tier", "tier_min", "tier_max", "resolved", "unresolved_reason"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = source.copy()
    frame = frame.rename(
        columns={
            "tier": "tier_4",
            "tier_min": "tier_min_4",
            "tier_max": "tier_max_4",
            "resolved": "resolved_4",
            "unresolved_reason": "unresolved_reason_4",
            "beyond_tier4_possible": "beyond_tier4_possible_4",
        }
    )
    frame["tier"] = frame.tier_4.map(map_tier)
    frame["tier_min"] = frame.tier_min_4.map(map_tier).astype("Int64")
    frame["tier_max"] = frame.tier_max_4.map(map_tier).astype("Int64")
    frame["resolved"] = frame.resolved_4.astype(bool)
    frame["unresolved_reason"] = frame.unresolved_reason_4
    frame["three_tier_interval_collapsed"] = (
        ~frame.resolved
        & frame.tier_max.notna()
        & (frame.tier_min == frame.tier_max)
    )
    old_beyond = frame.get("beyond_tier4_possible_4", pd.Series(False, index=frame.index)).fillna(False)
    frame["beyond_tier3_possible"] = old_beyond.astype(bool)
    frame["tier_policy"] = "phase1_three_tier_merge_v1"
    frame["tier_mapping"] = "T1=T1;T2=T2+T3;T3=T4"

    if frame.loc[frame.resolved, "tier"].isna().any():
        raise ValueError("Resolved row lost its exact tier")
    if frame.loc[~frame.resolved, "tier"].notna().any():
        raise ValueError("Unresolved row received an exact tier")
    if not set(frame.tier.dropna().astype(int)).issubset({1, 2, 3}):
        raise ValueError("Three-tier label outside 1..3")
    if len(frame) != len(source) or frame.prompt_id.nunique() != source.prompt_id.nunique():
        raise ValueError("Prompt preservation check failed")
    return frame


def build_audit(frame: pd.DataFrame) -> dict:
    resolved = frame[frame.resolved]
    counts = {f"T{tier}": int((resolved.tier == tier).sum()) for tier in (1, 2, 3)}
    counts.update({"unresolved": int((~frame.resolved).sum()), "total": int(len(frame))})
    return {
        "policy": "New T1=Old T1; New T2=Old T2+Old T3; New T3=Old T4",
        "counts": counts,
        "interval_collapsed_but_preserved_unresolved": int(frame.three_tier_interval_collapsed.sum()),
        "beyond_tier3_possible": int(frame.beyond_tier3_possible.sum()),
        "checks": {
            "all_prompts_preserved": True,
            "unresolved_remain_unresolved": True,
            "old_four_tier_columns_preserved": True,
        },
    }


def export(source_path: Path, output_dir: Path) -> dict:
    source = pd.read_parquet(source_path)
    frame = relabel_frame(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "final_labeled_dataset_3tier.parquet", index=False)
    frame[frame.resolved].to_json(
        output_dir / "final_resolved_training_dataset_3tier.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    frame[~frame.resolved].to_parquet(output_dir / "final_unresolved_prompts_3tier.parquet", index=False)

    domain = (
        frame.groupby(["domain", "resolved", "tier"], dropna=False).size().rename("count").reset_index()
    )
    source_dist = (
        frame.groupby(["source_dataset", "resolved", "tier"], dropna=False).size().rename("count").reset_index()
    )
    domain.to_csv(output_dir / "three_tier_distribution_by_domain.csv", index=False)
    source_dist.to_csv(output_dir / "three_tier_distribution_by_source.csv", index=False)
    audit = build_audit(frame)
    (output_dir / "three_tier_export_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Create auditable 3-tier artifacts from the frozen 4-tier dataset")
    parser.add_argument("--source", default="output/final_labeled_dataset.parquet")
    parser.add_argument("--output-dir", default="output/three_tier")
    args = parser.parse_args()
    print(json.dumps(export(Path(args.source), Path(args.output_dir)), indent=2))


if __name__ == "__main__":
    main()
