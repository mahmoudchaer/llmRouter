from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from .build_baseline3_model import build, parameter_report
from .hierarchical_chunking import chunk_token_ids
from .streaming_gradient import streaming_request_backward


def structural_vector(token_count: int, chunk_count: int, character_count: int, device):
    values = [
        math.log1p(token_count), math.log1p(character_count), math.log1p(1),
        math.log1p(chunk_count), 0.0, math.log1p(token_count), 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ]
    return torch.tensor(values, dtype=torch.bfloat16, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/final_labeled_dataset.parquet")
    ap.add_argument("--config", default="configs/baseline3.yaml")
    ap.add_argument("--split", default="splits/grouped_split_17.json")
    ap.add_argument("--output", default="reports/baseline3_cuda_seed17_prep.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    df = pd.read_parquet(args.data)
    split = json.loads(Path(args.split).read_text())["datasets"]
    df["partition"] = df.source_dataset.map(split)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["model_id"], revision=cfg["model"]["revision"]
    )
    lengths = [len(tokenizer.encode(str(x), add_special_tokens=False)) for x in df.prompt]
    df["token_count"] = lengths
    chunk_size = cfg["request"]["chunk_size_tokens"] = 2048
    overlap = cfg["request"]["chunk_overlap_tokens"] = 128
    eos = tokenizer.eos_token_id

    quantile_targets = {
        "short": 0.50, "medium": 0.90, "long": 0.99, "very_long": 1.0,
    }
    selected = []
    for label, q in quantile_targets.items():
        target = float(df.token_count.quantile(q))
        idx = (df.token_count - target).abs().idxmin()
        selected.append((label, df.loc[idx]))

    cfg["training"]["mode"] = "lora"
    model = build(cfg, device="cuda", aggregation="mean", structural=True)
    model.train()
    results = []
    torch.cuda.reset_peak_memory_stats()
    total_tokens = 0
    total_seconds = 0.0
    for i, (label, row) in enumerate(selected):
        ids = tokenizer.encode(str(row.prompt), add_special_tokens=False)
        chunks = chunk_token_ids(ids, chunk_size, overlap, eos)
        structural = structural_vector(len(ids), len(chunks), len(str(row.prompt)), "cuda")
        domain_label = torch.tensor(0, device="cuda")
        tier_label = torch.tensor(int(row.tier) - 1 if bool(row.resolved) else -1, device="cuda")
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses = streaming_request_backward(
            model, chunks, structural, domain_label, tier_label,
            tokenizer.pad_token_id or eos, max_tokens=2048, seed=1700 + i,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        processed = sum(map(len, chunks)) * 2  # no-grad pass plus gradient replay
        total_tokens += processed
        total_seconds += elapsed
        results.append({
            "sample": label, "prompt_id": str(row.prompt_id),
            "source_dataset": str(row.source_dataset), "raw_tokens": len(ids),
            "chunks": len(chunks), "processed_encoder_tokens": processed,
            "seconds": elapsed, "tokens_per_second": processed / elapsed,
            "loss": losses["loss"],
        })

    train = df[df.partition == "train"]
    val = df[df.partition == "validation"]
    test = df[df.partition == "test"]
    stride = chunk_size - 1 - overlap
    train_chunks = ((train.token_count.clip(lower=1) - overlap).clip(lower=1) / stride).apply(math.ceil)
    total_train_chunk_tokens = int(sum(
        sum(map(len, chunk_token_ids(tokenizer.encode(str(text), add_special_tokens=False), chunk_size, overlap, eos)))
        for text in train.prompt
    ))
    effective_requests = cfg["training"]["gradient_accumulation_requests"]
    report = {
        "model": cfg["model"], "parameters": parameter_report(model),
        "device": torch.cuda.get_device_name(0),
        "bf16": torch.cuda.is_bf16_supported(),
        "samples": results,
        "aggregate_tokens_per_second": total_tokens / total_seconds,
        "aggregate_requests_per_second": len(results) / total_seconds,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "seed17": {
            "train_examples": len(train), "validation_examples": len(val),
            "test_examples": len(test), "resolved_train_examples": int(train.resolved.sum()),
            "train_chunks": int(train_chunks.sum()),
            "train_chunk_tokens_one_encoder_pass": total_train_chunk_tokens,
            "streaming_encoder_tokens_per_epoch": 2 * total_train_chunk_tokens,
            "optimizer_steps_per_epoch_estimate": math.ceil(len(train) / effective_requests),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
