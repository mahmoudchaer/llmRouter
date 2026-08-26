from __future__ import annotations

import numpy as np
from .aggregate_context import aggregate_context


def semantic_feature(task_embedding: np.ndarray, context_embeddings: np.ndarray,
                     temperature: float = 0.10) -> np.ndarray:
    task = np.asarray(task_embedding, dtype=np.float32)
    if len(context_embeddings) == 0:
        # Benchmark short-prompt path: no threefold duplication.
        return np.concatenate([task, np.zeros_like(task), np.zeros_like(task)])
    mean, relevant, _ = aggregate_context(task, context_embeddings, temperature)
    return np.concatenate([task, mean - task, relevant - task]).astype(np.float32)


def combine_features(semantic: np.ndarray, structural_scaled: np.ndarray,
                     domain_probs: np.ndarray | None = None) -> np.ndarray:
    blocks = [np.asarray(semantic), np.asarray(structural_scaled)]
    if domain_probs is not None: blocks.append(np.asarray(domain_probs))
    return np.concatenate(blocks, axis=-1).astype(np.float32)

