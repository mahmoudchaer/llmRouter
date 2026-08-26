from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def aggregate_context(task: np.ndarray, chunks: np.ndarray, temperature: float = 0.10):
    task = l2_normalize(np.asarray(task).reshape(1, -1))[0]
    chunks = l2_normalize(np.asarray(chunks))
    if len(chunks) == 0:
        z = np.zeros_like(task)
        return z, z, np.empty(0, dtype=np.float32)
    mean = l2_normalize(chunks.mean(axis=0, keepdims=True))[0]
    similarities = chunks @ task
    logits = (similarities - similarities.max()) / temperature
    weights = np.exp(logits); weights /= weights.sum()
    relevant = l2_normalize((weights[:, None] * chunks).sum(axis=0, keepdims=True))[0]
    return mean, relevant, weights.astype(np.float32)

