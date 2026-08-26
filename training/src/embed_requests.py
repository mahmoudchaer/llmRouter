from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np


def token_chunks(token_ids: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    if not token_ids:
        return [[]]
    step = chunk_size - overlap
    return [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), step)]


def cache_fingerprint(config: dict) -> str:
    keys = {k: config.get(k) for k in (
        "model_id", "revision", "pooling", "normalize", "safe_max_tokens",
        "chunk_size_tokens", "chunk_overlap_tokens", "task_instruction", "device",
        "dtype", "mps_dtype"
    )}
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:16]


class QwenEmbedder:
    """Lazy loader: importing the pipeline never downloads model weights."""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._tokenizer = self._model = None

    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg["model_id"], revision=self.cfg["revision"], padding_side="left", trust_remote_code=False)
        requested=self.cfg.get("device","auto")
        if requested=="auto":
            if torch.cuda.is_available(): device="cuda"
            elif torch.backends.mps.is_available(): device="mps"
            else: device="cpu"
        else: device=requested
        dtype_name=self.cfg.get("mps_dtype","float16") if device=="mps" else self.cfg.get("dtype","bfloat16")
        dtype={"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}[dtype_name]
        self._model = AutoModel.from_pretrained(
            self.cfg["model_id"], revision=self.cfg["revision"], torch_dtype=dtype,
            trust_remote_code=False).to(device).eval()
        self.runtime={"device":device,"dtype":dtype_name}
        for p in self._model.parameters(): p.requires_grad_(False)

    def task_input(self, text: str) -> str:
        return f'Instruct: {self.cfg["task_instruction"]}\nQuery:{text}'

    def tokenize_task(self, text: str) -> list[int]:
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer=AutoTokenizer.from_pretrained(
                self.cfg["model_id"],revision=self.cfg["revision"],padding_side="left",trust_remote_code=False)
        return self._tokenizer.encode(self.task_input(text),add_special_tokens=True)

    def tokenize_text(self, text: str) -> list[int]:
        if self._tokenizer is None: self.tokenize_task("")
        return self._tokenizer.encode(text,add_special_tokens=False)

    def decode_tokens(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids,skip_special_tokens=False)

    @staticmethod
    def _last_token_pool(hidden, attention_mask):
        import torch
        lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        if self._model is None: self.load()
        batch = self._tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if int(batch["attention_mask"].sum(1).max()) > self.cfg["safe_max_tokens"]:
            raise ValueError("Input exceeds safe limit; chunk before encoding")
        device = next(self._model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            out = self._model(**batch)
            emb = self._last_token_pool(out.last_hidden_state, batch["attention_mask"])
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)


def save_cache(path: Path, ids: np.ndarray, embeddings: np.ndarray, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prompt_id=ids, embedding=embeddings)
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
