"""OpenAI text embeddings with a local hash cache.

Used for chunk stances, queries, anchors, and generated persona responses alike. Every text is
embedded at most once: results are keyed by a hash of (model, dimensions, text) and persisted under
the gitignored cache, so reruns and texts shared across figures are free.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

MODEL = "text-embedding-3-large"
DIM = 1024

_CACHE_DIR = Path(os.environ.get("FOMC_CACHE", Path(__file__).resolve().parent.parent / ".cache"))
_EMB_CACHE = _CACHE_DIR / "embeddings"
_client = None


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily so non-API figures (fig_data) need no key
        _client = OpenAI()  # reads OPENAI_API_KEY
    return _client


def _key(text: str) -> str:
    return hashlib.sha1(f"{MODEL}:{DIM}:{text}".encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    # shard by first 2 hex chars to avoid huge directories
    return _EMB_CACHE / key[:2] / f"{key}.npy"


def embed(texts, batch_size: int = 256) -> np.ndarray:
    """Embed a list of strings -> (n, 1024) float32 array, in input order. Cached on disk."""
    if isinstance(texts, str):
        texts = [texts]
    texts = list(texts)
    keys = [_key(t) for t in texts]
    out: list[np.ndarray | None] = [None] * len(texts)

    # load whatever is cached
    missing = []
    for i, k in enumerate(keys):
        p = _cache_path(k)
        if p.exists():
            out[i] = np.load(p)
        else:
            missing.append(i)

    # embed the misses in batches
    for s in range(0, len(missing), batch_size):
        idx = missing[s:s + batch_size]
        resp = _openai().embeddings.create(
            model=MODEL, dimensions=DIM, input=[texts[i] for i in idx]
        )
        for j, i in enumerate(idx):
            v = np.asarray(resp.data[j].embedding, dtype=np.float32)
            out[i] = v
            p = _cache_path(keys[i])
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, v)

    return np.vstack(out).astype(np.float32)
