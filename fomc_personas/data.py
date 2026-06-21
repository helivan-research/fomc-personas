"""Load the FOMC-personas dataset and its config files.

The chunk corpus + embeddings live as a Hugging Face dataset (`helivan/fomc-personas`, sharded by
member). The small config files (bios, query sets, anchors, reputational ordering) ship in this repo
under `data/`.

For local development before the HF dataset exists, set `FOMC_LOCAL_DATASET` to a directory laid out
like the HF repo (`<dir>/chunks/*.parquet`, `<dir>/embeddings/*.parquet`); the loader reads those
shards directly and makes no network calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

HF_REPO = os.environ.get("FOMC_HF_REPO", "helivan/fomc-personas")
_DATA = Path(os.environ.get("FOMC_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
_LOCAL = os.environ.get("FOMC_LOCAL_DATASET")  # optional dir of parquet shards (dev)


def _load_config(config: str) -> pd.DataFrame:
    """Return the `chunks` or `embeddings` table as a DataFrame (local shards if set, else HF)."""
    if _LOCAL:
        shards = sorted(Path(_LOCAL).glob(f"{config}/*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no {config}/*.parquet under FOMC_LOCAL_DATASET={_LOCAL}")
        return pd.concat((pd.read_parquet(p) for p in shards), ignore_index=True)
    from datasets import load_dataset  # lazy: only needed for the HF path
    return load_dataset(HF_REPO, config, split="train").to_pandas()


def load_chunks(embeddings: str = "cached") -> pd.DataFrame:
    """Load the chunk corpus as a DataFrame.

    embeddings:
      "none"    -> text + metadata only (no API key, no large download).
      "cached"  -> attach precomputed embeddings (downloaded from the HF dataset).
      "compute" -> embed each chunk's stance with OpenAI text-embedding-3-large (cached locally).
    """
    df = _load_config("chunks").sort_values("chunk_id").reset_index(drop=True)
    if embeddings == "none":
        return df
    if embeddings == "cached":
        emb = _load_config("embeddings")[["chunk_id", "embedding"]]
        return df.merge(emb, on="chunk_id", how="left")
    if embeddings == "compute":
        from .embeddings import embed
        df = df.copy()
        df["embedding"] = list(embed(df["stance"].fillna("").tolist()))
        return df
    raise ValueError(f"embeddings must be 'none'|'cached'|'compute', got {embeddings!r}")


# --- small config files shipped in this repo's data/ -------------------------------------------

def load_bios() -> dict:
    """{member: biography} — each persona's system prompt."""
    return json.loads((_DATA / "bios.json").read_text())


def load_queries(which: str = "curated"):
    """which='curated' -> the 30 monetary-policy questions (list[str]);
    which='pool' -> the 72-query facet pool (list of {q, subtopic, dove_end, hawk_end})."""
    name = {"curated": "curated.json", "pool": "pool_72.json"}[which]
    return json.loads((_DATA / "queries" / name).read_text())


def load_anchors() -> dict:
    """{'hawk': [...], 'dove': [...]} canonical anchor statements defining the projection axis."""
    return json.loads((_DATA / "queries" / "anchors.json").read_text())


def load_reputational() -> list:
    """External news-derived hawk->dove ordering of the 16 scored members (most hawkish first)."""
    return json.loads((_DATA / "reputational_ordering.json").read_text())
