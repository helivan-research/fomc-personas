"""fomc-personas: digital-twin personas of the FOMC.

Load the dataset, embed/retrieve text, build personas, and reproduce the paper's figures.
"""
from .data import (
    load_chunks,
    load_bios,
    load_queries,
    load_anchors,
    load_reputational,
)
from .embeddings import embed

__all__ = [
    "load_chunks",
    "load_bios",
    "load_queries",
    "load_anchors",
    "load_reputational",
    "embed",
]
