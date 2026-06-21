"""fomc-personas: digital-twin personas of the FOMC.

Load the dataset, embed/retrieve text, build personas, recover stance, and reproduce the paper.
"""
from .data import (
    load_chunks, load_bios, load_queries, load_anchors, load_reputational,
)
from .embeddings import embed
from .persona import retrieve, respond, generate, system_prompt
from .stance import (
    axis, project, kendall_vs_external,
    meanpool_corpus, meanpool_retrieved, meanpool_generated,
)
from . import roles, macro, likeness

__all__ = [
    "load_chunks", "load_bios", "load_queries", "load_anchors", "load_reputational",
    "embed", "retrieve", "respond", "generate", "system_prompt",
    "axis", "project", "kendall_vs_external",
    "meanpool_corpus", "meanpool_retrieved", "meanpool_generated",
    "roles", "macro", "likeness",
]
