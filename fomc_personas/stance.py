"""Hawk-dove stance: a fixed dove->hawk axis, member representations, and projection onto it.

A member's stance is the projection of a member representation onto a fixed, model-independent
dove->hawk axis built from canonical anchor statements. The same axis is held constant across the
three representations (corpus / retrieved / generated), so only the representation varies. Validated
by Kendall's tau-b against an external news-derived hawk->dove reputational ordering.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sstats

from .embeddings import embed


def axis(anchors: dict) -> np.ndarray:
    """Unit hawk-minus-dove direction from the anchor statements ({'hawk': [...], 'dove': [...]})."""
    u = embed(anchors["hawk"]).mean(0) - embed(anchors["dove"]).mean(0)
    return (u / (np.linalg.norm(u) + 1e-12)).astype(np.float32)


def project(reps: dict, u: np.ndarray) -> dict:
    """{member: <representation, axis>} — a scalar stance, higher = more hawkish."""
    return {m: float(np.asarray(v) @ u) for m, v in reps.items()}


def kendall_vs_external(scores: dict, external: list):
    """Rank members by score (desc = hawkish); Kendall tau-b vs the external hawk->dove ordering.
    Returns (ranked_members, tau, p)."""
    ranked = sorted(scores, key=lambda n: -scores[n])
    ext_rank = {n: i for i, n in enumerate(external)}
    pr = {n: i for i, n in enumerate(ranked)}
    common = [n for n in ranked if n in ext_rank]
    tau, p = sstats.kendalltau([pr[n] for n in common], [ext_rank[n] for n in common])
    return ranked, float(tau), float(p)


# --- the three member representations ---------------------------------------

def meanpool_corpus(df) -> dict:
    """{member: mean of all of the member's chunk embeddings} (not query-conditioned)."""
    return {m: np.vstack(g["embedding"].values).mean(0) for m, g in df.groupby("member")}


def meanpool_retrieved(df, queries, k: int) -> dict:
    """{member: mean over queries of the top-k retrieved chunk embeddings} (no generation)."""
    qv = embed(list(queries))
    out = {}
    for m, g in df.groupby("member"):
        emb = np.vstack(g["embedding"].values)
        out[m] = np.mean([emb[np.argsort(-(emb @ qv[i]))[:k]].mean(0)
                          for i in range(len(queries))], axis=0)
    return out


def meanpool_generated(responses: dict) -> dict:
    """{member: mean embedding of the member's generated persona responses}."""
    return {m: embed([r or " " for r in rs]).mean(0) for m, rs in responses.items()}
