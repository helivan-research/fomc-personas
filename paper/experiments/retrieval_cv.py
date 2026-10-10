"""Shared point-in-time generation utilities for the persona index.

  retrieve_weighted  top-k chunks by standardized relevance + beta * standardized recency
                     (score = z(rel) + beta * z(exp(-age / tau)); beta = 0 is pure relevance)
  index_series       per-meeting committee index for one (beta, tau): generates (and caches under
                     paper/.cache/retrieval_cv/beta{b}_tau{t}_pit/) each in-office persona's answers
                     to the 15-question battery with the as-of-date briefing, point-in-time corpus

The paper's hawk-dove index uses beta = 0 (pure relevance); the vote batteries use beta = 0.6 /
tau = 2 yr. Imported by scripts/update_index.py, cv_pit.py, cut_battery.py, vote_battery.py and
others. (This module originally also ran a pre-point-in-time beta grid; that analysis is superseded
by cv_pit.py and was removed.)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

import fomc_personas as fp                       # noqa: E402
from fomc_personas import macro, roles, persona  # noqa: E402
from fomc_personas import embeddings as _emb     # noqa: E402
import fig_index as F                            # noqa: E402

# A robust shared client (retries 429s with backoff instead of silently dropping a call -- important
# at high concurrency) and a higher worker count. gpt-4o-mini has ample headroom; 12 was leaving 3-4x
# on the table. Tune with CV_WORKERS.
WORKERS = int(os.environ.get("CV_WORKERS", 48))
if os.environ.get("OPENAI_API_KEY"):
    from openai import OpenAI
    # tight timeout so a hung request fails fast and retries instantly, instead of stalling the whole
    # 48-call meeting batch for ~2 min (each generation returns in 1-3s, so 20s = "this one hung").
    _client = OpenAI(max_retries=8, timeout=20.0)
    persona._client = _client
    _emb._client = _client

CACHE = ROOT / "paper" / ".cache" / "retrieval_cv"
CACHE.mkdir(parents=True, exist_ok=True)
TOPK = 3
# The live website window = the paper's frozen 2018-2025 list + completed post-window meetings
# (macro.completed_live_meetings: date passed AND outcome visible in the FRED target series).


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x * 0.0


def retrieve_weighted(emb, ages, q, texts, k, beta, tau):
    """Top-k chunk texts by standardized relevance + beta * standardized recency."""
    rel = emb @ np.asarray(q, dtype=np.float32).ravel()
    rec = np.exp(-np.maximum(ages, 0.0) / tau)
    score = _z(rel) + beta * _z(rec)
    top = np.argsort(-score)[:k]
    return [texts[i] for i in top]


def _gen_meeting(df_t, rost, bios, briefing, battery, qv, d, beta, tau):
    tT = pd.Timestamp(d)
    metas, messages = [], []
    for m in rost:
        g = df_t[df_t["member"] == m]
        emb = np.vstack(g["embedding"].values)
        texts = g["text"].values
        ages = (tT - pd.to_datetime(g["postedAt"], errors="coerce")).dt.days.values / 365.25
        sys_p = persona.system_prompt(m, bios.get(m, ""))
        for qi, q in enumerate(battery):
            retr = retrieve_weighted(emb, ages, qv[qi], texts, TOPK, beta, tau)
            metas.append((m, qi))
            messages.append([{"role": "system", "content": sys_p},
                             {"role": "user", "content": persona.index_prompt(battery[qi], retr, briefing)}])
    comps = persona.generate(messages, workers=WORKERS)
    resp = {m: [""] * len(battery) for m in rost}
    for (m, qi), c in zip(metas, comps):
        resp[m][qi] = c
    return resp


def index_series(df, dec, bios, u, beta, tau):
    """Conditioned committee index per meeting under recency-weighted retrieval (cached per setting)."""
    series = macro.load_fred()
    battery = fp.load_queries("curated")[:15]
    qv = fp.embed(battery)
    cdir = CACHE / f"beta{beta}_tau{tau}_pit"      # _pit: point-in-time filter (see F._public_asof)
    cdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        df_t = F._public_asof(df, d)
        rost = [m for m in F._roster(df_t) if roles.office_at(m, d) is not None]
        if not rost:
            continue
        _, briefing = macro.macro_briefing(series, d)
        cp = cdir / f"resp_{d}.json"
        if cp.exists():
            resp = json.loads(cp.read_text())
        else:
            resp = _gen_meeting(df_t, rost, bios, briefing, battery, qv, d, beta, tau)
            cp.write_text(json.dumps(resp))
            print(f"    [{beta}|{tau}] {d}: generated {sum(len([r for r in v if r]) for v in resp.values())} answers")
        # Embed all of the meeting's answers in ONE batched call (was one call per member -> ~17
        # serial round-trips/meeting; with fresh, uncached answers that serial tail dominated the
        # per-meeting time). Collect with per-member spans, embed once, then split.
        spans, texts = [], []
        for m, rs in resp.items():
            valid = [r for r in rs if r]
            if valid:
                spans.append((m, len(texts), len(texts) + len(valid)))
                texts.extend(valid)
        pos = {}
        if texts:
            embs = fp.embed(texts)
            pos = {m: float(embs[lo:hi].mean(0) @ u) for m, lo, hi in spans}
        if pos:
            out[d] = {"index": float(np.mean(list(pos.values()))), "bps": dec[d]["bps"]}
    return out
