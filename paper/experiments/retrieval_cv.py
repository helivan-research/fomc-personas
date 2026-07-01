"""Cross-validate the retrieval function: how should recency trade off against relevance?

The index's personas answer a fixed battery with top-k retrieval over each member's prior statements.
Baseline retrieval ranks purely by *relevance* (embedding dot product to the query). Here we add a
*recency* term and search for the weighting that maximizes the index's out-of-sample decision-
forecasting skill (the same walk-forward hike/hold/cut accuracy the paper reports).

For a member's candidate chunks dated <= meeting date t, with relevance r_i (dot product to the query)
and age a_i (years before t), we score:

    score_i = z(r_i) + beta * z( exp(-a_i / tau) )            # z = standardize within the candidate set

and take the top-k. beta = 0 recovers pure relevance; larger beta tilts toward recent statements; tau
(years) sets how fast "recent" decays. We run a FULL generation backtest per (beta, tau) -- new
retrieval changes which quotes are shown, hence the generations -- and compare OOS accuracy. Each
setting's generations are cached under paper/.cache/retrieval_cv/beta{b}_tau{t}/ so reruns are free.

    OPENAI_API_KEY=sk-...  python paper/experiments/retrieval_cv.py 0.6 1.5        # betas, tau via --tau

This is EXPENSIVE: each new (beta, tau) is a full per-meeting generation pass (~tens of thousands of
gpt-4o-mini calls). beta=0 reuses the figure_index cache (free).
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
# Match the live website window (paper's frozen 2018-2025 list + the completed 2026 meetings).
LIVE_MEETINGS = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17"]


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
    cdir = CACHE / f"beta{beta}_tau{tau}"
    cdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        df_t = df[df["postedAt"].astype(str) <= d]
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


def _oos_acc(out):
    dates = [d for d in macro.FOMC_MEETINGS if d in out]
    idx = np.array([out[d]["index"] for d in dates])
    bps = np.array([float(out[d]["bps"]) for d in dates])
    y = np.sign(bps).astype(int)
    m22 = np.array([d >= "2022" for d in dates])
    pred = F._walkfwd([idx, F._mom(idx)], bps)
    return F._acc(pred, y, m22)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tau = float(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--tau=")), 2.0))
    betas = [float(a) for a in args] or [0.6, 1.5]

    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + LIVE_MEETINGS
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    u = fp.axis(fp.load_anchors())
    series = macro.load_fred()
    dec = macro.decisions(series)

    results = {}
    # beta = 0 (pure relevance): reuse the existing figure_index conditioned cache -- free.
    print("beta=0.0 (relevance only) from figure_index cache ...")
    base = F._index_series(df, dec, bios, u, condition=True)
    acc0 = _oos_acc({d: {"index": base[d]["index"], "bps": base[d]["bps"]} for d in base})
    results["0.0"] = {"tau": None, "oos_acc": acc0, "n": len(base)}
    print(f"  beta=0.0: OOS acc={acc0:.3f}  (n={len(base)})")

    for beta in betas:
        print(f"beta={beta} tau={tau} ...")
        out = index_series(df, dec, bios, u, beta, tau)
        acc = _oos_acc(out)
        results[f"{beta}"] = {"tau": tau, "oos_acc": acc, "n": len(out)}
        print(f"  beta={beta} tau={tau}: OOS acc={acc:.3f}  (n={len(out)})")
        (CACHE / "results.json").write_text(json.dumps(results, indent=2))

    best = max(results, key=lambda k: results[k]["oos_acc"])
    print("\n=== retrieval CV (OOS 3-class decision accuracy) ===")
    for b, r in sorted(results.items(), key=lambda kv: float(kv[0])):
        star = "  <-- best" if b == best else ""
        print(f"  beta={b:>4}  tau={r['tau']}  acc={r['oos_acc']:.3f}{star}")
    (CACHE / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
