"""Table 1 of the resubmission: every baseline in its best probabilistic form, proper scoring.

Scores walk-forward forecasts of the FOMC decision (cut/hold/hike) with RPS (ordered classes)
and log-loss over 2022-25, plus:
  - RPS at decision changes and at cycle turns (2022-03-16, 2024-09-18, 2025-09-17);
  - stratification by gpt-4o-mini's late-2023 training cutoff (2022-23 vs 2024-25), the
    reviewers' contamination check.

Fairness protocol: feature-based baselines get a walk-forward multinomial logit on their own
features; every DIRECTIONAL signal additionally gets the identical hazard+tilt architecture the
persona index uses (walk-forward z-scored scalar -> f = [-z, 0, +z]), so no baseline can lose
for want of the harness. `direct-vote` is an ablation of our system (single question, beta=0,
modal votes), not an external baseline.

Run:  python paper/experiments/proper_scoring_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT / "paper" / "experiments"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import fomc_personas as fp                                     # noqa: E402
from fomc_personas import macro                                # noqa: E402
import fig_index as F                                          # noqa: E402
from probability_index import (CLS, START, build, forecast,    # noqa: E402
                               hazard_prior, logloss, rps, streaks)

FI = ROOT / "paper" / ".cache" / "figure_index_pit"
BETA = ROOT / "paper" / ".cache" / "retrieval_cv" / "beta0.0_tau2.0_pit"


def committee_proj(path, u):
    texts = [r for rs in json.loads(path.read_text()).values() for r in rs if r]
    return float(fp.embed(texts).mean(0) @ u) if texts else np.nan


def zwf(x):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        h = x[:i]
        out[i] = 0.0 if len(h) < 4 or np.std(h) < 1e-9 else (x[i] - np.mean(h)) / np.std(h)
    return np.clip(out, -3, 3)


def scalar_f(x):
    z = zwf(np.nan_to_num(x))
    return np.stack([-z, np.zeros_like(z), z], axis=1)


def wf_logit(X, y, start=START):
    from sklearn.linear_model import LogisticRegression
    X = np.nan_to_num(np.asarray(X, float))
    P = np.full((len(y), 3), np.nan)
    for i in range(start, len(y)):
        cl = np.array([(y[:i] == c).mean() for c in CLS]) + 1e-6
        if len(np.unique(y[:i])) < 3:
            P[i] = cl / cl.sum()
            continue
        mdl = LogisticRegression(max_iter=3000).fit(X[:i], y[:i])
        pr = np.zeros(3)
        for j, c in enumerate(mdl.classes_):
            pr[CLS.index(int(c))] = mdl.predict_proba(X[i:i + 1])[0][j]
        P[i] = pr
    return P


def climatology(y, start=START):
    P = np.full((len(y), 3), np.nan)
    for i in range(start, len(y)):
        cl = np.array([(y[:i] == c).mean() for c in CLS]) + 1e-6
        P[i] = cl / cl.sum()
    return P


def persist_hedged(y, start=START):
    P = np.full((len(y), 3), np.nan)
    for i in range(start, len(y)):
        clim = np.array([(y[:i] == c).mean() for c in CLS]) + 1e-6
        best_e, best_ll = 0.1, -1e18
        for e in (0.02, 0.05, 0.1, 0.2, 0.3):
            ll = 0.0
            for t in range(1, i):
                pr = e * clim / clim.sum()
                pr[CLS.index(int(y[t - 1]))] += 1 - e
                ll += np.log(pr[CLS.index(y[t])] + 1e-12)
            if ll > best_ll:
                best_ll, best_e = ll, e
        pr = best_e * clim / clim.sum()
        pr[CLS.index(int(y[i - 1]))] += 1 - best_e
        P[i] = pr
    return P


def main():
    b = build()
    dates, y, sha, sca = b["dates"], b["y"], b["sha"], b["sca"]
    st = b["streak"]
    n = len(dates)
    mom = lambda x: np.array([x[i] - np.mean(x[max(0, i - 3):i]) if i >= 1 else 0.0 for i in range(len(x))])

    series = macro.load_fred()
    dec = b["dec"]
    u = fp.axis(fp.load_anchors())
    xu = np.array([committee_proj(BETA / f"resp_{d}.json", u) for d in dates])
    noc = np.array([committee_proj(FI / f"resp_{d}_noc.json", u) for d in dates])
    LAB = {"hike": 1, "hold": 0, "cut": -1}
    dv_h, dv_c = [], []
    for d in dates:
        votes = json.loads((FI / f"vote_{d}.json").read_text())
        vs = [LAB.get(v, 0) for v in votes.values()]
        dv_h.append(np.mean([v == 1 for v in vs]))
        dv_c.append(np.mean([v == -1 for v in vs]))
    dv_h, dv_c = np.array(dv_h), np.array(dv_c)
    snaps = [macro.macro_briefing(series, d)[0] for d in dates]
    cpi = np.array([s["cpi_yoy"] for s in snaps])
    pce = np.array([s["core_pce_yoy"] for s in snaps])
    un = np.array([s["unrate"] for s in snaps])
    cur = np.array([s["target_upper"] for s in snaps])
    gap = 2 + pce + 0.5 * (pce - 2) - (un - 4.4) - cur
    print("computing retrieved-text index (embeddings cached) ...")
    df = fp.load_chunks(embeddings="cached")
    ridx_d = F._retrieved_index(df, dec, u)
    ridx = np.array([ridx_d.get(d, np.nan) for d in dates])

    def trailing_center(x, prior):
        return x - np.array([np.mean(x[:i]) if i >= 4 else prior for i in range(len(x))])

    f_ours = np.stack([sca, np.zeros(n), sha], axis=1)
    f_dv = np.stack([trailing_center(dv_c, 0.15), np.zeros(n), trailing_center(dv_h, 0.5)], axis=1)

    MODELS = {
        "climatology":              climatology(y),
        "persistence (hedged)":     persist_hedged(y),
        "hazard prior only":        forecast(y, st, np.zeros((n, 3)))[0],
        "Taylor logit":             wf_logit(np.column_stack([gap]), y),
        "Taylor hazard+tilt":       forecast(y, st, scalar_f(gap))[0],
        "macro logit":              wf_logit(np.column_stack([cpi, pce, un, cur, mom(cpi), mom(pce), mom(un)]), y),
        "retrieved hazard+tilt":    forecast(y, st, scalar_f(ridx))[0],
        "static hazard+tilt":       forecast(y, st, scalar_f(noc))[0],
        "hawk-dove hazard+tilt":    forecast(y, st, scalar_f(xu))[0],
        "direct-vote (ablation)":   forecast(y, st, f_dv)[0],
        "PBI-vote (ours)":          b["P"],
    }
    m22 = np.array([d >= "2022" for d in dates])
    pre = m22 & np.array([d < "2024" for d in dates])      # in-training-window (cutoff late 2023)
    post = np.array([d >= "2024" for d in dates])          # post-cutoff
    chg = np.concatenate([[False], y[1:] != y[:-1]])
    turns = np.array([d in ("2022-03-16", "2024-09-18", "2025-09-17") for d in dates])

    hdr = (f"{'model':<24} {'RPS':>6} {'logloss':>8} {'RPS@chg':>8} {'RPS@turn':>8} "
           f"{'RPS pre-cut.':>12} {'RPS post-cut.':>13}")
    print("\n" + hdr)
    for name, P in MODELS.items():
        print(f"{name:<24} {rps(P, y, m22):>6.3f} {logloss(P, y, m22):>8.3f} "
              f"{rps(P, y, m22 & chg):>8.3f} {rps(P, y, m22 & turns):>8.3f} "
              f"{rps(P, y, pre):>12.3f} {rps(P, y, post):>13.3f}")


if __name__ == "__main__":
    main()
