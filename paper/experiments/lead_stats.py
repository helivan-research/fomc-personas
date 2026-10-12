"""Lead-lag statistics for the paper (reviewer requests R1-W2, R4-W3).

For the hawk-dove index (paper Figure 4d) and the net vote-share index, over the paper window:
  - Kendall tau of index(t) with the target-rate LEVEL at t+s for s in -2..12 meetings;
  - multiplicity correction for reporting the PEAK over that scan: the null distribution of the
    maximum tau under a stationary block bootstrap of the index (mean block 6 meetings, target held
    fixed), B = 2000;
  - a rate-persistence control: tau between index(t) and target(t+s) after both are residualized on
    target(t) (linear), i.e. does the index anticipate the FUTURE level beyond today's level?

Run:  python paper/experiments/lead_stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import fomc_personas as fp                 # noqa: E402
from fomc_personas import macro            # noqa: E402
import probability_index as PI            # noqa: E402

BETA = ROOT / "paper" / ".cache" / "retrieval_cv" / "beta0.0_tau2.0_pit"
SHIFTS = np.arange(-2, 13)
B, MEAN_BLOCK, SEED = 2000, 6, 7


def lead_curve(x, tgt):
    out = []
    for s in SHIFTS:
        a, b = (x[:len(x) - s or None], tgt[s:]) if s >= 0 else (x[-s:], tgt[:s])
        out.append(kendalltau(a, b).correlation)
    return np.array(out)


def partial_curve(x, tgt):
    out = []
    for s in SHIFTS:
        if s <= 0:
            out.append(np.nan)
            continue
        a, b, c = x[:len(x) - s], tgt[s:], tgt[:len(tgt) - s]
        ra = a - np.polyval(np.polyfit(c, a, 1), c)
        rb = b - np.polyval(np.polyfit(c, b, 1), c)
        out.append(kendalltau(ra, rb).correlation)
    return np.array(out)


def max_tau_null(x, tgt, rng):
    n, mx = len(x), []
    for _ in range(B):
        idx, pos = [], rng.integers(n)
        while len(idx) < n:
            L = rng.geometric(1 / MEAN_BLOCK)
            idx.extend(((pos + np.arange(L)) % n).tolist())
            pos = rng.integers(n)
        mx.append(np.nanmax(lead_curve(x[np.array(idx[:n])], tgt)))
    return np.array(mx)


def main():
    series = macro.load_fred()
    dec = macro.decisions(series)
    u = fp.axis(fp.load_anchors())
    sup, _ = PI.load_support()                         # paper window (2018-2025)
    dates = [d for d in sorted(sup) if (BETA / f"resp_{d}.json").exists()]
    xu, net = [], []
    for d in dates:
        texts = [r for rs in json.loads((BETA / f"resp_{d}.json").read_text()).values() for r in rs if r]
        xu.append(float(fp.embed(texts).mean(0) @ u))
        vals = list(sup[d].values())
        net.append(np.nanmean([a for a, _ in vals]) - np.nanmean([b for _, b in vals]))
    xu, net = np.array(xu), np.array(net)
    tgt = np.array([macro.macro_briefing(series, d)[0]["target_upper"] for d in dates])
    rng = np.random.default_rng(SEED)
    print(f"{len(dates)} meetings ({dates[0]} .. {dates[-1]}); shifts {SHIFTS[0]}..{SHIFTS[-1]}; B={B}")
    for name, x in (("hawk-dove index", xu), ("net vote-share index", net)):
        lc = lead_curve(x, tgt)
        k = int(SHIFTS[np.nanargmax(lc)])
        null = max_tau_null(x, tgt, rng)
        p = float((null >= np.nanmax(lc)).mean())
        pc = partial_curve(x, tgt)
        print(f"--- {name}")
        print(f"  contemporaneous tau {lc[list(SHIFTS).index(0)]:+.2f}; peak {np.nanmax(lc):+.2f} at +{k} meetings; "
              f"multiplicity-corrected p = {p:.4f} (p < {1 / B:.4f} if 0; null 95th pct {np.percentile(null, 95):.2f})")
        print("  partial on target(t) @ +2/+4/+6/+8: " +
              "  ".join(f"{pc[list(SHIFTS).index(s)]:+.2f}" for s in (2, 4, 6, 8)))


if __name__ == "__main__":
    main()
