"""EXPERIMENT 6 (pre-registered): a stronger hike channel for the probabilistic index.

Motivation (a priori, from the paper's own results): the vote-share index fixed the cut channel
but traded away hike separation -- d'(hike, hold) is 1.65 for vote shares vs 2.94 for the
hawk-dove projection. The canonical forecast (probability_index.py) also ties hike and cut
evidence to ONE tilt weight. Variants, fixed before scoring:

  A  canonical: f = [cut anomaly, 0, hike anomaly], one lambda
  B  separate weights: lambda_cut, lambda_hike tuned independently
  C  B + hawk-dove level in the hike channel: f_hike += (lambda_hd/lambda_hike) * z(hawk-dove)
     -- implemented as a third weight on a walk-forward z-scored committee projection
  D  C + hawk-dove momentum (3-meeting) in the hike channel

Selection rule (same as the paper): walk-forward log-loss over 2022-25. The 2026 live meetings,
including the 2026-09-16 hike, are a HOLDOUT -- reported, never used to choose. All weights are
tuned walk-forward on history only, grid {0, 1, 2, 4, 8} per weight.

Run:  python paper/experiments/hike_channel.py
"""
import itertools
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

import fomc_personas as fp                                    # noqa: E402
from fomc_personas import macro                               # noqa: E402
import probability_index as PI                                # noqa: E402

BETA = ROOT / "paper" / ".cache" / "retrieval_cv" / "beta0.0_tau2.0_pit"
GRID = (0.0, 1.0, 2.0, 4.0, 8.0)
CLS = PI.CLS


def zwf(x):
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        h = x[:i]
        out[i] = 0.0 if len(h) < 4 or np.std(h) < 1e-9 else (x[i] - np.mean(h)) / np.std(h)
    return np.clip(out, -3, 3)


def mom(x):
    return np.array([x[i] - np.mean(x[max(0, i - 3):i]) if i >= 1 else 0.0 for i in range(len(x))])


def forecast_multi(y, streak, channels, n_weights, start=PI.START):
    """P(a) prop. to hazard(a) * exp(sum_k w_k * channels[k][:, a]); weights tuned walk-forward
    by log-loss on t < i over GRID^n_weights."""
    combos = list(itertools.product(GRID, repeat=n_weights))
    priors = [PI.hazard_prior(t, y, streak) if t >= 1 else None for t in range(len(y))]
    F = np.stack(channels, axis=0)                              # (k, n, 3)
    P = np.full((len(y), 3), np.nan)
    chosen = []
    for i in range(start, len(y)):
        best, best_ll = combos[0], -1e18
        for w in combos:
            ll = 0.0
            for t in range(1, i):
                q = priors[t] * np.exp(np.tensordot(np.array(w), F[:, t, :], axes=1))
                ll += np.log(q[CLS.index(y[t])] / q.sum() + 1e-12)
            if ll > best_ll:
                best_ll, best = ll, w
        q = priors[i] * np.exp(np.tensordot(np.array(best), F[:, i, :], axes=1))
        P[i] = q / q.sum()
        chosen.append(best)
    return P, chosen


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    sup, dec = PI.load_support()
    dates = sorted(sup)
    sha, sca = PI.anomaly_series(sup, dates)
    y = np.array([int(np.sign(dec[d]["bps"])) for d in dates])
    st = PI.streaks(y)
    n = len(dates)

    u = fp.axis(fp.load_anchors())
    xu = []
    for d in dates:
        texts = [r for rs in json.loads((BETA / f"resp_{d}.json").read_text()).values() for r in rs if r]
        xu.append(float(fp.embed(texts).mean(0) @ u))
    xu = np.array(xu)

    z0 = np.zeros(n)
    hike_only = lambda v: np.stack([z0, z0, v], axis=1)
    cut_only = lambda v: np.stack([v, z0, z0], axis=1)
    both = np.stack([sca, z0, sha], axis=1)
    hd_level = hike_only(zwf(xu))                               # hike channel only (pre-registered)
    hd_mom = hike_only(zwf(mom(xu)))

    variants = {
        "A canonical (1 weight)":    forecast_multi(y, st, [both], 1),
        "B separate hike/cut":       forecast_multi(y, st, [hike_only(sha), cut_only(sca)], 2),
        "C B + hawk-dove level":     forecast_multi(y, st, [hike_only(sha), cut_only(sca), hd_level], 3),
        "D C + hawk-dove momentum":  forecast_multi(y, st, [hike_only(sha), cut_only(sca), hd_level, hd_mom], 4),
    }

    m22 = np.array(["2022" <= d < "2026" for d in dates])
    m26 = np.array([d >= "2026" for d in dates])
    hike22 = m22 & (y == 1)
    print(f"{'variant':<26} {'RPS 22-25':>9} {'logloss':>8} {'P(hike)|hike':>12} "
          f"{'2026 logloss':>12} {'P(hike) 2022-03-16':>18} {'P(hike) 2026-09-16':>18}  last weights")
    for name, (P, w) in variants.items():
        ph = np.nanmean(P[hike22, 2])
        i22 = dates.index("2022-03-16")
        i26 = dates.index("2026-09-16")
        print(f"{name:<26} {PI.rps(P, y, m22):>9.3f} {PI.logloss(P, y, m22):>8.3f} {ph:>12.2f} "
              f"{PI.logloss(P, y, m26):>12.3f} {P[i22, 2]:>18.2f} {P[i26, 2]:>18.2f}  {w[-1]}")


if __name__ == "__main__":
    main()
