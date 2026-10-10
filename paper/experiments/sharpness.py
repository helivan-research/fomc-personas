"""EXPERIMENT 9 (pre-registered): sharpen the forecast without manufacturing confidence.

Diagnosis: the canonical forecast is calibrated in the tails but UNDER-confident in the middle
(forecasts of 50-70% came true 77% of the time) and less sharp than persistence (Brier resolution
0.098 vs 0.112). Two suspects, both in the decision-history prior / output layer:
  - the hazard prior shrinks every continuation rate toward 0.5 with a fixed pseudo-count of 2,
    applied separately within thin (state x run-length) buckets;
  - there is no recalibration of the final probabilities.

All variants apply the ZERO-LOWER-BOUND rule (no cut when the target ceiling is <= 0.25%), an
institutional fact applied to every model including persistence. Variants (fixed before scoring):
  S0  canonical prior (pseudo-count 2 toward 0.5) + persona tilt
  S1  pseudo-count strength k tuned walk-forward, still toward 0.5
  S2  hierarchical: each run-length bucket shrinks toward ITS STATE's overall continuation rate
      (empirical Bayes) with strength k tuned walk-forward
  S3  S0 + temperature T on the final probabilities (P ** T, renormalized), tuned walk-forward
  S4  S2 + temperature T
All hyperparameters (k, lambda, T) are tuned jointly, walk-forward, by log-loss on history only.
Selection rule: log-loss over 2022-25. 2026 holdout and the full 2019-26 record reported.

Run:  python paper/experiments/sharpness.py
"""
import itertools
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

from fomc_personas import macro                       # noqa: E402
import probability_index as PI                        # noqa: E402

CLS = PI.CLS
K_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
T_GRID = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
LAM_GRID = PI.LAMBDA_GRID
ZLB_CEILING = 0.25


def prior(i, y, streak, k, mode):
    """Hazard prior with configurable shrinkage. mode 'flat' -> toward 0.5; 'eb' -> toward the
    state's overall continuation rate. Destination split pseudo-count k/4 per class (k=2 ->
    canonical 0.5)."""
    st, b = int(y[i - 1]), PI._bucket(streak[i])
    n_s = stay_s = n_b = stay_b = 0
    trans = {c: k / 4 for c in CLS if c != st}
    for t in range(1, i):
        if int(y[t - 1]) != st:
            continue
        n_s += 1
        stay_s += (y[t] == st)
        if PI._bucket(streak[t]) == b:
            n_b += 1
            stay_b += (y[t] == st)
        if y[t] != st:
            trans[int(y[t])] += 1
    target = 0.5 if mode == "flat" else (stay_s + 1.0) / (n_s + 2.0)
    p_stay = (stay_b + k * target) / (n_b + k)
    tot = sum(trans.values())
    pr = np.zeros(3)
    pr[CLS.index(st)] = p_stay
    for c, v in trans.items():
        pr[CLS.index(c)] = (1 - p_stay) * v / tot
    return pr


def finalize(pr, lam, f, T, zlb):
    q = pr * np.exp(lam * f)
    q = q / q.sum()
    q = q ** T
    if zlb:
        q[0] = 0.0
    return q / q.sum()


def run(y, streak, F, zlb, mode, k_grid, t_grid, start=PI.START):
    """Walk-forward: at meeting i choose (k, lambda, T) maximizing log-likelihood on t < i."""
    n = len(y)
    pri = {k: [prior(t, y, streak, k, mode) if t >= 1 else None for t in range(n)] for k in k_grid}
    P = np.full((n, 3), np.nan)
    chosen = []
    combos = list(itertools.product(k_grid, LAM_GRID, t_grid))
    for i in range(start, n):
        best, best_ll = combos[0], -1e18
        for k, lam, T in combos:
            ll = 0.0
            for t in range(1, i):
                q = finalize(pri[k][t], lam, F[t], T, zlb[t])
                ll += np.log(q[CLS.index(y[t])] + 1e-12)
            if ll > best_ll:
                best_ll, best = ll, (k, lam, T)
        k, lam, T = best
        P[i] = finalize(pri[k][i], lam, F[i], T, zlb[i])
        chosen.append(best)
    return P, chosen


def persistence_zlb(y, zlb, start=PI.START):
    from proper_scoring_table import persist_hedged
    P = persist_hedged(y, start)
    for i in range(len(y)):
        if zlb[i] and not np.isnan(P[i, 0]):
            P[i, 0] = 0.0
            P[i] /= P[i].sum()
    return P


def reliability(P, y, mask):
    ok = mask & ~np.isnan(P[:, 0])
    O = np.stack([(y == c).astype(float) for c in CLS], axis=1)
    p, o = P[ok].ravel(), O[ok].ravel()
    rows, rel, res = [], 0.0, 0.0
    for lo, hi in zip((0, .1, .3, .5, .7, .9), (.1, .3, .5, .7, .9, 1.0001)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            rows.append(f"{lo:.1f}-{min(hi, 1):.1f}: {p[m].mean():.2f}->{o[m].mean():.2f} (n={m.sum()})")
            rel += m.sum() * (p[m].mean() - o[m].mean()) ** 2
            res += m.sum() * (o[m].mean() - o.mean()) ** 2
    return rows, rel / len(p), res / len(p)


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    sup, dec = PI.load_support()
    dates = sorted(sup)
    sha, sca = PI.anomaly_series(sup, dates)
    y = np.array([int(np.sign(dec[d]["bps"])) for d in dates])
    st = PI.streaks(y)
    F = np.stack([sca, np.zeros_like(sha), sha], axis=1)
    zlb = np.array([macro.macro_briefing(series, d)[0]["target_upper"] <= ZLB_CEILING for d in dates])

    variants = {
        "S0 canonical + ZLB":        ("flat", (2.0,), (1.0,)),
        "S1 flat, k tuned":          ("flat", K_GRID, (1.0,)),
        "S2 hierarchical, k tuned":  ("eb", K_GRID, (1.0,)),
        "S3 S0 + temperature":       ("flat", (2.0,), T_GRID),
        "S4 S2 + temperature":       ("eb", K_GRID, T_GRID),
    }
    masks = {
        "22-25": np.array(["2022" <= d < "2026" for d in dates]),
        "2026": np.array([d >= "2026" for d in dates]),
        "19-21": np.array(["2019" <= d < "2022" for d in dates]),
        "19-26": np.array(["2019" <= d for d in dates]),
    }
    results = {"persistence + ZLB": (persistence_zlb(y, zlb), None)}
    for name, (mode, kg, tg) in variants.items():
        results[name] = run(y, st, F, zlb, mode, kg, tg)

    print(f"{'model':<26} " + " ".join(f"{'LL ' + k:>9}" for k in masks) +
          f" {'RPS 19-26':>9} {'resol.':>7} {'reliab.':>8}  last (k, lambda, T)")
    for name, (P, ch) in results.items():
        _, rel, res = reliability(P, y, masks["19-26"])
        print(f"{name:<26} " + " ".join(f"{PI.logloss(P, y, m):>9.3f}" for m in masks.values()) +
              f" {PI.rps(P, y, masks['19-26']):>9.3f} {res:>7.3f} {rel:>8.4f}  {ch[-1] if ch else ''}")
    print()
    for name in ("S0 canonical + ZLB", "S2 hierarchical, k tuned", "S4 S2 + temperature", "persistence + ZLB"):
        rows, _, _ = reliability(results[name][0], y, masks["19-26"])
        print(f"{name:<26} " + " | ".join(rows))


if __name__ == "__main__":
    main()
