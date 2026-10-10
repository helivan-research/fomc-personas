"""EXPERIMENT 11 (pre-registered): walk-forward isotonic recalibration of the forecast.

Base: S0 (canonical forecast + zero-lower-bound rule; sharpness.py). Its probabilities are
calibrated in the tails but under-confident mid-range (50-70% realizes ~85%).

  RAW       S0 as is
  ISO       at meeting i, fit IsotonicRegression on all PAST (forecast, outcome) pairs, the three
            outcomes pooled one-vs-rest, clipped to [0.02, 0.98]; map each class probability,
            renormalize, re-apply the zero-lower-bound rule. Starts once 10 past forecasts exist.
  ISO-blend 0.5 * ISO + 0.5 * RAW (hedge against step-function overfitting at small n)

All compared on the SAME meetings (those with >= 10 prior forecasts), with persistence.
Selection rule (declared before running): log-loss over the full record; 2022-25, 2026 also shown.

Run:  python paper/experiments/isotonic.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

from fomc_personas import macro           # noqa: E402
import probability_index as PI            # noqa: E402
import sharpness as S                     # noqa: E402

MIN_PAST, CLIP = 10, (0.02, 0.98)


def isotonic_walkforward(P, y, zlb):
    O = np.stack([(y == c).astype(float) for c in PI.CLS], axis=1)
    have = np.where(~np.isnan(P[:, 0]))[0]
    Q = np.full_like(P, np.nan)
    for k, i in enumerate(have):
        past = have[:k]
        if len(past) < MIN_PAST:
            continue
        iso = IsotonicRegression(y_min=CLIP[0], y_max=CLIP[1], out_of_bounds="clip")
        iso.fit(P[past].ravel(), O[past].ravel())
        q = iso.predict(P[i])
        if zlb[i]:
            q[0] = 0.0
        Q[i] = q / q.sum()
    return Q


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    sup, dec = PI.load_support()
    dates = sorted(sup)
    sha, sca = PI.anomaly_series(sup, dates)
    y = np.array([int(np.sign(dec[d]["bps"])) for d in dates])
    st = PI.streaks(y)
    F = np.stack([sca, np.zeros_like(sha), sha], axis=1)
    zlb = np.array([macro.macro_briefing(series, d)[0]["target_upper"] <= S.ZLB_CEILING for d in dates])

    raw, _ = S.run(y, st, F, zlb, "flat", (2.0,), (1.0,))
    iso = isotonic_walkforward(raw, y, zlb)
    blend = 0.5 * iso + 0.5 * raw
    pers = S.persistence_zlb(y, zlb)

    common = ~np.isnan(iso[:, 0])
    first = dates[np.where(common)[0][0]]
    masks = {"full": common,
             "22-25": common & np.array(["2022" <= d < "2026" for d in dates]),
             "2026": common & np.array([d >= "2026" for d in dates])}
    turns = ["2022-03-16", "2024-09-18", "2025-09-17", "2026-09-16"]
    print(f"common evaluation set: {common.sum()} meetings from {first}")
    print(f"{'forecast':<14} " + " ".join(f"{'LL ' + k:>9}" for k in masks) +
          f" {'RPS full':>9} {'resol.':>7} {'reliab.':>8}  P(actual) at turns " + " ".join(t[:7] for t in turns))
    for name, P in (("RAW (S0)", raw), ("ISO", iso), ("ISO-blend", blend), ("persistence", pers)):
        rows, rel, res = S.reliability(P, y, common)
        at = " ".join(f"{P[dates.index(t), PI.CLS.index(y[dates.index(t)])]:.2f}" for t in turns)
        print(f"{name:<14} " + " ".join(f"{PI.logloss(P, y, m):>9.3f}" for m in masks.values()) +
              f" {PI.rps(P, y, common):>9.3f} {res:>7.3f} {rel:>8.4f}  {at}")
    print()
    for name, P in (("RAW (S0)", raw), ("ISO", iso), ("ISO-blend", blend)):
        rows, _, _ = S.reliability(P, y, common)
        print(f"{name:<14} " + " | ".join(rows))


if __name__ == "__main__":
    main()
