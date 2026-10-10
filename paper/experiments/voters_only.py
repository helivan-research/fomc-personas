"""EXPERIMENT 10 (pre-registered): average only the personas of VOTING members.

The decision is made by the ~12 voters; the canonical index averages every persona in office.
Only the committee aggregation changes: each persona's baseline (member debiasing) still uses its
full history, voting or not. Base model: S0 (canonical + zero-lower-bound rule), weights tuned
walk-forward as in sharpness.py.

  V0  all personas in office (canonical)
  V1  voting personas only (roles.is_voting at the meeting date)

Coverage caveat recorded in advance: voters-only rests on 3 personas in 2019 and 4 in early 2022
(12 in 2026).
Selection rule (declared before running): log-loss over the FULL 2019-26 record; 2022-25 and the
2026 holdout also reported.

Run:  python paper/experiments/voters_only.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

from fomc_personas import macro, roles   # noqa: E402
import probability_index as PI            # noqa: E402
import sharpness as S                     # noqa: E402


def anomalies(sup, dates, voters_only):
    """Member-debiased committee anomalies; history always updated with every member."""
    hist, sha, sca = {}, [], []
    for d in dates:
        hs, cs = [], []
        for m, (sh, sc) in sup[d].items():
            if voters_only and not roles.is_voting(m, d):
                continue
            ph = hist.get(m, {}).get("hike", [])
            pc = hist.get(m, {}).get("cut", [])
            if not np.isnan(sh):
                hs.append(sh - (np.mean(ph) if len(ph) >= PI.MIN_HIST else PI.PRIOR_HIKE))
            if not np.isnan(sc):
                cs.append(sc - (np.mean(pc) if len(pc) >= PI.MIN_HIST else PI.PRIOR_CUT))
        sha.append(np.mean(hs) if hs else 0.0)
        sca.append(np.mean(cs) if cs else 0.0)
        for m, (sh, sc) in sup[d].items():
            if not np.isnan(sh):
                hist.setdefault(m, {}).setdefault("hike", []).append(sh)
            if not np.isnan(sc):
                hist.setdefault(m, {}).setdefault("cut", []).append(sc)
    return np.array(sha), np.array(sca)


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    sup, dec = PI.load_support()
    dates = sorted(sup)
    y = np.array([int(np.sign(dec[d]["bps"])) for d in dates])
    st = PI.streaks(y)
    zlb = np.array([macro.macro_briefing(series, d)[0]["target_upper"] <= S.ZLB_CEILING for d in dates])
    masks = {"19-26": np.array(["2019" <= d for d in dates]),
             "22-25": np.array(["2022" <= d < "2026" for d in dates]),
             "2026": np.array([d >= "2026" for d in dates]),
             "19-21": np.array(["2019" <= d < "2022" for d in dates])}
    turns = ["2022-03-16", "2024-09-18", "2025-09-17", "2026-09-16"]
    print(f"{'aggregation':<22} " + " ".join(f"{'LL ' + k:>9}" for k in masks) +
          f" {'RPS 19-26':>9}  P(actual) at turns " + " ".join(t[:7] for t in turns))
    for label, vo in (("V0 all personas", False), ("V1 voters only", True)):
        sha, sca = anomalies(sup, dates, vo)
        F = np.stack([sca, np.zeros_like(sha), sha], axis=1)
        P, ch = S.run(y, st, F, zlb, "flat", (2.0,), (1.0,))
        at = " ".join(f"{P[dates.index(t), PI.CLS.index(y[dates.index(t)])]:.2f}" for t in turns)
        print(f"{label:<22} " + " ".join(f"{PI.logloss(P, y, m):>9.3f}" for m in masks.values()) +
              f" {PI.rps(P, y, masks['19-26']):>9.3f}  {at}   lambda last {ch[-1][1]:g}")


if __name__ == "__main__":
    main()
