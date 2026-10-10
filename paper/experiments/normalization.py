"""EXPERIMENT 7 (pre-registered): how to normalize persona vote shares without erasing signal.

Diagnosis (2026-09-16 hike): raw committee hike support rose from 43% (Mar 2026) to 70-72%
(Jun-Sep 2026) -- a hawkish turn three months before the hike -- but the canonical
normalization (each persona minus its EXPANDING all-history mean) measured it against the
2022-23 hike era, when support ran ~95%, so the evidence entering the forecast stayed near zero.

Methods (fixed before scoring; the 8-meeting window = one year, not tuned):
  N0  member - expanding member mean                 (canonical)
  N1  member - member's last-8-meeting mean
  N2  member last-8-meeting z-score (std floored at 0.05)
  N3  within-member contrast: s_hike - s_cut, same persona, same meeting (no history)
  N4  N3 - committee's last-8-meeting mean of N3

Everything else is the canonical forecast: hazard(state, run-length) prior x exp(lambda * f),
one lambda tuned walk-forward by log-loss. Per-member channels: N0-N2 give separate hike / cut
anomalies (f = [cut, 0, hike]); N3-N4 give one signed contrast (f = [-c, 0, +c]).
Selection rule: walk-forward log-loss over 2022-25. 2026 (incl. the 2026-09-16 hike) is a
HOLDOUT, reported only.

EXPERIMENT 8 (added AFTER seeing N0-N4, declared as such): N0 won 2022-25 but N1 won the 2026
holdout -- long memory suits the 2022-25 cycles, short memory the milder 2026 regime. Rather than
choose on the holdout:
  N5  both channels (N0 all-history and N1 last-8), each with its own lambda, tuned walk-forward
      on history only (grid = probability_index.LAMBDA_GRID per weight).
Same rule: 2022-25 log-loss decides; 2026 holdout reported only.

Run:  python paper/experiments/normalization.py
"""
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

from fomc_personas import macro                    # noqa: E402
import probability_index as PI                     # noqa: E402
import hike_channel as HC                          # noqa: E402

WIN, MIN_HIST, STD_FLOOR = 8, 3, 0.05
PRIOR = {"hike": PI.PRIOR_HIKE, "cut": PI.PRIOR_CUT}


def member_channels(sup, dates, method):
    """Committee-level features per meeting under one normalization method.
    Returns an (n, 3) feature matrix f = [cut, hold(0), hike]. History is updated AFTER use."""
    hist = {}                                      # member -> {"hike": [...], "cut": [...], "net": [...]}
    contrast_hist = []                             # committee N3 history (for N4)
    F = np.zeros((len(dates), 3))
    for i, d in enumerate(dates):
        hk, ct, nets = [], [], []
        for m, (sh, sc) in sup[d].items():
            h = hist.get(m, {})
            for key, val, out in (("hike", sh, hk), ("cut", sc, ct)):
                if np.isnan(val):
                    continue
                past = h.get(key, [])
                if method == "N0":
                    base = np.mean(past) if len(past) >= MIN_HIST else PRIOR[key]
                    out.append(val - base)
                elif method == "N1":
                    w = past[-WIN:]
                    base = np.mean(w) if len(w) >= MIN_HIST else PRIOR[key]
                    out.append(val - base)
                elif method == "N2":
                    w = past[-WIN:]
                    if len(w) >= MIN_HIST:
                        out.append((val - np.mean(w)) / max(np.std(w), STD_FLOOR))
                    else:
                        out.append((val - PRIOR[key]) / 0.25)
            if not (np.isnan(sh) or np.isnan(sc)):
                nets.append(sh - sc)
        if method in ("N0", "N1", "N2"):
            F[i] = [np.mean(ct) if ct else 0.0, 0.0, np.mean(hk) if hk else 0.0]
        else:
            c = float(np.mean(nets)) if nets else 0.0
            if method == "N4":
                w = contrast_hist[-WIN:]
                c = c - (np.mean(w) if len(w) >= MIN_HIST else 0.0)
            contrast_hist.append(float(np.mean(nets)) if nets else 0.0)
            F[i] = [-c, 0.0, c]
        for m, (sh, sc) in sup[d].items():
            h = hist.setdefault(m, {})
            if not np.isnan(sh):
                h.setdefault("hike", []).append(sh)
            if not np.isnan(sc):
                h.setdefault("cut", []).append(sc)
    return F


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    sup, dec = PI.load_support()
    dates = sorted(sup)
    y = np.array([int(np.sign(dec[d]["bps"])) for d in dates])
    st = PI.streaks(y)

    m22 = np.array(["2022" <= d < "2026" for d in dates])
    m26 = np.array([d >= "2026" for d in dates])
    turns = ["2022-03-16", "2024-09-18", "2025-09-17", "2026-09-16"]
    print(f"{'method':<34} {'RPS 22-25':>9} {'logloss':>8} {'P(hike)|hike':>12} {'P(cut)|cut':>10} "
          f"{'2026 logloss':>12}  P(actual) at turns " + " ".join(t[:7] for t in turns))
    names = {"N0": "N0 member vs all-history (current)", "N1": "N1 member vs last 8 meetings",
             "N2": "N2 member last-8 z-score", "N3": "N3 within-member hike-cut contrast",
             "N4": "N4 N3 vs committee last 8"}
    names["N5"] = "N5 N0 + N1 (two weights)"
    HC.GRID = PI.LAMBDA_GRID
    for meth, label in names.items():
        if meth == "N5":
            P, ws = HC.forecast_multi(y, st, [member_channels(sup, dates, "N0"),
                                              member_channels(sup, dates, "N1")], 2)
            lams = [f"{a:g}/{b:g}" for a, b in ws]
        else:
            F = member_channels(sup, dates, meth)
            P, lams = PI.forecast(y, st, F)
        hk = m22 & (y == 1)
        ct = m22 & (y == -1)
        at_turns = " ".join(f"{P[dates.index(t), PI.CLS.index(y[dates.index(t)])]:.2f}" for t in turns)
        print(f"{label:<34} {PI.rps(P, y, m22):>9.3f} {PI.logloss(P, y, m22):>8.3f} "
              f"{np.nanmean(P[hk, 2]):>12.2f} {np.nanmean(P[ct, 0]):>10.2f} {PI.logloss(P, y, m26):>12.3f}  "
              f"{at_turns}   (lambda {'last ' + lams[-1] if meth == 'N5' else 'median %g' % np.median(lams)})")


if __name__ == "__main__":
    main()
