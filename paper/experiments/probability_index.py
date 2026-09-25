"""The canonical probabilistic persona index: hazard prior x member-debiased vote-share tilt.

The per-meeting forecast P(hike), P(hold), P(cut) for the upcoming FOMC decision:

  1. VOTE SHARES. Each persona answers the timing-explicit hike and cut batteries
     (cut_battery.py / vote_battery.py, recency retrieval beta=0.6/tau=2yr); answers are parsed
     into support/oppose votes (cached under .cache/vote_parse/).
  2. MEMBER DEBIASING. Personas over-endorse action (acquiescence bias, roughly stable per
     member). Each member's support share is centered on their own trailing mean (walk-forward;
     priors 0.5 hike / 0.15 cut until 3 observations). Committee anomaly = mean over members.
  3. HAZARD PRIOR. Decision-history base rate: empirical continuation probability by
     (previous decision, run-length bucket 1-2/3-5/6+), Laplace-smoothed, with historical
     change-destination split. Strictly richer than persistence (= the state alone); phase- and
     magnitude-augmented variants were checked and do not help (and do not move lambda).
  4. TILT. P(a) prop. to prior(a) * exp(lambda * f_a), f = [cut anomaly, 0, hike anomaly];
     lambda tuned walk-forward by log-loss (grid below; selected lambda = 4 at every step).

Everything is walk-forward from meeting index START; scores are proper (RPS for the ordered
classes cut < hold < hike, and log-loss). See proper_scoring_table.py for the full baseline
comparison and the pre/post-training-cutoff stratification.

Run:  python paper/experiments/probability_index.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fomc_personas import macro          # noqa: E402

PARSE_CACHE = ROOT / "paper" / ".cache" / "vote_parse"
CLS = [-1, 0, 1]                          # cut < hold < hike (ordered)
START = 16                                # first forecasted meeting index (matches the paper)
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
PRIOR_HIKE, PRIOR_CUT, MIN_HIST = 0.5, 0.15, 3


def load_support():
    """{date: {member: (s_hike, s_cut)}} from the cached battery parses."""
    series = macro.load_fred()
    dec = macro.decisions(series)
    sup = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        row = {}
        for direction in ("hike", "cut"):
            p = PARSE_CACHE / f"{direction}_{d}.json"
            if not p.exists():
                continue
            for m, qs in json.loads(p.read_text()).items():
                clear = [v for v in qs.values() if v != "unclear"]
                if clear:
                    row.setdefault(m, {})[direction] = float(np.mean([v == "support" for v in clear]))
        if row:
            sup[d] = {m: (v.get("hike", np.nan), v.get("cut", np.nan)) for m, v in row.items()}
    return sup, dec


def anomaly_series(sup, dates):
    """Walk-forward member-debiased committee anomalies (s~_hike, s~_cut) per meeting."""
    hist = {}
    sha, sca = [], []
    for d in dates:
        hs, cs = [], []
        for m, (sh, sc) in sup[d].items():
            ph = hist.get(m, {}).get("hike", [])
            pc = hist.get(m, {}).get("cut", [])
            if not np.isnan(sh):
                hs.append(sh - (np.mean(ph) if len(ph) >= MIN_HIST else PRIOR_HIKE))
            if not np.isnan(sc):
                cs.append(sc - (np.mean(pc) if len(pc) >= MIN_HIST else PRIOR_CUT))
        sha.append(np.mean(hs) if hs else 0.0)
        sca.append(np.mean(cs) if cs else 0.0)
        for m, (sh, sc) in sup[d].items():           # update history AFTER use
            if not np.isnan(sh):
                hist.setdefault(m, {}).setdefault("hike", []).append(sh)
            if not np.isnan(sc):
                hist.setdefault(m, {}).setdefault("cut", []).append(sc)
    return np.array(sha), np.array(sca)


def streaks(y):
    s = np.zeros(len(y), int)
    for i in range(1, len(y)):
        s[i] = s[i - 1] + 1 if y[i - 1] == (y[i - 2] if i >= 2 else y[i - 1]) else 1
    return s


def _bucket(s):
    return 0 if s <= 2 else (1 if s <= 5 else 2)


def hazard_prior(i, y, streak):
    """P(y_i) from history < i: continuation by (state, run-length bucket) + destination split."""
    st, b = int(y[i - 1]), _bucket(streak[i])
    cont_n, cont_k = 1.0, 2.0                        # Laplace toward 0.5
    trans = {c: 0.5 for c in CLS if c != st}
    for t in range(1, i):
        if int(y[t - 1]) == st and _bucket(streak[t]) == b:
            cont_k += 1
            cont_n += (y[t] == st)
        if int(y[t - 1]) == st and y[t] != st:
            trans[int(y[t])] = trans.get(int(y[t]), 0.5) + 1
    p_stay = cont_n / cont_k
    tot = sum(trans.values())
    pr = np.zeros(3)
    pr[CLS.index(st)] = p_stay
    for c, v in trans.items():
        pr[CLS.index(c)] = (1 - p_stay) * v / tot
    return pr


def tilt(pr, lam, f):
    q = pr * np.exp(lam * f)
    return q / q.sum()


def forecast(y, streak, fmat, start=START, lam_grid=LAMBDA_GRID):
    """Walk-forward hazard+tilt forecast; returns (P[n,3], selected lambdas)."""
    P = np.full((len(y), 3), np.nan)
    lams = []
    for i in range(start, len(y)):
        best_lam, best_ll = 0.0, -1e18
        for lam in lam_grid:
            ll = sum(np.log(tilt(hazard_prior(t, y, streak), lam, fmat[t])[CLS.index(y[t])] + 1e-12)
                     for t in range(1, i))
            if ll > best_ll:
                best_ll, best_lam = ll, lam
        lams.append(best_lam)
        P[i] = tilt(hazard_prior(i, y, streak), best_lam, fmat[i])
    return P, lams


def rps(P, y, mask):
    tot, n = 0.0, 0
    for i in np.where(mask & ~np.isnan(P[:, 0]))[0]:
        o = np.zeros(3)
        o[CLS.index(y[i])] = 1
        tot += np.sum((np.cumsum(P[i]) - np.cumsum(o)) ** 2)
        n += 1
    return tot / n if n else float("nan")


def logloss(P, y, mask):
    idx = np.where(mask & ~np.isnan(P[:, 0]))[0]
    return -np.mean([np.log(P[i][CLS.index(y[i])] + 1e-12) for i in idx]) if len(idx) else float("nan")


def build():
    """Assemble everything; returns a dict of the series and the forecast."""
    sup, dec = load_support()
    dates = sorted(sup)
    sha, sca = anomaly_series(sup, dates)
    bps = np.array([float(dec[d]["bps"]) for d in dates])
    y = np.sign(bps).astype(int)
    fmat = np.stack([sca, np.zeros_like(sha), sha], axis=1)
    st = streaks(y)
    P, lams = forecast(y, st, fmat)
    return {"dates": dates, "y": y, "bps": bps, "sha": sha, "sca": sca,
            "streak": st, "P": P, "lams": lams, "dec": dec}


def main():
    b = build()
    dates, y, P = b["dates"], b["y"], b["P"]
    m22 = np.array([d >= "2022" for d in dates])
    chg = np.concatenate([[False], y[1:] != y[:-1]])
    print(f"meetings: {len(dates)} ({dates[0]}..{dates[-1]});  "
          f"lambda: median {np.median(b['lams']):.0f}, all={sorted(set(b['lams']))}")
    print(f"RPS 2022-25:      {rps(P, y, m22):.3f}")
    print(f"log-loss 2022-25: {logloss(P, y, m22):.3f}")
    print(f"RPS at changes:   {rps(P, y, m22 & chg):.3f}")
    print("\nlatest forecasts:")
    for i in range(len(dates) - 3, len(dates)):
        print(f"  {dates[i]} [{b['dec'][dates[i]]['label']:<4}]  "
              f"P(cut)={P[i,0]:.2f}  P(hold)={P[i,1]:.2f}  P(hike)={P[i,2]:.2f}")


if __name__ == "__main__":
    main()
