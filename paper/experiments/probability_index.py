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


HISTORY_FROM = 2006      # decision history prepended to the hazard prior (experiment 13 / Phase 0)
ZLB_CEILING = 0.25       # no cut when the target ceiling is at/below this (zero lower bound)
HIST = ROOT / "paper" / ".cache" / "history"


def history_events(start_year=HISTORY_FROM):
    """Decisions before 2018 from the committed history cache (built by history_prior.py):
    scheduled meetings labelled by the target change across the meeting, plus intermeeting moves.
    Returns (signs, zlb_flags) in date order."""
    import pandas as pd
    if start_year is None:
        return np.zeros(0, int), np.zeros(0, bool)
    meetings = json.loads((HIST / "scheduled_1994_2017.json").read_text())
    tgt = json.loads((HIST / "target_1993_2017.json").read_text())
    meetings = [d for d in meetings if d >= f"{start_year}-01-01"]
    keys = sorted(tgt)
    ev = []
    for d in meetings:
        before = macro._asof(tgt, d)
        after = macro._asof(tgt, (pd.Timestamp(d) + pd.Timedelta(days=8)).strftime("%Y-%m-%d"))
        ev.append((d, int(np.sign(round((after - before) * 100))), before))
    sched = [pd.Timestamp(d) for d in meetings]
    for k_prev, k in zip(keys, keys[1:]):
        if k < f"{start_year}-01-01" or tgt[k] == tgt[k_prev]:
            continue
        dk = pd.Timestamp(k)
        if any(pd.Timedelta(0) <= dk - m <= pd.Timedelta(days=2) for m in sched):
            continue
        ev.append((k, int(np.sign(tgt[k] - tgt[k_prev])), tgt[k_prev]))
    ev.sort()
    return np.array([e[1] for e in ev], int), np.array([e[2] <= ZLB_CEILING for e in ev], bool)


def zlb_flags(dates, series=None):
    series = series or macro.load_fred()
    return np.array([macro.macro_briefing(series, d)[0]["target_upper"] <= ZLB_CEILING for d in dates])


def history_context(y_cur, zlb_cur, start_year=HISTORY_FROM):
    """Prepend pre-2018 decision history. Returns (y_full, zlb_full, offset)."""
    y_h, z_h = history_events(start_year)
    return (np.concatenate([y_h, np.asarray(y_cur, int)]),
            np.concatenate([z_h, np.asarray(zlb_cur, bool)]), len(y_h))


def with_history(y_cur, fmat_cur, zlb_cur, start=START, lam_grid=None):
    """The canonical forecast: hazard prior learned over the prepended history x persona tilt,
    zero-lower-bound rule; returns (P for the current window, selected lambdas)."""
    y, z, off = history_context(y_cur, zlb_cur)
    F = np.concatenate([np.zeros((off, 3)), np.asarray(fmat_cur, float)])
    P, lams = forecast(y, streaks(y), F, start=off + start, zlb=z,
                       **({"lam_grid": lam_grid} if lam_grid is not None else {}))
    return P[off:], lams


def load_support():
    """{date: {member: (s_hike, s_cut)}} from the cached battery parses."""
    series = macro.load_fred()
    dec = macro.decisions(series)
    sup = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        row = member_support({direction: PARSE_CACHE / f"{direction}_{d}.json"
                              for direction in ("hike", "cut")})
        if row:
            sup[d] = row
    return sup, dec


def member_support(parse_paths):
    """{member: (s_hike, s_cut)} from parse files {"hike": path, "cut": path} (missing ok)."""
    row = {}
    for direction, p in parse_paths.items():
        if not Path(p).exists():
            continue
        for m, qs in json.loads(Path(p).read_text()).items():
            clear = [v for v in qs.values() if v != "unclear"]
            if clear:
                row.setdefault(m, {})[direction] = float(np.mean([v == "support" for v in clear]))
    return {m: (v.get("hike", np.nan), v.get("cut", np.nan)) for m, v in row.items()}


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


def zwf(x):
    """Walk-forward z-score: each value standardized by the mean/std of the values BEFORE it
    (0 until 4 observations), clipped to [-3, 3]."""
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        h = x[:i]
        out[i] = 0.0 if len(h) < 4 or np.std(h) < 1e-9 else (x[i] - np.mean(h)) / np.std(h)
    return np.clip(out, -3, 3)


def mom(x):
    """3-meeting momentum: value minus the mean of the previous (up to) three values."""
    return np.array([x[i] - np.mean(x[max(0, i - 3):i]) if i >= 1 else 0.0 for i in range(len(x))])


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


def _zero_cut(p):
    q = p.copy()
    q[0] = 0.0
    return q / q.sum()


def forecast(y, streak, fmat, start=START, lam_grid=LAMBDA_GRID, zlb=None):
    """Walk-forward hazard+tilt forecast; returns (P[n,3], selected lambdas). With `zlb` (bool per
    meeting), cut probability is zero at the lower bound, in fitting and in forecasting."""
    P = np.full((len(y), 3), np.nan)
    lams = []
    pri = [hazard_prior(t, y, streak) if t >= 1 else None for t in range(len(y))]

    def p_at(t, lam):
        q = tilt(pri[t], lam, fmat[t])
        return _zero_cut(q) if zlb is not None and zlb[t] else q

    for i in range(start, len(y)):
        best_lam, best_ll = 0.0, -1e18
        for lam in lam_grid:
            ll = sum(np.log(p_at(t, lam)[CLS.index(y[t])] + 1e-12) for t in range(1, i))
            if ll > best_ll:
                best_ll, best_lam = ll, lam
        lams.append(best_lam)
        P[i] = p_at(i, best_lam)
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
    zlb = zlb_flags(dates)
    P, lams = with_history(y, fmat, zlb)
    return {"dates": dates, "y": y, "bps": bps, "sha": sha, "sca": sca,
            "streak": st, "zlb": zlb, "P": P, "lams": lams, "dec": dec}


def build_with_pending(pending_date, pending_support):
    """build() plus one not-yet-decided meeting appended at the end.

    The pending meeting's forecast uses only history before it (the hazard prior reads y[:i],
    lambda is tuned on t < i, member baselines on earlier meetings), so the placeholder outcome
    appended for it never enters its own forecast."""
    sup, dec = load_support()
    dates = sorted(sup)
    sup = dict(sup)
    sup[pending_date] = pending_support
    all_dates = dates + [pending_date]
    sha, sca = anomaly_series(sup, all_dates)
    y = np.array([int(np.sign(dec[d]["bps"])) for d in dates] + [0])
    fmat = np.stack([sca, np.zeros_like(sha), sha], axis=1)
    zlb = zlb_flags(all_dates)
    P, lams = with_history(y, fmat, zlb)
    return {"dates": all_dates, "y": y, "sha": sha, "sca": sca, "P": P, "lams": lams, "dec": dec}


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
