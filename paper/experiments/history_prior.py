"""EXPERIMENT 13 / PHASE 0 (pre-registered): a longer decision history for the hazard prior.

Experiments 6-12 found the binding constraint is history: 2018-2025 contains no gradual, contested
first hike after a long hold like 2026's. Longer U.S. decision history does (1994, 1999, 2004,
2015, 2016). This experiment extends ONLY the decision history the hazard prior learns from --
no personas for former members, no new generation.

History = every decision before 2018: scheduled meetings (federalreserve.gov historical calendars,
"Meeting" entries) labelled from the target change across the meeting, plus intermeeting moves
(target changes on dates that are not within 2 days after a scheduled meeting) as decisions in
their own right, so the "previous decision" state is right. Target = FRED DFEDTAR (single target,
through 2008-12-15) spliced with DFEDTARU (range ceiling, from 2008-12-16).

Variants (prior history start), each hazard-only and hazard x persona tilt, zero-lower-bound rule:
  H0  2018 (current)    H1  2006    H2  1994
Evaluated on the SAME meetings as before (2019-26 full record; 2022-25 and 2026 shown).
Selection rule (declared before running): full-record log-loss. Also reports each variant's
forecast for the next meeting (2026-10-28, current live-forecast inputs).

Run:  python paper/experiments/history_prior.py
"""
import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fomc_personas import macro           # noqa: E402
import probability_index as PI            # noqa: E402
import sharpness as S                     # noqa: E402

HIST = ROOT / "paper" / ".cache" / "history"
HIST.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research; fomc-personas decision history)"}
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}
LIVE = ROOT / "paper" / ".cache" / "live_forecast"


def scheduled_meetings(y0=1994, y1=2017):
    p = HIST / f"scheduled_{y0}_{y1}.json"
    if p.exists():
        return json.loads(p.read_text())
    out = []
    for y in range(y0, y1 + 1):
        raw = requests.get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm",
                           headers=UA, timeout=40).text
        t = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)))
        for m in re.finditer(r"(January|February|March|April|May|June|July|August|September|October|"
                             r"November|December) (\d{1,2})(?:-(?:(January|February|March|April|May|June|"
                             r"July|August|September|October|November|December) )?(\d{1,2}))? Meeting - "
                             + str(y), t):
            month = m.group(3) or m.group(1)
            day = int(m.group(4) or m.group(2))
            out.append(f"{y}-{MONTHS[month]:02d}-{day:02d}")
        time.sleep(0.4)
    out = sorted(set(out))
    p.write_text(json.dumps(out))
    return out


def target_series():
    p = HIST / "target_1993_2017.json"
    if p.exists():
        return json.loads(p.read_text())
    s = {}
    for y in range(1993, 2009):
        s.update(macro._fred("DFEDTAR", f"{y}-01-01", f"{y}-12-31"))
    for y in range(2008, 2018):
        s.update(macro._fred("DFEDTARU", f"{y}-01-01", f"{y}-12-31"))
    s = {k: v for k, v in s.items() if k <= "2008-12-15" or k >= "2008-12-16"}
    p.write_text(json.dumps(s))
    return s


def history_events(start_year):
    """[(date, sign, is_scheduled, target_ceiling_before)] for decisions before 2018."""
    meetings = [d for d in scheduled_meetings() if d >= f"{start_year}-01-01"]
    tgt = target_series()
    keys = sorted(tgt)
    ev = []
    for d in meetings:
        before = macro._asof(tgt, d)
        after = macro._asof(tgt, (pd.Timestamp(d) + pd.Timedelta(days=8)).strftime("%Y-%m-%d"))
        ev.append((d, int(np.sign(round((after - before) * 100))), True, before))
    sched = [pd.Timestamp(d) for d in meetings]
    for k_prev, k in zip(keys, keys[1:]):
        if k < f"{start_year}-01-01" or tgt[k] == tgt[k_prev]:
            continue
        dk = pd.Timestamp(k)
        if any(pd.Timedelta(0) <= dk - m <= pd.Timedelta(days=2) for m in sched):
            continue                                                    # a scheduled decision
        ev.append((k, int(np.sign(tgt[k] - tgt[k_prev])), False, tgt[k_prev]))
    return sorted(ev)


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    sup, dec = PI.load_support()
    dates = sorted(sup)
    y_cur = np.array([int(np.sign(dec[d]["bps"])) for d in dates])

    # pending meeting (live forecast inputs) appended at the end, like build_with_pending
    fc = json.loads((LIVE / "forecast.json").read_text())
    tag = f"{fc['meeting']}_asof_{fc['asof']}"
    pending = PI.member_support({k: LIVE / f"{k}_parse_{tag}.json" for k in ("cut", "hike")})
    sup2 = dict(sup)
    sup2[fc["meeting"]] = pending
    dates2 = dates + [fc["meeting"]]
    sha, sca = PI.anomaly_series(sup2, dates2)
    F_cur = np.stack([sca, np.zeros_like(sha), sha], axis=1)
    zlb_cur = np.array([macro.macro_briefing(series, d)[0]["target_upper"] <= S.ZLB_CEILING for d in dates2])
    y_cur2 = np.concatenate([y_cur, [0]])                             # placeholder for the pending meeting

    masks = {"full": np.array(["2019" <= d for d in dates]),
             "22-25": np.array(["2022" <= d < "2026" for d in dates]),
             "2026": np.array([d >= "2026" for d in dates]),
             "19-21": np.array(["2019" <= d < "2022" for d in dates])}
    turns = ["2022-03-16", "2024-09-18", "2025-09-17", "2026-09-16"]

    print(f"{'prior history':<16} {'personas':<8} " + " ".join(f"{'LL ' + k:>9}" for k in masks) +
          f" {'RPS full':>9}  P(actual) at turns " + " ".join(t[:7] for t in turns) +
          f"   Oct 28 forecast (cut/hold/hike)")
    for label, start_year in (("H0 from 2018", None), ("H1 from 2006", 2006), ("H2 from 1994", 1994)):
        ev = history_events(start_year) if start_year else []
        y_h = np.array([e[1] for e in ev], dtype=int)
        zlb_h = np.array([e[3] <= S.ZLB_CEILING for e in ev], dtype=bool)
        off = len(y_h)
        y = np.concatenate([y_h, y_cur2]).astype(int)
        zlb = np.concatenate([zlb_h, zlb_cur])
        F = np.concatenate([np.zeros((off, 3)), F_cur])
        st = PI.streaks(y)
        for use_p in (False, True):
            Fx = F if use_p else np.zeros_like(F)
            P_all, ch = S.run(y, st, Fx, zlb, "flat", (2.0,), (1.0,), start=off + PI.START)
            P = P_all[off:off + len(dates)]
            nxt = P_all[off + len(dates)]
            at = " ".join(f"{P[dates.index(t), PI.CLS.index(y_cur[dates.index(t)])]:.2f}" for t in turns)
            print(f"{label:<16} {'yes' if use_p else 'no':<8} " +
                  " ".join(f"{PI.logloss(P, y_cur, m):>9.3f}" for m in masks.values()) +
                  f" {PI.rps(P, y_cur, masks['full']):>9.3f}  {at}   "
                  f"{nxt[0]:.2f}/{nxt[1]:.2f}/{nxt[2]:.2f}  (lambda {ch[-1][1]:g})")
        if start_year:
            n_sched = sum(e[2] for e in ev)
            print(f"   {start_year}-2017 history: {len(ev)} decisions ({n_sched} scheduled, "
                  f"{len(ev) - n_sched} intermeeting: {[e[0] for e in ev if not e[2]]})")


if __name__ == "__main__":
    main()
