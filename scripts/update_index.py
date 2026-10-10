#!/usr/bin/env python3
"""Weekly index update: extend the index over completed FOMC meetings and forecast the next one.

Run weekly by CI after the corpus refresh (and safe to run any time by hand):

  1. merge new FRED observations into the cached macro series;
  2. completed live meetings (macro.completed_live_meetings: date passed AND the target series
     already carries the outcome) missing from the hawk-dove caches are generated
     (retrieval_cv/beta0.0_tau2.0_pit + figure_index_pit cond/noc/vote) -- the site's index chart;
  3. the same meetings get the hike/cut vote batteries (recency retrieval) and their parses --
     the inputs of the probabilistic forecast (paper/experiments/probability_index.py);
  4. the NEXT scheduled meeting is forecast from this week's data: every persona answers both
     batteries with the corpus and briefing as of today; the vote shares tilt the decision-history
     hazard prior into P(cut) / P(hold) / P(hike). Written with the walk-forward history to
     paper/.cache/live_forecast/forecast.json;
  5. outputs for the workflow: new_meetings=true|false and changed=true|false (new meeting OR a
     new weekly forecast), so CI knows whether to commit caches and re-publish the site data.

Everything cached is skipped, so a quiet week costs one battery run (~350 short generations).
The paper's frozen FOMC_MEETINGS list and its published caches are never modified, only extended.
"""
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

import fomc_personas as fp                 # noqa: E402
from fomc_personas import macro            # noqa: E402
import fig_index as F                      # noqa: E402
import retrieval_cv as R                   # noqa: E402
import vote_battery as V                   # noqa: E402
import probability_index as PI             # noqa: E402
from cut_battery import CUT_BATTERY        # noqa: E402

BETA_DIR = R.CACHE / "beta0.0_tau2.0_pit"
LIVE_DIR = ROOT / "paper" / ".cache" / "live_forecast"
BATTERIES = {"cut": (CUT_BATTERY, V.CUT_CACHE), "hike": (V.HIKE_BATTERY, V.HIKE_CACHE)}


def _emit(**flags) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    for k, v in flags.items():
        line = f"{k}={'true' if v else 'false'}"
        print(line)
        if out:
            with open(out, "a") as f:
                f.write(line + "\n")


def next_meeting(completed, today):
    """First scheduled decision date after the last completed meeting that is on/after today."""
    last = completed[-1] if completed else ""
    for d in macro.LIVE_DECISIONS:
        if d > last and d >= today:
            return d
    return None


def main() -> None:
    today = os.environ.get("INDEX_ASOF") or _dt.date.today().isoformat()
    print("refreshing FRED ...")
    macro.refresh_fred()
    series = macro.load_fred()
    live = macro.completed_live_meetings(series, today=today)
    print("completed live meetings:", live)
    # --backfill: fill gaps for EVERY meeting (paper window + live), not just live ones -- used after
    # a corpus change to regenerate all caches (see paper/regenerate.sh).
    backfill = "--backfill" in sys.argv
    scope = list(macro.FOMC_MEETINGS) + live if backfill else live

    def _missing_hawkdove():
        miss = set()
        for d in scope:
            if not (BETA_DIR / f"resp_{d}.json").exists():
                miss.add(d)
            for tag in (f"resp_{d}_noc", f"vote_{d}"):
                if not (F.CACHE / f"{tag}.json").exists():
                    miss.add(d)
        return sorted(miss)

    def _missing_votes():
        return sorted({d for d in scope for direction in ("cut", "hike")
                       if not (V.PARSE_CACHE / f"{direction}_{d}.json").exists()})

    nxt = next_meeting(live, today)
    fc_tag = f"{nxt}_asof_{today}" if nxt else None
    need_forecast = bool(nxt) and not all(
        (LIVE_DIR / f"{direction}_parse_{fc_tag}.json").exists() for direction in ("cut", "hike"))
    missing_hd, missing_votes = _missing_hawkdove(), _missing_votes()
    print("missing hawk-dove:", missing_hd, " missing vote batteries:", missing_votes,
          " next meeting:", nxt, " forecast needed:", need_forecast)

    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + live
    dec = macro.decisions(series)
    if missing_hd or missing_votes or need_forecast:
        df = fp.load_chunks(embeddings="cached")
        bios = fp.load_bios()

        if missing_hd:
            u = fp.axis(fp.load_anchors())
            R.index_series(df, dec, bios, u, 0.0, 2.0)          # -> beta0.0_tau2.0_pit (site chart)
            F._index_series(df, dec, bios, u, condition=False)  # -> figure_index_pit resp_*_noc
            F._direct_vote(df, dec, bios)                       # -> figure_index_pit vote_*

        for d in missing_votes:                                 # backtest-identical (asof = meeting)
            for direction, (battery, cdir) in BATTERIES.items():
                rp = cdir / f"resp_{d}.json"
                V.generate_battery_at(df, bios, series, battery, d, rp)
                V.parse_file(direction, rp, V.PARSE_CACHE / f"{direction}_{d}.json")
            print(f"  vote batteries: {d}")

        if need_forecast:                                       # live: corpus + briefing as of today
            for direction, (battery, _) in BATTERIES.items():
                rp = LIVE_DIR / f"{direction}_resp_{fc_tag}.json"
                V.generate_battery_at(df, bios, series, battery, nxt, rp, asof=today)
                V.parse_file(direction, rp, LIVE_DIR / f"{direction}_parse_{fc_tag}.json")
            print(f"  live forecast batteries: {fc_tag}")

    still = [d for d in _missing_hawkdove() + _missing_votes()]
    if still:
        raise SystemExit(f"generation incomplete for {sorted(set(still))} -- not marking success")

    wrote_forecast = False
    if nxt:
        pending = PI.member_support({direction: LIVE_DIR / f"{direction}_parse_{fc_tag}.json"
                                     for direction in ("cut", "hike")})
        b = PI.build_with_pending(nxt, pending)
        P, dates = b["P"], b["dates"]
        hist = [{"date": d, "label": b["dec"][d]["label"],
                 "p_cut": round(float(P[i, 0]), 4), "p_hold": round(float(P[i, 1]), 4),
                 "p_hike": round(float(P[i, 2]), 4)}
                for i, d in enumerate(dates[:-1]) if d >= "2022" and not np.isnan(P[i, 0])]
        out = {
            "meeting": nxt, "asof": today,
            "p_cut": round(float(P[-1, 0]), 4), "p_hold": round(float(P[-1, 1]), 4),
            "p_hike": round(float(P[-1, 2]), 4),
            "s_hike": round(float(np.nanmean([v[0] for v in pending.values()])), 4),
            "s_cut": round(float(np.nanmean([v[1] for v in pending.values()])), 4),
            "n_members": len(pending), "last_decision": dates[-2], "history": hist,
        }
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        fpath = LIVE_DIR / "forecast.json"
        prev = json.loads(fpath.read_text()) if fpath.exists() else {}
        wrote_forecast = {k: prev.get(k) for k in out} != out
        fpath.write_text(json.dumps(out, indent=1))
        print(f"forecast {nxt} (as of {today}): P(cut)={out['p_cut']:.2f} "
              f"P(hold)={out['p_hold']:.2f} P(hike)={out['p_hike']:.2f}  [{len(pending)} members]")

    new = bool(missing_hd or missing_votes)
    _emit(new_meetings=new, changed=new or wrote_forecast)


if __name__ == "__main__":
    main()
