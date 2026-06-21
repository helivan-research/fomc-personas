"""FRED macro data, the as-of-date market briefing c^(t), and realized per-meeting decisions.

All data comes from the public FRED CSV endpoint (no API key). Series are cached under the gitignored
cache so figures don't refetch.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(os.environ.get("FOMC_CACHE", Path(__file__).resolve().parent.parent / ".cache")) / "fred"

FOMC_MEETINGS = [
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]

FRED_SERIES = {
    "target": "DFEDTARU",   # fed funds target range, upper bound (daily)
    "cpi": "CPIAUCSL",      # CPI all items (monthly index) -> YoY
    "core_pce": "PCEPILFE", # core PCE price index (monthly) -> YoY
    "unrate": "UNRATE",     # unemployment rate (monthly, %)
}


def _fred(series_id, start="2016-01-01", end="2025-12-31"):
    """Fetch a FRED series as a {date: float} dict via the public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
    raw = requests.get(url, timeout=40).text
    out = {}
    for row in csv.DictReader(io.StringIO(raw)):
        date = row.get("observation_date") or row.get("DATE") or list(row.values())[0]
        val = row.get(series_id) or list(row.values())[1]
        try:
            out[date] = float(val)
        except (ValueError, TypeError):
            continue
    return out


def _fred_yearly(series_id, y0=2016, y1=2025):
    """Long daily ranges return empty from the CSV endpoint, so fetch year by year and merge."""
    out = {}
    for y in range(y0, y1 + 1):
        out.update(_fred(series_id, f"{y}-01-01", f"{y}-12-31"))
    return out


def load_fred(use_cache=True):
    """Return {name: {date: value}} for each FRED series, cached under .cache/fred/."""
    series = {}
    for k, sid in FRED_SERIES.items():
        cp = CACHE / f"{k}.json"
        if use_cache and cp.exists():
            series[k] = json.loads(cp.read_text())
            continue
        series[k] = _fred_yearly(sid) if k == "target" else _fred(sid)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(series[k]))
    return series


def _asof(s, date):
    keys = sorted(k for k in s if k <= date)
    return s[keys[-1]] if keys else None


def _yoy(monthly, date):
    keys = sorted(k for k in monthly if k <= date)
    if not keys:
        return None
    latest = keys[-1]
    y, m, d = latest.split("-")
    prior = f"{int(y) - 1}-{m}-{d}"
    if prior not in monthly:
        cand = [k for k in monthly if k.startswith(f"{int(y) - 1}-{m}")]
        if not cand:
            return None
        prior = cand[0]
    return (monthly[latest] / monthly[prior] - 1.0) * 100.0


def macro_briefing(series, date):
    """(snapshot dict, briefing string c^(t)) of as-of-date conditions for the given meeting date."""
    cpi, pce = _yoy(series["cpi"], date), _yoy(series["core_pce"], date)
    un, tgt = _asof(series["unrate"], date), _asof(series["target"], date)
    snap = {"date": date, "cpi_yoy": cpi, "core_pce_yoy": pce, "unrate": un, "target_upper": tgt}
    briefing = (
        f"Current U.S. economic conditions as of {date}: "
        f"CPI inflation is running at about {cpi:.1f}% year over year; "
        f"core PCE inflation is about {pce:.1f}% year over year (the FOMC targets 2%); "
        f"the unemployment rate is {un:.1f}%; "
        f"the current federal funds target range tops out at {tgt:.2f}%."
    )
    return snap, briefing


def decisions(series):
    """Realized per-meeting decision from the target series: {date: {target, before, bps, label}}.
    The new target is effective the day after the decision; compare pre-meeting vs ~a week later."""
    tgt = series["target"]
    out = {}
    for d in FOMC_MEETINGS:
        before = _asof(tgt, d)
        after = _asof(tgt, (pd.Timestamp(d) + pd.Timedelta(days=8)).strftime("%Y-%m-%d"))
        if before is None or after is None:
            out[d] = {"target": after, "before": before, "bps": None, "label": None}
            continue
        bps = round((after - before) * 100)
        out[d] = {"target": after, "before": before, "bps": bps,
                  "label": "hike" if bps > 0 else ("cut" if bps < 0 else "hold")}
    return out
