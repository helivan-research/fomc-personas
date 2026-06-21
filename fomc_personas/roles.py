"""As-of-date FOMC roles: was a member a voting member, and the Chair, on a given date?

Both are functions of (member, date), derived from each member's office timeline, the chair tenure,
and the FOMC voting-rotation schedule. These tags are also precomputed on every chunk
(`is_voting`, `is_chair`); this module recomputes them for arbitrary dates (e.g. the per-meeting
roster used by the rate-action index).

Rotation (verified against federalreserve.gov FOMC minutes/roster, 2019-2026): the 7 governors, the
Chair, and the New York Fed president vote every year; the other 11 presidents rotate, one per group
per year, on a deterministic schedule.
"""
import datetime

D = datetime.date

# --- FOMC voting rotation: one bank per group votes each year (NY votes every year) ---
_ROTATION = {
    "Chicago":       lambda y: y % 2 == 1,   # odd years
    "Cleveland":     lambda y: y % 2 == 0,   # even years
    "Boston":        lambda y: y % 3 == 0,
    "Philadelphia":  lambda y: y % 3 == 1,
    "Richmond":      lambda y: y % 3 == 2,
    "St. Louis":     lambda y: y % 3 == 0,
    "Dallas":        lambda y: y % 3 == 1,
    "Atlanta":       lambda y: y % 3 == 2,
    "Kansas City":   lambda y: y % 3 == 0,
    "Minneapolis":   lambda y: y % 3 == 1,
    "San Francisco": lambda y: y % 3 == 2,
}

# --- per-member office timeline: sorted (start_date, office) segments ---
# office is "governor", "chair", "pres:NY" (always votes), "pres:<bank>" (per _ROTATION), or None.
_OFFICE = {
    "Jerome H. Powell":     [(D(2012, 5, 25), "governor"), (D(2018, 2, 5), "chair"),
                             (D(2026, 5, 22), "governor")],
    "Kevin M. Warsh":       [(D(2006, 2, 24), "governor"), (D(2011, 4, 2), None),
                             (D(2026, 5, 22), "chair")],
    "John C. Williams":     [(D(2011, 3, 1), "pres:San Francisco"), (D(2018, 6, 18), "pres:NY")],
    "Philip N. Jefferson":  [(D(2022, 5, 23), "governor")],
    "Michelle W. Bowman":   [(D(2018, 11, 26), "governor")],
    "Michael S. Barr":      [(D(2022, 7, 19), "governor")],
    "Lisa D. Cook":         [(D(2022, 5, 23), "governor")],
    "Christopher J. Waller":[(D(2020, 12, 18), "governor")],
    "Mary C. Daly":         [(D(2018, 10, 1), "pres:San Francisco")],
    "Neel Kashkari":        [(D(2016, 1, 1), "pres:Minneapolis")],
    "Thomas I. Barkin":     [(D(2018, 1, 4), "pres:Richmond")],
    "Lorie K. Logan":       [(D(2022, 8, 22), "pres:Dallas")],
    "Susan M. Collins":     [(D(2022, 7, 1), "pres:Boston")],
    "Austan D. Goolsbee":   [(D(2023, 1, 9), "pres:Chicago")],
    "Beth M. Hammack":      [(D(2024, 8, 21), "pres:Cleveland")],
    "Jeffrey R. Schmid":    [(D(2023, 8, 21), "pres:Kansas City")],
    "Anna Paulson":         [(D(2025, 7, 1), "pres:Philadelphia")],
}

# Chair tenure intervals [start, end) (end None = open).
_CHAIR = {
    "Jerome H. Powell": [(D(2018, 2, 5), D(2026, 5, 22))],
    "Kevin M. Warsh":   [(D(2026, 5, 22), None)],
}

MEMBERS = list(_OFFICE)


def _as_date(x):
    import pandas as pd
    if x is None or (not isinstance(x, str) and pd.isna(x)):
        return None
    if isinstance(x, datetime.datetime):
        return x.date()
    if isinstance(x, datetime.date):
        return x
    t = pd.to_datetime(x, errors="coerce", utc=True)
    return None if pd.isna(t) else t.date()


def office_at(member, date):
    """Office held by ``member`` on ``date`` (None if member/date out of range)."""
    segs = _OFFICE.get(member)
    d = _as_date(date)
    if not segs or d is None or d < segs[0][0]:
        return None
    office = None
    for start, off in segs:
        if d >= start:
            office = off
        else:
            break
    return office


def is_voting(member, date):
    off = office_at(member, date)
    if off in ("governor", "chair", "pres:NY"):
        return True
    if off and off.startswith("pres:"):
        return bool(_ROTATION[off.split(":", 1)[1]](_as_date(date).year))
    return False


def is_chair(member, date):
    d = _as_date(date)
    if d is None:
        return False
    return any(d >= start and (end is None or d < end) for start, end in _CHAIR.get(member, []))


def roster(date, voting_only=False):
    """Members holding an FOMC office on ``date`` (optionally only voters)."""
    out = [m for m in _OFFICE if office_at(m, date) is not None]
    return [m for m in out if is_voting(m, date)] if voting_only else out
