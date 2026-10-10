"""EXPERIMENT 12 (pre-registered): recorded dissents as forecast information.

Dissents are public when the statement is released, so the PREVIOUS meeting's dissents are fair
information for forecasting the next decision. Each dissent is classified relative to the
Committee's action at that meeting:
  hawkish  wanted tighter policy (higher rate, smaller cut, larger hike, or opposed easier language)
  dovish   wanted looser policy
  neutral  not about the policy stance (e.g. balance-sheet operations)
Features for meeting i: counts of hawkish / dovish dissents at meeting i-1 -> f = [dovish, 0, hawkish].

Variants (all with the hazard prior and the zero-lower-bound rule; weights tuned walk-forward):
  D0  personas only (canonical S0)
  D1  dissents only (a stronger history-only baseline)
  D2  personas + dissents (separate weights)
Selection rule (declared before running): log-loss over the full record; 2022-25 and 2026 shown.
D2 vs D1 measures what the personas add beyond the public vote record.

Statement formats: through May 2026 the voting paragraph lists "Voting for ... Voting against ...";
from June 2026 the statement says "approved ... by a N - M vote" and names only dissenters
("Voting against ..."). Both are handled; a unanimous meeting has no dissents.

Classification fix (declared): a first pass asked the model for the lean directly; the audit against
the known record found 3 errors, all same-direction magnitude dissents (Mester 2020-03, George
2022-06, Bowman 2024-09). The model now only EXTRACTS each dissenter's preferred change in basis
points (or, for language-only dissents, whether they wanted tighter or looser wording); the lean is
computed in code by comparing with the action. Correctness fix, not tuning.

Run:  python paper/experiments/dissents.py
"""
import html
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fomc_personas import macro, persona   # noqa: E402
import probability_index as PI             # noqa: E402
import sharpness as S                      # noqa: E402

CACHE = ROOT / "paper" / ".cache" / "fomc_votes"
JUDGE = "gpt-4.1-mini"
UA = {"User-Agent": "Mozilla/5.0 (research; fomc-personas dissent features)"}
ACTION = {1: "raised the target range", 0: "kept the target range unchanged", -1: "lowered the target range"}
MU_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

PROMPT = """At this FOMC meeting the Committee {action} ({bps:+d} basis points).
Statement excerpt:
"{text}"

For EACH person who voted AGAINST the action, extract:
- "preferred_bps": the change in the target range they preferred, in signed basis points
  (e.g. "preferred to lower ... by 1/4 percentage point" -> -25; "preferred to raise ... by 1/2" -> 50;
  "supported maintaining" / "preferred no change" -> 0). null if their dissent is not about the size
  of the rate move.
- "language": only when preferred_bps is null -- "tighter" if they wanted more hawkish policy language or
  balance-sheet policy, "looser" if more dovish, "none" otherwise.
Reply with ONLY a JSON array: [{{"name": "...", "preferred_bps": <int or null>, "language": "tighter"|"looser"|"none"}}].
Reply [] if nobody voted against."""


def vote_text(d):
    """The voting / dissent text of the statement for decision date d (cached)."""
    p = CACHE / f"dissent_text_{d}.txt"
    if p.exists():
        return p.read_text()
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{d.replace('-', '')}a.htm"
    raw = requests.get(url, headers=UA, timeout=30).text
    t = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)))
    m = re.search(r"Voting (?:for|against) .{0,1500}", t)
    if m:
        txt = m.group(0)
    else:
        v = re.search(r"by an? (\d+)\s*[–-]\s*(\d+) vote", t)
        txt = f"Unanimous: {v.group(0)}" if v and v.group(2) == "0" else ""
    p.write_text(txt)
    time.sleep(0.5)
    return txt


def classify(d, bps):
    p = CACHE / f"dissent_lean_{d}.json"
    if p.exists():
        return json.loads(p.read_text())
    txt = vote_text(d)
    if not txt or txt.startswith("Unanimous") or "against" not in txt:
        out = []
    else:
        a = int(np.sign(bps))
        raw = persona.generate([[{"role": "user", "content": PROMPT.format(
            action=ACTION[a], bps=int(bps), text=txt[:1500])}]], model=JUDGE, max_tokens=400)[0]
        m = re.search(r"\[.*\]", raw or "", re.S)
        ext = json.loads(m.group(0)) if m else None
        if ext is None:
            raise SystemExit(f"could not parse dissent extraction for {d}: {raw!r}")
        out = []
        for x in ext:
            pb = x.get("preferred_bps")
            if pb is not None and int(pb) != int(bps):
                lean = "hawkish" if int(pb) > int(bps) else "dovish"
            elif pb is None:
                lean = {"tighter": "hawkish", "looser": "dovish"}.get(x.get("language"), "neutral")
            else:
                # preferred the adopted rate move itself: the dissent is about something else
                lean = {"tighter": "hawkish", "looser": "dovish"}.get(x.get("language"), "neutral")
            out.append({"name": x.get("name", ""), "preferred_bps": pb, "lean": lean})
    p.write_text(json.dumps(out))
    return out


def run(y, st, Fp, Fd, zlb, lam_grid, mu_grid, start=PI.START):
    n = len(y)
    pri = [PI.hazard_prior(t, y, st) if t >= 1 else None for t in range(n)]

    def fin(t, lam, mu):
        q = pri[t] * np.exp(lam * Fp[t] + mu * Fd[t])
        if zlb[t]:
            q[0] = 0.0
        return q / q.sum()

    P = np.full((n, 3), np.nan)
    chosen = []
    combos = list(itertools.product(lam_grid, mu_grid))
    for i in range(start, n):
        best, best_ll = combos[0], -1e18
        for lam, mu in combos:
            ll = sum(np.log(fin(t, lam, mu)[PI.CLS.index(y[t])] + 1e-12) for t in range(1, i))
            if ll > best_ll:
                best_ll, best = ll, (lam, mu)
        P[i] = fin(i, *best)
        chosen.append(best)
    return P, chosen


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    sup, dec = PI.load_support()
    dates = sorted(sup)
    y = np.array([int(np.sign(dec[d]["bps"])) for d in dates])

    # --- dissent features (and an audit listing for manual checking) ---
    hawk, dove = np.zeros(len(dates)), np.zeros(len(dates))
    print("dissent classifications (audit):")
    for i, d in enumerate(dates):
        leans = classify(d, dec[d]["bps"])
        hawk[i] = sum(x.get("lean") == "hawkish" for x in leans)
        dove[i] = sum(x.get("lean") == "dovish" for x in leans)
        if leans:
            print(f"  {d} [{dec[d]['label']:<4}] " + ", ".join(f"{x['name'].split()[-1]}={x['lean']}" for x in leans))
    prev_h = np.concatenate([[0], hawk[:-1]])
    prev_d = np.concatenate([[0], dove[:-1]])
    Fd = np.stack([prev_d, np.zeros(len(dates)), prev_h], axis=1)

    sha, sca = PI.anomaly_series(sup, dates)
    Fp = np.stack([sca, np.zeros_like(sha), sha], axis=1)
    st = PI.streaks(y)
    zlb = np.array([macro.macro_briefing(series, d)[0]["target_upper"] <= S.ZLB_CEILING for d in dates])

    masks = {"full": np.array(["2019" <= d for d in dates]),
             "22-25": np.array(["2022" <= d < "2026" for d in dates]),
             "2026": np.array([d >= "2026" for d in dates]),
             "19-21": np.array(["2019" <= d < "2022" for d in dates])}
    turns = ["2022-03-16", "2024-09-18", "2025-09-17", "2026-09-16"]
    models = {
        "D0 personas only":       run(y, st, Fp, Fd, zlb, PI.LAMBDA_GRID, (0.0,)),
        "D1 dissents only":       run(y, st, Fp, Fd, zlb, (0.0,), MU_GRID),
        "D2 personas + dissents": run(y, st, Fp, Fd, zlb, PI.LAMBDA_GRID, MU_GRID),
        "persistence":            (S.persistence_zlb(y, zlb), None),
    }
    print(f"\n{'model':<24} " + " ".join(f"{'LL ' + k:>9}" for k in masks) +
          f" {'RPS full':>9}  P(actual) at turns " + " ".join(t[:7] for t in turns) + "   last (lambda, mu)")
    for name, (P, ch) in models.items():
        at = " ".join(f"{P[dates.index(t), PI.CLS.index(y[dates.index(t)])]:.2f}" for t in turns)
        print(f"{name:<24} " + " ".join(f"{PI.logloss(P, y, m):>9.3f}" for m in masks.values()) +
              f" {PI.rps(P, y, masks['full']):>9.3f}  {at}   {ch[-1] if ch else ''}")
    i = dates.index("2026-09-16")
    for name, (P, _) in models.items():
        print(f"  {name:<24} Sep 16 2026: P(cut) {P[i,0]:.2f}  P(hold) {P[i,1]:.2f}  P(hike) {P[i,2]:.2f}")


if __name__ == "__main__":
    main()
