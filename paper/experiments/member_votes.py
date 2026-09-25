"""EXPERIMENT 5: member-level vote prediction against the recorded FOMC votes.

Ground truth: the voting paragraph of each post-meeting statement
(federalreserve.gov/newsevents/pressreleases/monetary{YYYYMMDD}a.htm) names every voter and
every dissent with the dissenter's preferred action. This is the sharpest testable form of the
digital-twin claim: does the persona of the member who dissented, dissent?

Persona vote: from the (already-cached) recency vote batteries, per member-meeting
  net_m(t) = share(hike-battery answers endorsing) - share(cut-battery answers endorsing).

FIXED evaluation rules (chosen once, before scoring):
  - predicted member vote: hike if net_m > +1/3, cut if net_m < -1/3, else hold
    (symmetric thirds of the [-1,1] range; not tuned);
  - member's true vote: realized action sign for majority voters; a dissenter's sign is their
    preferred direction (magnitude-only dissents -- smaller/larger same-direction moves -- keep
    the action's sign and are flagged, not treated as direction errors);
  - populations: (member, meeting) pairs where the member is a LISTED VOTER and has a persona;
    dissent ranking is within the persona-covered voters of that meeting;
  - threshold-free check: AUC of -net_m for "votes cut" and of net_m for "votes hike".

Run:  python paper/experiments/member_votes.py   (scrape + parse cached per meeting; then eval)
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT / "paper" / "experiments"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fomc_personas import macro, persona   # noqa: E402

VOTE_CACHE = ROOT / "paper" / ".cache" / "fomc_votes"
VOTE_CACHE.mkdir(parents=True, exist_ok=True)
PARSE_CACHE = ROOT / "paper" / ".cache" / "vote_parse"
UA = {"User-Agent": "Mozilla/5.0 (research; fomc-personas member-vote evaluation)"}

PARSE_PROMPT = (
    "Below is the voting paragraph from an FOMC statement. List every voter and how they voted.\n\n"
    "Paragraph: \"{para}\"\n\n"
    "Reply with ONLY a JSON array, one object per person: "
    '{{"name": "<full name as written>", "vote": "for"|"against", '
    '"preferred": null | "raise" | "lower" | "no change" | "larger move" | "smaller move"}}. '
    "\"preferred\" is null for majority voters; for dissenters it is the action they preferred."
)


def fetch_voting_paragraph(d: str) -> str | None:
    hp = VOTE_CACHE / f"raw_{d}.txt"
    if hp.exists():
        return hp.read_text() or None
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{d.replace('-', '')}a.htm"
    try:
        html = requests.get(url, headers=UA, timeout=30).text
    except Exception as e:
        print(f"  ! fetch failed {d}: {e}")
        return None
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    # Take a generous window from "Voting for" (sentence-boundary regexes break on the periods in
    # middle initials, e.g. "Jerome H. Powell"); the LLM parser ignores any trailing boilerplate.
    m = re.search(r"Voting for .{0,1200}", text)
    para = m.group(0).strip() if m else ""
    hp.write_text(para)
    time.sleep(0.5)                                   # polite to federalreserve.gov
    return para or None


def parse_votes(d: str, para: str) -> list | None:
    pp = VOTE_CACHE / f"votes_{d}.json"
    if pp.exists():
        return json.loads(pp.read_text())
    out = persona.generate([[{"role": "user", "content": PARSE_PROMPT.format(para=para)}]],
                           max_tokens=800)[0]
    m = re.search(r"\[.*\]", out or "", re.S)
    if not m:
        print(f"  ! parse failed {d}")
        return None
    votes = json.loads(m.group(0))
    pp.write_text(json.dumps(votes))
    return votes


PREF_SIGN = {"raise": 1, "lower": -1, "no change": 0}


def member_net(d: str) -> dict:
    """Per-member net vote share from the cached battery parses at meeting d."""
    out = {}
    for direction, sign in (("hike", 1), ("cut", -1)):
        p = PARSE_CACHE / f"{direction}_{d}.json"
        if not p.exists():
            continue
        for m, qs in json.loads(p.read_text()).items():
            clear = [v for v in qs.values() if v != "unclear"]
            if clear:
                out.setdefault(m, {})[direction] = float(np.mean([v == "support" for v in clear]))
    return {m: v.get("hike", 0.0) - v.get("cut", 0.0) for m, v in out.items()
            if "hike" in v or "cut" in v}


def main():
    series = macro.load_fred()
    dec = macro.decisions(series)

    rows = []                     # (date, member, true_sign, net_m, is_dissent, magnitude_only)
    dissent_ranks = []
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        para = fetch_voting_paragraph(d)
        if not para:
            continue
        votes = parse_votes(d, para)
        if not votes:
            continue
        nets = member_net(d)
        act = int(np.sign(dec[d]["bps"]))
        meeting_rows = []
        for v in votes:
            name = (v.get("name") or "").strip()
            match = next((m for m in nets if m.split()[-1] == name.split()[-1]), None)
            if match is None:
                continue
            pref = v.get("preferred")
            if v.get("vote") == "against" and pref in PREF_SIGN and PREF_SIGN[pref] != act:
                true_sign, dis, mag = PREF_SIGN[pref], True, False   # direction dissent
            elif v.get("vote") == "against":
                # magnitude-only or non-rate dissent (same preferred direction as the action,
                # e.g. a smaller cut, or a balance-sheet disagreement): keeps the action's sign
                true_sign, dis, mag = act, True, True
            else:
                true_sign, dis, mag = act, False, False
            meeting_rows.append((d, match, true_sign, nets[match], dis, mag))
        rows.extend(meeting_rows)
        # dissent ranking within this meeting's persona-covered voters
        for (dd, m, ts, nm, dis, mag) in meeting_rows:
            if dis and not mag:
                nets_here = [r[3] for r in meeting_rows]
                if ts < act:      # dissented toward cutting: is their net among the lowest?
                    rank = 1 + sum(1 for x in nets_here if x < nm)
                else:
                    rank = 1 + sum(1 for x in nets_here if x > nm)
                dissent_ranks.append((dd, m, "cut" if ts < act else "hike", rank, len(nets_here)))

    dates = sorted({r[0] for r in rows})
    print(f"meetings with votes+personas: {len(dates)}   member-vote pairs: {len(rows)}")
    tr = np.array([r[2] for r in rows]); net = np.array([r[3] for r in rows])
    pred = np.where(net > 1/3, 1, np.where(net < -1/3, -1, 0))
    m22 = np.array([r[0] >= "2022" for r in rows])
    print(f"member-vote 3-class acc (all): {(pred == tr).mean():.3f}   "
          f"(2022-25): {(pred[m22] == tr[m22]).mean():.3f}")
    base = max((tr[m22] == c).mean() for c in (-1, 0, 1))
    print(f"member-level base rate 22-25: {base:.3f}")

    def auc(scores, labels):
        pos, neg = scores[labels], scores[~labels]
        if not len(pos) or not len(neg):
            return float("nan")
        return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    print(f"AUC(net_m -> votes hike): {auc(net[m22], tr[m22] == 1):.3f}")
    print(f"AUC(-net_m -> votes cut): {auc(-net[m22], tr[m22] == -1):.3f}")

    print(f"\ndirection dissents captured (rank of dissenter's persona among {'{n}'} covered voters, 1=most extreme toward dissent):")
    from math import log
    from scipy.stats import chi2
    chi, k = 0.0, 0
    for dd, m, drc, rank, n in dissent_ranks:
        print(f"  {dd}  {m:<24} toward {drc:<4}  rank {rank}/{n}")
        chi += -2.0 * log(rank / n)          # Fisher combination of per-dissent rank p-values
        k += 1
    if k:
        p = float(chi2.sf(chi, 2 * k))
        print(f"combined (Fisher over {k} direction dissents): p = {p:.4f}")


if __name__ == "__main__":
    main()
