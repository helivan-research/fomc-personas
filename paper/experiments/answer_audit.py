"""Qualitative audit of persona answers: grounding, boilerplate, and factual error rates.

Samples answers from the (recency) vote batteries, recomputes each answer's actual retrieval
context (deterministic), and has a JUDGE model different from the generator (gpt-4.1-mini vs
gpt-4o-mini -- no self-grading) code each answer:

  grounded    -- takes a position consistent with the member's retrieved statements and the
                 briefing's numbers; no invented specifics
  boilerplate -- generic central-bank language; not anchored in the member's retrieved record
  error       -- contradicts the retrieved statements or the briefing, or invents specifics
                 (figures, votes, events) not present in either

Writes rates to stdout and a human-readable sample (audit_sample.txt) for the appendix and for
manual spot-checking.

Run:  python paper/experiments/answer_audit.py [n_samples]
"""
import json
import random
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

import pandas as pd                                    # noqa: E402
import fomc_personas as fp                             # noqa: E402
from fomc_personas import macro, persona               # noqa: E402
import fig_index as F                                  # noqa: E402
from retrieval_cv import retrieve_weighted             # noqa: E402
from cut_battery import CUT_BATTERY                    # noqa: E402
from vote_battery import HIKE_BATTERY                  # noqa: E402

JUDGE = "gpt-4.1-mini"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
BETA, TAU, TOPK = 0.6, 2.0, 3
BATTERIES = {"cut": (CUT_BATTERY, ROOT / "paper/.cache/cut_battery_b0.6"),
             "hike": (HIKE_BATTERY, ROOT / "paper/.cache/hike_battery_b0.6")}

JUDGE_PROMPT = """You are auditing a retrieval-augmented persona of a named FOMC member.

The persona was shown these retrieved statements the member actually made:
{retrieved}

And this market briefing:
{briefing}

It was asked: "{question}"
It answered: "{answer}"

Code the answer with exactly one label:
- "grounded": the position is consistent with the retrieved statements and the briefing's numbers; no invented specifics.
- "boilerplate": generic central-bank language; could have been said by anyone; not anchored in the retrieved record.
- "error": contradicts the retrieved statements or the briefing, or invents specifics (figures, votes, events) found in neither.

Reply with JSON only: {{"label": "...", "reason": "<one sentence>"}}"""


def main():
    rng = random.Random(11)
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    series = macro.load_fred()
    dec = macro.decisions(series)

    pool = []
    for direction, (battery, cdir) in BATTERIES.items():
        for p in sorted(cdir.glob("resp_*.json")):
            d = p.stem[5:]
            for m, answers in json.loads(p.read_text()).items():
                for qi, a in enumerate(answers):
                    if a:
                        pool.append((direction, d, m, qi, a))
    sample = rng.sample(pool, min(N, len(pool)))

    metas, messages = [], []
    for direction, d, m, qi, a in sample:
        battery = BATTERIES[direction][0]
        df_t = F._public_asof(df, d)
        g = df_t[df_t["member"] == m]
        emb = np.vstack(g["embedding"].values)
        texts = g["text"].values
        ages = (pd.Timestamp(d) - pd.to_datetime(g["postedAt"], errors="coerce")).dt.days.values / 365.25
        qv = fp.embed([battery[qi]])[0]
        retr = retrieve_weighted(emb, ages, qv, texts, TOPK, beta=BETA, tau=TAU)
        _, briefing = macro.macro_briefing(series, d)
        metas.append((direction, d, m, qi, a, retr, briefing))
        messages.append([{"role": "user", "content": JUDGE_PROMPT.format(
            retrieved="\n".join(f"- {t}" for t in retr), briefing=briefing,
            question=battery[qi], answer=a)}])
    outs = persona.generate(messages, model=JUDGE, max_tokens=120)

    rows, counts = [], {"grounded": 0, "boilerplate": 0, "error": 0, "unparsed": 0}
    for (direction, d, m, qi, a, retr, briefing), o in zip(metas, outs):
        try:
            j = json.loads(o[o.index("{"):o.rindex("}") + 1])
            lab = j.get("label", "unparsed")
            reason = j.get("reason", "")
        except Exception:
            lab, reason = "unparsed", ""
        counts[lab] = counts.get(lab, 0) + 1
        rows.append((direction, d, m, qi, a, retr, lab, reason))

    n_ok = sum(counts[k] for k in ("grounded", "boilerplate", "error"))
    print(f"audited {len(rows)} answers (judge: {JUDGE})")
    for k in ("grounded", "boilerplate", "error"):
        print(f"  {k:>11}: {counts[k]:>3}  ({counts[k] / max(n_ok, 1):.0%})")
    if counts["unparsed"]:
        print(f"  {'unparsed':>11}: {counts['unparsed']}")

    out = ROOT / "paper" / ".cache" / "answer_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(
        [{"direction": r[0], "date": r[1], "member": r[2], "q": r[3], "answer": r[4],
          "label": r[6], "reason": r[7]} for r in rows], indent=1))
    with open(out / "audit_sample.txt", "w") as f:
        for r in rng.sample(rows, min(20, len(rows))):
            f.write(f"[{r[6].upper()}] {r[2]} @ {r[1]} ({r[0]} q{r[3] + 1})\n"
                    f"  answer: {r[4]}\n  judge: {r[7]}\n\n")
    print(f"wrote {out}/audit.json and audit_sample.txt")


if __name__ == "__main__":
    main()
