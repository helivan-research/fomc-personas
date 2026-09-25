"""EXPERIMENT 4 (pre-registered): the net-vote-share index -- a PBI that separates hike/hold/cut.

The cut-battery answers are explicit votes in prose ("I would (not) support a 25 basis point
cut..."); embedding-projection onto a mood axis discards that. Here each persona is asked the
timing-explicit battery in BOTH directions and each answer is parsed into a vote:

  s_cut(t)  = share of (member, question) answers endorsing easing at meeting t
  s_hike(t) = share endorsing tightening
  PBI-vote(t) = s_hike(t) - s_cut(t)      in [-1, 1]

By construction hikes should sit positive, cuts negative, holds near zero -- three-way separation
on one interpretable line, with the pair (s_hike, s_cut) additionally exposing committee splits.

FIXED before scoring: the five hike questions below (mirrors of CUT_BATTERY, direction words
swapped); recency retrieval beta=0.6/tau=2yr (the pre-committed live-chat setting; see
cut_battery.py EXPERIMENT 3 for the mechanism); the parse prompt; the two-threshold walk-forward
decision rule (same tune-on-history spirit as the paper's _cls).

This is `direct_vote` repaired: that baseline (0.53) used stale beta=0 retrieval, a single
question, and modal aggregation -- each now diagnosed as a defect.

Run:  python paper/experiments/vote_battery.py     (generate hike battery + parse both + score)
"""
import json
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

import pandas as pd                    # noqa: E402
import fomc_personas as fp             # noqa: E402
from fomc_personas import macro, persona, roles  # noqa: E402
import fig_index as F                  # noqa: E402
from retrieval_cv import retrieve_weighted, WORKERS  # noqa: E402
from cut_battery import CUT_BATTERY    # noqa: E402

BETA, TAU, TOPK = 0.6, 2.0, 3
CUT_CACHE = ROOT / "paper" / ".cache" / "cut_battery_b0.6"      # reused from experiment 3
HIKE_CACHE = ROOT / "paper" / ".cache" / "hike_battery_b0.6"
PARSE_CACHE = ROOT / "paper" / ".cache" / "vote_parse"
HIKE_CACHE.mkdir(parents=True, exist_ok=True)
PARSE_CACHE.mkdir(parents=True, exist_ok=True)

# Mirrors of CUT_BATTERY with direction words swapped. FIXED -- written once, before any scoring.
HIKE_BATTERY = [
    "Should the Committee raise the federal funds rate at its next meeting?",
    "Is it appropriate to tighten policy further now, or should the Committee wait for more data?",
    "How urgent is it to increase the current level of policy restriction?",
    "Would you support a 25 basis point hike at the coming meeting?",
    "Has the time come to firm the policy stance further?",
]

PARSE_PROMPT = (
    'You will read a statement by a Federal Reserve official answering whether they support '
    '{action} at the coming FOMC meeting.\n\nStatement: "{answer}"\n\n'
    'Does the speaker support {action}? Reply with exactly one word: "support", "oppose", or "unclear".'
)
ACTIONS = {"cut": "cutting the federal funds rate", "hike": "raising the federal funds rate"}


def generate_hike_battery():
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    series = macro.load_fred()
    dec = macro.decisions(series)
    qv = fp.embed(HIKE_BATTERY)
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        cp = HIKE_CACHE / f"resp_{d}.json"
        if cp.exists():
            continue
        df_t = F._public_asof(df, d)
        rost = [m for m in F._roster(df_t) if roles.office_at(m, d) is not None]
        if not rost:
            continue
        _, briefing = macro.macro_briefing(series, d)
        tT = pd.Timestamp(d)
        metas, messages = [], []
        for m in rost:
            g = df_t[df_t["member"] == m]
            emb = np.vstack(g["embedding"].values)
            texts = g["text"].values
            ages = (tT - pd.to_datetime(g["postedAt"], errors="coerce")).dt.days.values / 365.25
            sys_p = persona.system_prompt(m, bios.get(m, ""))
            for qi, q in enumerate(HIKE_BATTERY):
                retr = retrieve_weighted(emb, ages, qv[qi], texts, TOPK, beta=BETA, tau=TAU)
                metas.append((m, qi))
                messages.append([{"role": "system", "content": sys_p},
                                 {"role": "user", "content": persona.index_prompt(q, retr, briefing)}])
        comps = persona.generate(messages, workers=WORKERS)
        resp = {m: [""] * len(HIKE_BATTERY) for m in rost}
        for (m, qi), c in zip(metas, comps):
            resp[m][qi] = c
        cp.write_text(json.dumps(resp))
        print(f"  hike battery {d}: {sum(1 for v in resp.values() for r in v if r)} answers")


def parse_votes(direction: str, cache_dir: Path):
    """Parse every cached answer for `direction` into support/oppose/unclear (cached per meeting)."""
    out = {}
    for d in macro.FOMC_MEETINGS:
        rp = cache_dir / f"resp_{d}.json"
        if not rp.exists():
            continue
        pp = PARSE_CACHE / f"{direction}_{d}.json"
        if pp.exists():
            out[d] = json.loads(pp.read_text())
            continue
        resp = json.loads(rp.read_text())
        metas, messages = [], []
        for m, answers in resp.items():
            for qi, a in enumerate(answers):
                if a:
                    metas.append((m, qi))
                    messages.append([{"role": "user", "content": PARSE_PROMPT.format(
                        action=ACTIONS[direction], answer=a)}])
        labels = persona.generate(messages, workers=WORKERS, max_tokens=4)
        parsed = {}
        for (m, qi), lab in zip(metas, labels):
            low = (lab or "").lower()
            parsed.setdefault(m, {})[str(qi)] = ("support" if "support" in low
                                                 else "oppose" if "oppose" in low else "unclear")
        pp.write_text(json.dumps(parsed))
        out[d] = parsed
        n = sum(1 for m in parsed.values() for v in m.values() if v != "unclear")
        print(f"  parsed {direction} {d}: {n} clear votes")
    return out


def share(parsed_meeting) -> float:
    votes = [v for m in parsed_meeting.values() for v in m.values() if v != "unclear"]
    return float(np.mean([v == "support" for v in votes])) if votes else np.nan


def main():
    generate_hike_battery()
    cut = parse_votes("cut", CUT_CACHE)
    hike = parse_votes("hike", HIKE_CACHE)

    series = macro.load_fred()
    dec = macro.decisions(series)
    dates = sorted(set(cut) & set(hike))
    s_cut = np.array([share(cut[d]) for d in dates])
    s_hike = np.array([share(hike[d]) for d in dates])
    net = s_hike - s_cut
    bps = np.array([float(dec[d]["bps"]) for d in dates])
    y = np.sign(bps).astype(int)
    m22 = np.array([d >= "2022" for d in dates])
    mpost = np.array([d >= "2024" for d in dates])
    m2021 = np.array([("2020" <= d < "2022") for d in dates])
    chg = np.concatenate([[False], y[1:] != y[:-1]])

    print("\n=== net-vote index by class (2022-25) ===")
    print(f"{'class':>6} {'s_hike':>14} {'s_cut':>14} {'net':>14}")
    for cls, name in [(1, "hike"), (0, "hold"), (-1, "cut")]:
        m = (y == cls) & m22
        print(f"{name:>6} {s_hike[m].mean():+.3f}±{s_hike[m].std():.3f} "
              f"{s_cut[m].mean():+.3f}±{s_cut[m].std():.3f} {net[m].mean():+.3f}±{net[m].std():.3f}")
    m24h = (y == 0) & (np.array(dates) >= "2024")
    print(f"{'hold24+':>6} {s_hike[m24h].mean():+.3f}±{s_hike[m24h].std():.3f} "
          f"{s_cut[m24h].mean():+.3f}±{s_cut[m24h].std():.3f} {net[m24h].mean():+.3f}±{net[m24h].std():.3f}")

    # walk-forward two-threshold rule (tune-on-history, same spirit as the paper's _cls)
    def predict(start=16):
        pred = np.full(len(y), 99)
        grid = np.linspace(0.05, 0.95, 19)
        for i in range(start, len(y)):
            best, best_acc = (0.5, 0.5), -1.0
            for th in grid:
                for tc in grid:
                    p = np.where((s_hike[:i] > th) & (s_hike[:i] >= s_cut[:i]), 1,
                                 np.where(s_cut[:i] > tc, -1, 0))
                    a = float((p == y[:i]).mean())
                    if a > best_acc:
                        best_acc, best = a, (th, tc)
            th, tc = best
            pred[i] = 1 if (s_hike[i] > th and s_hike[i] >= s_cut[i]) else (-1 if s_cut[i] > tc else 0)
        return pred

    from fig_index import _walkfwd, _mom, _acc
    print(f"\n{'model':<22} {'acc 22-25':>9} {'cuts':>6} {'chg':>5} {'post24':>7} {'20-21':>7}")
    for name, pred in {
        "PBI-vote (2-threshold)": predict(),
        "persistence": np.where(np.arange(len(y)) >= 16,
                                np.array([y[i-1] if i >= 1 else 99 for i in range(len(y))]), 99),
    }.items():
        mcut = m22 & (y == -1) & (pred != 99)
        print(f"{name:<22} {_acc(pred, y, m22):>9.3f} "
              f"{int((pred[mcut] == -1).sum())}/{int(mcut.sum()):<4} "
              f"{_acc(pred, y, m22 & chg):>5.2f} {_acc(pred, y, mpost):>7.3f} {_acc(pred, y, m2021):>7.3f}")


if __name__ == "__main__":
    main()
