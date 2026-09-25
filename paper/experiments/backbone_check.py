"""Backbone sensitivity for the vote-share index: regenerate both batteries under another model.

Only the GENERATOR changes (default gpt-4.1-mini); retrieval, prompts, briefings, the parse model
(gpt-4o-mini, a labeling utility), member debiasing, and the hazard+tilt forecast are all held
fixed, so any score change is attributable to the generation backbone alone.

Run:  python paper/experiments/backbone_check.py [model]      (resumable per meeting)
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

import pandas as pd                                    # noqa: E402
import fomc_personas as fp                             # noqa: E402
from fomc_personas import macro, persona, roles        # noqa: E402
import fig_index as F                                  # noqa: E402
from retrieval_cv import retrieve_weighted, WORKERS    # noqa: E402
from cut_battery import CUT_BATTERY                    # noqa: E402
from vote_battery import HIKE_BATTERY, PARSE_PROMPT, ACTIONS  # noqa: E402
from probability_index import (anomaly_series, forecast, streaks, rps, logloss)  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4.1-mini"
TAG = MODEL.replace(".", "").replace("-", "_")
BETA, TAU, TOPK = 0.6, 2.0, 3


def gen_battery(battery, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    series = macro.load_fred()
    dec = macro.decisions(series)
    qv = fp.embed(battery)
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        cp = cache_dir / f"resp_{d}.json"
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
            for qi, q in enumerate(battery):
                retr = retrieve_weighted(emb, ages, qv[qi], texts, TOPK, beta=BETA, tau=TAU)
                metas.append((m, qi))
                messages.append([{"role": "system", "content": sys_p},
                                 {"role": "user", "content": persona.index_prompt(q, retr, briefing)}])
        comps = persona.generate(messages, model=MODEL, workers=WORKERS)
        resp = {m: [""] * len(battery) for m in rost}
        for (m, qi), c in zip(metas, comps):
            resp[m][qi] = c
        cp.write_text(json.dumps(resp))
        print(f"  [{MODEL}] {cache_dir.name} {d}: {sum(1 for v in resp.values() for r in v if r)} answers")


def parse_battery(direction, resp_dir, parse_dir):
    parse_dir.mkdir(parents=True, exist_ok=True)
    for d in macro.FOMC_MEETINGS:
        rp = resp_dir / f"resp_{d}.json"
        if not rp.exists():
            continue
        pp = parse_dir / f"{direction}_{d}.json"
        if pp.exists():
            continue
        resp = json.loads(rp.read_text())
        metas, messages = [], []
        for m, answers in resp.items():
            for qi, a in enumerate(answers):
                if a:
                    metas.append((m, qi))
                    messages.append([{"role": "user", "content": PARSE_PROMPT.format(
                        action=ACTIONS[direction], answer=a)}])
        labels = persona.generate(messages, workers=WORKERS, max_tokens=4)   # parse model = default
        parsed = {}
        for (m, qi), lab in zip(metas, labels):
            low = (lab or "").lower()
            parsed.setdefault(m, {})[str(qi)] = ("support" if "support" in low
                                                 else "oppose" if "oppose" in low else "unclear")
        pp.write_text(json.dumps(parsed))


def main():
    cut_dir = ROOT / "paper" / ".cache" / f"cut_battery_{TAG}"
    hike_dir = ROOT / "paper" / ".cache" / f"hike_battery_{TAG}"
    parse_dir = ROOT / "paper" / ".cache" / f"vote_parse_{TAG}"
    gen_battery(CUT_BATTERY, cut_dir)
    gen_battery(HIKE_BATTERY, hike_dir)
    parse_battery("cut", cut_dir, parse_dir)
    parse_battery("hike", hike_dir, parse_dir)

    series = macro.load_fred()
    dec = macro.decisions(series)
    sup = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        row = {}
        for direction in ("hike", "cut"):
            p = parse_dir / f"{direction}_{d}.json"
            if not p.exists():
                continue
            for m, qs in json.loads(p.read_text()).items():
                clear = [v for v in qs.values() if v != "unclear"]
                if clear:
                    row.setdefault(m, {})[direction] = float(np.mean([v == "support" for v in clear]))
        if row:
            sup[d] = {m: (v.get("hike", np.nan), v.get("cut", np.nan)) for m, v in row.items()}
    dates = sorted(sup)
    sha, sca = anomaly_series(sup, dates)
    y = np.sign([float(dec[d]["bps"]) for d in dates]).astype(int)
    P, lams = forecast(y, streaks(y), np.stack([sca, np.zeros_like(sha), sha], axis=1))
    m22 = np.array([d >= "2022" for d in dates])
    print(f"\n[{MODEL}] RPS 2022-25: {rps(P, y, m22):.3f}   log-loss: {logloss(P, y, m22):.3f}   "
          f"lambda median: {np.median(lams):.0f}")
    print("(gpt-4o-mini reference: RPS 0.187, log-loss 0.612; persistence 0.199 / 0.720)")


if __name__ == "__main__":
    main()
