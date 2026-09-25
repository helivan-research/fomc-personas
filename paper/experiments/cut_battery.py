"""Cut-conditioned query battery: can timing-explicit questions recover easing signal?

PRE-REGISTERED DESIGN (fixed before scoring; no iteration on queries, anchors, or features):

  Motivation. On the hawk-dove axis the committee's conditioned language at CUT meetings is
  statistically identical to hold-at-restrictive meetings (cut +0.037 +/- 0.004 vs 2024+ holds
  +0.039 +/- 0.007), so the paper's classifier calls 0/6 cuts. Hypothesis: the production battery
  asks *disposition* questions; answers to *timing-explicit easing* questions may separate cuts
  from holds even though disposition answers do not.

  Design. Everything mirrors the paper's PBI generation exactly -- point-in-time corpus
  (fig_index._public_asof), pure-relevance top-k retrieval (beta=0), the same index_prompt with
  the same as-of-date briefing c^(t), gpt-4o-mini at the production temperature -- EXCEPT the five
  queries below, which ask directly about the timing of easing. Answers are scored by projection
  onto the easing-urgency anchor axis written (once, blind to outcomes) for the second-axis
  experiment; that axis failed to separate cuts when applied to disposition answers, so any
  separation here is attributable to the queries, not the axis.

  Outcome, either way, goes in the paper: separation -> the index gains a cut channel;
  no separation -> measured evidence that FOMC public communication does not telegraph easing
  (asymmetric forward guidance), since even cut-conditioned persona language stays hold-like.

Run:  python paper/experiments/cut_battery.py [beta]   (generate + score; resumable per meeting)

  EXPERIMENT 3 (beta=0.6): the beta=0 run exposed a retrieval failure at the 2024 easing turn --
  the corpus contains Powell's 2024-08-23 "time has come" pivot, but pure-relevance retrieval for
  cut queries surfaces 2017/2023 "25 basis point" HIKE chunks instead (lexical anchoring, no
  recency). beta=0.6/tau=2yr is the PRE-COMMITTED live-chat recency setting (chosen 2026-07 for
  the product, before this experiment); with it, all top-3 retrieved chunks at 2024-09-18 are
  Powell's July-2024 "cut could be on the table in September" statements.
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

BETA = 0.0
if __name__ == "__main__" and len(sys.argv) > 1:      # argv only when run directly (importable otherwise)
    BETA = float(sys.argv[1])
CACHE = ROOT / "paper" / ".cache" / ("cut_battery" if BETA == 0.0 else f"cut_battery_b{BETA}")
CACHE.mkdir(parents=True, exist_ok=True)
TOPK = 3

# The five timing-explicit queries. FIXED -- written once, before any scoring.
CUT_BATTERY = [
    "Should the Committee begin lowering the federal funds rate at its next meeting?",
    "Is it appropriate to start easing policy now, or should the Committee wait for more data?",
    "How urgent is it to reduce the current level of policy restriction?",
    "Would you support a 25 basis point cut at the coming meeting?",
    "Has the time come to dial back policy restraint?",
]

# The easing-urgency anchor axis, verbatim from the (failed) second-axis experiment. FIXED.
EASE = [
    "It is time to begin lowering the policy rate; maintaining this level of restriction risks unnecessary damage to the labor market.",
    "The balance of risks has shifted; we should recalibrate policy toward a less restrictive stance at the coming meeting.",
    "With inflation clearly on a path to target, holding rates this high is no longer warranted; a rate cut is appropriate soon.",
    "Downside risks to employment now outweigh inflation risks, and policy should begin easing without delay.",
    "The policy rate is well above neutral; we can reduce restriction now while remaining restrictive overall.",
]
HOLD = [
    "There is no urgency to adjust the policy rate; we can afford to be patient and wait for greater confidence on inflation.",
    "Policy is well positioned; I see no need to change the stance of policy at this time.",
    "We should hold the policy rate at its current level until the data clearly warrant a move.",
    "The risks are balanced and the economy is solid; maintaining the current stance remains appropriate.",
    "It would be premature to adjust rates now; I prefer to see several more months of data before changing course.",
]


def generate_all():
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    series = macro.load_fred()
    dec = macro.decisions(series)
    qv = fp.embed(CUT_BATTERY)
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        cp = CACHE / f"resp_{d}.json"
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
            for qi, q in enumerate(CUT_BATTERY):
                retr = retrieve_weighted(emb, ages, qv[qi], texts, TOPK, beta=BETA, tau=2.0)
                metas.append((m, qi))
                messages.append([{"role": "system", "content": sys_p},
                                 {"role": "user", "content": persona.index_prompt(q, retr, briefing)}])
        comps = persona.generate(messages, workers=WORKERS)
        resp = {m: [""] * len(CUT_BATTERY) for m in rost}
        for (m, qi), c in zip(metas, comps):
            resp[m][qi] = c
        cp.write_text(json.dumps(resp))
        n = sum(1 for v in resp.values() for r in v if r)
        print(f"  {d}: generated {n} answers ({len(rost)} members)")


def score():
    from fig_index import _mom, _walkfwd, _acc
    v = fp.embed(EASE).mean(0) - fp.embed(HOLD).mean(0)
    v = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
    u = fp.axis(fp.load_anchors())
    series = macro.load_fred()
    dec = macro.decisions(series)

    xc_d, xu_d = {}, {}
    for d in macro.FOMC_MEETINGS:
        cp = CACHE / f"resp_{d}.json"
        bp = F.CACHE.parent / "retrieval_cv" / "beta0.0_tau2.0_pit" / f"resp_{d}.json"
        if not (cp.exists() and bp.exists()):
            continue
        texts = [r for rs in json.loads(cp.read_text()).values() for r in rs if r]
        if not texts:
            continue
        E = fp.embed(texts)
        xc_d[d] = float(E.mean(0) @ v)
        tb = [r for rs in json.loads(bp.read_text()).values() for r in rs if r]
        xu_d[d] = float(fp.embed(tb).mean(0) @ u)

    dates = sorted(xc_d)
    xc = np.array([xc_d[d] for d in dates])
    xu = np.array([xu_d[d] for d in dates])
    bps = np.array([float(dec[d]["bps"]) for d in dates])
    y = np.sign(bps).astype(int)
    m22 = np.array([d >= "2022" for d in dates])
    mpost = np.array([d >= "2024" for d in dates])
    chg = np.concatenate([[False], y[1:] != y[:-1]])

    print("\n=== cut-battery committee feature by class (2022-25) ===")
    for cls, name in [(1, "hike"), (0, "hold"), (-1, "cut")]:
        m = (y == cls) & m22
        print(f"  {name:>4}: {xc[m].mean():+.4f} +/- {xc[m].std():.4f}  (n={m.sum()})")
    m24h = (y == 0) & (np.array(dates) >= "2024")
    print(f"  holds 2024+: {xc[m24h].mean():+.4f} +/- {xc[m24h].std():.4f}")

    prev = np.array([y[i - 1] if i >= 1 else 0 for i in range(len(y))], float)
    print(f"\n{'model':<28} {'acc 22-25':>9} {'cuts':>6} {'chg':>5} {'post24':>7}")
    for name, fl in {
        "1D (paper) [xu,mom]":       [xu, _mom(xu)],
        "cut-battery [xc,mom]":      [xc, _mom(xc)],
        "2D [xu,mom,xc,mom]":        [xu, _mom(xu), xc, _mom(xc)],
        "2D + prev":                 [xu, _mom(xu), xc, _mom(xc), prev],
    }.items():
        pred = _walkfwd(fl, bps)
        mcut = m22 & (y == -1) & (pred != 99)
        print(f"{name:<28} {_acc(pred, y, m22):>9.3f} "
              f"{int((pred[mcut] == -1).sum())}/{int(mcut.sum()):<4} "
              f"{_acc(pred, y, m22 & chg):>5.2f} {_acc(pred, y, mpost):>7.3f}")
    pp = np.array([y[i - 1] if i >= 1 else 99 for i in range(len(y))])
    pp = np.where(np.arange(len(y)) >= 16, pp, 99)
    print(f"{'persistence':<28} {_acc(pp, y, m22):>9.3f} {'0/6':>6} "
          f"{_acc(pp, y, m22 & chg):>5.2f} {_acc(pp, y, mpost):>7.3f}")


if __name__ == "__main__":
    generate_all()
    score()
