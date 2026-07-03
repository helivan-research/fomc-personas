"""Re-run the retrieval sensitivity grid under the POINT-IN-TIME corpus filter.

The original CV selected (beta=0.6, tau=2yr) on the pre-PIT (leaked) corpus; PIT changes the
retrieval pool (embargoed transcripts excluded until release), so both the sensitivity curve and the
"unchanged at beta=0" claim need re-verification. Honesty protocol: we do NOT re-select a max on the
eval window. We report, per setting: full-window metrics (sensitivity), the pre-cutoff window
(selection-legal), and the post-LLM-cutoff window (contamination-proof validation).

    OPENAI_API_KEY=... python paper/experiments/cv_pit.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fomc_personas as fp                        # noqa: E402
from fomc_personas import macro                   # noqa: E402
import retrieval_cv as CV                         # noqa: E402  (robust client, PIT index_series)
import fig_index as F                             # noqa: E402

SETTINGS = [(0.0, 2.0), (0.3, 2.0), (0.6, 2.0), (0.6, 4.0), (1.2, 2.0)]   # 0.6/2.0 already cached
CUTOFF = "2023-10-01"                             # gpt-4o-mini knowledge cutoff


def metrics(out, dates_all):
    dates = [d for d in dates_all if d in out]
    idx = np.array([out[d]["index"] for d in dates])
    bps = np.array([float(out[d]["bps"]) for d in dates])
    y = np.sign(bps).astype(int)
    e22 = np.array([d >= "2022" for d in dates])
    post = np.array([d >= CUTOFF for d in dates])
    pre = e22 & ~post
    pred = F._walkfwd([idx, F._mom(idx)], bps)

    def acc(mask):
        m = mask & (pred != 99)
        return float((pred[m] == y[m]).mean()) if m.any() else float("nan")

    def auc(score, label):
        pos, neg = score[label == 1], score[label == 0]
        if not len(pos) or not len(neg):
            return float("nan")
        o = np.concatenate([pos, neg]).argsort().argsort() + 1
        return float((o[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    tau_move = kendalltau(idx[e22], bps[e22]).correlation
    # lead-lag peak vs target level (full series)
    series = macro.load_fred()
    tgt = np.array([macro.macro_briefing(series, d)[0]["target_upper"] for d in dates])
    shifts = np.arange(0, 13)
    _, taus = F._lead_lag(idx, tgt, shifts)
    kbest = int(shifts[np.nanargmax(taus)])
    return {"n": len(dates), "acc22": acc(e22), "acc_pre": acc(pre), "acc_post": acc(post),
            "tau_move": tau_move, "auc_cut": auc(-idx[e22], (y[e22] == -1).astype(int)),
            "lead_tau": float(np.nanmax(taus)), "lead_k": kbest}


def main():
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + CV.LIVE_MEETINGS
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    u = fp.axis(fp.load_anchors())
    series = macro.load_fred()
    dec = macro.decisions(series)
    dates_all = [d for d in macro.FOMC_MEETINGS if dec[d]["bps"] is not None]

    rows = {}
    for beta, tau in SETTINGS:
        print(f"=== beta={beta} tau={tau} (PIT) ===", flush=True)
        out = CV.index_series(df, dec, bios, u, beta, tau)
        rows[f"b{beta}_t{tau}"] = metrics(out, dates_all)

    cols = ["n", "acc22", "acc_pre", "acc_post", "tau_move", "auc_cut", "lead_tau", "lead_k"]
    print("\n=== PIT retrieval sensitivity ===")
    print(f"{'setting':<14}" + "".join(f"{c:>9}" for c in cols))
    for k, m in rows.items():
        print(f"{k:<14}" + "".join(
            f"{m[c]:>9}" if isinstance(m[c], int) else f"{m[c]:>9.3f}" for c in cols))
    import json
    (ROOT / "paper" / ".cache" / "retrieval_cv" / "sensitivity_pit.json").write_text(
        json.dumps(rows, indent=2))
    print("\nwrote sensitivity_pit.json")


if __name__ == "__main__":
    main()
