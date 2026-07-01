"""Re-score the retrieval-CV settings with continuous metrics (sensitivity analysis).

The headline OOS 3-class accuracy is coarse: over ~32 evaluation meetings it can only move in steps
of ~1/32, so a retrieval change can shift the underlying index without crossing a decision boundary --
accuracy reads "unchanged" even when the index improved. To actually see how sensitive results are to
the recency/relevance weighting, this reads each setting's cached generations (NO new OpenAI calls --
the answers are already cached) and reports several continuous scores alongside accuracy:

  acc3       walk-forward 3-class hike/hold/cut accuracy (the headline; discrete)
  r_pred     OOS Pearson r between the walk-forward's continuous predicted bps and realized bps
  r_idx      Pearson r between the index level and realized bps (over the eval window)
  auc_hike   AUC of the index discriminating hikes vs. the rest
  auc_cut    AUC of (-index) discriminating cuts vs. the rest
  sep        class separation: mean index at hikes minus mean index at cuts

All over the 2022+ evaluation window. Run after retrieval_cv.py has cached one or more settings.

    python paper/experiments/rescore_cv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

import fomc_personas as fp                       # noqa: E402
from fomc_personas import macro                  # noqa: E402
import fig_index as F                            # noqa: E402

CVCACHE = ROOT / "paper" / ".cache" / "retrieval_cv"
FIGCACHE = ROOT / "paper" / ".cache" / "figure_index"
LIVE_MEETINGS = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17"]


def _index_from(resp_path_fn, dates, u):
    """Committee index per meeting from cached responses (project mean answer embedding onto u)."""
    out = {}
    for d in dates:
        p = resp_path_fn(d)
        if not p.exists():
            continue
        resp = json.loads(p.read_text())
        spans, texts = [], []                         # one batched embed call per meeting
        for m, rs in resp.items():
            valid = [r for r in rs if r]
            if valid:
                spans.append((len(texts), len(texts) + len(valid)))
                texts.extend(valid)
        if texts:
            embs = fp.embed(texts)
            out[d] = float(np.mean([embs[lo:hi].mean(0) @ u for lo, hi in spans]))
    return out


def _walkfwd_cont(feats, bps, start=16):
    X = np.column_stack(feats)
    n = len(bps)
    pred = np.full(n, np.nan)
    for i in range(start, n):
        reg = LinearRegression().fit(X[:i], bps[:i])
        pred[i] = reg.predict(X[i:i + 1])[0]
    return pred


def _auc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.concatenate([pos, neg]).argsort().argsort() + 1
    rp = order[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _r(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _tau(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return float("nan")
    return float(kendalltau(a[m], b[m]).correlation)


def _metrics(idx_map, dec, tgt):
    dates = [d for d in macro.FOMC_MEETINGS if d in idx_map and dec[d]["bps"] is not None]
    idx = np.array([idx_map[d] for d in dates])
    bps = np.array([float(dec[d]["bps"]) for d in dates])
    tg = np.array([tgt[d] for d in dates], dtype=float)
    y = np.sign(bps).astype(int)
    e = np.array([d >= "2022" for d in dates])               # eval window
    acc3 = F._acc(F._walkfwd([idx, F._mom(idx)], bps), y, e)
    pred = _walkfwd_cont([idx, F._mom(idx)], bps)
    r_pred = _r(pred[e], bps[e])
    r_idx = _r(idx[e], bps[e])
    auc_h = _auc(idx[e], (y[e] == 1).astype(int))
    auc_c = _auc(-idx[e], (y[e] == -1).astype(int))
    hi, cu = idx[e][y[e] == 1], idx[e][y[e] == -1]
    sep = float(hi.mean() - cu.mean()) if len(hi) and len(cu) else float("nan")
    # index-tracking validation over the FULL series (as in the paper): does the index level move
    # with the fed funds target? -- Pearson r and Kendall tau-b vs the target, and tau vs realized bps.
    r_tgt = _r(idx, tg)
    tau_tgt = _tau(idx, tg)
    tau_bps = _tau(idx, bps)
    return {"n": len(dates), "acc3": acc3, "r_pred": r_pred, "r_idx": r_idx,
            "auc_hike": auc_h, "auc_cut": auc_c, "sep": sep,
            "r_tgt": r_tgt, "tau_tgt": tau_tgt, "tau_bps": tau_bps}


def main():
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + LIVE_MEETINGS
    u = fp.axis(fp.load_anchors())
    series = macro.load_fred()
    dec = macro.decisions(series)
    dates = [d for d in macro.FOMC_MEETINGS if dec[d]["bps"] is not None]
    tgt = {d: macro.macro_briefing(series, d)[0]["target_upper"] for d in dates}

    settings = {"0.0 (relevance)": lambda d: FIGCACHE / f"resp_{d}_cond.json"}
    for cdir in sorted(CVCACHE.glob("beta*_tau*")):
        beta = cdir.name.replace("beta", "").split("_")[0]
        tau = cdir.name.split("tau")[1]
        settings[f"{beta} (tau={tau})"] = (lambda c: (lambda d: c / f"resp_{d}.json"))(cdir)

    rows = {}
    for name, fn in settings.items():
        idx_map = _index_from(fn, dates, u)
        if len(idx_map) < 40:                                 # incomplete setting -> skip
            print(f"  (skip {name}: only {len(idx_map)} meetings cached)")
            continue
        rows[name] = _metrics(idx_map, dec, tgt)

    cols = ["n", "acc3", "r_pred", "r_idx", "auc_hike", "auc_cut", "sep", "r_tgt", "tau_tgt", "tau_bps"]
    print("\n=== retrieval sensitivity (acc/r_pred/r_idx/auc/sep: 2022+; r_tgt/tau_*: full series) ===")
    print(f"{'setting':<18} " + "  ".join(f"{c:>8}" for c in cols))
    for name, m in rows.items():
        print(f"{name:<18} " + "  ".join(
            f"{m[c]:>8.3f}" if isinstance(m[c], float) else f"{m[c]:>8}" for c in cols))
    (CVCACHE / "sensitivity.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {CVCACHE/'sensitivity.json'}")


if __name__ == "__main__":
    main()
