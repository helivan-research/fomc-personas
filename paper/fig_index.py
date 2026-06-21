"""Figure 4 — the persona-based rate-action index (PBI).

(a) PBI vs the fed funds target over 2018-2025; (b) PBI by realized decision (2022-25);
(c) walk-forward OOS decision accuracy vs baselines.

At each meeting the in-office roster's personas answer a fixed 15-question battery, prepended with an
as-of-date macro briefing c^(t), with retrieval restricted to chunks dated <= t. Their projected
stances are averaged into the committee index. Per-meeting generation is cached under
paper/.cache/figure_index/. This is the costliest figure (per-meeting generation for PBI + the
static-query ablation): first run is several $ and ~30-40 min; reruns plot instantly.

    OPENAI_API_KEY=sk-...  python paper/fig_index.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fomc_personas as fp
from fomc_personas import macro, roles, persona

CACHE = Path(__file__).resolve().parent / ".cache" / "figure_index"
CACHE.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})

MIN_OPINIONS, TOPK = 5, 3
RED, BLUE, GREY = "#C44E52", "#4C72B0", "#999999"


def _roster(df_t):
    """In-office members at this meeting with at least MIN_OPINIONS chunks dated <= t."""
    counts = df_t["member"].value_counts()
    return [m for m in counts.index if counts[m] >= MIN_OPINIONS]


def _cached(tag, fn):
    p = CACHE / f"{tag}.json"
    if p.exists():
        return json.loads(p.read_text())
    out = fn()
    p.write_text(json.dumps(out))
    return out


def _index_series(df, dec, bios, u, condition):
    """Per-meeting committee index: roster personas answer the battery (with/without c^(t) briefing),
    retrieval restricted to <= t; project mean response embedding onto u, average over the roster."""
    series = macro.load_fred()
    battery = fp.load_queries("curated")[:15]
    out = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        df_t = df[df["postedAt"].astype(str) <= d]
        rost = [m for m in _roster(df_t) if roles.office_at(m, d) is not None]
        if not rost:
            continue
        _, briefing = macro.macro_briefing(series, d)
        tag = f"resp_{d}_{'cond' if condition else 'noc'}"
        resp = _cached(tag, lambda: persona.respond(
            df_t, rost, battery, bios, k=TOPK, briefing=(briefing if condition else None)))
        pos = {}
        for m, rs in resp.items():
            valid = [r for r in rs if r]
            if valid:
                pos[m] = float(fp.embed(valid).mean(0) @ u)
        if pos:
            out[d] = {"index": float(np.mean(list(pos.values()))), "positions": pos,
                      "bps": dec[d]["bps"], "label": dec[d]["label"]}
    return out


def _retrieved_index(df, dec, u):
    """No-generation control: each member's position is the projection of the mean top-k retrieved
    chunk embedding (for the fixed battery, <= t) onto u; averaged over the roster."""
    battery = fp.load_queries("curated")[:15]
    qv = fp.embed(battery)
    out = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        df_t = df[df["postedAt"].astype(str) <= d]
        pos = []
        for m in _roster(df_t):
            g = df_t[df_t["member"] == m]
            emb = np.vstack(g["embedding"].values)
            rows = [emb[np.argsort(-(emb @ qv[i]))[:TOPK]].mean(0) for i in range(len(battery))]
            pos.append(float(np.mean(rows, axis=0) @ u))
        if pos:
            out[d] = float(np.mean(pos))
    return out


def _direct_vote(df, dec, bios):
    """Each persona votes hike/hold/cut directly (no stance projection); committee = modal vote."""
    series = macro.load_fred()
    lab = {"hike": 1, "hold": 0, "cut": -1}
    out = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        df_t = df[df["postedAt"].astype(str) <= d]
        rost = [m for m in _roster(df_t) if roles.office_at(m, d) is not None]
        if not rost:
            continue
        _, briefing = macro.macro_briefing(series, d)
        qv = fp.embed(["how would you vote on the federal funds rate at this meeting?"])[0]
        votes = _cached(f"vote_{d}", lambda: _gen_votes(df_t, rost, bios, briefing, qv))
        modal = Counter(votes.values()).most_common(1)[0][0] if votes else None
        out[d] = lab.get(modal, 0)
    return out


def _gen_votes(df_t, rost, bios, briefing, qv):
    metas, messages = [], []
    for m in rost:
        g = df_t[df_t["member"] == m]
        emb = np.vstack(g["embedding"].values)
        retr = "\n".join(f"- {t}" for t in g["text"].values[np.argsort(-(emb @ qv))[:TOPK]])
        user = (f"{briefing}\n\nHere are positions you have actually expressed:\n{retr}\n\n"
                f"In light of these conditions, how would you vote at this FOMC meeting on the federal "
                f'funds rate? Answer with exactly one word: "hike", "hold", or "cut".')
        metas.append(m)
        messages.append([{"role": "system", "content": persona.system_prompt(m, bios.get(m, ""))},
                         {"role": "user", "content": user}])
    raw = persona.generate(messages, max_tokens=8)
    out = {}
    for m, c in zip(metas, raw):
        cl = c.lower()
        out[m] = "hike" if "hike" in cl else ("cut" if "cut" in cl else "hold")
    return out


# --- walk-forward OOS classifier (index level + 3-meeting momentum) ---
def _cls(b, db):
    return np.where(b > db, 1, np.where(b < -db, -1, 0))


def _walkfwd(feats, bps, start=16):
    X = np.column_stack(feats)
    n = len(bps)
    y = np.sign(bps).astype(int)
    pred = np.full(n, 99)
    for i in range(start, n):
        reg = LinearRegression().fit(X[:i], bps[:i])
        tr = reg.predict(X[:i])
        db = max(np.linspace(0, 40, 81), key=lambda g: (_cls(tr, g) == y[:i]).mean())
        pred[i] = _cls(reg.predict(X[i:i + 1]), db)[0]
    return pred


def _mom(idx):
    return np.array([idx[i] - np.mean(idx[max(0, i - 3):i]) if i >= 1 else 0.0 for i in range(len(idx))])


def _acc(pred, y, mask):
    m = mask & (pred != 99)
    return float((pred[m] == y[m]).mean())


def main():
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    u = fp.axis(fp.load_anchors())
    series = macro.load_fred()
    dec = macro.decisions(series)

    print("PBI backtest (conditioned) ...")
    pbi = _index_series(df, dec, bios, u, condition=True)
    print("static query set (no c^(t)) ...")
    noc = _index_series(df, dec, bios, u, condition=False)
    print("baselines ...")
    ridx_d = _retrieved_index(df, dec, u)
    dvote_d = _direct_vote(df, dec, bios)

    dates = [d for d in macro.FOMC_MEETINGS if d in pbi]
    idx = np.array([pbi[d]["index"] for d in dates])
    bps = np.array([float(pbi[d]["bps"]) for d in dates])
    y = np.sign(bps).astype(int)
    m22 = np.array([d >= "2022" for d in dates])
    labels = [pbi[d]["label"] for d in dates]

    # macro + Taylor features
    snaps = [macro.macro_briefing(series, d)[0] for d in dates]
    pi = np.array([s["core_pce_yoy"] for s in snaps]); un = np.array([s["unrate"] for s in snaps])
    cur = np.array([s["target_upper"] for s in snaps])
    gap = 2 + pi + 0.5 * (pi - 2) - (un - 4.4) - cur
    macro_feats = [pi, un, cur, _mom(pi), _mom(un)]
    noc_idx = np.array([noc.get(d, {"index": 0.0})["index"] for d in dates])
    ridx = np.array([ridx_d.get(d, 0.0) for d in dates])
    dvote = np.array([dvote_d.get(d, 0) for d in dates])
    prev = np.array([y[i - 1] if i >= 1 else 99 for i in range(len(y))])
    oos = np.arange(len(y)) >= 16
    base = max((y[m22 & oos] == c).mean() for c in (-1, 0, 1))

    preds = {
        "chance": np.where(oos, 0, 99),
        "Taylor": _walkfwd([gap], bps),
        "retr.": _walkfwd([ridx, _mom(ridx)], bps),
        "direct": dvote,
        "macro": _walkfwd(macro_feats, bps),
        "static": _walkfwd([noc_idx, _mom(noc_idx)], bps),
        "PBI": _walkfwd([idx, _mom(idx)], bps),
        "persist.": prev,
    }
    accs = {k: _acc(p, y, m22) for k, p in preds.items()}
    print("  2022-25 OOS 3-class:", {k: round(v, 2) for k, v in accs.items()})

    tgt = np.array([macro.macro_briefing(series, d)[0]["target_upper"] for d in dates])
    _plot(dates, idx, tgt, labels, accs, base)
    print(f"wrote {FIG/'fig_index.pdf'}  (PBI acc={accs['PBI']:.2f} vs base {base:.2f})")


def _plot(dates, idx, tgt, labels, accs, base):
    x = [np.datetime64(d) for d in dates]
    fig = plt.figure(figsize=(7.2, 3.85))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.55],
                          hspace=0.45, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])
    ax.axhline(0, color="k", lw=.6, alpha=.4)
    for d, l in zip(x, labels):
        if l == "hike":
            ax.axvspan(d - np.timedelta64(20, "D"), d + np.timedelta64(20, "D"), color=RED, alpha=.06)
        elif l == "cut":
            ax.axvspan(d - np.timedelta64(20, "D"), d + np.timedelta64(20, "D"), color=BLUE, alpha=.08)
    ax.plot(x, idx, "-", color=RED, lw=1.7, label="PBI (left)")
    ax.set_ylabel("PBI", color=RED, fontsize=8)
    ax2 = ax.twinx(); ax2.plot(x, tgt, "-", color="#333", lw=1.2, label="fed funds target (right)")
    ax2.set_ylabel("target upper (%)", fontsize=8); ax2.grid(False)
    ax.set_title("(a) PBI vs. fed funds target", fontsize=8)
    ax.legend(fontsize=6, loc="upper left"); ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)

    axc = fig.add_subplot(gs[1, 0])
    m22 = [d for d in dates if d >= "2022"]
    rng = np.random.default_rng(0)
    ccol = {"hike": RED, "hold": "#888", "cut": BLUE}
    for i, c in enumerate(["hike", "hold", "cut"]):
        ys = np.array([idx[dates.index(d)] for d in m22 if labels[dates.index(d)] == c])
        axc.scatter(rng.normal(i, .07, len(ys)), ys, s=12, color=ccol[c], alpha=.75, edgecolor="none")
        if len(ys):
            axc.hlines(np.median(ys), i - .26, i + .26, color="k", lw=1.3)
    cnt = {c: sum(1 for d in m22 if labels[dates.index(d)] == c) for c in ["hike", "hold", "cut"]}
    axc.set_xticks(range(3)); axc.set_xticklabels([f"{c}\n($n{{=}}{cnt[c]}$)" for c in ["hike", "hold", "cut"]], fontsize=6)
    axc.set_ylabel("PBI", fontsize=7); axc.set_title("(b) PBI by decision (2022--25)", fontsize=8)
    axc.tick_params(labelsize=6)

    axb = fig.add_subplot(gs[1, 1])
    order = ["chance", "Taylor", "retr.", "direct", "macro", "static", "PBI", "persist."]
    vals = [accs[k] for k in order]
    cols = [RED if k == "PBI" else (GREY if k == "persist." else "#CBCBCB") for k in order]
    axb.bar(range(len(order)), vals, color=cols, width=.72)
    for i, v in enumerate(vals):
        if order[i] != "chance":
            axb.text(i, v + .015, f"{v:.2f}", ha="center", fontsize=5.5)
    axb.axhline(base, ls=":", color="0.4", lw=.8)
    axb.text(0.02, base + .008, f"base rate ${base:.2f}$", transform=axb.get_yaxis_transform(),
             ha="left", va="bottom", fontsize=5, color="0.4")
    axb.set_xticks(range(len(order))); axb.set_xticklabels(order, fontsize=6)
    axb.set_ylim(0, .92); axb.set_ylabel("OOS accuracy", fontsize=7); axb.tick_params(labelsize=6)
    axb.set_title("(c) decision accuracy vs. baselines", fontsize=8)

    fig.savefig(FIG / "fig_index.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
