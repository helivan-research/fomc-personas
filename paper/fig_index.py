"""Figure 4 — the persona-based rate-action index (PBI).

(a) PBI vs the fed funds target over 2018-2025; (b) PBI by realized decision (2022-25);
(c) walk-forward OOS decision accuracy vs baselines (+ accuracy at decision changes, where the
persistence baseline scores 0 by construction); (d) lead-lag -- Kendall tau of the PBI with the
target LEVEL as the index is slid forward (tau 0.38 contemporaneous -> ~0.79 at +9 meetings),
against two mechanical controls (raw CPI; the no-briefing static index), both of which it exceeds.

At each meeting the in-office roster's personas answer a fixed 15-question battery, prepended with an
as-of-date macro briefing c^(t). Retrieval is POINT-IN-TIME (see _public_asof: embargoed FOMC
transcripts are dated at their ~5-year release, strict day-before cutoff) and recency-weighted
(relevance + beta*recency, exp(-age/tau); beta=0.6, tau=2yr from paper/experiments/retrieval_cv.py).
Honesty notes: (beta, tau) was selected on the same 2022+ eval window the paper reports, so treat the
cut-discrimination gain vs pure relevance as in-sample-selected; and the walk-forward accuracy (0.66)
sits below the persistence baseline (0.78) on this hold-heavy window -- persistence is blind at
turning points, which is where the index carries its value.

Projected stances are averaged into the committee index. Generations are cached under
paper/.cache/retrieval_cv/beta{b}_tau{t}_pit/ (built by retrieval_cv.py) and the ablations under
paper/.cache/figure_index_pit/. With those caches present the figure plots in seconds, no paid calls.

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
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fomc_personas as fp
from fomc_personas import macro, roles, persona

CACHE = Path(__file__).resolve().parent / ".cache" / "figure_index_pit"   # v2: point-in-time filter
CACHE.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})

MIN_OPINIONS, TOPK = 5, 3
RED, BLUE, GREY = "#C44E52", "#4C72B0", "#999999"

# The PBI uses the CV-tuned recency-weighted retrieval (relevance + beta*recency, exp(-age/tau)).
# Those conditioned per-meeting generations are cached by paper/experiments/retrieval_cv.py under
# .cache/retrieval_cv/beta{b}_tau{t}/. Set to "0.0"/None to fall back to the pure-relevance cache.
RETRIEVAL_BETA, RETRIEVAL_TAU = "0.6", "2.0"
BETA_DIR = Path(__file__).resolve().parent / ".cache" / "retrieval_cv" / f"beta{RETRIEVAL_BETA}_tau{RETRIEVAL_TAU}_pit"


def _public_asof(df, d):
    """Chunks *publicly available* strictly before meeting date d (point-in-time filter).

    Two fixes over the previous `postedAt.astype(str) <= d` string compare:
      - FOMC meeting *transcripts* are embargoed ~5 years (year Y is released ~Jan of Y+6). They are
        dated at the meeting they record, so filtering on postedAt alone let the backtest retrieve
        text that was not public at time t. Their availability date is the release, not the meeting.
      - The cutoff is an explicit datetime strict-< (day-before). The old string compare happened to
        exclude same-day rows only because most timestamps carry a time component -- a data property,
        not a guarantee. Same-day statements (e.g. the post-decision press conference) must never be
        retrievable when "predicting" that decision.
    Recency *ages* still use postedAt (when the words were said); only availability changes.
    """
    ts = pd.to_datetime(df["postedAt"], errors="coerce")
    avail = ts.copy()
    tr = (df["source"] == "fomc_transcript").values
    avail[tr] = pd.to_datetime((ts[tr].dt.year + 6).astype(str) + "-01-15")
    return df[avail < pd.Timestamp(d)]


def _roster(df_t):
    """In-office members at this meeting with at least MIN_OPINIONS chunks available <= t."""
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
        df_t = _public_asof(df, d)
        rost = [m for m in _roster(df_t) if roles.office_at(m, d) is not None]
        if not rost:
            continue
        _, briefing = macro.macro_briefing(series, d)
        tag = f"resp_{d}_{'cond' if condition else 'noc'}"
        resp = _cached(tag, lambda: persona.respond(
            df_t, rost, battery, bios, k=TOPK, briefing=(briefing if condition else None),
            prompt_fn=persona.index_prompt))
        pos = {}
        for m, rs in resp.items():
            valid = [r for r in rs if r]
            if valid:
                pos[m] = float(fp.embed(valid).mean(0) @ u)
        if pos:
            out[d] = {"index": float(np.mean(list(pos.values()))), "positions": pos,
                      "bps": dec[d]["bps"], "label": dec[d]["label"]}
    return out


def _pbi_series(dec, u):
    """Per-meeting committee index from the CV-tuned recency-weighted generations (cached, no calls).
    Same structure as _index_series(condition=True) but reads .cache/retrieval_cv/beta{b}_tau{t}/, so
    the PBI uses relevance + beta*recency retrieval instead of pure relevance."""
    out = {}
    for d in macro.FOMC_MEETINGS:
        if dec[d]["bps"] is None:
            continue
        p = BETA_DIR / f"resp_{d}.json"
        if not p.exists():
            continue
        resp = json.loads(p.read_text())
        spans, texts = [], []                         # one batched embed call per meeting
        for m, rs in resp.items():
            valid = [r for r in rs if r]
            if valid:
                spans.append((m, len(texts), len(texts) + len(valid)))
                texts.extend(valid)
        if not texts:
            continue
        embs = fp.embed(texts)
        pos = {m: float(embs[lo:hi].mean(0) @ u) for m, lo, hi in spans}
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
        df_t = _public_asof(df, d)
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
        df_t = _public_asof(df, d)
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


def _lead_lag(idx, tgt, shifts):
    """Correlation of the PBI with the fed-funds target level when the PBI is shifted forward by k
    meetings (idx[t] vs target[t+k], k>0 = PBI leads). Returns Pearson r and Kendall tau per shift."""
    rs, taus = [], []
    for s in shifts:
        a, b = (idx[:len(idx) - s], tgt[s:]) if s >= 0 else (idx[-s:], tgt[:len(idx) + s])
        rs.append(float(np.corrcoef(a, b)[0, 1]))
        taus.append(float(kendalltau(a, b).correlation))
    return np.array(rs), np.array(taus)


def main():
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    u = fp.axis(fp.load_anchors())
    series = macro.load_fred()
    dec = macro.decisions(series)

    print(f"PBI backtest (conditioned, recency-weighted beta={RETRIEVAL_BETA} tau={RETRIEVAL_TAU}) ...")
    if BETA_DIR.exists():
        pbi = _pbi_series(dec, u)
    else:                                             # no tuned cache -> generate pure-relevance PBI
        print(f"  ! {BETA_DIR} missing; run paper/experiments/retrieval_cv.py. Falling back to relevance.")
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

    # macro + Taylor features. The macro classifier receives every variable in the conditioning
    # briefing c^(t) the personas read (headline CPI, core PCE, unemployment, the funds-rate ceiling)
    # plus the momentum of the inflation/employment conditions -- a SUPERSET of PBI's information,
    # just raw instead of compressed into the behavioural index.
    snaps = [macro.macro_briefing(series, d)[0] for d in dates]
    cpi = np.array([s["cpi_yoy"] for s in snaps]); pi = np.array([s["core_pce_yoy"] for s in snaps])
    un = np.array([s["unrate"] for s in snaps]); cur = np.array([s["target_upper"] for s in snaps])
    gap = 2 + pi + 0.5 * (pi - 2) - (un - 4.4) - cur
    macro_feats = [cpi, pi, un, cur, _mom(cpi), _mom(pi), _mom(un)]
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

    # Accuracy AT DECISION CHANGES: the honest counterpoint to the persistence baseline. Persistence
    # (repeat the last decision) wins on the long holds but scores 0 by construction at every turning
    # point -- exactly where a forecast has value. Report both.
    chg = np.concatenate([[False], y[1:] != y[:-1]])
    acc_chg = {k: _acc(p, y, m22 & chg) for k, p in preds.items()}
    n_chg = int((m22 & chg & (preds["PBI"] != 99)).sum())
    print(f"  at decision changes (n={n_chg}): PBI={acc_chg['PBI']:.2f}  persist.={acc_chg['persist.']:.2f}"
          f"  macro={acc_chg['macro']:.2f}  Taylor={acc_chg['Taylor']:.2f}")

    tgt = np.array([macro.macro_briefing(series, d)[0]["target_upper"] for d in dates])

    # Lead-lag: the PBI leads the funds-rate LEVEL. Scanned wide enough that the peak is interior, and
    # against two controls that could inherit the same profile mechanically: raw CPI (which led the
    # funds rate this cycle and sits in the personas' briefing c^(t)) and the no-briefing (static)
    # index. If the PBI's profile only matched the controls, "the PBI leads" would deflate to "the Fed
    # reacts to inflation with a lag."
    ll_shifts = np.arange(-2, 13)
    ll_r, ll_tau = _lead_lag(idx, tgt, ll_shifts)
    _, cpi_tau = _lead_lag(cpi, tgt, ll_shifts)
    _, noc_tau = _lead_lag(noc_idx, tgt, ll_shifts)
    kbest = int(ll_shifts[np.nanargmax(ll_tau)])
    i0 = list(ll_shifts).index(0)
    print(f"  lead-lag vs target: contemporaneous r={ll_r[i0]:.2f} tau={ll_tau[i0]:.2f} ; "
          f"peak +{kbest} mtgs tau={np.nanmax(ll_tau):.2f} "
          f"(controls at +{kbest}: CPI tau={cpi_tau[list(ll_shifts).index(kbest)]:.2f}, "
          f"static tau={noc_tau[list(ll_shifts).index(kbest)]:.2f})")

    _plot(dates, idx, tgt, labels, accs, base, (ll_shifts, ll_tau, cpi_tau, noc_tau),
          (acc_chg["PBI"], n_chg))
    print(f"wrote {FIG/'fig_index.pdf'}  (PBI acc={accs['PBI']:.2f} vs base {base:.2f})")


def _plot(dates, idx, tgt, labels, accs, base, leadlag, chg_stat=None):
    x = [np.datetime64(d) for d in dates]
    fig = plt.figure(figsize=(7.4, 4.45))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.02], width_ratios=[1.0, 1.28, 1.02],
                          hspace=0.55, wspace=0.42)

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
    axb.set_xticks(range(len(order)))
    axb.set_xticklabels(order, fontsize=6, rotation=35, ha="right", rotation_mode="anchor")
    axb.set_ylim(0, .92); axb.set_ylabel("OOS accuracy", fontsize=7); axb.tick_params(labelsize=6)
    axb.set_title("(c) decision accuracy vs. baselines", fontsize=8)
    if chg_stat is not None:                                     # persistence wins the holds, not the turns
        axb.text(.02, .96, f"at decision changes (n={chg_stat[1]}):\nPBI {chg_stat[0]:.2f}, persist. 0.00",
                 transform=axb.transAxes, fontsize=5.2, va="top", color="0.25",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=.5))

    # (d) lead-lag: Kendall tau of each series with the fed-funds target level as the series is slid
    # forward. The PBI's profile vs two mechanical controls: raw CPI (in the personas' briefing, and a
    # natural leader of the rate this cycle) and the static (no-briefing) index.
    axd = fig.add_subplot(gs[1, 2])
    shifts, ll_tau, cpi_tau, noc_tau = leadlag
    axd.axvspan(0, shifts.max() + .4, color=RED, alpha=.05)      # "leads" region
    axd.axvline(0, color="k", lw=.7, alpha=.45)
    axd.plot(shifts, cpi_tau, ":", color="#999", lw=1.1, label="CPI (control)")
    axd.plot(shifts, noc_tau, "--", color=BLUE, lw=1.0, alpha=.75, label="static idx (control)")
    axd.plot(shifts, ll_tau, "-o", color=RED, lw=1.5, ms=2.6, label="PBI")
    kbest = int(shifts[np.nanargmax(ll_tau)])
    ib = list(shifts).index(kbest)
    axd.annotate(rf"$\tau{{=}}{ll_tau[ib]:.2f}$ at +{kbest}", xy=(kbest, ll_tau[ib]),
                 xytext=(kbest - .5, min(ll_tau[ib] + .1, .93)), fontsize=5.6, color=RED, ha="center")
    axd.text(shifts.max(), .045, "series leads $\\rightarrow$", ha="right", va="bottom", fontsize=5.6,
             color="0.35", style="italic")
    axd.set_xlabel("lead (meetings)", fontsize=7)
    axd.set_ylabel(r"Kendall $\tau$ w/ target level", fontsize=7)
    axd.set_title("(d) the PBI leads the rate level", fontsize=8)
    axd.set_ylim(0, .98); axd.set_xticks(range(-2, 13, 2)); axd.tick_params(labelsize=6)
    axd.legend(fontsize=5.2, loc="lower right", handlelength=1.5, framealpha=.85)

    fig.savefig(FIG / "fig_index.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
