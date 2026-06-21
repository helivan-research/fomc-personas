"""Figure 3 — stance recovery against the external hawk->dove reputational ordering.

Row 1: rank-rank recovery at k=3 for the three member representations (retrieval / corpus / persona).
Row 2: (d) tau vs retrieval depth k, (e) tau vs query-set size, (f) per-query signal by query type.

Persona generation is cached under paper/.cache/figure_stance/ — the first run calls OpenAI
(generation + embeddings, a few minutes / ~$1-2); reruns plot instantly.

    OPENAI_API_KEY=sk-...  python paper/fig_stance.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fomc_personas as fp

CACHE = Path(__file__).resolve().parent / ".cache" / "figure_stance"
CACHE.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})

# 3 topics -> the 12 hawk-dove subtopics of the 72-query pool (panel f grouping)
STANCE_GROUPS = {
    "mandate priorities": [
        "weight on labor-market slack versus wage-driven inflation",
        "priority of price stability versus maximum employment",
        "willingness to accept higher unemployment to lower inflation",
        "weight on financial-stability and growth risks"],
    "inflation strictness": [
        "tolerance for inflation above the 2 percent target",
        "concern about inflation expectations becoming unanchored",
        "urgency of returning inflation to the 2 percent target",
        "how restrictive policy should be"],
    "tactics & timing": [
        "how preemptively to tighten against anticipated inflation",
        "patience before cutting rates as inflation eases",
        "tolerance for recession risk in pursuit of disinflation",
        "appetite for a higher neutral rate / restrictive stance for longer"],
}


def cached_respond(df, members, queries, bios, k, tag):
    p = CACHE / f"resp_{tag}.json"
    if p.exists():
        return json.loads(p.read_text())
    print(f"  generating responses ({tag}: {len(members)}x{len(queries)} @ k={k}) ...")
    out = fp.respond(df, members, queries, bios, k=k)
    p.write_text(json.dumps(out))
    return out


def main():
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    ext = fp.load_reputational()
    u = fp.axis(fp.load_anchors())
    allmem = sorted(df["member"].unique())
    members = [n for n in allmem if n in ext]          # the 16 scored members
    ext_rank = {n: i for i, n in enumerate(ext)}
    queries = fp.load_queries("curated")               # 30

    def tau_of(rep):
        return fp.kendall_vs_external(fp.project(rep, u), ext)[1]

    # --- row 1: rank-rank at k=3 ---
    resp3 = cached_respond(df, allmem, queries, bios, 3, "curated_k3")
    reps = [("retrieval-based", fp.meanpool_retrieved(df, queries, 3)),
            ("corpus average", fp.meanpool_corpus(df)),
            ("persona-based", fp.meanpool_generated(resp3))]
    rankrank = []
    for name, rep in reps:
        score = fp.project(rep, u)
        order = sorted(members, key=lambda n: -score[n])
        rr = {n: i for i, n in enumerate(order)}
        rankrank.append((name, [rr[m] for m in members], tau_of(rep)))

    # --- (d) tau vs retrieval depth k ---
    ks = list(range(6))
    corpus_tau = tau_of(fp.meanpool_corpus(df))
    ret_tau, per_tau = [], []
    for k in ks:
        resp_k = resp3 if k == 3 else cached_respond(df, allmem, queries, bios, k, f"curated_k{k}")
        per_tau.append(tau_of(fp.meanpool_generated(resp_k)))
        if k == 0:
            ret_tau.append(tau_of({n: fp.embed([bios[n]])[0] for n in allmem}))
        else:
            ret_tau.append(tau_of(fp.meanpool_retrieved(df, queries, k)))

    # --- (e) query-set size within the curated 30 (persona, k=3) ---
    proj = {n: fp.embed([r or " " for r in resp3[n]]) @ u for n in members}
    mq = min(len(resp3[n]) for n in members)
    rng = default_rng(0)

    def rho_sub(qidx):
        return fp.kendall_vs_external({n: float(np.mean(proj[n][qidx])) for n in members}, ext)[1]
    qcount = {}
    for s in [1, 2, 3, 5, 8, 12, 16, 20]:
        if s < mq:
            vals = [rho_sub(rng.choice(mq, s, replace=False)) for _ in range(40)]
            qcount[s] = (float(np.mean(vals)), float(np.std(vals)))

    # --- (f) per-query signal over the 72-query pool, grouped by 3 topics ---
    pool = fp.load_queries("pool")
    pool_q = [p["q"] for p in pool]
    resp_pool = cached_respond(df, allmem, pool_q, bios, 3, "pool72_k3")
    pproj = {n: fp.embed([r or " " for r in resp_pool[n]]) @ u for n in members}
    sub2grp = {s: g for g, ss in STANCE_GROUPS.items() for s in ss}
    grp = {g: [] for g in STANCE_GROUPS}
    for i, meta in enumerate(pool):
        g = sub2grp.get(meta["subtopic"])
        if g is None:
            continue
        rho = fp.kendall_vs_external({n: float(pproj[n][i]) for n in members}, ext)[1]
        grp[g].append(rho)
    gmean = {g: float(np.mean(v)) for g, v in grp.items() if v}

    _plot(len(members), [ext_rank[m] for m in members], rankrank, ks, corpus_tau, ret_tau, per_tau,
          qcount, mq, gmean)
    print(f"wrote {FIG/'fig_stance.pdf'}  (corpus tau={corpus_tau:.2f}, persona k=3 tau={per_tau[3]:.2f})")


def _plot(nmem, ext_ranks, rankrank, ks, corpus_tau, ret_tau, per_tau, qcount, nq, gmean):
    blue, orange, grey = "#4C72B0", "#DD8452", "0.55"
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.34, height_ratios=[1.3, 1])

    for j, (name, est_ranks, tau) in enumerate(rankrank):
        a = fig.add_subplot(gs[0, j])
        a.plot([0, nmem - 1], [0, nmem - 1], ls="--", color="0.6", lw=0.7, zorder=0)
        a.scatter(ext_ranks, est_ranks, c=ext_ranks, cmap="coolwarm_r", s=22, edgecolor="k", lw=.25, zorder=2)
        a.set_title(f"({chr(97 + j)}) {name}", fontsize=7, pad=2)
        a.text(0.05, 0.95, f"$\\tau{{=}}{tau:.2f}$", transform=a.transAxes, fontsize=6, va="top")
        a.set_xticks([0, nmem - 1]); a.set_xticklabels(["hawk", "dove"], fontsize=5.5)
        a.set_yticks([0, nmem - 1]); a.set_xlabel("reputation rank", fontsize=6.5, labelpad=1)
        a.tick_params(length=2, pad=1.5); a.set_xlim(-0.7, nmem - 0.3); a.set_ylim(-0.7, nmem - 0.3)
        a.set_aspect("equal", adjustable="box")
        if j == 0:
            a.set_yticklabels(["hawk", "dove"], fontsize=5.5); a.set_ylabel("estimate rank", fontsize=6.5, labelpad=1)
        else:
            a.set_yticklabels([])

    ad = fig.add_subplot(gs[1, 0])
    ad.axhline(corpus_tau, ls="--", color=grey, lw=1.0, label="corpus average")
    ad.plot(ks, ret_tau, "-o", color=orange, ms=3, lw=1.1, label="retrieval-based")
    ad.plot(ks, per_tau, "-o", color=blue, ms=3, lw=1.1, label="persona-based")
    ad.set_xlabel("retrieval depth $k$", fontsize=7); ad.set_ylabel(r"Kendall $\tau$", fontsize=7)
    ad.set_title("(d) recovery vs. retrieval depth", fontsize=7); ad.set_ylim(0, 0.75)
    ad.tick_params(labelsize=6); ad.legend(fontsize=5, frameon=False, loc="lower center")

    ae = fig.add_subplot(gs[1, 1])
    xs = sorted(qcount)
    mean = np.array([qcount[s][0] for s in xs]); sd = np.array([qcount[s][1] for s in xs])
    ae.fill_between(xs, mean - sd, mean + sd, color=blue, alpha=0.15)
    ae.plot(xs, mean, "-o", color=blue, ms=3, lw=1.1, label="random subsets")
    ae.scatter([nq], [per_tau[3]], marker="*", s=70, color=orange, zorder=5, edgecolor="k", lw=.3)
    ae.text(nq - 1, per_tau[3] + 0.02, "full set", fontsize=5.5, va="bottom", ha="right", color=orange)
    ae.set_xlabel("number of queries", fontsize=7); ae.set_ylabel(r"Kendall $\tau$", fontsize=7)
    ae.set_title("(e) recovery vs. query-set size", fontsize=7); ae.set_ylim(0, 0.75); ae.set_xlim(0, nq + 1)
    ae.tick_params(labelsize=6); ae.legend(fontsize=5, frameon=False, loc="lower right")

    af = fig.add_subplot(gs[1, 2])
    order = sorted(gmean, key=lambda g: gmean[g]); vals = [gmean[g] for g in order]
    af.barh(range(len(order)), vals, color=blue, height=0.5)
    for i, (g, v) in enumerate(zip(order, vals)):
        af.text(0.005, i + 0.32, g, va="bottom", ha="left", fontsize=6.5)
        af.text(v + 0.006, i, f"{v:.2f}", va="center", fontsize=6)
    af.set_yticks([]); af.set_ylim(-0.5, len(order) - 0.2)
    af.set_xlabel(r"mean per-query Kendall's $\tau$", fontsize=7)
    af.set_title("(f) per-query signal by type", fontsize=7)
    af.tick_params(labelsize=6); af.set_xlim(0, max(vals) * 1.2)

    fig.savefig(FIG / "fig_stance.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
