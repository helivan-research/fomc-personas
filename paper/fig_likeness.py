"""Figure 2 — likeness of the persona collection: identifiability (blue) and detectability (orange).

Col 1: high/low example densities. Col 2: each axis vs retrieval depth k. Col 3: (c) the two axes
are uncorrelated; (f) neither tracks corpus size.

Identifiability reuses the curated-30 persona responses (shared with fig_stance's cache). Detectability
generates seeded held-out completions across k=0..5 — the heaviest figure; cached under
paper/.cache/figure_likeness/. First run ~$3-5 / ~20 min; reruns plot instantly.

    OPENAI_API_KEY=sk-...  python paper/fig_likeness.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fomc_personas as fp

KS = (0, 1, 2, 3, 4, 5)
CACHE = Path(__file__).resolve().parent / ".cache" / "figure_likeness"
CACHE.mkdir(parents=True, exist_ok=True)
RESP_DIRS = [CACHE, Path(__file__).resolve().parent / ".cache" / "figure_stance"]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})


def _responses(df, members, queries, bios, k):
    """Curated-30 persona responses at depth k — reused from fig_stance's cache if present."""
    for d in RESP_DIRS:
        p = d / f"resp_curated_k{k}.json"
        if p.exists():
            return json.loads(p.read_text())
    print(f"  generating identifiability responses (k={k}) ...")
    out = fp.respond(df, members, queries, bios, k=k)
    (CACHE / f"resp_curated_k{k}.json").write_text(json.dumps(out))
    return out


def _completions(pool, test, bios, k):
    p = CACHE / f"completions_k{k}.json"
    if p.exists():
        return json.loads(p.read_text())
    print(f"  generating seeded completions (k={k}) ...")
    comps = fp.likeness.complete_seeded(pool, test, bios, k)
    p.write_text(json.dumps(comps))
    return comps


def main():
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    queries = fp.load_queries("curated")
    allmem = sorted(df["member"].unique())
    pool, test = fp.likeness.split(df)

    # detectability across k
    detect = {}
    for k in KS:
        detect[str(k)] = fp.likeness.detect_taus(_completions(pool, test, bios, k))
    comps3, comps0 = _completions(pool, test, bios, 3), _completions(pool, test, bios, 0)
    _, fl = fp.likeness.detect_taus(comps3, with_floor=True)
    floor = float(np.median(list(fl.values())))
    lo_d = min(detect["3"], key=detect["3"].get)
    hi_d = max(detect["0"], key=detect["0"].get)
    detect_example = {"low": {**fp.likeness.overlap_example(comps3, lo_d), "member": lo_d, "k": 3},
                      "high": {**fp.likeness.overlap_example(comps0, hi_d), "member": hi_d, "k": 0}}

    # identifiability (uniqueness) across k
    uniq = {}
    members_u = None
    for k in KS:
        members_u, recall, _ = fp.likeness.loqo_recall(_responses(df, allmem, queries, bios, k))
        uniq[str(k)] = recall
    u3 = uniq["3"]
    hi_u, lo_u = max(u3, key=u3.get), min(u3, key=u3.get)
    s_hi, r_hi = fp.likeness.nway_scores(_responses(df, allmem, queries, bios, 3), hi_u)
    s_lo, r_lo = fp.likeness.nway_scores(_responses(df, allmem, queries, bios, 3), lo_u)
    uniq_example = {"high": {"member": hi_u, "recall": u3[hi_u], "self": s_hi, "rest": r_hi},
                    "low": {"member": lo_u, "recall": u3[lo_u], "self": s_lo, "rest": r_lo}}

    # corpus-size relation
    corpus = {m: int((df["member"] == m).sum()) for m in members_u}
    cm = [m for m in members_u if m in u3 and m in detect["3"]]
    id_t, id_p = kendalltau([corpus[m] for m in cm], [u3[m] for m in cm])
    dt_t, dt_p = kendalltau([corpus[m] for m in cm], [detect["3"][m] for m in cm])
    corpus_rel = {"chunks": corpus, "id_tau": float(id_t), "id_p": float(id_p),
                  "det_tau": float(dt_t), "det_p": float(dt_p)}

    L = {"members": members_u, "ks": list(KS), "floor": floor, "detect": detect,
         "detect_example": detect_example,
         "uniqueness": {"chance": 1.0 / len(members_u), "ks": list(KS), "per_member": uniq,
                        "example": uniq_example, "corpus_size": corpus_rel}}
    _plot(L)
    print(f"wrote {FIG/'fig_likeness.pdf'}  (identifiability recall@3 mean="
          f"{np.mean(list(u3.values())):.2f}, detectability tau@3 median="
          f"{np.median(list(detect['3'].values())):.2f}, floor={floor:.2f})")


def _example_hist(a, d, sk, rk, color, slabel, rlabel, tag):
    s = np.array(d[sk], float); r = np.array(d[rk], float)
    allv = np.concatenate([s, r]); mu, sd = allv.mean(), allv.std() + 1e-9
    s, r = (s - mu) / sd, (r - mu) / sd
    bins = np.linspace(-3, 3, 14)
    a.hist(r, bins=bins, density=True, color="grey", alpha=0.55, label=rlabel)
    a.hist(s, bins=bins, density=True, color=color, alpha=0.72, label=slabel)
    a.set_xlim(-3, 3); a.set_xticks([-3, 0, 3]); a.set_yticks([]); a.grid(False)
    a.tick_params(axis="x", labelsize=6)
    a.text(0.03, 0.95, tag, transform=a.transAxes, fontsize=6, va="top", linespacing=1.2)


def _member_panel(a, pm, ks, members, color, refs, ylabel, title, ylim, xlabel=None):
    for mi, name in enumerate(members):
        xs = [k for k in ks if pm[str(k)].get(name) is not None]
        ys = [pm[str(k)][name] for k in xs]
        if len(xs) >= 2:
            a.plot(xs, ys, "-", color=color, lw=0.5, alpha=0.16)
        a.scatter([x + (mi % 5 - 2) * 0.03 for x in xs], ys, s=6, color=color, alpha=0.4, edgecolors="none")
    agg = [np.mean([v for v in pm[str(k)].values() if v is not None]) for k in ks]
    a.plot(ks, agg, "o-", color=color, lw=1.7, ms=4, label="mean", zorder=5)
    for val, lbl, c in refs:
        a.axhline(val, color=c, ls=":", lw=.8)
        a.text(min(ks), val, lbl + " ", fontsize=6, color=c, ha="left", va="bottom")
    a.set_ylabel(ylabel, fontsize=7); a.set_title(title, fontsize=8); a.set_ylim(*ylim)
    a.set_xticks(ks); a.tick_params(axis="both", labelsize=6)
    if xlabel:
        a.set_xlabel(xlabel, fontsize=7)
    a.legend(fontsize=6, loc="best")


def _plot(L):
    floor, members = L["floor"], L["members"]
    uchance = L["uniqueness"]["chance"]
    blue, orange = "#4C72B0", "#DD8452"
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.72, 1.12, 1.0], hspace=0.42, wspace=0.42)

    ue = L["uniqueness"]["example"]
    gsa = gs[0, 0].subgridspec(2, 1, hspace=0.4)
    a_hi, a_lo = fig.add_subplot(gsa[0]), fig.add_subplot(gsa[1])
    _example_hist(a_hi, ue["high"], "self", "rest", blue, "member", "others",
                  f"{ue['high']['member'].split()[-1]}\n(high, {ue['high']['recall']:.2f})")
    _example_hist(a_lo, ue["low"], "self", "rest", blue, "member", "others",
                  f"{ue['low']['member'].split()[-1]}\n(low, {ue['low']['recall']:.2f})")
    a_hi.set_title("(a) identifiability", fontsize=8); a_hi.set_xticklabels([]); a_lo.set_xticklabels([])

    de = L["detect_example"]
    gsd = gs[1, 0].subgridspec(2, 1, hspace=0.4)
    d_lo, d_hi = fig.add_subplot(gsd[0]), fig.add_subplot(gsd[1])
    _example_hist(d_lo, de["low"], "gen", "real", orange, "generated", "real",
                  f"{de['low']['member'].split()[-1]}\n($k{{=}}{de['low']['k']}$, $\\hat\\tau{{=}}{de['low']['tau']:.2f}$)")
    _example_hist(d_hi, de["high"], "gen", "real", orange, "generated", "real",
                  f"{de['high']['member'].split()[-1]}\n($k{{=}}{de['high']['k']}$, $\\hat\\tau{{=}}{de['high']['tau']:.2f}$)")
    d_lo.set_title("(d) detectability", fontsize=8); d_lo.set_xticklabels([]); d_hi.set_xlabel("discriminant", fontsize=6)

    ax_b = fig.add_subplot(gs[1, 2])
    cs = L["uniqueness"]["corpus_size"]; u3 = L["uniqueness"]["per_member"]["3"]; t3 = L["detect"]["3"]
    cm = [m for m in members if m in cs["chunks"] and m in u3 and m in t3]
    cx = np.log10([cs["chunks"][m] for m in cm])
    for yvals, col, lab in (([u3[m] for m in cm], blue, "identifiability"),
                            ([t3[m] for m in cm], orange, "detectability $\\hat\\tau$")):
        ax_b.scatter(cx, yvals, s=11, color=col, alpha=0.85, edgecolors="none", label=lab)
        af, bf = np.polyfit(cx, yvals, 1); xs = np.array([cx.min(), cx.max()])
        ax_b.plot(xs, af * xs + bf, color=col, lw=1.0, ls="--", alpha=0.9)
    star = "n.s." if cs["id_p"] >= 0.1 else f"$p{{=}}{cs['id_p']:.2f}$"
    ax_b.text(0.96, 0.96, f"id $\\tau={cs['id_tau']:+.2f}$ ({star})\ndet $\\tau={cs['det_tau']:+.2f}$ (n.s.)",
              transform=ax_b.transAxes, fontsize=6, va="top", ha="right", linespacing=1.3)
    ax_b.legend(fontsize=5.5, loc="lower left", frameon=False, handletextpad=0.3, borderpad=0.1)
    ax_b.set_xlabel("corpus size (chunks, log scale)", fontsize=7); ax_b.set_ylabel("likeness axis", fontsize=7)
    ax_b.set_ylim(-0.02, 1.02); ax_b.set_xticks([2, 3, 4]); ax_b.set_xticklabels(["$10^2$", "$10^3$", "$10^4$"])
    ax_b.set_title("(f) neither axis tracks corpus size", fontsize=8); ax_b.tick_params(labelsize=6)

    ax_c = fig.add_subplot(gs[0, 1])
    _member_panel(ax_c, L["uniqueness"]["per_member"], L["uniqueness"]["ks"], members, blue,
                  [(uchance, "chance", "red")], "attribution recall", "(b) identifiability vs $k$", (0, 1.02))
    ax_c.tick_params(labelbottom=False)

    ax_e = fig.add_subplot(gs[0, 2])
    mm = [m for m in members if m in u3 and m in t3]
    ux, ty = [u3[m] for m in mm], [t3[m] for m in mm]
    ax_e.scatter(ux, ty, s=12, color="#7B68A6", alpha=0.85, edgecolors="none")
    af, bf = np.polyfit(ux, ty, 1); xs = np.array([min(ux), max(ux)])
    ax_e.plot(xs, af * xs + bf, color="0.4", lw=1.0, ls="--")
    ax_e.text(0.96, 0.04, f"Kendall $\\tau={kendalltau(ux, ty)[0]:+.2f}$ (n.s.)",
              transform=ax_e.transAxes, fontsize=6, va="bottom", ha="right")
    ax_e.set_ylabel("detectability $\\hat\\tau$", fontsize=7)
    ax_e.set_title("(c) the two axes are uncorrelated", fontsize=8); ax_e.tick_params(labelsize=6, labelbottom=False)

    ax_f = fig.add_subplot(gs[1, 1])
    _member_panel(ax_f, L["detect"], L["ks"], members, orange, [(floor, "real--real floor", "grey")],
                  "detectability $\\hat\\tau$", "(e) detectability vs $k$", (0, 1.02),
                  xlabel="retrieval depth $k$ ($k{=}0$: no retrieval)")

    fig.savefig(FIG / "fig_likeness.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
