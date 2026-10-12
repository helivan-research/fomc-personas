"""Figure 5 -- the vote-share index and the probabilistic forecast.

(a) Committee vote shares over time: share of persona answers endorsing a hike (s_hike) and a cut
    (s_cut) at each meeting, with realized decisions marked.
(b) Net vote share (s_hike - s_cut) by realized decision, 2022-25.
(c) The canonical forecast's probability of the realized outcome class by decision -- P(hike) at hike
    meetings, P(hold) at holds, P(cut) at cuts -- against hedged persistence.

Reads the cached vote-battery parses (paper/.cache/vote_parse) and the canonical forecast in
paper/experiments/probability_index.py; no paid calls. Prints the class means / d' used in the text.

    python paper/fig_votes.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fomc_personas import macro                     # noqa: E402
import probability_index as PI                      # noqa: E402
from proper_scoring_table import persist_hedged     # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
RED, BLUE, GREY = "#C44E52", "#4C72B0", "#8a8f98"
CLS_COLOR = {1: RED, 0: GREY, -1: BLUE}
CLS_NAME = {1: "hike", 0: "hold", -1: "cut"}


def dprime(v, mask_a, mask_b):
    a, b = v[mask_a], v[mask_b]
    return abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()))


def main():
    series = macro.load_fred()
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + macro.completed_live_meetings(series)
    b = PI.build()
    dates, y, P, sup = b["dates"], b["y"], b["P"], PI.load_support()[0]
    s_h = np.array([np.nanmean([v[0] for v in sup[d].values()]) for d in dates])
    s_c = np.array([np.nanmean([v[1] for v in sup[d].values()]) for d in dates])
    net = s_h - s_c
    y_full, z_full, off = PI.history_context(y, b["zlb"])
    Pp = persist_hedged(y_full, start=off + PI.START)[off:]

    m22 = np.array(["2022" <= d < "2026" for d in dates])
    print("net vote share by decision (2022-25):")
    for c in (1, 0, -1):
        m = m22 & (y == c)
        print(f"  {CLS_NAME[c]:>4}: mean {net[m].mean():+.3f} sd {net[m].std():.3f} (n={m.sum()})")
    print(f"  d'(hike,hold)={dprime(net, m22 & (y == 1), m22 & (y == 0)):.2f}  "
          f"d'(cut,hold)={dprime(net, m22 & (y == -1), m22 & (y == 0)):.2f}")
    ok = ~np.isnan(P[:, 0])
    p_true = np.array([P[i, PI.CLS.index(y[i])] if ok[i] else np.nan for i in range(len(y))])
    p_true_pers = np.array([Pp[i, PI.CLS.index(y[i])] if ok[i] else np.nan for i in range(len(y))])
    for c in (1, 0, -1):
        m = ok & (y == c) & np.array([d >= "2019" for d in dates])
        print(f"  median P(realized) at {CLS_NAME[c]} meetings: ours {np.nanmedian(p_true[m]):.2f} "
              f"| persistence {np.nanmedian(p_true_pers[m]):.2f} (n={m.sum()})")

    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1, 1.15], wspace=0.38)

    a = fig.add_subplot(gs[0, 0])
    x = np.array([np.datetime64(d) for d in dates])
    a.plot(x, s_h, color=RED, lw=1.2, label=r"$s_{\mathrm{hike}}$")
    a.plot(x, s_c, color=BLUE, lw=1.2, label=r"$s_{\mathrm{cut}}$")
    for c in (1, -1):
        m = y == c
        a.scatter(x[m], np.where(c == 1, s_h, s_c)[m], s=9, color=CLS_COLOR[c], zorder=3)
    a.set_ylim(0, 1); a.set_ylabel("share endorsing", fontsize=7)
    a.set_title("(a) committee vote shares (dots: hikes, cuts)", fontsize=7)
    a.tick_params(labelsize=6); a.legend(fontsize=6, frameon=False, loc="upper left", ncol=2)

    a2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    for j, c in enumerate((1, 0, -1)):
        m = m22 & (y == c)
        a2.scatter(j + rng.uniform(-0.13, 0.13, m.sum()), net[m], s=11, color=CLS_COLOR[c], alpha=0.8, lw=0)
        a2.hlines(net[m].mean(), j - 0.28, j + 0.28, color=CLS_COLOR[c], lw=2)
    a2.set_xticks([0, 1, 2]); a2.set_xticklabels(["hike", "hold", "cut"], fontsize=6.5)
    a2.set_ylabel(r"$s_{\mathrm{hike}}-s_{\mathrm{cut}}$", fontsize=7)
    a2.set_title("(b) net share by decision (2022-25)", fontsize=7); a2.tick_params(labelsize=6)

    a3 = fig.add_subplot(gs[0, 2])
    for j, c in enumerate((1, 0, -1)):
        m = ok & (y == c) & np.array([d >= "2019" for d in dates])
        xs = j + rng.uniform(-0.1, 0.1, m.sum())
        a3.scatter(xs - 0.16, p_true_pers[m], s=8, color="0.7", lw=0, label="persistence" if j == 0 else None)
        a3.scatter(xs + 0.16, p_true[m], s=10, color=CLS_COLOR[c], lw=0, label="ours" if j == 0 else None)
    a3.set_xticks([0, 1, 2]); a3.set_xticklabels(["hike", "hold", "cut"], fontsize=6.5)
    a3.set_ylim(0, 1); a3.set_ylabel("P(realized decision)", fontsize=7)
    a3.set_title("(c) forecast vs. persistence", fontsize=7); a3.tick_params(labelsize=6)
    a3.legend(fontsize=5.5, frameon=False, loc="lower left")

    for ax in (a, a2, a3):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "fig_votes.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {FIG / 'fig_votes.pdf'}")


if __name__ == "__main__":
    main()
