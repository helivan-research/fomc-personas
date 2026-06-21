"""Figure 1 — corpus composition: chunks by source, per member, by year, and by theme.

Metadata only: no OpenAI key or embeddings needed.

    python paper/fig_data.py
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make fomc_personas importable
import fomc_personas as fp

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})

SRC_LABEL = {"fomc_presconf": "press conf.", "fed_speech": "speeches",
             "fomc_transcript": "transcripts", "fed_testimony": "testimony"}


def main():
    df = fp.load_chunks(embeddings="none")
    member_id = {m: i + 1 for i, m in enumerate(fp.roles.MEMBERS)}  # stable seniority order

    bysrc = df["source"].value_counts()
    by_id = df["member"].map(member_id).value_counts()
    years = pd.to_datetime(df["postedAt"], errors="coerce", utc=True).dt.year.dropna()
    years = Counter(int(y) for y in years if 2006 <= y <= 2026)
    themes = df["theme"].value_counts()

    fig, ax = plt.subplots(2, 2, figsize=(3.4, 2.95))

    items = bysrc.sort_values(ascending=False)
    ax[0, 0].bar([SRC_LABEL.get(k, k) for k in items.index], items.values, color="#4C72B0")
    ax[0, 0].set_ylabel("chunks"); ax[0, 0].set_title("(a) by source", fontsize=8)
    ax[0, 0].set_yticks([0, 5000, 10000]); ax[0, 0].set_yticklabels(["0", "5k", "10k"])
    ax[0, 0].tick_params(axis="x", rotation=30, labelsize=6.5)

    ids = sorted(by_id.index)
    ax[0, 1].bar(ids, [by_id[i] for i in ids], color="#55A868")
    ax[0, 1].set_yscale("log"); ax[0, 1].set_xlabel("member id")
    ax[0, 1].set_ylabel("chunks (log)"); ax[0, 1].set_title("(b) per member", fontsize=8)
    ax[0, 1].set_xticks([1, 5, 10, 15]); ax[0, 1].tick_params(axis="x", labelsize=6.5)

    yk = sorted(years)
    ax[1, 0].bar(yk, [years[y] for y in yk], color="#C44E52")
    ax[1, 0].set_ylabel("chunks"); ax[1, 0].set_title("(c) by year", fontsize=8)
    ax[1, 0].set_yticks([0, 2000]); ax[1, 0].set_yticklabels(["0", "2k"])
    ax[1, 0].tick_params(axis="x", rotation=0, labelsize=6.5)

    short = [t.split(" ")[0].rstrip("&") for t in themes.index]
    ax[1, 1].pie(themes.values, labels=short, autopct="%1.0f%%", labeldistance=1.12,
                 pctdistance=0.62, textprops={"fontsize": 5.5}, colors=plt.cm.tab20.colors, radius=1.0)
    ax[1, 1].set_title("(d) by topic", fontsize=8)

    fig.tight_layout()
    out = FIG / "fig_data.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}  ({len(df)} chunks; sources {dict(bysrc)})")


if __name__ == "__main__":
    main()
