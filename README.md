# fomc-personas

Data and reproduction code for **"A Persona-Based Rate-Action Index"** — digital-twin personas of the
U.S. Federal Open Market Committee (FOMC) built from members' public record, used to recover their
monetary-policy stance and to construct an index that tracks and anticipates rate decisions.

The personas power an interactive site: **[federalreserve.ai](https://federalreserve.ai)**.

The repository contains:

- **`data/`** — the retrievable-chunk dataset (17 of 19 sitting members, 24,333 chunks, 2006–2026) plus
  the scraping and chunk-extraction pipeline used to build it.
- **`fomc_personas/`** — a small library to load the data, embed/retrieve text, and build personas.
- **`paper/`** — one standalone script per paper figure. Each regenerates its intermediates into a
  gitignored cache and writes the figure PDF; nothing precomputed is committed.

The chunk corpus and embeddings are published as a Hugging Face dataset —
**[`helivan/fomc-personas`](https://huggingface.co/datasets/helivan/fomc-personas)** — which this
library downloads on demand (`load_chunks(embeddings="cached")`).

## Quickstart

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...          # your own key

python paper/fig_data.py             # Figure 1 — corpus composition (no API key needed)
python paper/fig_likeness.py         # Figure 2 — persona likeness
python paper/fig_stance.py           # Figure 3 — stance recovery
python paper/fig_index.py            # Figure 4 — the rate-action index
```

```python
from fomc_personas import load_chunks, embed, retrieve

df = load_chunks(embeddings="cached")           # DataFrame; df["embedding"] populated from the hosted asset
q  = embed(["Is inflation too high right now?"])
hits = retrieve(df, q, member="Jerome H. Powell", k=3)
```

## The dataset (`data/`)

| file | where | what |
|---|---|---|
| `chunks.parquet` | [HF dataset](https://huggingface.co/datasets/helivan/fomc-personas) (not in git) | one row per retrievable chunk (text + metadata; see schema below) |
| `embeddings.parquet` | HF dataset (not in git) | 1024-d `text-embedding-3-large` embedding per chunk, keyed by `chunk_id` (~100 MB); fetched on demand or recomputed |
| `bios.json` | committed | `{member: biography}` — used as each persona's system prompt |
| `queries/curated.json` | committed | 30 curated monetary-policy questions |
| `queries/pool_72.json` | committed | a balanced 72-query pool with hawk/dove facet labels |
| `queries/anchors.json` | committed | canonical hawkish / dovish anchor statements (the projection axis) |
| `reputational_ordering.json` | committed | external news-derived hawk→dove ordering of 16 scored members |
| `scrape/` | committed | the scraping + LLM chunk-extraction pipeline (`build_dataset.py`) |

**`chunks.parquet` schema:** `chunk_id, member, text, topic, quote, stance, handle, postedAt, source,
sourceId, postUrl, accessedAt, probabilitySpeaker, theme, is_voting, is_chair`. `stance` is an
LLM-paraphrased, self-contained sentence stating the member's position — it is what gets *embedded*
(retrieval similarity runs over the paraphrases; the original `text` is what personas read);
`theme` is one of six macro themes; `is_voting`/`is_chair` are as-of the statement date.

**Point-in-time caveat.** FOMC meeting *transcripts* (`source == "fomc_transcript"`, ~32% of chunks)
are dated at the meeting they record but are published with a ~5-year embargo (year *Y* releases
~Jan *Y+6*). Any backtest filtering on `postedAt` alone would retrieve text that was not public at
the time; the paper's backtests use the availability-corrected filter in `paper/fig_index.py`
(`_public_asof`) instead. Live use today is unaffected (all transcripts in the corpus are now public).

### Embeddings: cached vs computed

`load_chunks(embeddings=...)`:

- `"cached"` — download the embeddings from the [Hugging Face dataset](https://huggingface.co/datasets/helivan/fomc-personas) once into the local cache (no OpenAI calls).
- `"compute"` — embed the chunks with your own key (`text-embedding-3-large`, ≈ \$0.20 for the full corpus); cached locally so it only happens once.
- `"none"` — text + metadata only (e.g. Figure 1).

## Reproducing the figures

Each `paper/fig_*.py` is self-contained: it loads the data, regenerates any intermediates into
`paper/.cache/` (reused on subsequent runs), and writes `paper/figures/fig_<name>.pdf`. Cost ladder:

- **free, no API key** — `fig_data.py` (corpus composition).
- **cheap (embeddings only, ~cents)** — `fig_likeness.py` / `fig_stance.py` after a first run has
  cached the generations; first run of each is a few dollars of `gpt-4o-mini`.
- **the costly one** — `fig_index.py`: per-meeting generation across the historical roster
  (~10k+ generations for the PBI pass; the CV in `experiments/retrieval_cv.py` is a full pass *per
  (beta, tau) setting*). With the caches under `paper/.cache/retrieval_cv/beta0.6_tau2.0_pit/` and
  `paper/.cache/figure_index_pit/` present it replots in seconds with no paid calls. Note that a
  fresh regeneration is *stochastic* (temperature 0.2), so exact headline numbers reproduce only
  from the caches.

## Evaluation protocol (and honest caveats)

The headline decision-forecasting numbers come from a walk-forward 3-class (hike/hold/cut)
classifier on the committee index + its momentum: train on meetings 1..i-1, predict meeting i,
starting at meeting 17; the evaluation window is 2022+ (n≈32). Retrieval is point-in-time
(`_public_asof`: embargoed transcripts dated at release; strict day-before cutoff). Current numbers:

- **OOS accuracy 0.66** vs a 0.47 majority-class base rate. The **persistence baseline
  ("repeat the last decision") scores 0.78** on this hold-heavy window — but 0.00 at the 7
  decision *changes*, where the PBI scores 0.29 (Taylor 0.43); no signal is good at turns yet.
- **Lead-lag (the main result):** slid forward, the PBI's Kendall tau against the funds-rate
  *level* rises from 0.38 (contemporaneous) to **~0.79 at +9 meetings (~3 quarters)**, exceeding
  both a raw-CPI control (0.67) and the no-briefing static index (0.57) at the same lead.
- **Discrimination (2022+):** hike-vs-rest AUC 0.98, cut-vs-rest AUC 0.83.
- **Selection caveat:** the retrieval weighting (beta=0.6, tau=2yr) was chosen by
  `experiments/retrieval_cv.py` on the *same* 2022+ window, so cross-setting deltas are
  in-sample-selected; treat them as sensitivity analysis, not out-of-sample gains.

## License

Code: MIT (see `LICENSE`). The dataset is derived entirely from public records; please cite the paper
if you use it.
