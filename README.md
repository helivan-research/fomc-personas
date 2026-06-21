# fomc-personas

Data and reproduction code for **"A Persona-Based Rate-Action Index"** — digital-twin personas of the
U.S. Federal Open Market Committee (FOMC) built from members' public record, used to recover their
monetary-policy stance and to construct an index that tracks and anticipates rate decisions.

The repository contains:

- **`data/`** — the retrievable-chunk dataset (17 of 19 sitting members, 24,333 chunks, 2006–2026) plus
  the scraping and chunk-extraction pipeline used to build it.
- **`fomc_personas/`** — a small library to load the data, embed/retrieve text, and build personas.
- **`paper/`** — one standalone script per paper figure. Each regenerates its intermediates into a
  gitignored cache and writes the figure PDF; nothing precomputed is committed.

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

| file | what |
|---|---|
| `chunks.parquet` | one row per retrievable chunk (text + metadata; see schema below) |
| `embeddings.parquet` | 1024-d `text-embedding-3-large` embedding per chunk, keyed by `chunk_id` — a ~100 MB asset, **not committed**; fetched on demand (or recomputed) |
| `bios.json` | `{member: biography}` — used as each persona's system prompt |
| `queries/curated.json` | 30 curated monetary-policy questions |
| `queries/pool_72.json` | a balanced 72-query pool with hawk/dove facet labels |
| `queries/anchors.json` | canonical hawkish / dovish anchor statements (the projection axis) |
| `reputational_ordering.json` | external news-derived hawk→dove ordering of 16 scored members |
| `scrape/` | the scraping + LLM chunk-extraction pipeline (`build_dataset.py`) |

**`chunks.parquet` schema:** `chunk_id, member, text, topic, quote, stance, handle, postedAt, source,
sourceId, postUrl, accessedAt, probabilitySpeaker, theme, is_voting, is_chair`. `stance` is a
self-contained sentence stating the member's position; `theme` is one of six macro themes;
`is_voting`/`is_chair` are as-of the statement date.

### Embeddings: cached vs computed

`load_chunks(embeddings=...)`:

- `"cached"` — download the hosted `embeddings.parquet` once into the local cache (no OpenAI calls).
- `"compute"` — embed the chunks with your own key (`text-embedding-3-large`, ≈ \$0.20 for the full corpus); cached locally so it only happens once.
- `"none"` — text + metadata only (e.g. Figure 1).

## Reproducing the figures

Each `paper/fig_*.py` is self-contained: it loads the data, regenerates any intermediates into
`paper/.cache/figure_<name>/` (reused on subsequent runs), and writes `paper/figures/fig_<name>.pdf`.
`fig_data` needs no API key; the others call OpenAI (generation + embeddings). `fig_index` is the
costly one (per-meeting generation across the historical roster).

## License

Code: MIT (see `LICENSE`). The dataset is derived entirely from public records; please cite the paper
if you use it.
