# data/scrape — building the chunk dataset

The pipeline that produced `helivan/fomc-personas`: scrape each member's public documents, extract
retrievable **chunks** with an LLM, embed them, and write per-member parquet shards.

You do **not** need to run this to use the dataset (it's already published on the Hub). It's here for
transparency and so the dataset can be **kept current**.

## Pipeline

```
scrape (public sources) -> extract_chunks (gpt-4o-mini) -> embed (text-embedding-3-large)
  -> chunks/<member>.parquet (text + metadata + chunk_id) + embeddings/<member>.parquet (chunk_id + vector)
```

The dataset is **split**: a `chunks/` shard (text + metadata) and an `embeddings/` shard
(`chunk_id`, `embedding`) per member, joined on a global integer `chunk_id`. This is exactly the
published HF layout, so `build_dataset.py` reads and writes it directly.

| file | role |
|---|---|
| `fed_reserve_scraper.py` | federalreserve.gov speeches, testimony, bio (governors / chair) |
| `fed_bank_scraper.py` | BIS speech archive + home-bank sites (regional presidents) |
| `fomc_transcript_scraper.py` | FOMC meeting transcripts (5-year embargo; all members) |
| `fomc_pressconf_scraper.py` | FOMC press conferences (chair) |
| `extract_chunks.py` | LLM extraction of stance/quote/topic per passage (OpenAI structured outputs) |
| `topic_themes.py` | bucket a free-text topic into one of six macro themes |
| `build_dataset.py` | the orchestrator (below) |

All scrapers fetch only public data and take a `start_date` lower bound. Extraction and embedding use
`OPENAI_API_KEY`.

## Incremental & idempotent

`build_dataset.py` does **not** re-scrape from scratch. For each member it reads the shards already
present, scrapes only documents published **after** the latest one, skips URLs already seen (document
level — one document yields many chunks, so there is no chunk-level dedup), and extracts + embeds
**only the new passages**. New chunks get sequential `chunk_id`s continuing from the global maximum.
Re-running when nothing new has been published does no work; refreshing one member touches only that
member's two shards.

It operates on a working copy of the dataset (`--data-dir`, default `$FOMC_DATASET_DIR` or
`data/.hf_dataset/`, gitignored). `--pull` downloads the current dataset from the Hub first; `--push`
uploads the changed shards back when done.

```bash
# update one member into the local working copy (cheap if nothing new has been published)
OPENAI_API_KEY=sk-...  python data/scrape/build_dataset.py --member "Jerome H. Powell"

# pull the published dataset, update everyone, push the changed shards back to the Hub
OPENAI_API_KEY=sk-...  HF_TOKEN=hf_...  python data/scrape/build_dataset.py --all --pull --push
```

Because each member is two small shards keyed by a stable `chunk_id`, refreshing one (or adding a
not-yet-onboarded member) is a one-shard change, and `load_dataset` still returns one joined table.

## Scheduled refresh (GitHub Actions)

`.github/workflows/update-dataset.yml` runs the `--all --pull --push` command **weekly** (Mondays
06:00 UTC; also runnable on demand via *workflow_dispatch*). A run with nothing new pushes nothing.
It needs two repository secrets:

| secret | used for |
|---|---|
| `OPENAI_API_KEY` | chunk extraction (gpt-4o-mini) + embeddings (text-embedding-3-large) |
| `HF_TOKEN` | write access to the `helivan/fomc-personas` dataset (also reads it for `--pull`) |
