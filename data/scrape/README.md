# data/scrape — building the chunk dataset

The pipeline that produced `helivan/fomc-personas`: scrape each member's public documents, extract
retrievable **chunks** with an LLM, embed them, and write per-member parquet shards.

You do **not** need to run this to use the dataset (it's already published on the Hub). It's here for
transparency and so the dataset can be **kept current**.

## Pipeline

```
scrape (public sources)  ->  extract_chunks (gpt-4o-mini)  ->  embed (text-embedding-3-large)  ->  chunks/<member>.parquet
```

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

`build_dataset.py` does **not** re-scrape from scratch. For each member it reads the chunks already
present, scrapes only documents published **after** the latest one, deduplicates by URL, and extracts
+ embeds **only the new passages**. Re-running when nothing new has been published does no work;
refreshing one member touches only that member's shard.

```bash
# update one member (cheap if nothing new has been published)
OPENAI_API_KEY=sk-...  python data/scrape/build_dataset.py --member "Jerome H. Powell"

# update everyone, then push the changed shards to the Hub
OPENAI_API_KEY=sk-...  python data/scrape/build_dataset.py --all
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='data/chunks', \
  path_in_repo='chunks', repo_id='helivan/fomc-personas', repo_type='dataset')"
```

Because chunks are sharded per member and keyed by a stable `chunk_id`, adding the two not-yet-onboarded
members or refreshing an existing one is a one-shard change, and `load_dataset` still returns one table.
