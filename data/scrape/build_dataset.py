"""Build or **incrementally update** the FOMC-personas chunk dataset.

For each member the pipeline is: scrape public documents -> extract retrievable chunks (LLM) ->
embed -> per-member parquet shard. It is **incremental and idempotent**: it reads whatever chunks a
member already has, scrapes only documents published *after* the latest one present, deduplicates by
URL, and extracts + embeds only the new passages. Re-running when nothing new has been published does
no work; refreshing one member touches only that member's shard.

Sources per office (all public, no API key for the scrape itself):
  governor / chair -> federalreserve.gov speeches, testimony, bio
  chair            -> FOMC press conferences
  regional pres.   -> BIS speech archive (+ optionally the member's home-bank site)
  all members      -> FOMC meeting transcripts (5-year embargo)

Chunk extraction and embedding use OpenAI (OPENAI_API_KEY): gpt-4o-mini for extraction,
text-embedding-3-large for the 1024-d vectors.

    OPENAI_API_KEY=sk-...  python data/scrape/build_dataset.py --member "Jerome H. Powell"
    OPENAI_API_KEY=sk-...  python data/scrape/build_dataset.py --all      # update every member
"""
import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # sibling scrapers + topic_themes
sys.path.insert(0, str(HERE.parent.parent))         # the fomc_personas package

import fomc_personas as fp
from fomc_personas import roles
from topic_themes import bucket_topic
from extract_chunks import extract_opinions
from fed_reserve_scraper import run_fed_reserve_scraper
from fed_bank_scraper import run_fed_president_speeches, run_bank_speeches
from fomc_transcript_scraper import run_fomc_transcript_scraper
from fomc_pressconf_scraper import run_fomc_presconf_scraper

OUT = HERE.parent / "chunks"          # per-member parquet shards (the HF dataset layout)

# Home-bank site keys for the optional per-bank speech scraper (BIS already covers presidents).
BANK_KEY = {
    "Lorie K. Logan": "logan", "Susan M. Collins": "collins",
    "Austan D. Goolsbee": "goolsbee", "Beth M. Hammack": "hammack",
}


def scrape_member(member, start_date=None):
    """Scrape a member's documents published on/after start_date (None = full history)."""
    office = roles.office_at(member, datetime.date.today().isoformat())
    records = {}
    records.update(run_fomc_transcript_scraper(member, start_date))            # everyone
    if office in ("governor", "chair"):
        records.update(run_fed_reserve_scraper(member, start_date))
    if office == "chair":
        records.update(run_fomc_presconf_scraper(member, start_date))
    if office and office.startswith("pres:"):
        records.update(run_fed_president_speeches(member, start_date))         # BIS
        if member in BANK_KEY:
            records.update(run_bank_speeches(member, BANK_KEY[member], start_date))
    return records


def _existing(member):
    p = OUT / f"{member.lower().replace(' ', '_').replace('.', '')}.parquet"
    return (pd.read_parquet(p), p) if p.exists() else (None, p)


def update_member(member):
    """Incrementally scrape -> extract -> embed -> append new chunks for one member. Idempotent."""
    existing, path = _existing(member)
    start_date, seen = None, set()
    if existing is not None and len(existing):
        start_date = str(existing["postedAt"].max())[:10]
        seen = set(existing["postUrl"].dropna())

    records = {k: v for k, v in scrape_member(member, start_date).items() if v.get("postUrl") not in seen}
    if not records:
        print(f"  {member}: no new documents since {start_date or 'inception'}")
        return existing

    chunks = extract_opinions(records)                       # LLM extraction (gpt-4o-mini)
    rows = []
    for rec in chunks.values():
        r = dict(rec)
        r["member"] = member
        r.setdefault("handle", member)
        r["theme"] = bucket_topic(r.get("topic", ""))
        rows.append(r)
    new = pd.DataFrame(rows)
    new["embedding"] = list(fp.embed(new["stance"].fillna("").tolist()))      # text-embedding-3-large

    out = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    out = out.drop_duplicates("postUrl", keep="first").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"  {member}: +{len(new)} chunks ({len(records)} new docs) -> {path.name} (total {len(out)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", help="a single FOMC member's full name")
    ap.add_argument("--all", action="store_true", help="update every onboarded member")
    args = ap.parse_args()
    members = roles.MEMBERS if args.all else ([args.member] if args.member else [])
    if not members:
        ap.error("pass --member \"<name>\" or --all")
    for m in members:
        update_member(m)
    print("Done. Push updated shards to the HF dataset with `huggingface_hub.upload_folder`.")


if __name__ == "__main__":
    main()
