"""Build or **incrementally update** the FOMC-personas Hugging Face dataset.

For each member the pipeline is: scrape public documents -> extract retrievable chunks (LLM) ->
embed -> append to the per-member parquet shards, in the dataset's native **split** layout:

    <data-dir>/chunks/<member>.parquet       text + metadata + a global integer ``chunk_id``
    <data-dir>/embeddings/<member>.parquet    ``chunk_id`` + the 1024-d embedding

(joined on ``chunk_id`` by ``fomc_personas.load_chunks`` — same as the published HF dataset.)

It is **incremental and idempotent**: it reads the shards a member already has, scrapes only documents
published *after* the latest one present, skips URLs already seen, and extracts + embeds only the new
passages. New chunks get sequential ``chunk_id``s continuing from the global maximum. Re-running when
nothing new has been published does no work.

Sources per office (all public, no API key for the scrape itself):
  governor / chair -> federalreserve.gov speeches, testimony, bio
  chair            -> FOMC press conferences
  regional pres.   -> BIS speech archive (+ optionally the member's home-bank site)
  all members      -> FOMC meeting transcripts (5-year embargo)

Chunk extraction and embedding use OpenAI (OPENAI_API_KEY): gpt-4o-mini for extraction,
text-embedding-3-large for the 1024-d vectors.

    # update a local working copy ($FOMC_DATASET_DIR, default data/.hf_dataset)
    OPENAI_API_KEY=sk-...  python data/scrape/build_dataset.py --all
    # CI / publish: pull the current dataset from HF, update, push changed shards back
    OPENAI_API_KEY=sk-...  HF_TOKEN=hf_...  python data/scrape/build_dataset.py --all --pull --push
"""
import argparse
import datetime
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # sibling scrapers + topic_themes
sys.path.insert(0, str(HERE.parent.parent))         # the fomc_personas package
ROOT = HERE.parent.parent

import fomc_personas as fp
from fomc_personas import roles
from topic_themes import bucket_topic
from extract_chunks import extract_opinions
from fed_reserve_scraper import run_fed_reserve_scraper
from fed_bank_scraper import run_fed_president_speeches, run_bank_speeches
from fomc_transcript_scraper import run_fomc_transcript_scraper
from fomc_pressconf_scraper import run_fomc_presconf_scraper

HF_REPO = os.environ.get("FOMC_HF_REPO", "helivan/fomc-personas")

# Canonical chunk-shard schema (matches the published dataset; embedding lives in the sibling shard).
CHUNK_COLS = ["chunk_id", "member", "text", "topic", "quote", "stance", "handle", "postedAt",
              "source", "sourceId", "postUrl", "accessedAt", "probabilitySpeaker", "theme",
              "is_voting", "is_chair"]

# Home-bank site keys for the optional per-bank speech scraper (BIS already covers presidents).
BANK_KEY = {
    "Lorie K. Logan": "logan", "Susan M. Collins": "collins",
    "Austan D. Goolsbee": "goolsbee", "Beth M. Hammack": "hammack",
}


def _shard(member):
    return f"{member.lower().replace(' ', '_').replace('.', '')}.parquet"


def _coerce_schema(df):
    """Cast freshly-extracted rows to the published shard dtypes so appended rows concat cleanly
    (the scrapers emit string dates / Python bools; the dataset stores typed datetime/boolean cols)."""
    df = df.copy()
    df["chunk_id"] = df["chunk_id"].astype("int64")
    df["postedAt"] = pd.to_datetime(df["postedAt"], errors="coerce", utc=True).dt.tz_localize(None)
    df["accessedAt"] = pd.to_datetime(df["accessedAt"], errors="coerce", utc=True)
    df["probabilitySpeaker"] = pd.to_numeric(df["probabilitySpeaker"], errors="coerce").astype("float64")
    for c in ("is_voting", "is_chair"):
        df[c] = df[c].astype("boolean")
    return df


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


def _next_chunk_id(data_dir):
    """One past the global maximum chunk_id across all chunk shards (ids are global, not per-shard)."""
    mx = -1
    for p in sorted((data_dir / "chunks").glob("*.parquet")):
        s = pd.read_parquet(p, columns=["chunk_id"])["chunk_id"]
        if len(s):
            mx = max(mx, int(s.max()))
    return mx + 1


def update_member(data_dir, member, next_id):
    """Incrementally scrape -> extract -> embed -> append new chunks for one member.

    Returns (next_id, n_new). Document-level dedup is handled by the ``seen`` URL set; a single
    document yields many chunks, so we never dedup at the chunk level."""
    cpath, epath = data_dir / "chunks" / _shard(member), data_dir / "embeddings" / _shard(member)
    existing = pd.read_parquet(cpath) if cpath.exists() else None

    start_date, seen = None, set()
    if existing is not None and len(existing):
        start_date = str(existing["postedAt"].max())[:10]
        seen = set(existing["postUrl"].dropna())

    # Keep only well-formed record dicts (a scraper that fails returns {'error': ...}, whose value
    # is a string) and drop documents already seen.
    records = {k: v for k, v in scrape_member(member, start_date).items()
               if isinstance(v, dict) and v.get("postUrl") not in seen}
    if not records:
        print(f"  {member}: no new documents since {start_date or 'inception'}")
        return next_id, 0

    chunks = extract_opinions(records)                       # LLM extraction (gpt-4o-mini)
    if not chunks:                                           # new doc(s) but nothing extractable
        print(f"  {member}: {len(records)} new doc(s), no extractable chunks")
        return next_id, 0
    rows = []
    for rec in chunks.values():
        r = dict(rec)
        r["member"] = member
        r.setdefault("handle", member)
        r["theme"] = bucket_topic(r.get("topic", ""))
        rows.append(r)
    new = pd.DataFrame(rows)
    new["chunk_id"] = range(next_id, next_id + len(new))     # continue the global id sequence
    next_id += len(new)
    embeddings = fp.embed(new["stance"].fillna("").tolist())  # text-embedding-3-large

    new_meta = _coerce_schema(new.reindex(columns=CHUNK_COLS))
    new_emb = pd.DataFrame({"chunk_id": new["chunk_id"].astype("int64").values, "embedding": list(embeddings)})
    meta_out = pd.concat([existing.reindex(columns=CHUNK_COLS), new_meta], ignore_index=True) \
        if existing is not None else new_meta
    eold = pd.read_parquet(epath) if epath.exists() else None
    emb_out = pd.concat([eold, new_emb], ignore_index=True) if eold is not None else new_emb

    cpath.parent.mkdir(parents=True, exist_ok=True)
    epath.parent.mkdir(parents=True, exist_ok=True)
    meta_out.to_parquet(cpath, index=False)
    emb_out.to_parquet(epath, index=False)
    print(f"  {member}: +{len(new)} chunks ({len(records)} new docs) -> total {len(meta_out)}")
    return next_id, len(new)


def _pull(data_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(HF_REPO, repo_type="dataset", local_dir=str(data_dir),
                      allow_patterns=["chunks/*.parquet", "embeddings/*.parquet"],
                      token=os.environ.get("HF_TOKEN"))


def _push(data_dir):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    stamp = datetime.date.today().isoformat()
    for sub in ("chunks", "embeddings"):
        api.upload_folder(folder_path=str(data_dir / sub), path_in_repo=sub, repo_id=HF_REPO,
                          repo_type="dataset", commit_message=f"weekly refresh ({stamp}): {sub}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", help="a single FOMC member's full name")
    ap.add_argument("--all", action="store_true", help="update every onboarded member")
    ap.add_argument("--data-dir", default=os.environ.get("FOMC_DATASET_DIR", str(ROOT / "data" / ".hf_dataset")),
                    help="working copy of the dataset (chunks/ + embeddings/ shards)")
    ap.add_argument("--pull", action="store_true", help="download the current dataset from HF first")
    ap.add_argument("--push", action="store_true", help="upload changed shards back to HF when done")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if args.pull:
        print(f"pulling {HF_REPO} -> {data_dir}")
        _pull(data_dir)

    members = roles.MEMBERS if args.all else ([args.member] if args.member else [])
    if not members:
        ap.error('pass --member "<name>" or --all')

    (data_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (data_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    next_id = _next_chunk_id(data_dir)
    total_new = 0
    for m in members:
        next_id, n = update_member(data_dir, m, next_id)
        total_new += n
    print(f"added {total_new} chunks; max chunk_id now {next_id - 1}")

    if args.push:
        if total_new == 0:
            print("no new chunks — skipping HF push")
        else:
            print(f"pushing to {HF_REPO}")
            _push(data_dir)
    elif total_new:
        print("Done. Re-run with --push (or upload <data-dir>/{chunks,embeddings} via huggingface_hub).")


if __name__ == "__main__":
    main()
