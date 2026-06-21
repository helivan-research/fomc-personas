"""
Opinion extraction stage.

Turns long source passages (speeches, testimony, FOMC turns) into discrete,
retrievable "opinion" units via the OpenAI API. Each opinion is a
{topic, stance, quote} triple grounded in the source text. The resulting
records conform to the standard storage contract (text / postedAt / source /
postUrl / probabilitySpeaker) so they flow straight into create_persona().

Requires OPENAI_API_KEY. Default model is gpt-4o-mini; override with the
OPINION_MODEL env var. Uses OpenAI structured outputs (chat.completions.parse
with a Pydantic schema) and a small concurrent fan-out over passages.
"""

import os
import re
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from topic_themes import bucket_topic          # sibling module in data/scrape/
from fomc_personas.roles import roles as fomc_roles  # single source of truth for role tags

DEFAULT_MODEL = os.getenv('OPINION_MODEL', 'gpt-4o-mini')

SYSTEM_PROMPT = (
    "You extract the discrete, attributable opinions a public figure expresses in a "
    "passage of their own words (a speech, testimony, or meeting transcript turn).\n\n"
    "An 'opinion' is a stance the speaker takes on a topic — a claim, judgment, "
    "preference, prediction, or recommendation they personally assert. Extract only "
    "views the speaker themselves holds; ignore procedural remarks, pleasantries, "
    "pure factual recitation with no evaluative content, and views attributed to "
    "other people.\n\n"
    "For each opinion return:\n"
    "  - topic: a short noun phrase naming the subject (e.g. 'inflation', "
    "'interest rate policy', 'bank regulation').\n"
    "  - stance: one self-contained sentence stating the speaker's position on that "
    "topic, phrased so it stands alone without the surrounding passage.\n"
    "  - quote: a short verbatim excerpt from the passage that grounds the stance. "
    "Copy it exactly; do not paraphrase.\n\n"
    "If the passage contains no genuine opinions, return an empty list. Do not invent "
    "opinions that are not supported by a verbatim quote."
)


class Opinion(BaseModel):
    topic: str = Field(description="Short noun phrase naming the subject of the opinion")
    stance: str = Field(description="One self-contained sentence stating the speaker's position")
    quote: str = Field(description="Verbatim excerpt from the passage that grounds the stance")


class OpinionList(BaseModel):
    opinions: List[Opinion]


def chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    """Split a long passage into <= max_chars chunks on paragraph/sentence boundaries."""
    text = (text or '').strip()
    if len(text) <= max_chars:
        return [text] if text else []
    # Prefer paragraph boundaries, then fall back to sentence boundaries.
    paras = re.split(r'\n\s*\n', text)
    chunks: List[str] = []
    cur = ''
    for para in paras:
        if len(para) > max_chars:
            # Paragraph itself too long: split on sentence enders.
            for sent in re.split(r'(?<=[.!?])\s+', para):
                if len(cur) + len(sent) + 1 > max_chars and cur:
                    chunks.append(cur.strip())
                    cur = ''
                cur += sent + ' '
            continue
        if len(cur) + len(para) + 2 > max_chars and cur:
            chunks.append(cur.strip())
            cur = ''
        cur += para + '\n\n'
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def _extract_one(client, model: str, passage: str) -> List[Opinion]:
    """Single passage -> list of Opinion (validated). Returns [] on empty/refused parse."""
    resp = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract the opinions expressed in this passage:\n\n{passage}"},
        ],
        response_format=OpinionList,
    )
    parsed = resp.choices[0].message.parsed
    return parsed.opinions if parsed else []


def extract_opinions(records: Dict[str, Any],
                     model: Optional[str] = None,
                     max_workers: int = 8,
                     max_passage_chars: int = 6000) -> Dict[str, Any]:
    """Extract opinion records from a dict of source records.

    Args:
        records: source records keyed by id, each with at least 'text'; carries
            through 'source', 'postUrl', 'postedAt', 'handle', 'probabilitySpeaker'.
        model: OpenAI model id (defaults to OPINION_MODEL env or gpt-4o-mini).
        max_workers: concurrency for the per-passage requests.
        max_passage_chars: long source texts are chunked to this size first.

    Returns:
        Opinion records keyed "opinion-{source_id}-{n}".
    """
    from openai import OpenAI

    model = model or DEFAULT_MODEL
    client = OpenAI()

    # Build the work list: (source_id, chunk_index, passage, source_record).
    work: List[tuple] = []
    for sid, rec in records.items():
        if not isinstance(rec, dict):
            continue
        for ci, passage in enumerate(chunk_text(rec.get('text', ''), max_passage_chars)):
            work.append((sid, ci, passage, rec))

    if not work:
        return {}

    print(f"🧠 Extracting opinions from {len(work)} passages with {model}")

    def run(item):
        sid, ci, passage, _rec = item
        try:
            return (sid, ci), _extract_one(client, model, passage)
        except Exception as ex:
            print(f"   ⚠️  opinion extraction failed for {sid} chunk {ci}: {ex}")
            return (sid, ci), []

    results: Dict[tuple, List[Opinion]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for key, ops in ex.map(run, work):
            results[key] = ops

    # Assemble opinion records keyed opinion-{source_id}-{n}.
    accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    opinion_records: Dict[str, Any] = {}
    counter: Dict[str, int] = {}
    for (sid, ci), ops in sorted(results.items()):
        rec = records[sid]
        for op in ops:
            counter[sid] = counter.get(sid, 0) + 1
            n = counter[sid]
            record = {
                'text': op.stance,
                'topic': op.topic,
                'theme': bucket_topic(op.topic),
                'quote': op.quote,
                'stance': op.stance,
                'handle': rec.get('handle'),
                'postedAt': rec.get('postedAt'),
                'source': rec.get('source', 'opinion'),
                'sourceId': sid,
                'postUrl': rec.get('postUrl'),
                'accessedAt': accessed_at,
                'probabilitySpeaker': rec.get('probabilitySpeaker', 1.0),
            }
            # As-of-date FOMC role tags (only for recognized FOMC members).
            _rl = fomc_roles(rec.get('handle'), rec.get('postedAt'))
            if _rl is not None:
                record['is_voting'] = _rl['is_voting']
                record['is_chair'] = _rl['is_chair']
            opinion_records[f"opinion-{sid}-{n}"] = record
    print(f"✅ Extracted {len(opinion_records)} opinions from {len(records)} source records")
    return opinion_records


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'data/fed/powell_fed_board.json'
    data = json.load(open(src))
    out = extract_opinions(data)
    print(json.dumps({k: out[k] for k in list(out)[:5]}, indent=2, default=str))
    print(f"\nTotal opinions: {len(out)}")
