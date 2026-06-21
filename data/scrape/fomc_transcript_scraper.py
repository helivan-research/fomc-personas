"""
FOMC Meeting Transcript Scraper

FOMC meeting transcripts are released on a ~5-year lag and are verbatim and
speaker-attributed by name ("CHAIR POWELL.", "MR. WILLIAMS." ...). That makes
them the best source of an individual policymaker's spoken views and requires no
audio diarization — we parse the named speaker turns out of the PDF directly.

Conforms to the standard scraper contract: returns a flat dict keyed
"fomc-{YYYYMMDD}-{n}" with one record per target-speaker turn, and writes the
result under data/fed/.
"""

import os
import re
import json
import time
import shutil
import datetime
import tempfile
import subprocess
from typing import Optional, Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://www.federalreserve.gov"
HIST_YEAR = f"{BASE}/monetarypolicy/fomchistorical%d.htm"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

# Matches a speaker turn header at line start, e.g. "CHAIR POWELL.", "MR. CLARIDA.",
# "VICE CHAIR WILLIAMS.", "GOVERNOR BRAINARD.". Captures (title, surname).
SPEAKER_RE = re.compile(
    r'(?m)^\s*((?:CHAIR|VICE CHAIR|MR|MS|MRS|GOVERNOR|PRESIDENT|DR)\.?\s+'
    r'(?:VICE CHAIR\s+)?([A-Z][A-Z\'’\-]+))\.\s')

# Transcript PDF links look like /monetarypolicy/files/FOMC20200129meeting.pdf
PDF_RE = re.compile(r'FOMC(\d{8})meeting\.pdf', re.I)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _surname(fed_speaker_name: str) -> str:
    tokens = [t for t in re.split(r'\s+', fed_speaker_name.strip()) if t]
    return (tokens[-1] if tokens else fed_speaker_name).upper()


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using poppler's pdftotext (no -layout: clean turn breaks)."""
    if not shutil.which('pdftotext'):
        raise RuntimeError("pdftotext (poppler-utils) not found; cannot parse FOMC PDFs")
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, 'in.pdf')
        txt_path = os.path.join(td, 'out.txt')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        subprocess.run(['pdftotext', pdf_path, txt_path], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(txt_path, encoding='utf-8', errors='ignore') as f:
            return f.read()


_NOISE_LINE = re.compile(
    r'^\s*(\d+\s+of\s+\d+|page\s+\d+|january|february|march|april|may|june|july|'
    r'august|september|october|november|december)\b.*meeting', re.I)


def _clean_turn(body: str) -> str:
    """Collapse a turn body to clean prose, dropping page-header/footer artifacts."""
    lines = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            continue
        if _NOISE_LINE.match(s):
            continue
        if re.fullmatch(r'\d{1,4}', s):  # stray page numbers
            continue
        lines.append(s)
    text = ' '.join(lines)
    return re.sub(r'\s+', ' ', text).strip()


def _parse_turns(text: str, surname: str, min_chars: int) -> List[str]:
    """Return the cleaned turn passages spoken by the target surname."""
    matches = list(SPEAKER_RE.finditer(text))
    passages: List[str] = []
    for i, m in enumerate(matches):
        spk_surname = m.group(2).upper()
        if spk_surname != surname:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _clean_turn(text[m.end():end])
        if len(body) >= min_chars:
            passages.append(body)
    return passages


def _meeting_pdf_links(session: requests.Session, year: int) -> List[Tuple[str, str]]:
    """Return [(yyyymmdd, pdf_url)] of meeting transcripts for a historical year page."""
    url = HIST_YEAR % year
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, 'html.parser')
    out = []
    seen = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        mm = PDF_RE.search(href)
        if mm and mm.group(1) not in seen:
            seen.add(mm.group(1))
            full = href if href.startswith('http') else f"{BASE}{href}"
            out.append((mm.group(1), full))
    return out


def run_fomc_transcript_scraper(fed_speaker_name: str,
                                start_date: Optional[str] = None,
                                min_chars: int = 80,
                                max_meetings: Optional[int] = None) -> Dict[str, Any]:
    """Parse a policymaker's named turns out of FOMC meeting transcript PDFs.

    Args:
        fed_speaker_name: Full name; the surname is used to match speaker headers.
        start_date: Optional YYYY-MM-DD lower bound on meeting date.
        min_chars: Drop turns shorter than this (procedural one-liners).
        max_meetings: Optional cap on meetings processed (useful for testing).

    Returns:
        Flat dict keyed "fomc-{YYYYMMDD}-{n}" or {} / {'error': ...}.
    """
    try:
        print(f"\n{'='*60}")
        print(f"🏦 Starting FOMC transcript scraper for: {fed_speaker_name}")
        print(f"{'='*60}\n")

        surname = _surname(fed_speaker_name)
        start_dt = None
        if start_date:
            try:
                start_dt = pd.Timestamp(start_date).tz_localize(None)
            except Exception:
                print(f"   ⚠️  bad start_date {start_date!r}; ignoring")

        # Transcripts have a ~5-year release lag; scan back to a reasonable floor.
        current_year = datetime.datetime.now().year
        latest_available = current_year - 5
        floor_year = start_dt.year if start_dt is not None else latest_available - 6
        years = range(floor_year, latest_available + 2)  # +2 for partial early releases

        session = _session()
        meetings: List[Tuple[str, str]] = []
        for y in years:
            for ymd, url in _meeting_pdf_links(session, y):
                meetings.append((ymd, url))
        meetings.sort()
        if start_dt is not None:
            meetings = [(d, u) for d, u in meetings
                        if pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}") >= start_dt]
        if max_meetings:
            meetings = meetings[:max_meetings]
        print(f"   {len(meetings)} meeting transcripts to parse")

        records: Dict[str, Any] = {}
        accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for ymd, url in meetings:
            try:
                pdf = session.get(url, timeout=90).content
                text = _pdf_to_text(pdf)
            except Exception as ex:
                print(f"   ⚠️  failed {url}: {ex}")
                continue
            posted = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
            passages = _parse_turns(text, surname, min_chars)
            for n, body in enumerate(passages, 1):
                records[f"fomc-{ymd}-{n}"] = {
                    'text': body,
                    'caption': f"FOMC meeting {posted.date()} — {fed_speaker_name}",
                    'handle': fed_speaker_name,
                    'postedAt': posted,
                    'source': 'fomc_transcript',
                    'postUrl': url,
                    'postId': f"{ymd}-{n}",
                    'accessedAt': accessed_at,
                    'probabilitySpeaker': 1.0,
                }
            print(f"   {posted.date()}: {len(passages)} {fed_speaker_name} turns")
            time.sleep(0.5)

        if records:
            output_dir = os.path.join('data', 'fed')
            os.makedirs(output_dir, exist_ok=True)
            slug = surname.lower()
            json_path = os.path.join(output_dir, f'{slug}_fomc_transcripts.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 Saved {len(records)} FOMC turn records to {json_path}")
            print("\n✅ FOMC transcript scraping completed!")
            return records
        print("\n❌ No FOMC transcript data retrieved")
        return {}
    except Exception as e:
        print(f"\n⚠️  FOMC transcript scraper failed: {str(e)}")
        print("   Continuing with other scrapers...")
        return {'error': str(e)}


if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Jerome H. Powell"
    sd = sys.argv[2] if len(sys.argv) > 2 else "2020-01-01"
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out = run_fomc_transcript_scraper(name, sd, max_meetings=cap)
    print(f"\nTotal records: {len(out)}")
