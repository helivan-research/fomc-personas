"""
FOMC Press-Conference Scraper

Post-meeting FOMC press conferences are the Chair's largest spoken corpus and,
unlike meeting transcripts, are published immediately (no ~5-year lag). The Chair
reads a statement then answers reporter questions. Only the Chair speaks among Fed
officials, so attribution is trivial — we keep the Chair's turns and drop the
reporters.

Transcript PDFs: /mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf. Meeting dates come
from the historical year pages (<=2020) and fomccalendars.htm (2021+); we probe the
deterministic PDF URL (404 = that meeting had no press conference).

Conforms to the standard scraper contract: returns a flat dict keyed
"presconf-{YYYYMMDD}-{n}", one record per Chair turn, written under data/fed/.
"""

import os
import re
import json
import time
import datetime
from typing import Optional, Dict, Any, List

import requests
import pandas as pd

# Reuse the PDF + cleanup helpers from the meeting-transcript scraper.
from fomc_transcript_scraper import _pdf_to_text, _clean_turn, _surname, HEADERS

BASE = "https://www.federalreserve.gov"
CALENDARS = f"{BASE}/monetarypolicy/fomccalendars.htm"
HIST_YEAR = f"{BASE}/monetarypolicy/fomchistorical%d.htm"
PRESCONF_PDF = f"{BASE}/mediacenter/files/FOMCpresconf%s.pdf"

# Press-conference speaker headers are plain ALL-CAPS names (the Chair is
# "CHAIR POWELL."; reporters are untitled, e.g. "STEVE LIESMAN."). Split on any
# 1-4 word ALL-CAPS label followed by a period, then keep only the Chair's turns.
PC_SPEAKER_RE = re.compile(r"(?m)^\s*([A-Z][A-Z.'’\-]*(?:\s+[A-Z][A-Z.'’\-]*){0,3})\.\s")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _discover_meeting_dates(session: requests.Session, start_dt: Optional[pd.Timestamp]) -> List[str]:
    """Candidate FOMC meeting dates (YYYYMMDD) from historical pages + the calendar."""
    dates = set()
    current_year = datetime.datetime.now().year
    floor_year = start_dt.year if start_dt is not None else current_year - 8
    # Historical year pages (older meetings) — list presconf/meeting links directly.
    for y in range(floor_year, current_year - 3):
        try:
            r = session.get(HIST_YEAR % y, timeout=30)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        dates.update(re.findall(r'presconf(\d{8})', r.text))
        dates.update(re.findall(r'FOMC(\d{8})meeting', r.text))
    # Calendar page (recent meetings) — derive dates from statement/minutes links.
    try:
        r = session.get(CALENDARS, timeout=30)
        if r.status_code == 200:
            dates.update(re.findall(r'(?:monetary|fomcminutes)(\d{8})', r.text))
    except Exception:
        pass
    out = sorted(dates)
    if start_dt is not None:
        out = [d for d in out if pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}") >= start_dt]
    return out


def _parse_chair_turns(text: str, surname: str, min_chars: int) -> List[str]:
    """Return the cleaned turns spoken by the Chair (label contains the surname)."""
    matches = list(PC_SPEAKER_RE.finditer(text))
    passages: List[str] = []
    for i, m in enumerate(matches):
        label = m.group(1).upper()
        if surname not in label.split():
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _clean_turn(text[m.end():end])
        if len(body) >= min_chars:
            passages.append(body)
    return passages


def run_fomc_presconf_scraper(fed_speaker_name: str,
                              start_date: Optional[str] = None,
                              min_chars: int = 80,
                              max_meetings: Optional[int] = None) -> Dict[str, Any]:
    """Parse a Chair's statement + Q&A answers out of FOMC press-conference PDFs.

    Yields content only for members who were Chair (only the Chair speaks among Fed
    officials); for everyone else it returns {} after finding no matching turns.

    Args:
        fed_speaker_name: Full name; the surname matches the "CHAIR <SURNAME>." header.
        start_date: Optional YYYY-MM-DD lower bound on meeting date.
        min_chars: Drop turns shorter than this.
        max_meetings: Optional cap (useful for testing).
    """
    try:
        print(f"\n{'='*60}")
        print(f"🎤 Starting FOMC press-conference scraper for: {fed_speaker_name}")
        print(f"{'='*60}\n")

        surname = _surname(fed_speaker_name)
        start_dt = None
        if start_date:
            try:
                start_dt = pd.Timestamp(start_date).tz_localize(None)
            except Exception:
                print(f"   ⚠️  bad start_date {start_date!r}; ignoring")

        session = _session()
        meeting_dates = _discover_meeting_dates(session, start_dt)
        if max_meetings:
            meeting_dates = meeting_dates[:max_meetings]
        print(f"   {len(meeting_dates)} candidate meetings to probe for press conferences")

        records: Dict[str, Any] = {}
        accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        n_pc = 0
        for ymd in meeting_dates:
            url = PRESCONF_PDF % ymd
            try:
                resp = session.get(url, timeout=90)
                if resp.status_code != 200:
                    continue  # no press conference for this meeting
                text = _pdf_to_text(resp.content)
            except Exception as ex:
                print(f"   ⚠️  failed {url}: {ex}")
                continue
            posted = pd.Timestamp(f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}")
            passages = _parse_chair_turns(text, surname, min_chars)
            if not passages:
                continue
            n_pc += 1
            for k, body in enumerate(passages, 1):
                records[f"presconf-{ymd}-{k}"] = {
                    'text': body,
                    'caption': f"FOMC press conference {posted.date()} — {fed_speaker_name}",
                    'handle': fed_speaker_name,
                    'postedAt': posted,
                    'source': 'fomc_presconf',
                    'postUrl': url,
                    'postId': f"{ymd}-{k}",
                    'accessedAt': accessed_at,
                    'probabilitySpeaker': 1.0,
                }
            print(f"   {posted.date()}: {len(passages)} {fed_speaker_name} turns")
            time.sleep(0.4)

        if records:
            output_dir = os.path.join('data', 'fed')
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f'{surname.lower()}_fomc_presconf.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 Saved {len(records)} press-conference turn records from {n_pc} press conferences to {json_path}")
            print("\n✅ FOMC press-conference scraping completed!")
            return records
        print("\n❌ No FOMC press-conference data retrieved (member was likely never Chair)")
        return {}
    except Exception as e:
        print(f"\n⚠️  FOMC press-conference scraper failed: {str(e)}")
        print("   Continuing with other scrapers...")
        return {'error': str(e)}


if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Jerome H. Powell"
    sd = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out = run_fomc_presconf_scraper(name, sd, max_meetings=cap)
    print(f"\nTotal records: {len(out)}")
