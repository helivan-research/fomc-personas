"""
Federal Reserve Board Scraper

Collects an individual Board governor's public corpus from federalreserve.gov:
speeches, congressional testimony, and official bio. Governors' material lives on
the Board domain (regional bank presidents live on their own bank domains and are
handled separately).

Conforms to the standard scraper contract: returns a flat dict keyed
"fed-{kind}-{slug}" with records carrying text/handle/postedAt/source/postUrl/
postId/accessedAt/probabilitySpeaker, and writes the result under data/fed/.
"""

import os
import re
import json
import time
import datetime
from typing import Optional, Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://www.federalreserve.gov"
SPEECHES_FEED = f"{BASE}/json/ne-speeches.json"
TESTIMONY_FEED = f"{BASE}/json/ne-testimony.json"
BOARD_BIOS = f"{BASE}/aboutthefed/bios/board/default.htm"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _name_parts(fed_speaker_name: str) -> Tuple[str, str]:
    """Return (first_name, last_name) lowercased from a full name like 'Jerome H. Powell'."""
    tokens = [t for t in re.split(r'\s+', fed_speaker_name.strip()) if t]
    first = tokens[0].lower() if tokens else ''
    last = tokens[-1].lower() if tokens else ''
    return first, last


def _name_matches(s_field: str, first: str, last: str) -> bool:
    """The feed's 's' field is 'Chair Jerome H. Powell' etc. Require first+last present."""
    s = (s_field or '').lower()
    return bool(first) and bool(last) and first in s and last in s


def _parse_feed_date(d: str) -> Optional[pd.Timestamp]:
    if not d:
        return None
    try:
        return pd.Timestamp(d).tz_localize(None)
    except Exception:
        return None


def _slug_from_link(link: str) -> str:
    """'/newsevents/speech/powell20260321a.htm' -> 'powell20260321a'."""
    return os.path.splitext(os.path.basename(link or ''))[0] or link


_VIDEO_NOISE = re.compile(
    r'accessible keys for video|toggles play/pause|seeks the video|'
    r'\[space bar\]|\[right/left arrows\]|\[up/down arrows\]|increase/decrease volume|'
    r'watch live|share\b.*\bfacebook', re.I)


def _extract_article_text(html: str) -> str:
    """Extract the prepared-remarks body from a Fed speech/testimony page.

    Body lives in div#article; we take its paragraphs, drop the video-player
    accessibility boilerplate, and cut the footnotes/references tail.
    """
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    # Drop screen-reader-only blocks (video keyboard help) and video/media containers.
    for el in soup.select('.sr-only'):
        el.decompose()
    for el in soup.find_all(['div', 'section']):
        idc = ((el.get('id') or '') + ' ' + ' '.join(el.get('class') or [])).lower()
        if any(w in idc for w in ('video', 'media', 'player')):
            el.decompose()
    article = soup.select_one('div#article') or soup.select_one('#content') or soup
    paras: List[str] = []
    for p in article.find_all('p'):
        txt = p.get_text(' ', strip=True)
        if not txt:
            continue
        if _VIDEO_NOISE.search(txt):
            continue
        # Footnotes / references tail: stop once we hit "Return to text" markers.
        if txt.lower().startswith('return to text'):
            break
        paras.append(txt)
    return '\n\n'.join(paras).strip()


def _collect_from_feed(session: requests.Session, feed_url: str, kind: str, source_tag: str,
                       first: str, last: str, start_dt: Optional[pd.Timestamp],
                       handle: str, delay: float = 0.5) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    resp = session.get(feed_url, timeout=30)
    resp.raise_for_status()
    raw = resp.content.decode('utf-8-sig', errors='ignore').strip()
    entries = json.loads(raw)

    matched = [e for e in entries if _name_matches(e.get('s', ''), first, last)]
    print(f"   {len(matched)} {kind} entries match {first.title()} {last.title()}")

    accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for e in matched:
        posted = _parse_feed_date(e.get('d', ''))
        if start_dt is not None and posted is not None and posted < start_dt:
            continue
        link = e.get('l', '')
        url = link if link.startswith('http') else f"{BASE}{link}"
        slug = _slug_from_link(link)
        try:
            page = session.get(url, timeout=30)
            page.raise_for_status()
            body = _extract_article_text(page.text)
        except Exception as ex:
            print(f"   ⚠️  failed to fetch {url}: {ex}")
            continue
        if not body:
            continue
        title = e.get('t', '')
        records[f"fed-{kind}-{slug}"] = {
            'text': body,
            'caption': title,
            'handle': handle,
            'postedAt': posted,
            'source': source_tag,
            'postUrl': url,
            'postId': slug,
            'location': e.get('lo', ''),
            'accessedAt': accessed_at,
            'probabilitySpeaker': 1.0,
        }
        time.sleep(delay)
    return records


def _collect_bio(session: requests.Session, fed_speaker_name: str, handle: str) -> Dict[str, Any]:
    try:
        resp = session.get(BOARD_BIOS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        first, last = _name_parts(fed_speaker_name)
        bio_link = None
        for a in soup.find_all('a', href=True):
            label = a.get_text(' ', strip=True).lower()
            if first in label and last in label:
                bio_link = a['href']
                break
        if not bio_link:
            print("   ⚠️  no bio link found on board bios page")
            return {}
        url = bio_link if bio_link.startswith('http') else f"{BASE}{bio_link}"
        page = session.get(url, timeout=30)
        page.raise_for_status()
        body = _extract_article_text(page.text)
        if not body:
            return {}
        slug = _slug_from_link(bio_link)
        return {
            f"fed-bio-{slug}": {
                'text': body,
                'caption': f"{fed_speaker_name} — official bio",
                'handle': handle,
                'postedAt': pd.Timestamp(datetime.datetime.now().date()),
                'source': 'fed_bio',
                'postUrl': url,
                'postId': slug,
                'accessedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'probabilitySpeaker': 1.0,
            }
        }
    except Exception as ex:
        print(f"   ⚠️  bio collection failed: {ex}")
        return {}


def run_fed_reserve_scraper(fed_speaker_name: str,
                            start_date: Optional[str] = None) -> Dict[str, Any]:
    """Scrape a Board governor's speeches, testimony, and bio from federalreserve.gov.

    Args:
        fed_speaker_name: Full name as it appears on the Board site (e.g. "Jerome H. Powell").
        start_date: Optional YYYY-MM-DD lower bound on publication date.

    Returns:
        Flat dict keyed "fed-{kind}-{slug}" or {} / {'error': ...}.
    """
    try:
        print(f"\n{'='*60}")
        print(f"🏛️  Starting Federal Reserve Board scraper for: {fed_speaker_name}")
        print(f"{'='*60}\n")

        start_dt = None
        if start_date:
            try:
                start_dt = pd.Timestamp(start_date).tz_localize(None)
            except Exception:
                print(f"   ⚠️  bad start_date {start_date!r}; ignoring")

        first, last = _name_parts(fed_speaker_name)
        handle = fed_speaker_name
        session = _session()

        records: Dict[str, Any] = {}
        records.update(_collect_from_feed(session, SPEECHES_FEED, 'speech', 'fed_speech',
                                          first, last, start_dt, handle))
        records.update(_collect_from_feed(session, TESTIMONY_FEED, 'testimony', 'fed_testimony',
                                          first, last, start_dt, handle))
        records.update(_collect_bio(session, fed_speaker_name, handle))

        if records:
            output_dir = os.path.join('data', 'fed')
            os.makedirs(output_dir, exist_ok=True)
            slug = last or fed_speaker_name.lower().replace(' ', '_')
            json_path = os.path.join(output_dir, f'{slug}_fed_board.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 Saved {len(records)} Fed Board records to {json_path}")
            print("\n✅ Federal Reserve Board scraping completed!")
            return records
        print("\n❌ No Federal Reserve Board data retrieved")
        return {}
    except Exception as e:
        print(f"\n⚠️  Federal Reserve Board scraper failed: {str(e)}")
        print("   Continuing with other scrapers...")
        return {'error': str(e)}


if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "Jerome H. Powell"
    sd = sys.argv[2] if len(sys.argv) > 2 else None
    out = run_fed_reserve_scraper(name, sd)
    print(f"\nTotal records: {len(out)}")
