"""
Regional Fed president speech scraper, via the BIS central-bankers' speech archive.

The 12 Reserve Bank presidents publish speeches on 12 separate (often JS-rendered)
bank sites. Rather than maintain 12 bespoke parsers, we use the BIS archive
(bis.org/cbspeeches), which aggregates every Fed regional speech in ONE uniform
structure with clean full text + citation metadata. BIS is JS-gated, so we render
it with Playwright/Chromium.

Per president:
  1. Discover their BIS author page by paging the global cbspeeches index
     (?cbspeeches_page=N) until a review whose speaker matches them appears, then
     read the /author/<slug>.htm link off that review page. (BIS slugs are not
     derivable from names, so we discover rather than guess.) Cached to disk.
  2. Collect their speeches from the author page; the review URL encodes the date
     (rYYMMDD…), so we filter by start_date without extra requests.
  3. Render each review -> citation_title + #cmsContent full text.

Conforms to the standard scraper contract: returns a flat dict keyed
"fed-speech-{id}" with source='fed_speech', written under data/fed/.
"""

import os
import re
import json
import time
import datetime
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

BASE = "https://www.bis.org"
INDEX = BASE + "/cbspeeches/index.htm?cbspeeches_page=%d"
SLUG_CACHE = os.path.join('data', 'fed', 'bis_slugs.json')


def _name_parts(name: str) -> Tuple[str, str]:
    toks = [t for t in re.split(r'\s+', name.replace('.', '').strip()) if t]
    return (toks[0].lower(), toks[-1].lower()) if toks else ('', '')


def _date_from_href(href: str) -> Optional[pd.Timestamp]:
    m = re.search(r'/review/r(\d{2})(\d{2})(\d{2})', href)
    if not m:
        return None
    yy, mm, dd = m.groups()
    try:
        return pd.Timestamp(f"20{yy}-{mm}-{dd}")
    except Exception:
        return None


def _load_slugs() -> Dict[str, str]:
    return json.load(open(SLUG_CACHE)) if os.path.exists(SLUG_CACHE) else {}


def _save_slug(name: str, slug: str):
    os.makedirs(os.path.dirname(SLUG_CACHE), exist_ok=True)
    d = _load_slugs(); d[name] = slug
    json.dump(d, open(SLUG_CACHE, 'w'))


def _render_review_links(page, url: str) -> List[str]:
    """Goto a BIS listing/author page and return its review hrefs (waits for JS).

    Review links render into the DOM but are not 'visible' to Playwright, so we
    wait for state='attached' (DOM presence), not visibility.
    """
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    try:
        page.wait_for_selector("a[href*='/review/r']", timeout=9000, state="attached")
    except Exception:
        pass
    page.wait_for_timeout(2000)
    return page.eval_on_selector_all("a[href*='/review/r']",
                                     "els=>els.map(e=>e.getAttribute('href'))")


def _candidate_slugs(first: str, mid_full: str) -> List[str]:
    parts = [p for p in mid_full.replace('.', '').lower().split() if p]
    out = []
    if len(parts) >= 2:
        out.append('_'.join(parts))                 # john_c_williams
        out.append(f"{parts[0]}_{parts[-1]}")        # john_williams
    return list(dict.fromkeys(out))


def _author_page_links(page, slug: str, last: str) -> List[str]:
    """Validate+collect a candidate author page: returns review links if it's the right person."""
    hrefs, seen = [], set()
    for n in range(1, 8):
        url = f"{BASE}/author/{slug}.htm" + (f"?cbspeeches_page={n}" if n > 1 else "")
        links = _render_review_links(page, url)
        if n == 1 and last not in (page.title() or '').lower():  # wrong/generic author page
            return []
        new = [h for h in links if h not in seen]
        if not new:
            break
        for h in new:
            seen.add(h); hrefs.append(h)
    return hrefs


def _discover_author_slug(page, first: str, last: str, max_pages: int) -> Optional[str]:
    """Fallback: page the global index until a review by this speaker appears; read author slug."""
    for n in range(1, max_pages + 1):
        page.goto(INDEX % n, wait_until="domcontentloaded", timeout=45000)
        try:
            page.wait_for_selector("a[href*='/review/r']", timeout=9000, state="attached")
        except Exception:
            pass
        page.wait_for_timeout(1500)
        items = page.eval_on_selector_all(
            "a[href*='/review/r']",
            "els=>els.map(e=>({href:e.getAttribute('href'),text:e.innerText.trim()}))")
        for it in items:
            speaker = it['text'].split(':')[0].lower()
            if last in speaker and first in speaker:
                page.goto(BASE + it['href'], wait_until="domcontentloaded", timeout=40000)
                page.wait_for_timeout(1500)
                a = page.query_selector("a[href*='/author/']")
                if a:
                    m = re.search(r'/author/([a-z0-9_]+)\.htm', a.get_attribute('href') or '')
                    if m:
                        return m.group(1)
    return None


def run_fed_president_speeches(fed_speaker_name: str,
                              start_date: Optional[str] = None,
                              max_discovery_pages: int = 40) -> Dict[str, Any]:
    """Scrape a Reserve Bank president's speeches from the BIS archive (headless)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("⚠️  playwright not installed; cannot scrape president speeches")
        return {'error': 'playwright not installed'}
    try:
        print(f"\n{'='*60}")
        print(f"🗣️  Starting BIS speech scraper for: {fed_speaker_name}")
        print(f"{'='*60}\n")

        first, last = _name_parts(fed_speaker_name)
        start_dt = None
        if start_date:
            try:
                start_dt = pd.Timestamp(start_date)
            except Exception:
                pass

        records: Dict[str, Any] = {}
        accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            page = b.new_page()

            # Resolve the author page: cached slug, then constructed candidates
            # (validated against the page title), then index-based discovery.
            cached = _load_slugs().get(fed_speaker_name)
            slug, links = None, []
            for cand in ([cached] if cached else []) + _candidate_slugs(first, fed_speaker_name):
                ls = _author_page_links(page, cand, last)
                if ls:
                    slug, links = cand, ls
                    break
            if slug is None:
                slug = _discover_author_slug(page, first, last, max_discovery_pages)
                if slug:
                    links = _author_page_links(page, slug, last)
            if not slug or not links:
                print(f"   ⚠️  could not find/collect a BIS author page for {fed_speaker_name}")
                b.close(); return {}
            _save_slug(fed_speaker_name, slug)

            links = [h for h in links if (_date_from_href(h) is None or start_dt is None
                                          or _date_from_href(h) >= start_dt)]
            print(f"   BIS author slug: {slug} | {len(links)} speeches to fetch")

            for href in links:
                url = BASE + href
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=40000)
                    try:
                        page.wait_for_selector("#cmsContent, article", timeout=9000, state="attached")
                    except Exception:
                        pass
                    page.wait_for_timeout(1200)
                    title = page.eval_on_selector(
                        "meta[name='citation_title']", "e=>e.content") if page.query_selector("meta[name='citation_title']") else ""
                    el = page.query_selector("#cmsContent") or page.query_selector("article")
                    text = el.inner_text() if el else ""
                except Exception as ex:
                    print(f"   ⚠️  failed {url}: {ex}")
                    continue
                text = re.sub(r'\s+\n', '\n', text).strip()
                if len(text) < 200:
                    continue
                slug_id = re.search(r'/review/(r[0-9a-z]+)\.htm', href).group(1)
                records[f"fed-speech-{slug_id}"] = {
                    'text': text,
                    'caption': title,
                    'handle': fed_speaker_name,
                    'postedAt': _date_from_href(href),
                    'source': 'fed_speech',
                    'postUrl': url,
                    'postId': slug_id,
                    'accessedAt': accessed_at,
                    'probabilitySpeaker': 1.0,
                }
                time.sleep(0.2)
            b.close()

        if records:
            output_dir = os.path.join('data', 'fed')
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f'{last}_bis_speeches.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 Saved {len(records)} BIS speeches to {json_path}")
            print("\n✅ BIS speech scraping completed!")
            return records
        print("\n❌ No BIS speeches retrieved")
        return {}
    except Exception as e:
        print(f"\n⚠️  BIS speech scraper failed: {str(e)}")
        return {'error': str(e)}


# ----------------------------------------------------------------------------
# Per-bank speech scrapers (the 11 regional presidents BIS does not cover).
# Each bank site is different; a small config per bank drives one generic
# headless render -> link-extract -> render -> text engine.
# ----------------------------------------------------------------------------

# date_re groups are concatenated then parsed; 2-digit years are assumed 20xx.
BANK_CONFIGS = {
    'logan':    {'listing': 'https://www.dallasfed.org/news/speeches/logan',
                 'link_re': r'/news/speeches/logan/\d{4}/lkl\d{6}',
                 'date_re': r'lkl(\d{2})(\d{2})(\d{2})'},
    'hammack':  {'listing': 'https://www.clevelandfed.org/collections/speeches',
                 'link_re': r'/collections/speeches/(?:\d{4}/)?sp-\d{8}-[\w\-]+',
                 'date_re': r'sp-(\d{4})(\d{2})(\d{2})'},
    'collins':  {'listing': 'https://www.bostonfed.org/news-and-events/speeches.aspx',
                 'link_re': r'/news-and-events/speeches/\d{4}/[\w%\.\-]+\.aspx',
                 'date_re': None},
    'goolsbee': {'listing': 'https://www.chicagofed.org/publications/speeches',
                 'link_re': r'/publications/speeches/\d{4}/[\w\-]+',
                 'date_re': None},
    'schmid':   {'listing': 'https://www.kansascityfed.org/speeches/',
                 'link_re': r'/speeches/[a-z0-9][\w\-]+/',
                 'date_re': None},
    'goolsbee': {'listing': 'https://www.chicagofed.org/publications/speeches',
                 'link_re': r'/publications/speeches/\d{4}/[\w\-]+',
                 'date_re': None},
    'venable':  {'listing': 'https://www.atlantafed.org/news-and-events/speeches',
                 'link_re': r'/news-and-events/speeches/\d{4}/[\w\-]+',
                 'date_re': None},
    # Chicago's listing is Akamai/JS-gated, but the sitemap exposes every speech URL
    # and the pages fetch via curl (requests gets TLS-fingerprinted). 2023+ = Goolsbee era.
    'goolsbee': {'sitemap': 'https://www.chicagofed.org/sitemap.xml',
                 'link_re': r'https://www\.chicagofed\.org/publications/speeches/202[3-9]/[a-z0-9-]+',
                 'date_re': None},
    # Philadelphia: correct listing URL; speech hrefs carry a YYMMDD date prefix.
    'paulson':  {'listing': 'https://www.philadelphiafed.org/the-economy/speeches-anna-paulson',
                 'link_re': r'/the-economy/[a-z\-]+/\d{6}-[\w\-]+',
                 'date_re': r'/(\d{2})(\d{2})(\d{2})-'},
}

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def _curl(url: str) -> str:
    """Fetch via the curl binary (passes Akamai TLS fingerprinting where requests does not)."""
    import subprocess
    try:
        r = subprocess.run(['curl', '-s', '-A', UA, '--max-time', '30', url],
                           capture_output=True, timeout=45)
        return r.stdout.decode('utf-8', 'ignore')
    except Exception:
        return ''


def _bs4_text(html: str) -> str:
    from bs4 import BeautifulSoup
    s = BeautifulSoup(html, 'html.parser')
    for t in s(['script', 'style', 'nav', 'footer', 'header']):
        t.decompose()
    best = ''
    for sel in ['article', 'main', '#content', '.content', '.page-content',
                '.bodycopy', '.cms-content', 'div[class*="content"]']:
        for el in s.select(sel):
            txt = el.get_text(' ', strip=True)
            if len(txt) > len(best):
                best = txt
    if len(best) < 400:
        best = ' '.join(p.get_text() for p in s.find_all('p'))
    return re.sub(r'\s+', ' ', best).strip()

_MONTHS = ('january february march april may june july august september october '
           'november december').split()
_DATE_TEXT = re.compile(r'(' + '|'.join(_MONTHS) + r')\s+(\d{1,2}),?\s+(\d{4})', re.I)


def _date_from_url(href: str, date_re: Optional[str]) -> Optional[pd.Timestamp]:
    if not date_re:
        return None
    m = re.search(date_re, href)
    if not m:
        return None
    g = m.groups()
    y = g[0]
    y = ('20' + y) if len(y) == 2 else y
    try:
        return pd.Timestamp(f"{y}-{g[1]}-{g[2]}")
    except Exception:
        return None


def _page_date(text: str) -> Optional[pd.Timestamp]:
    m = _DATE_TEXT.search(text[:1500])
    if not m:
        return None
    try:
        return pd.Timestamp(f"{m.group(1)} {m.group(2)}, {m.group(3)}")
    except Exception:
        return None


def _try_pdf(page, domain: str) -> str:
    """Some banks (e.g. Boston) put only an abstract on the page and the full text in
    a PDF. Find the speech text PDF, download it, and extract with pdftotext."""
    href = None
    for sel in ["a[href*='-text.pdf']", "a[href*='text.pdf']", "a[href$='.pdf']"]:
        el = page.query_selector(sel)
        if el:
            href = el.get_attribute('href'); break
    if not href:
        return ''
    url = href if href.startswith('http') else domain + href
    try:
        import requests
        from fomc_transcript_scraper import _pdf_to_text
        pdf = requests.get(url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'}).content
        return _pdf_to_text(pdf)
    except Exception:
        return ''


def _main_text(page) -> str:
    """Heuristic main-content extraction: largest of several candidate containers."""
    best = ''
    for sel in ['article', 'main', '#content', '.content', '.page-content',
                '#main-content', '.article-body', '.field--name-body', '.bodycopy']:
        try:
            for el in page.query_selector_all(sel):
                t = el.inner_text()
                if len(t) > len(best):
                    best = t
        except Exception:
            pass
    if len(best) < 500:  # fallback: join paragraph text (skips nav/footer)
        try:
            best = page.eval_on_selector_all("p", "els=>els.map(e=>e.innerText).join('\\n\\n')")
        except Exception:
            pass
    return re.sub(r'\n{3,}', '\n\n', best).strip()


def _collect_via_sitemap(fed_speaker_name: str, bank: str, cfg: dict,
                         start_dt, last: str) -> Dict[str, Any]:
    """Enumerate a president's speech URLs from the bank sitemap and fetch each via
    curl + bs4 (used where the listing is bot-gated but the sitemap + pages are not)."""
    from bs4 import BeautifulSoup
    print(f"\n{'='*60}\n🏦 Starting {bank} bank speech scraper (sitemap) for: {fed_speaker_name}\n{'='*60}\n")
    accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sm = _curl(cfg['sitemap'])
    urls = sorted(set(re.findall(cfg['link_re'], sm)))
    print(f"   {len(urls)} sitemap speech URLs")
    records: Dict[str, Any] = {}
    for url in urls:
        d = _date_from_url(url, cfg.get('date_re'))
        if d is not None and start_dt is not None and d < start_dt:
            continue
        html = _curl(url)
        if not html:
            continue
        text = _bs4_text(html)
        if d is None:
            d = _page_date(text)
        if d is not None and start_dt is not None and d < start_dt:
            continue
        if last not in text.lower()[:1500]:
            continue
        if len(text) < 400:
            continue
        title = ''
        try:
            t = BeautifulSoup(html, 'html.parser').title
            title = (t.get_text() if t else '').split('|')[0].strip()
        except Exception:
            pass
        sid = re.sub(r'[^a-z0-9]+', '-', url.lower()).strip('-')[-60:]
        records[f"fed-speech-{sid}"] = {
            'text': text, 'caption': title, 'handle': fed_speaker_name,
            'postedAt': d, 'source': 'fed_speech', 'postUrl': url, 'postId': sid,
            'accessedAt': accessed_at, 'probabilitySpeaker': 1.0,
        }
        time.sleep(0.1)
    if records:
        os.makedirs(os.path.join('data', 'fed'), exist_ok=True)
        path = os.path.join('data', 'fed', f'{last}_{bank}_speeches.json')
        json.dump(records, open(path, 'w'), ensure_ascii=False, indent=2, default=str)
        print(f"💾 Saved {len(records)} {bank} speeches to {path}")
    else:
        print(f"\n❌ No {bank} speeches retrieved for {fed_speaker_name}")
    return records


def run_bank_speeches(fed_speaker_name: str, bank: str,
                      start_date: Optional[str] = None) -> Dict[str, Any]:
    """Scrape a regional president's speeches from their Reserve Bank site (headless)."""
    cfg = BANK_CONFIGS.get(bank)
    if not cfg:
        print(f"⚠️  no bank config for '{bank}'")
        return {}
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {'error': 'playwright not installed'}
    try:
        print(f"\n{'='*60}")
        print(f"🏦 Starting {bank} bank speech scraper for: {fed_speaker_name}")
        print(f"{'='*60}\n")
        first, last = _name_parts(fed_speaker_name)
        start_dt = pd.Timestamp(start_date) if start_date else None
        if cfg.get('sitemap'):
            return _collect_via_sitemap(fed_speaker_name, bank, cfg, start_dt, last)
        accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        records: Dict[str, Any] = {}
        link_re = re.compile(cfg['link_re'])
        domain = cfg['listing'].split('.org')[0] + '.org'
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True, args=["--disable-http2"])
            page = b.new_page(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36")
            page.goto(cfg['listing'], wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(3500)
            hrefs = page.eval_on_selector_all("a", "els=>els.map(e=>e.getAttribute('href')).filter(Boolean)")
            links = []
            seen = set()
            for h in hrefs:
                if link_re.search(h) and h not in seen:
                    seen.add(h); links.append(h)
            print(f"   {len(links)} candidate speech links")

            for h in links:
                url = h if h.startswith('http') else domain + h
                d = _date_from_url(h, cfg.get('date_re'))
                if d is not None and start_dt is not None and d < start_dt:
                    continue
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=40000)
                    page.wait_for_timeout(1500)
                    text = _main_text(page)
                    title = (page.title() or '').split('|')[0].strip()
                    if len(text) < 1500:  # abstract-only page -> fetch the full-text PDF
                        pdf_text = _try_pdf(page, domain)
                        if len(pdf_text) > len(text):
                            text = pdf_text
                except Exception as ex:
                    print(f"   ⚠️  failed {url}: {ex}")
                    continue
                if d is None:
                    d = _page_date(text)
                if d is not None and start_dt is not None and d < start_dt:
                    continue
                # filter to this president (listings may include other speakers)
                if last not in text.lower()[:1200] and last not in title.lower():
                    continue
                if len(text) < 400:
                    continue
                sid = re.sub(r'[^a-z0-9]+', '-', h.lower()).strip('-')[-60:]
                records[f"fed-speech-{sid}"] = {
                    'text': text,
                    'caption': title,
                    'handle': fed_speaker_name,
                    'postedAt': d,
                    'source': 'fed_speech',
                    'postUrl': url,
                    'postId': sid,
                    'accessedAt': accessed_at,
                    'probabilitySpeaker': 1.0,
                }
                time.sleep(0.2)
            b.close()

        if records:
            output_dir = os.path.join('data', 'fed')
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f'{last}_{bank}_speeches.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 Saved {len(records)} {bank} speeches to {json_path}")
            return records
        print(f"\n❌ No {bank} speeches retrieved for {fed_speaker_name}")
        return {}
    except Exception as e:
        print(f"\n⚠️  {bank} bank scraper failed: {str(e)}")
        return {'error': str(e)}


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--bank':
        out = run_bank_speeches(sys.argv[3], sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        name = sys.argv[1] if len(sys.argv) > 1 else "John C. Williams"
        sd = sys.argv[2] if len(sys.argv) > 2 else "2023-01-01"
        out = run_fed_president_speeches(name, sd)
    print(f"\nTotal: {len(out)}")
