"""Speeches from Reserve Bank sites that refuse automated access, via the Internet Archive.

stlouisfed.org refuses connections from cloud and datacenter networks (this host and GitHub-hosted
runners both time out), so Alberto G. Musalem's remarks cannot be scraped directly. The Internet
Archive's Wayback Machine snapshots the St. Louis Fed's remarks pages and the PDFs of the prepared
remarks they link to, usually within a day of publication. This scraper:

  1. lists the archived remarks pages (CDX index) under the president's section,
  2. reads each archived page and finds the PDF of the prepared remarks,
  3. fetches the archived PDF ("id_" = the original bytes) and extracts its text,
  4. records the ORIGINAL stlouisfed.org URL as the citation (postUrl), so every chunk still points
     to the official source; the archive is only the transport.

Only documents attributed to the member in their opening lines and dated on/after the start of
their tenure are kept. Remarks pages with no prepared-remarks PDF (Q&As, roundtables, photo pages)
are skipped. Requests are paced and retried: the archive rate-limits rapid queries.
"""
import datetime
import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional

import pandas as pd
import requests

WB = "https://web.archive.org"
UA = {"User-Agent": "fomc-personas research (archive.org Wayback; official St. Louis Fed remarks)"}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                      "oct", "nov", "dec"], 1)}

SITES = {
    "musalem": {
        "pages": "stlouisfed.org/from-the-president/remarks/*",
        "page_re": r"/from-the-president/remarks/(\d{4})/[a-z0-9\-]+/?$",
        "pdf_re": r'/-/media/project/frbstl/stlouisfed/musalem/[^"\'\s>]+\.pdf',
        "origin": "https://www.stlouisfed.org",
        "tenure_start": "2024-04-02",
    },
}


def _get(url, tries=5, **kw):
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            if r.status_code == 200 and r.content:
                return r
            if r.status_code in (429, 503):
                time.sleep(10 * (k + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    return None


def _cdx(pattern):
    r = _get(f"{WB}/cdx/search/cdx", params={"url": pattern, "output": "json",
                                             "fl": "timestamp,original", "filter": "statuscode:200",
                                             "collapse": "urlkey", "limit": "5000"})
    if r is None or r.status_code != 200 or not r.text.strip():
        raise RuntimeError(f"CDX query failed for {pattern}")
    return r.json()[1:]


def _pdf_text(blob: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(blob)
        f.flush()
        out = subprocess.run(["pdftotext", "-layout", f.name, "-"], capture_output=True, text=True)
    return out.stdout


def _date_from_text(head: str) -> Optional[pd.Timestamp]:
    m = re.search(r"\b(Jan|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
                  r"(\d{1,2}),\s+(20\d{2})", head)
    if not m:
        return None
    return pd.Timestamp(int(m.group(3)), MONTHS[m.group(1)[:3].lower()], int(m.group(2)))


def _add(pdf_bytes, url, cfg, start, last, fed_speaker_name, accessed_at, records):
    text = _pdf_text(pdf_bytes)
    head = text[:1500]
    if last.lower() not in head.lower():
        return                                                   # not attributed to the member
    d = _date_from_text(head)
    if d is None or d < start:
        return
    title = next((ln.strip() for ln in head.splitlines() if ln.strip()), "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text).strip()
    if len(text) < 1500:
        return
    site = next(k for k, v in SITES.items() if v is cfg)
    sid = re.sub(r"[^a-z0-9]+", "-", url.rstrip("/").split("/")[-1])[:80]
    records[f"fed-speech-{site}-{sid}"] = {
        "text": text, "caption": title, "handle": fed_speaker_name, "postedAt": d,
        "source": "fed_speech", "postUrl": url, "postId": sid,
        "accessedAt": accessed_at, "probabilitySpeaker": 1.0,
    }
    print(f"   + {d.date()}  {title[:70]}")


def run_wayback_speeches(fed_speaker_name: str, site: str,
                         start_date: Optional[str] = None) -> Dict[str, Any]:
    cfg = SITES[site]
    last = fed_speaker_name.split()[-1]
    start = max(pd.Timestamp(cfg["tenure_start"]), pd.Timestamp(start_date) if start_date else pd.Timestamp.min)
    accessed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\n{'='*60}\n🗄️  Wayback speech scraper ({site}) for: {fed_speaker_name}\n{'='*60}")
    try:
        pages = [(ts, url) for ts, url in _cdx(cfg["pages"])
                 if re.search(cfg["page_re"], url) and int(re.search(cfg["page_re"], url).group(1)) >= start.year]
    except Exception as e:
        print(f"   ⚠️  {e}")
        return {"error": str(e)}
    print(f"   {len(pages)} archived remarks pages since {start.year}")
    records: Dict[str, Any] = {}
    todo, failed = sorted(pages), []
    for attempt, pause in ((1, 1.5), (2, 6.0)):                 # second, slower pass for throttled fetches
        if attempt == 2:
            if not failed:
                break
            print(f"   retrying {len(failed)} throttled fetches")
            time.sleep(30)
            todo, failed = failed, []
        for ts, url in todo:
            pdf_bytes = None
            r = _get(f"{WB}/web/{ts}id_/{url}")
            time.sleep(pause)
            if r is None or r.status_code != 200:
                failed.append((ts, url))
                continue
            pdfs = list(dict.fromkeys(re.findall(cfg["pdf_re"], r.text)))
            if not pdfs:
                continue                                         # Q&A / roundtable / photo page
            pr = _get(f"{WB}/web/{ts}id_/{cfg['origin']}{pdfs[0]}", allow_redirects=True)
            time.sleep(pause)
            if pr is None or pr.status_code != 200 or not pr.content.startswith(b"%PDF"):
                failed.append((ts, url))
                continue
            pdf_bytes = pr.content
            _add(pdf_bytes, url, cfg, start, last, fed_speaker_name, accessed_at, records)
    if failed:
        print(f"   {len(failed)} pages still unavailable after retry")
    if records:
        os.makedirs(os.path.join("data", "fed"), exist_ok=True)
        path = os.path.join("data", "fed", f"{last.lower()}_{site}_wayback.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 Saved {len(records)} speeches to {path}")
    else:
        print(f"❌ No archived speeches retrieved for {fed_speaker_name}")
    return records


if __name__ == "__main__":
    import sys
    out = run_wayback_speeches(sys.argv[1] if len(sys.argv) > 1 else "Alberto G. Musalem",
                               sys.argv[2] if len(sys.argv) > 2 else "musalem",
                               sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"\nTotal: {len(out)}")
