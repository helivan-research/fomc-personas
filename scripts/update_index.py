#!/usr/bin/env python3
"""Extend the live persona index over newly-completed FOMC meetings.

Run weekly by CI after a successful corpus refresh (and safe to run any time by hand):

  1. merge new FRED observations into the cached macro series,
  2. find completed live meetings (macro.completed_live_meetings: date passed AND the target
     series already carries the outcome) that are missing from the generation caches,
  3. generate ONLY the missing meetings into the caches the site export reads
     (retrieval_cv/beta0.0_tau2.0_pit + figure_index_pit cond/noc/vote),
  4. write `new_meetings=true|false` to $GITHUB_OUTPUT (when set) so the workflow knows whether
     to commit caches and re-export the website's pbi.json.

Everything cached is skipped, so a week with no new meeting costs no OpenAI work. The paper's
frozen FOMC_MEETINGS list and its published caches are never modified, only extended.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "paper", ROOT / "paper" / "experiments"):
    sys.path.insert(0, str(p))

import fomc_personas as fp                 # noqa: E402
from fomc_personas import macro            # noqa: E402
import fig_index as F                      # noqa: E402
import retrieval_cv as R                   # noqa: E402

BETA_DIR = R.CACHE / "beta0.0_tau2.0_pit"


def _emit(new: bool) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"new_meetings={'true' if new else 'false'}\n")
    print(f"new_meetings={new}")


def main() -> None:
    print("refreshing FRED ...")
    macro.refresh_fred()
    series = macro.load_fred()
    live = macro.completed_live_meetings(series)
    print("completed live meetings:", live)

    def _missing():
        miss = set()
        for d in live:
            if not (BETA_DIR / f"resp_{d}.json").exists():
                miss.add(d)
            for tag in (f"resp_{d}_cond", f"resp_{d}_noc", f"vote_{d}"):
                if not (F.CACHE / f"{tag}.json").exists():
                    miss.add(d)
        return sorted(miss)

    missing = _missing()
    if not missing:
        print("index caches already cover every completed meeting -- nothing to do")
        _emit(False)
        return

    print("generating missing meetings:", missing)
    macro.FOMC_MEETINGS = list(macro.FOMC_MEETINGS) + live
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    u = fp.axis(fp.load_anchors())
    dec = macro.decisions(series)

    R.index_series(df, dec, bios, u, 0.0, 2.0)          # -> beta0.0_tau2.0_pit (the site's PBI)
    F._index_series(df, dec, bios, u, condition=True)   # -> figure_index_pit resp_*_cond
    F._index_series(df, dec, bios, u, condition=False)  # -> figure_index_pit resp_*_noc (static)
    F._direct_vote(df, dec, bios)                       # -> figure_index_pit vote_* (baseline)

    still = _missing()
    if still:
        raise SystemExit(f"generation incomplete for {still} -- not marking success")
    print("caches complete for:", missing)
    _emit(True)


if __name__ == "__main__":
    main()
