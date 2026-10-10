#!/usr/bin/env bash
# Regenerate every paper figure and the index generation caches from the CURRENT dataset snapshot.
#
# Use after a corpus change (new members, new sources, date repairs): generations depend on what each
# persona can retrieve, so every cached answer is stale. Previous results stay reproducible from git
# (e.g. tag aaai27-results). Costs roughly $30-50 of API calls and a few hours.
#
#   bash paper/regenerate.sh            # from the repo root, with OPENAI_API_KEY / MISTRAL_API_KEY / HF_TOKEN set
set -euo pipefail
cd "$(dirname "$0")/.."
C=paper/.cache
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

python - <<'EOF'
import os
from huggingface_hub import HfApi
sha = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info("helivan/fomc-personas").sha
open("paper/.cache/DATASET_REVISION", "w").write(sha + "\n")
print("dataset revision:", sha)
EOF

# 1. stale generation caches out (committed ones remain in git history)
log "clearing generation caches"
rm -rf "$C/retrieval_cv/beta0.0_tau2.0_pit" "$C/figure_index_pit" \
       "$C/cut_battery_b0.6" "$C/hike_battery_b0.6" "$C/live_forecast" \
       "$C/figure_likeness" "$C/figure_stance"
rm -f "$C"/vote_parse/hike_*.json "$C"/vote_parse/cut_*.json

# 2. figures that do not depend on the index
log "fig_data";     python paper/fig_data.py
log "fig_stance";   python paper/fig_stance.py      # + fig_generator.pdf
log "fig_likeness"; python paper/fig_likeness.py    # + fig_confusion.pdf

# 3. index caches for every meeting (hawk-dove index, ablations, vote batteries) + next-meeting forecast
log "index caches (backfill)"; python scripts/update_index.py --backfill

# 4. the index figure
log "fig_index"; python paper/fig_index.py
log "done"
