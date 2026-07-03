"""One-shot driver: regenerate everything Figure 4 needs under the point-in-time filter.

1. The PBI's conditioned generations with recency-weighted retrieval (beta=0.6, tau=2) into
   .cache/retrieval_cv/beta0.6_tau2.0_pit/ -- including the 2026 live meetings (the website uses them).
2. fig_index.main() -- regenerates the noc (static-battery) ablation + direct-vote baselines into
   .cache/figure_index_pit/ and writes the updated figure.

    OPENAI_API_KEY=sk-... python paper/experiments/regen_pit.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fomc_personas as fp                        # noqa: E402
from fomc_personas import macro, persona          # noqa: E402
import retrieval_cv as CV                         # noqa: E402  (installs robust OpenAI client)
import fig_index as F                             # noqa: E402

# fig_index's own generation paths (noc ablation, direct votes) call persona.generate with the default
# 12 workers; run them at the CV worker count so the noc pass doesn't take 4x longer than the PBI pass.
_orig_generate = persona.generate
persona.generate = lambda messages, **kw: _orig_generate(messages, **{**kw, "workers": CV.WORKERS})

print("loading corpus ...")
df = fp.load_chunks(embeddings="cached")
bios = fp.load_bios()
u = fp.axis(fp.load_anchors())
series = macro.load_fred()

# --- pass 1: PBI generations (paper window + live 2026 meetings, for the website export) ---
frozen = list(macro.FOMC_MEETINGS)
macro.FOMC_MEETINGS = frozen + CV.LIVE_MEETINGS
dec = macro.decisions(series)                     # after extending, so 2026 meetings have decisions
print(f"pass 1: beta=0.6 tau=2.0 PIT generations ({len(macro.FOMC_MEETINGS)} meetings) ...")
out = CV.index_series(df, dec, bios, u, 0.6, 2.0)
print(f"  done: {len(out)} meetings cached under {CV.CACHE/'beta0.6_tau2.0_pit'}")

# --- pass 2: figure (paper window only), regenerating noc + votes under PIT ---
macro.FOMC_MEETINGS = frozen
print("pass 2: fig_index.main() (noc ablation + direct votes + plot) ...")
F.main()
