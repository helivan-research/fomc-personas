"""Does swapping the generation model move the PBI?

Regenerates ONLY the conditioned rate-action index (the headline metric) under several small
generation models and reports walk-forward 3-class OOS accuracy (2022-25). Everything else is held
fixed: the same retrieval, the same text-embedding-3-large response embeddings, the same hawk-dove
axis, the same 15-question battery, the same as-of-date briefings, the same walk-forward classifier.
So any change in accuracy is attributable to the generator alone.

The paper generates at temperature 0.2 (concentrated, cleanly-embeddable stances). Models that cannot
honor that (the base GPT-5 family locks temperature to 1.0) would confound "better model" with
"noisier sampling", so the slate sticks to models that run at temp 0.2. gpt-4o-mini is the paper
baseline (its conditioned PBI is already 0.72, reproduced in fig_index from the canonical
generations), so it is not re-run here.

Concurrency: for each model, ALL per-meeting messages are flattened into one thread pool (WORKERS
in-flight at once, the de-facto semaphore) so the pool never drains at meeting boundaries; calls use
exponential backoff on 429/5xx so the higher concurrency stays correct (a rate-limited call is
retried, never silently dropped to an empty response). Responses are then embedded in one batched
pass. Per-meeting caches make the whole thing resumable.

    # put your key in a gitignored .env at the repo root:  OPENAI_API_KEY=sk-...
    python paper/experiments/model_compare.py
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))

# load a gitignored .env (OPENAI_API_KEY); point at the local parquet shards so no HF token is needed
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass
os.environ.setdefault("FOMC_LOCAL_DATASET", str(ROOT.parent / "fomc-personas-hf"))

import openai
import fomc_personas as fp
from fomc_personas import macro, roles, persona
import fig_index as F

CACHE = ROOT / "paper" / ".cache" / "figure_index"
MIN_OPINIONS, TOPK = 5, 3
WORKERS = int(os.environ.get("MODEL_COMPARE_WORKERS", "32"))

# gpt-4o-mini is the paper baseline; we only generate the candidate upgrades.
BASELINE = ("gpt-4o-mini", 0.72)
MODELS = {
    "gpt-4.1-mini": {"temperature": 0.2},
    "gpt-4.1-nano": {"temperature": 0.2},
    "gpt-5.4-nano": {"temperature": 0.2, "reasoning_effort": "none"},  # newest accessible nano, temp 0.2
}


def _slug(m):
    return m.replace(".", "").replace("-", "_")


def _retryable(e):
    if isinstance(e, (openai.RateLimitError, openai.APITimeoutError,
                      openai.APIConnectionError, openai.InternalServerError)):
        return True
    return getattr(e, "status_code", None) in (429, 500, 502, 503, 504)


def _call(client, model, msg, kw_base, tok, max_retries=8):
    """One chat completion with exponential backoff on rate-limit/5xx and one-shot stripping of any
    parameter the model rejects. Raises after max_retries rather than returning a misleading "" — an
    empty string here would silently bias the index, so a persistent failure must surface loudly."""
    kw = {"model": model, "messages": msg, tok: 90, **kw_base}
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**kw)
            return persona._clean(r.choices[0].message.content or "")
        except Exception as e:
            s = str(e).lower()
            if "max_tokens" in s and "max_completion_tokens" in s:
                kw["max_completion_tokens"] = kw.pop("max_tokens", 90); continue
            if "temperature" in s and ("unsupported" in s or "does not support" in s or "only the default" in s):
                kw.pop("temperature", None); continue
            if "reasoning" in s and ("unsupported" in s or "unknown" in s or "does not support" in s):
                kw.pop("reasoning_effort", None); continue
            if _retryable(e) and attempt < max_retries - 1:
                time.sleep(delay + random.random() * 0.5)
                delay = min(delay * 2, 30.0)
                continue
            raise
    raise RuntimeError(f"{model}: generation failed after {max_retries} retries")


def _gen(messages, model, params):
    """Run all messages through one WORKERS-wide pool, with progress logging."""
    client = persona._openai()
    reasoning = model.startswith(("gpt-5", "o3", "o4"))
    tok = "max_completion_tokens" if reasoning else "max_tokens"
    out = [""] * len(messages)
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_call, client, model, messages[i], dict(params), tok): i
                for i in range(len(messages))}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 500 == 0 or done == len(messages):
                print(f"    [{model}] {done}/{len(messages)} calls", flush=True)
    return out


def generate_all(df, dec, bios, series, battery, q_emb, model, params):
    """Generate every uncached meeting for `model` in one flattened pool; write per-meeting caches."""
    mdir = CACHE / _slug(model)
    mdir.mkdir(parents=True, exist_ok=True)
    meetings = [d for d in macro.FOMC_MEETINGS if dec[d]["bps"] is not None]

    flat_msgs, flat_tags, plan, n_cached = [], [], {}, 0
    for d in meetings:
        if (mdir / f"resp_{d}_cond.json").exists():
            n_cached += 1
            continue
        df_t = df[df["postedAt"].astype(str) <= d]
        counts = df_t["member"].value_counts()
        rost = [m for m in counts.index if counts[m] >= MIN_OPINIONS and roles.office_at(m, d) is not None]
        if not rost:
            continue
        _, briefing = macro.macro_briefing(series, d)
        for m in rost:
            mdf = df_t[df_t["member"] == m]
            for qi, q in enumerate(battery):
                retrieved = persona.retrieve(mdf, q_emb[qi], TOPK)
                flat_msgs.append([{"role": "system", "content": persona.system_prompt(m, bios.get(m, ""))},
                                  {"role": "user", "content": persona.index_prompt(q, retrieved, briefing)}])
                flat_tags.append((d, m, qi))
        plan[d] = rost

    print(f"  [{model}] {n_cached} meetings cached; generating {len(flat_msgs)} calls "
          f"across {len(plan)} meetings at {WORKERS} workers", flush=True)
    if flat_msgs:
        comps = _gen(flat_msgs, model, params)
        buf = {d: {m: [""] * len(battery) for m in plan[d]} for d in plan}
        for (d, m, qi), c in zip(flat_tags, comps):
            buf[d][m][qi] = c
        for d, resp in buf.items():
            (mdir / f"resp_{d}_cond.json").write_text(json.dumps(resp))
    return mdir


def index_from_cache(df, dec, u, model):
    """Build the conditioned PBI series from cached responses, embedding every response in one pass."""
    mdir = CACHE / _slug(model)
    meetings = [d for d in macro.FOMC_MEETINGS if dec[d]["bps"] is not None]
    per_meeting, texts = {}, set()
    for d in meetings:
        p = mdir / f"resp_{d}_cond.json"
        if not p.exists():
            continue
        resp = json.loads(p.read_text())
        per_meeting[d] = resp
        for rs in resp.values():
            texts.update(r for r in rs if r)
    texts = list(texts)
    vecs = fp.embed(texts) if texts else np.zeros((0, 1024), np.float32)
    vmap = {t: vecs[i] for i, t in enumerate(texts)}

    out = {}
    for d, resp in per_meeting.items():
        pos = {}
        for m, rs in resp.items():
            valid = [r for r in rs if r]
            if valid:
                pos[m] = float(np.mean([vmap[r] for r in valid], axis=0) @ u)
        if pos:
            out[d] = {"index": float(np.mean(list(pos.values()))), "bps": dec[d]["bps"]}
    return out


def pbi_accuracy(pbi):
    """Walk-forward 3-class OOS accuracy over 2022-25 (identical protocol to fig_index)."""
    dates = [d for d in macro.FOMC_MEETINGS if d in pbi]
    idx = np.array([pbi[d]["index"] for d in dates])
    bps = np.array([float(pbi[d]["bps"]) for d in dates])
    y = np.sign(bps).astype(int)
    m22 = np.array([d >= "2022" for d in dates])
    pred = F._walkfwd([idx, F._mom(idx)], bps)
    return F._acc(pred, y, m22), int((m22 & (pred != 99)).sum())


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY (e.g. in a gitignored .env at the repo root)")
    df = fp.load_chunks(embeddings="cached")
    bios = fp.load_bios()
    u = fp.axis(fp.load_anchors())
    series = macro.load_fred()
    dec = macro.decisions(series)
    battery = fp.load_queries("curated")[:15]
    q_emb = fp.embed(battery)

    results = {BASELINE[0]: (BASELINE[1], None)}
    for model, params in MODELS.items():
        print(f"\n=== {model} ===", flush=True)
        t0 = time.time()
        generate_all(df, dec, bios, series, battery, q_emb, model, params)
        pbi = index_from_cache(df, dec, u, model)
        acc, n = pbi_accuracy(pbi)
        results[model] = (acc, n)
        print(f"  -> {model}: 3-class OOS (2022-25) = {acc:.3f}  (n={n})  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'model':16s} {'3-class OOS (2022-25)':>22s}   n")
    for model, (acc, n) in results.items():
        tag = "  (paper baseline)" if model == BASELINE[0] else ""
        print(f"{model:16s} {acc:>22.3f}   {str(n) if n is not None else '-':>3}{tag}")
    best = max(results, key=lambda m: results[m][0])
    print(f"\nbest: {best} ({results[best][0]:.3f}) vs {BASELINE[0]} baseline {BASELINE[1]:.3f}")


if __name__ == "__main__":
    main()
