"""Personas: retrieval over a member's chunks + bio-grounded generation.

A persona is the base model (`gpt-4o-mini`) conditioned on a member-specific biography (system prompt)
with top-k retrieval from that member's chunks. `respond()` elicits short, stance-forward answers to a
query battery — optionally prepended with an as-of-date market briefing for the rate-action index.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

GEN_MODEL = "gpt-4o-mini"
_client = None


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()  # reads OPENAI_API_KEY
    return _client


# --- retrieval ---------------------------------------------------------------

def retrieve(member_df, q_emb, k: int = 3):
    """Top-k chunk `text`s for one query, ranked by cosine to the member's chunk embeddings.
    `member_df` must carry an `embedding` column (load_chunks(embeddings=...))."""
    emb = np.vstack(member_df["embedding"].values)
    q = np.asarray(q_emb, dtype=np.float32).ravel()
    top = np.argsort(-(emb @ q))[:k]
    return member_df["text"].values[top].tolist()


# --- generation --------------------------------------------------------------

def system_prompt(member: str, bio: str = "") -> str:
    p = (f"You are {member}, a member of the U.S. Federal Open Market Committee (FOMC), "
         f"which sets U.S. monetary policy.")
    if bio:
        p += f" Background: {bio}"
    return p


def _user_prompt(question: str, retrieved: list[str], briefing: str | None = None) -> str:
    """Stance-elicitation prompt (used for the stance/likeness figures). The `briefing` argument is
    accepted for a uniform `prompt_fn` signature but unused here — stance elicitation is not
    market-conditioned. The index figure passes `index_prompt` instead (which does use it)."""
    rows = "\n".join(f"- {t}" for t in retrieved)
    return (f'Answer the following question: "{question}"\n\n'
            f'Here are statements/positions you have actually expressed related to it:\n'
            f'{rows}\n\nState your actual position in AT MOST TWO SENTENCES, plainly and directly, '
            f'making your stance on the policy direction (raise, hold, or cut rates, and inflation '
            f'vs. employment priority) unmistakable. Do not hedge, do not equivocate, do not use '
            f'emojis, do not use lists.')


def index_prompt(question: str, retrieved: list[str], briefing: str | None = None) -> str:
    """Rate-action-index prompt: with `briefing` (the as-of-date market conditions $c^{(t)}$) the
    member answers *in light of current conditions*; without it (the $c^{(t)}$ ablation) the same
    retrieval/persona runs unconditioned. Distinct from the stance prompt above and held fixed across
    meetings — its wording is what the paper's PBI numbers were generated under."""
    rows = "\n".join(f"- {t}" for t in retrieved)
    if briefing:
        head = (f"{briefing}\n\nHere are statements/positions you have actually expressed:\n{rows}\n\n"
                f'In light of these current conditions, answer the following: "{question}"')
    else:
        head = (f"Here are statements/positions you have actually expressed:\n{rows}\n\n"
                f'Answer the following: "{question}"')
    return (head + "\n\nState your position in AT MOST TWO SENTENCES, plainly and directly, making "
            "your stance unmistakable. Do not hedge, no emojis, no lists.")


def _clean(text: str) -> str:
    """Collapse whitespace and strip any wrapping quotes the model adds (e.g. when asked to continue
    a quoted statement); gpt-4o-mini returns the answer directly (no delimiters needed)."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().strip("\"“”").strip()


def generate(messages, model: str = GEN_MODEL, max_tokens: int = 90, workers: int = 12):
    """Run a list of chat-message lists through the model concurrently; returns cleaned strings."""
    client = _openai()

    def run(m):
        try:
            r = client.chat.completions.create(
                model=model, messages=m, max_tokens=max_tokens, temperature=0.2)
            return _clean(r.choices[0].message.content)
        except Exception:
            return ""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run, messages))


def respond(df, members, queries, bios, k: int = 3, briefing: str | None = None,
            model: str = GEN_MODEL, prompt_fn=None):
    """Elicit each member's answers to `queries` (a list of question strings).

    Returns {member: [response per query]}. `df` must have an `embedding` column; `bios` is
    {member: biography}; `briefing` (optional) is the as-of-date macro string passed to `prompt_fn`.
    `prompt_fn(question, retrieved, briefing) -> str` defaults to the stance prompt; the index figure
    passes `index_prompt` to reproduce the paper's PBI wording.
    """
    from .embeddings import embed
    prompt_fn = prompt_fn or _user_prompt
    q_emb = embed(list(queries))
    metas, messages = [], []
    for m in members:
        mdf = df[df["member"] == m]
        if mdf.empty:
            continue
        sys = system_prompt(m, bios.get(m, ""))
        for qi, q in enumerate(queries):
            retrieved = retrieve(mdf, q_emb[qi], k)
            metas.append((m, qi))
            messages.append([{"role": "system", "content": sys},
                             {"role": "user", "content": prompt_fn(q, retrieved, briefing)}])
    comps = generate(messages, model=model)
    out = {m: [""] * len(queries) for m in members}
    for (m, qi), c in zip(metas, comps):
        out[m][qi] = c
    return out
