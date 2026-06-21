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
_START, _END = "###Start", "###End"
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
    return p + f" Start your answer to the question with {_START} and end your answer with {_END}."


def _user_prompt(question: str, retrieved: list[str], briefing: str | None = None) -> str:
    rows = "\n".join(f"- {t}" for t in retrieved)
    head = (f"{briefing}\n\nHere are statements/positions you have actually expressed:\n{rows}\n\n"
            f"In light of these current conditions, answer the following: \"{question}\""
            if briefing else
            f"Answer the following question: \"{question}\"\n\nHere are statements/positions you have "
            f"actually expressed related to it:\n{rows}")
    return (head + "\n\nState your position in AT MOST TWO SENTENCES, plainly and directly, making "
            "your stance on the policy direction (raise, hold, or cut rates, and inflation vs. "
            "employment priority) unmistakable. Do not hedge, do not equivocate, no emojis, no lists.")


def _clean(text: str) -> str:
    if not text:
        return ""
    m = re.search(re.escape(_START) + r"(.*?)" + re.escape(_END), text, re.S)
    out = m.group(1) if m else text.replace(_START, "").replace(_END, "")
    return re.sub(r"\s+", " ", out).strip()


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
            model: str = GEN_MODEL):
    """Elicit each member's answers to `queries` (a list of question strings).

    Returns {member: [response per query]}. `df` must have an `embedding` column; `bios` is
    {member: biography}; `briefing` (optional) is the as-of-date macro string prepended to each query.
    """
    from .embeddings import embed
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
                             {"role": "user", "content": _user_prompt(q, retrieved, briefing)}])
    comps = generate(messages, model=model)
    out = {m: [""] * len(queries) for m in members}
    for (m, qi), c in zip(metas, comps):
        out[m][qi] = c
    return out
