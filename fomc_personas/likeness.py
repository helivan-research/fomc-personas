"""Likeness of the persona collection along two axes.

identifiability — can a response be attributed to its persona? A leave-one-query-out shrinkage-LDA
  over the persona responses to a fixed query set; per-member attribution recall vs chance.
detectability — is a persona's generated text distinguishable from the member's real text? Seeded
  held-out completions, then a cross-validated LDA separating generated vs real continuations
  (length-matched); per-member tau = 2*max(A, 1-A) - 1, with a real-vs-real floor.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from .embeddings import embed
from .persona import generate, system_prompt

N_TEST = 40
SEED_FRAC = 0.5


# --- identifiability: leave-one-query-out attribution over persona responses -------------------

def loqo_recall(responses: dict):
    """responses: {member: [response per query]}. Per-member LOQO shrinkage-LDA attribution recall.
    Returns (members, {member: recall}, confusion)."""
    members = sorted(responses)
    n = len(members)
    nq = min(len(responses[m]) for m in members)
    A = np.stack([embed(responses[m][:nq]) for m in members])  # (n, nq, dim)
    conf = np.zeros((n, n), int)
    for j in range(nq):
        Xtr = np.vstack([np.delete(A[i], j, axis=0) for i in range(n)])
        ytr = np.concatenate([[i] * (nq - 1) for i in range(n)])
        clf = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
        for t, p in zip(range(n), clf.predict(np.vstack([A[i][j] for i in range(n)]))):
            conf[t, p] += 1
    recall = {m: float(conf[i, i] / conf[i].sum()) for i, m in enumerate(members)}
    return members, recall, conf


def nway_scores(responses: dict, member: str):
    """The LOQO classifier's log-prob of class=member on each member's held-out responses:
    (self_scores, rest_scores) for the identifiability example panel."""
    members = sorted(responses)
    n = len(members)
    nq = min(len(responses[m]) for m in members)
    A = np.stack([embed(responses[m][:nq]) for m in members])
    mi = members.index(member)
    self_s, rest_s = [], []
    for j in range(nq):
        Xtr = np.vstack([np.delete(A[i], j, axis=0) for i in range(n)])
        ytr = np.concatenate([[i] * (nq - 1) for i in range(n)])
        clf = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
        lp = clf.predict_log_proba(np.vstack([A[i][j] for i in range(n)]))[:, list(clf.classes_).index(mi)]
        for t in range(n):
            (self_s if t == mi else rest_s).append(float(lp[t]))
    return self_s, rest_s


# --- detectability: seeded held-out completions ------------------------------------------------

def split(df, n_test: int = N_TEST, seed: int = 0):
    """Per member: held-out test set (n_test substantive quotes, >=8 words) + retrieval pool (rest)."""
    rng = random.Random(seed)
    pool, test = {}, {}
    for m, g in df.groupby("member"):
        g = g.reset_index(drop=True)
        idx = list(range(len(g)))
        rng.shuffle(idx)
        test_idx = [i for i in idx if len(str(g.iloc[i]["quote"]).split()) >= 8][:n_test]
        ts = set(test_idx)
        pool[m] = g[~g.index.isin(ts)].reset_index(drop=True)
        test[m] = g.iloc[test_idx].reset_index(drop=True)
    return pool, test


def _seed_of(quote, frac=SEED_FRAC):
    w = str(quote).split()
    return " ".join(w[:max(4, int(len(w) * frac))])


def complete_seeded(pool, test, bios, k: int, model: str = "gpt-4o-mini"):
    """Generate seeded continuations of each held-out quote. Returns
    [{governor, seed, real_quote, completion}]."""
    metas, messages = [], []
    for name in test:
        sysp = system_prompt(name, bios.get(name, ""))
        ptext = pool[name]["text"].values
        pemb = np.vstack(pool[name]["embedding"].values) if k else None
        for j in range(len(test[name])):
            quote = test[name].iloc[j]["quote"]
            seed = _seed_of(quote)
            nrem = max(1, len(str(quote).split()) - len(str(seed).split()))
            if k == 0:
                user = (f"Continue the following statement in your own voice, expressing the views you "
                        f"actually hold. Write approximately {nrem} more words; do not use emojis.\n\n"
                        f'Statement to continue: "{seed}"')
            else:
                top = np.argsort(-(pemb @ embed([seed])[0]))[:k]
                retrieved = "\n".join(f"- {ptext[int(i)]}" for i in top)
                user = (f"Here are statements/positions you have actually expressed:\n{retrieved}\n\n"
                        f"Continue the following statement in your own voice, expressing the views you "
                        f"actually hold. Write approximately {nrem} more words, matching the brevity and "
                        f"register of the examples; do not use emojis.\n\nStatement to continue: \"{seed}\"")
            metas.append({"governor": name, "seed": seed, "real_quote": quote})
            messages.append([{"role": "system", "content": sysp}, {"role": "user", "content": user}])
    comps = generate(messages, model=model, max_tokens=120)
    return [{**m, "completion": c} for m, c in zip(metas, comps) if c]


def _detect(ea, eb):
    X = np.vstack([ea, eb])
    y = np.array([0] * len(ea) + [1] * len(eb))
    k = min(5, len(ea), len(eb))
    if k < 2:
        return 0.5
    cv = StratifiedKFold(k, shuffle=True, random_state=0)
    return float(cross_val_score(LDA(solver="lsqr", shrinkage="auto"), X, y, cv=cv).mean())


def _tau(a):
    return 2 * max(a, 1 - a) - 1


def _matched(comps):
    """Length-matched (generated continuation, real continuation) per member."""
    gen_m, cont_m = defaultdict(list), defaultdict(list)
    for c in comps:
        gw = str(c["completion"]).split()
        rw = str(c["real_quote"]).split()[len(str(c["seed"]).split()):]
        m = min(len(gw), len(rw))
        if m >= 3:
            gen_m[c["governor"]].append(" ".join(gw[:m]))
            cont_m[c["governor"]].append(" ".join(rw[:m]))
    return gen_m, cont_m


def detect_taus(comps, with_floor: bool = False):
    """Per-member detectability tau (generated vs real continuation), optionally + real-real floor."""
    gen_m, cont_m = _matched(comps)
    taus, floor = {}, {}
    for name in gen_m:
        if len(gen_m[name]) < 8:
            continue
        eg, ec = embed(gen_m[name]), embed(cont_m[name])
        taus[name] = _tau(_detect(eg, ec))
        if with_floor:
            h = len(ec) // 2
            floor[name] = _tau(_detect(ec[:h], ec[h:]))
    return (taus, floor) if with_floor else taus


def overlap_example(comps, name):
    """One member's gen-vs-real continuations as out-of-fold LDA decision scores (panel d):
    {'gen': [...], 'real': [...], 'tau': ...}."""
    gen_m, cont_m = _matched(comps)
    eg, ec = embed(gen_m[name]), embed(cont_m[name])
    X = np.vstack([eg, ec])
    y = np.array([1] * len(eg) + [0] * len(ec))
    kf = min(5, len(eg), len(ec))
    s = cross_val_predict(LDA(solver="lsqr", shrinkage="auto"), X, y,
                          cv=StratifiedKFold(kf, shuffle=True, random_state=0),
                          method="decision_function")
    return {"gen": s[y == 1].tolist(), "real": s[y == 0].tolist(), "tau": _tau(_detect(eg, ec))}
