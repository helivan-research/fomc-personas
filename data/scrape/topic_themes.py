"""Canonical topic-theme taxonomy.

Each extracted chunk carries a free-text `topic`; we additionally tag it with a coarse
`theme` drawn from a fixed six-way taxonomy (five macro themes + ``Other''). The theme is
keyword-derived from the free-text topic and is the label used in the paper's Data section and
Figure 3(d). Keeping the taxonomy here makes ingestion (``dec.utils.opinion_extraction``) and
analysis (``analysis/figures.py``) share one definition.
"""

# Order matters: the first bucket whose keywords match the (lowercased) topic wins.
TOPIC_BUCKETS = [
    ("Inflation & prices", ["inflation", "price", "disinflation", "deflation", "cpi", "pce"]),
    ("Employment & labor", ["labor", "employ", "unemploy", "job", "wage"]),
    ("Rates & policy", ["monetary", "interest rate", "funds rate", "policy rate", "rate cut",
                        "rate hike", "rate path", "rate policy", "neutral rate", "tightening",
                        "easing", "restrictive", "accommodat", "asset purchase", "balance sheet",
                        "policy stance", "quantitative", "forward guidance", "taper", "mandate",
                        "guidance", "fomc", "communication", "transparency"]),
    ("Financial stability", ["financial", "bank", "credit", "systemic", "liquidity", "supervis",
                             "regulat", "capital", "stress", "yield", "asset", "risk", "bubble",
                             "stablecoin", "crypto", "debt", "market", "investor"]),
    ("Growth & outlook", ["growth", "gdp", "outlook", "recovery", "recession", "activity",
                          "demand", "econom", "consumer", "spending", "housing", "investment",
                          "business", "productiv", "manufactur", "trade", "tariff", "global",
                          "supply"]),
]

THEMES = [name for name, _ in TOPIC_BUCKETS] + ["Other"]


def bucket_topic(topic):
    """Map a free-text topic to one of the six themes (``Other`` if nothing matches)."""
    if not isinstance(topic, str):
        return "Other"
    t = topic.lower()
    for name, kws in TOPIC_BUCKETS:
        if any(k in t for k in kws):
            return name
    return "Other"
