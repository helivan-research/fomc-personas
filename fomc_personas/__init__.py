"""fomc-personas: digital-twin personas of the FOMC.

Load the dataset, embed/retrieve text, build personas, recover stance, and reproduce the paper.

Exports resolve lazily (PEP 562) so importing the package stays light: consumers that only need
`embed`/`axis`/`roles` (e.g. the website's serving process) don't pay for pandas, scipy, or
scikit-learn, which load with `data`/`stance`/`likeness` on first use.
"""
_EXPORTS = {
    "load_chunks": ".data", "load_bios": ".data", "load_queries": ".data",
    "load_anchors": ".data", "load_reputational": ".data",
    "embed": ".embeddings",
    "retrieve": ".persona", "respond": ".persona", "generate": ".persona",
    "system_prompt": ".persona",
    "axis": ".stance", "project": ".stance", "kendall_vs_external": ".stance",
    "meanpool_corpus": ".stance", "meanpool_retrieved": ".stance",
    "meanpool_generated": ".stance",
    "roles": None, "macro": None, "likeness": None,   # submodules
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    import importlib
    if name in _EXPORTS:
        mod = _EXPORTS[name]
        if mod is None:                                   # submodule
            return importlib.import_module(f".{name}", __name__)
        return getattr(importlib.import_module(mod, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
