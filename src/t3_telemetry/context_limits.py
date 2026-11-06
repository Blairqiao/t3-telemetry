"""Context window limits for supported models."""

from __future__ import annotations

# Recorded by T3 provider threads and Codex rollouts, checked 2026-11-06.
_LIMITS: list[tuple[str, int]] = [   # first prefix match wins
    ("claude-opus-5-5", 1_000_000),
    ("gemini-", 1_048_576),
    ("gpt-6.1-sol", 475_000),
    ("gpt-6-astra", 258_400),
]


def context_limit_for(model: str | None) -> int | None:
    """Return context token limit for model, or None for unknown/blank/<synthetic> models."""
    if not model:
        return None
    val = model.strip()
    if not val or val.startswith("<"):
        return None
    normalized = val.lower()
    for prefix, limit in _LIMITS:
        if normalized.startswith(prefix.lower()):
            return limit
    return None
