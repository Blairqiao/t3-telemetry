# PRICES_CHECKED: 2026-11-05
# Sources: https://github.com/rsaryev/tokendoctor/blob/main/lib/prices.mjs (lib/prices.mjs; Claude checked
# 2026-11-04 against https://platform.claude.com/docs/en/about-claude/pricing),
# https://aistudio.google.com/docs/pricing, https://openai.com/index/gpt-6-1-sol/,
# https://aitokenprice.com/pricing/gpt-6 (third-party, verify).
from dataclasses import dataclass
from datetime import date, datetime
import re

KINDS: tuple[str, ...] = ("cache_read", "cache_write", "output", "input")
PRICES_CHECKED: str = "2026-11-05"


@dataclass(frozen=True, slots=True)
class Rate:
    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(slots=True)
class CostBreakdown:
    input: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    output: float = 0.0
    unpriced_tokens: int = 0

    @property
    def total(self) -> float:
        return self.input + self.cache_read + self.cache_write + self.output

    def add(self, other: "CostBreakdown") -> None:
        for kind in (*KINDS, "unpriced_tokens"):
            setattr(self, kind, getattr(self, kind) + getattr(other, kind))


_TABLE: dict[str, list[tuple[date | None, Rate]]] = {
    "claude-fable-5-1": [(None, Rate(10, 50, 0.25, 12.5))],
    "claude-opus-5-5": [(None, Rate(4, 20, 0.2, 5))],
    "claude-opus-5": [(None, Rate(5, 25, 0.5, 6.25))],
    **{f"claude-opus-4-{n}": [(None, Rate(5, 25, 0.5, 6.25))] for n in (5, 6, 7, 8)},
    "claude-sonnet-5-5": [(None, Rate(2, 10, 0.1, 2.5))],
    "claude-sonnet-5": [(None, Rate(2, 10, 0.2, 2.5))],
    **{f"claude-sonnet-4-{n}": [(None, Rate(3, 15, 0.3, 3.75))] for n in (5, 6)},
    "claude-haiku-5-5": [(None, Rate(0.1, 0.5, 0.01, 0.125))],
    "claude-haiku-4-5": [(None, Rate(1, 5, 0.1, 1.25))],
    "gpt-6-astra": [(None, Rate(10, 50, 1, 12.5))],  # third-party, verify
    "gpt-6.1-sol": [(None, Rate(2, 10, 0.1, 2.5))],
    "gpt-6.1-terra": [(None, Rate(1, 5, 0.05, 1.25))],  # third-party, verify
    "gpt-6-luna": [(None, Rate(0.1, 0.5, 0.01, 0.125))],  # third-party, verify
    "gemini-3.8-flash": [(None, Rate(0.75, 3.75, 0.075, 0.9375)), (date(2027, 1, 1), Rate(1.5, 7.5, 0.15, 1.875))],
    "gemini-4-argon": [(None, Rate(2, 10, 0.1, 2.5)), (date(2026, 12, 1), Rate(4, 20, 0.2, 5))],
}
_LONG: dict[str, tuple[int, Rate]] = {
    "claude-haiku-5-5": (100_000, Rate(0.5, 2.5, 0.05, 0.625)),
    "gpt-6-astra": (272_000, Rate(20, 75, 1, 12.5)),  # third-party, verify
    "gpt-6.1-sol": (272_000, Rate(4, 15, 0.2, 5)),
    "gpt-6.1-terra": (272_000, Rate(2, 7.5, 0.05, 1.25)),  # third-party, verify
    "gpt-6-luna": (272_000, Rate(0.2, 0.75, 0.01, 0.125)),  # third-party, verify
}


def model_key(model: str | None) -> str | None:
    if not model or model.startswith("<"):
        return None
    normalized = re.sub(r"-\d{8}$", "", model.lower())
    return max((key for key in _TABLE if normalized == key or normalized.startswith(key + "-")), key=len, default=None)


def rate_for(model: str | None, when: datetime | None, prompt_tokens: int = 0) -> Rate | None:
    key = model_key(model)
    if key is None:
        return None
    if key in _LONG and prompt_tokens > _LONG[key][0]:
        return _LONG[key][1]
    entries = _TABLE[key]
    if when is None:
        return entries[0][1]
    local_date = when.astimezone().date()
    return next((rate for effective, rate in reversed(entries) if effective is None or effective <= local_date), None)


def step_cost(model: str | None, when: datetime | None, input: int, cache_read: int, cache_write: int, output: int) -> CostBreakdown:
    rate = rate_for(model, when, input + cache_read + cache_write)
    if rate is None:
        return CostBreakdown(unpriced_tokens=input + cache_read + cache_write + output)
    return CostBreakdown(input=input * rate.input / 1_000_000, cache_read=cache_read * rate.cache_read / 1_000_000, cache_write=cache_write * rate.cache_write / 1_000_000, output=output * rate.output / 1_000_000)
