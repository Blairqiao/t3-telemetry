from __future__ import annotations

from dataclasses import dataclass

from .prices import KINDS
from .usage import UsageStats

MEASURES: tuple[str, ...] = ("tokens", "usd")

PHRASES: dict[str, str] = {
    "cache_read": "re-reading cached context",
    "cache_write": "writing new context to the cache",
    "output": "model output, thinking included",
    "input": "uncached input",
}


@dataclass(frozen=True, slots=True)
class Kpi:
    value: str
    label: str


def _usd(x: float) -> str:
    if x <= 0:
        return "$0"
    if x < 100:
        return f"${x:.2f}"
    if x < 10_000:
        return f"${x:,.0f}"
    return f"${x / 1000:.1f}K"


def _fmt_tokens(value: int | float | None) -> str:
    if value is None or value <= 0:
        return "0"
    if value < 1000:
        return f"{int(value)}"
    if value < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.2f}M"


def kind_values(stats: UsageStats, measure: str) -> dict[str, float]:
    if measure == "usd":
        return {
            "cache_read": stats.cost.cache_read,
            "cache_write": stats.cost.cache_write,
            "output": stats.cost.output,
            "input": stats.cost.input,
        }
    return {
        "cache_read": stats.tokens.cache_read_tokens,
        "cache_write": stats.tokens.cache_creation_tokens,
        "output": stats.tokens.output_tokens,
        "input": stats.tokens.input_tokens,
    }


def share_label(part: float, whole: float) -> str:
    if whole <= 0 or part <= 0:
        return "0%"
    pct = (part / whole) * 100
    if 0 < pct < 1:
        return "<1%"
    return f"{round(pct)}%"


def headline(stats: UsageStats, measure: str, scope: str) -> str:
    values = kind_values(stats, measure)
    whole = sum(values.values())
    if whole <= 0:
        if measure == "usd" and stats.cost.unpriced_tokens > 0:
            return f"No priced usage ({_fmt_tokens(stats.cost.unpriced_tokens)} tokens unpriced)."
        return "No usage recorded."

    noun = "tokens" if measure == "tokens" else "estimated spend"
    sub = stats.subagent_tokens.total if measure == "tokens" else stats.subagent_cost.total
    top = max(KINDS, key=lambda k: values.get(k, 0.0))

    if sub / whole >= 0.5 and sub >= values[top]:
        msg = f"{share_label(sub, whole)} of {scope} {noun} went to subagents"
    else:
        msg = f"{share_label(values[top], whole)} of {scope} {noun} was {PHRASES[top]}"

    if measure == "usd" and stats.cost.unpriced_tokens > 0:
        msg += f" ({_fmt_tokens(stats.cost.unpriced_tokens)} tokens unpriced)"

    return msg + "."


def kpis(stats: UsageStats, measure: str) -> list[Kpi]:
    whole = sum(kind_values(stats, measure).values())
    sub = stats.subagent_tokens.total if measure == "tokens" else stats.subagent_cost.total

    models = stats.model_tokens if measure == "tokens" else stats.model_cost
    if not models:
        top_model_kpi = Kpi("–", "no model data")
    else:
        top_model = max(models, key=lambda m: models[m])
        top_model_value = models[top_model]
        whole_models = sum(models.values())
        top_model_kpi = Kpi(share_label(top_model_value, whole_models), f"on {top_model}")

    prompt_tokens = stats.tokens.input_tokens + stats.tokens.cache_read_tokens + stats.tokens.cache_creation_tokens

    return [
        Kpi(_usd(stats.cost.total), "est. at API list prices"),
        Kpi(share_label(sub, whole), "went to subagents"),
        top_model_kpi,
        Kpi(share_label(stats.tokens.cache_read_tokens, prompt_tokens), "of prompt tokens from cache"),
        Kpi(share_label(stats.large_context_steps, stats.steps), "of steps over 200K context"),
    ]
