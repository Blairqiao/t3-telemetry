from datetime import datetime

import pytest

from t3_telemetry.prices import CostBreakdown, model_key, rate_for, step_cost


@pytest.mark.parametrize(("model", "expected"), [
    ("gemini-3.8-flash-high", "gemini-3.8-flash"),
    ("claude-opus-5-5", "claude-opus-5-5"),
    ("claude-opus-5-20260101", "claude-opus-5"),
    ("codex-auto-review", None), ("<synthetic>", None), (None, None),
    ("GPT-6.1-SOL-high", "gpt-6.1-sol"), ("gpt-6-astra-high", "gpt-6-astra"),
    ("gpt-6.1-astra", None), ("gpt-6-sol", None),
])
def test_model_key(model, expected):
    assert model_key(model) == expected


def test_effective_date():
    assert rate_for("gemini-4-argon", datetime(2026, 11, 30, 12)).input == 2
    assert rate_for("gemini-4-argon", datetime(2026, 12, 1, 12)).input == 4
    assert rate_for("gemini-4-argon", None).input == 2


def test_long_tier():
    assert rate_for("claude-haiku-5-5", None, 100_000).input == pytest.approx(0.1)
    assert rate_for("claude-haiku-5-5", None, 100_001).input == pytest.approx(0.5)


def test_step_cost_and_add():
    cost = step_cost("claude-opus-5-5", None, 1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert (cost.input, cost.cache_read, cost.cache_write, cost.output, cost.total) == pytest.approx((4, 0.2, 5, 20, 29.2))
    other = step_cost("unknown", None, 1, 2, 3, 4)
    assert other.unpriced_tokens == 10
    assert other.total == 0
    cost.add(other)
    assert cost.unpriced_tokens == 10
    assert cost.total == pytest.approx(29.2)
    assert CostBreakdown().total == 0
