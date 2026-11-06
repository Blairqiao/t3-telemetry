from datetime import datetime

import pytest

from t3_telemetry.insights import (
    KINDS,
    MEASURES,
    Kpi,
    _fmt_tokens,
    _usd,
    headline,
    kind_values,
    kpis,
    share_label,
)
from t3_telemetry.usage import UsageStats


def test_share_label():
    assert share_label(0, 100) == "0%"
    assert share_label(-5, 100) == "0%"
    assert share_label(10, 0) == "0%"
    assert share_label(10, -100) == "0%"
    assert share_label(0.4, 100) == "<1%"
    assert share_label(0.99, 100) == "<1%"
    assert share_label(1, 100) == "1%"
    assert share_label(50, 100) == "50%"
    assert share_label(99.6, 100) == "100%"
    assert share_label(100, 100) == "100%"


def test_usd_boundaries():
    assert _usd(0) == "$0"
    assert _usd(-1) == "$0"
    assert _usd(0.001) == "$0.00"
    assert _usd(0.05) == "$0.05"
    assert _usd(99.99) == "$99.99"
    assert _usd(100) == "$100"
    assert _usd(1234) == "$1,234"
    assert _usd(1234.6) == "$1,235"
    assert _usd(9999) == "$9,999"
    assert _usd(10_000) == "$10.0K"
    assert _usd(12_345) == "$12.3K"
    assert _usd(250_000) == "$250.0K"


def test_fmt_tokens():
    assert _fmt_tokens(None) == "0"
    assert _fmt_tokens(0) == "0"
    assert _fmt_tokens(-10) == "0"
    assert _fmt_tokens(500) == "500"
    assert _fmt_tokens(1500) == "1.5k"
    assert _fmt_tokens(2_500_000) == "2.50M"


def test_kind_values():
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=10, cache_read=20, cache_write=30, output=40, subagent=False)
    assert kind_values(stats, "tokens") == {
        "cache_read": 20,
        "cache_write": 30,
        "output": 40,
        "input": 10,
    }
    usd_vals = kind_values(stats, "usd")
    assert tuple(usd_vals.keys()) == KINDS
    assert usd_vals["cache_read"] == stats.cost.cache_read
    assert usd_vals["cache_write"] == stats.cost.cache_write
    assert usd_vals["output"] == stats.cost.output
    assert usd_vals["input"] == stats.cost.input


def test_headline_no_usage():
    stats = UsageStats()
    assert headline(stats, "tokens", "this week's") == "No usage recorded."
    assert headline(stats, "usd", "this week's") == "No usage recorded."


def test_headline_token_mode_kind():
    # 90 cache_read + 10 output gives "90% of this week's tokens was re-reading cached context."
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=0, cache_read=90, cache_write=0, output=10, subagent=False)
    assert headline(stats, "tokens", "this week's") == "90% of this week's tokens was re-reading cached context."


def test_headline_all_kind_phrases():
    # cache_write
    stats_cw = UsageStats()
    stats_cw.add(model="claude-sonnet-5", when=None, input=0, cache_read=0, cache_write=100, output=0, subagent=False)
    assert headline(stats_cw, "tokens", "today's") == "100% of today's tokens was writing new context to the cache."

    # output
    stats_out = UsageStats()
    stats_out.add(model="claude-sonnet-5", when=None, input=0, cache_read=0, cache_write=0, output=100, subagent=False)
    assert headline(stats_out, "tokens", "this month's") == "100% of this month's tokens was model output, thinking included."

    # input
    stats_in = UsageStats()
    stats_in.add(model="claude-sonnet-5", when=None, input=100, cache_read=0, cache_write=0, output=0, subagent=False)
    assert headline(stats_in, "tokens", "this chat's") == "100% of this chat's tokens was uncached input."


def test_headline_tie_breaking():
    # KINDS = ("cache_read", "cache_write", "output", "input")
    # Equal cache_read and cache_write -> cache_read wins
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=0, cache_read=50, cache_write=50, output=0, subagent=False)
    assert headline(stats, "tokens", "this week's") == "50% of this week's tokens was re-reading cached context."

    # Equal cache_write and output -> cache_write wins
    stats2 = UsageStats()
    stats2.add(model="claude-sonnet-5", when=None, input=0, cache_read=0, cache_write=50, output=50, subagent=False)
    assert headline(stats2, "tokens", "this week's") == "50% of this week's tokens was writing new context to the cache."


def test_headline_subagent_wins():
    # Subagents are 60% and also at least the top kind
    # Main agent: 20 input + 20 output = 40
    # Subagent: 10 input + 30 cache_read + 20 output = 60
    # Total: 100. Kinds: input 30, cache_read 30, output 40, cache_write 0.
    # Top kind is output (40). Subagent is 60 >= 40 and 60 / 100 >= 0.5.
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=20, cache_read=0, cache_write=0, output=20, subagent=False)
    stats.add(model="claude-sonnet-5", when=None, input=10, cache_read=30, cache_write=0, output=20, subagent=True)
    assert headline(stats, "tokens", "this week's") == "60% of this week's tokens went to subagents."


def test_headline_subagent_does_not_win_if_top_kind_larger():
    # Subagents are 60%, but top kind is 70% (e.g. cache_read has 30 from main + 40 from subagent)
    # Main: 30 cache_read + 10 input = 40
    # Subagent: 40 cache_read + 20 output = 60
    # Total: 100. Kinds: cache_read 70, output 20, input 10.
    # Subagent is 60 (>= 50%), but top kind is 70 > 60. Top kind wins.
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=10, cache_read=30, cache_write=0, output=0, subagent=False)
    stats.add(model="claude-sonnet-5", when=None, input=0, cache_read=40, cache_write=0, output=20, subagent=True)
    assert headline(stats, "tokens", "this week's") == "70% of this week's tokens was re-reading cached context."


def test_headline_subagent_does_not_win_if_under_half():
    # Subagents are 40% (40 < 50%)
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=30, cache_read=30, cache_write=0, output=0, subagent=False)
    stats.add(model="claude-sonnet-5", when=None, input=20, cache_read=0, cache_write=0, output=20, subagent=True)
    # Total 100, kinds: input 50, cache_read 30, output 20. Top kind input (50).
    assert headline(stats, "tokens", "this week's") == "50% of this week's tokens was uncached input."


def test_headline_usd_mode():
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=1_000_000, cache_read=0, cache_write=0, output=0, subagent=False)
    assert headline(stats, "usd", "this week's") == "100% of this week's estimated spend was uncached input."


def test_headline_usd_mode_with_unpriced_suffix():
    stats = UsageStats()
    stats.add(model="claude-sonnet-5", when=None, input=1_000_000, cache_read=0, cache_write=0, output=0, subagent=False)
    stats.add(model="unpriced-model-xyz", when=None, input=500, cache_read=0, cache_write=0, output=0, subagent=False)
    assert headline(stats, "usd", "this week's") == "100% of this week's estimated spend was uncached input (500 tokens unpriced)."

    # Subagent win with unpriced suffix in usd mode
    # Main: input 1_000_000 ($2.00 input)
    # Sub: output 300_000 ($3.00 output)
    # Total spend: $5.00. Subagent spend: $3.00 (60%).
    # Top kind is output ($3.00). Subagent ($3.00) >= top kind ($3.00) and 3/5 >= 0.5.
    stats_sub = UsageStats()
    stats_sub.add(model="claude-sonnet-5", when=None, input=1_000_000, cache_read=0, cache_write=0, output=0, subagent=False)
    stats_sub.add(model="claude-sonnet-5", when=None, input=0, cache_read=0, cache_write=0, output=300_000, subagent=True)
    stats_sub.add(model="unpriced-model-xyz", when=None, input=1_500, cache_read=0, cache_write=0, output=0, subagent=False)
    assert headline(stats_sub, "usd", "this week's") == "60% of this week's estimated spend went to subagents (1.5k tokens unpriced)."


def test_kpis_empty():
    stats = UsageStats()
    items = kpis(stats, "tokens")
    assert len(items) == 5
    assert items == [
        Kpi("$0", "est. at API list prices"),
        Kpi("0%", "went to subagents"),
        Kpi("–", "no model data"),
        Kpi("0%", "of prompt tokens from cache"),
        Kpi("0%", "of steps over 200K context"),
    ]


def test_kpis_populated_tokens():
    stats = UsageStats()
    # Step 1: main, prompt tokens = 100k + 150k + 0 = 250k (> 200k large context)
    stats.add(
        model="claude-sonnet-5",
        when=None,
        input=100_000,
        cache_read=150_000,
        cache_write=0,
        output=50_000,
        subagent=False,
    )
    # Step 2: subagent, prompt tokens = 50k (< 200k)
    stats.add(
        model="claude-opus-5",
        when=None,
        input=50_000,
        cache_read=0,
        cache_write=0,
        output=50_000,
        subagent=True,
    )
    items = kpis(stats, "tokens")
    assert len(items) == 5
    # Total tokens: 300k main + 100k sub = 400k
    # Subagent tokens: 100k / 400k = 25%
    assert items[1] == Kpi("25%", "went to subagents")
    # Top model by tokens: claude-sonnet-5 has 300k, claude-opus-5 has 100k -> 300k/400k = 75%
    assert items[2] == Kpi("75%", "on claude-sonnet-5")
    # Cache share: cache_read (150k) / (input 150k + cache_read 150k + cache_write 0) = 150k / 300k = 50%
    assert items[3] == Kpi("50%", "of prompt tokens from cache")
    # Large context share: 1 large context out of 2 steps = 50%
    assert items[4] == Kpi("50%", "of steps over 200K context")


def test_kpis_populated_usd():
    stats = UsageStats()
    stats.add(
        model="claude-sonnet-5",
        when=None,
        input=1_000_000,
        cache_read=0,
        cache_write=0,
        output=0,
        subagent=False,
    )
    items = kpis(stats, "usd")
    assert len(items) == 5
    assert items[0].label == "est. at API list prices"
    assert items[1].label == "went to subagents"
    assert items[2].label == "on claude-sonnet-5"
    assert items[3].label == "of prompt tokens from cache"
    assert items[4].label == "of steps over 200K context"


def test_headline_usd_mode_all_unpriced_is_not_empty():
    stats = UsageStats()
    stats.add(model=None, when=None, input=2_000, cache_read=0, cache_write=0, output=0, subagent=False)
    assert headline(stats, "usd", "this chat's") == "No priced usage (2.0k tokens unpriced)."
    assert headline(stats, "tokens", "this chat's") == "100% of this chat's tokens was uncached input."
