"""Tests for provider keys, family labels, and context window limits."""

from t3_telemetry.context_limits import _LIMITS, context_limit_for
from t3_telemetry.providers import FAMILY_LABELS, PROVIDER_KEYS, family_label, provider_key
from t3_telemetry.usage import provider_family


def test_shared_contracts_constants():
    assert PROVIDER_KEYS == ("claude", "codex", "antigravity")
    assert FAMILY_LABELS == {
        "claude": "Claude",
        "codex": "Codex",
        "antigravity": "Antigravity",
    }


def test_provider_key_mappings_and_precedence():
    # Antigravity takes precedence (e.g. antigravity_gemini_* should not be confused)
    assert provider_key("antigravity") == "antigravity"
    assert provider_key("antigravity_gemini_main") == "antigravity"
    assert provider_key("antigravity_gemini_secondary") == "antigravity"
    assert provider_key("antigravity_harness", "inst") == "antigravity"
    assert provider_key("unknown_driver", "antigravity_gemini") == "antigravity"
    assert provider_key(None, "antigravity_inst") == "antigravity"

    # Claude
    assert provider_key("claude") == "claude"
    assert provider_key("claudeAgent") == "claude"
    assert provider_key("claude-3-7-sonnet") == "claude"
    assert provider_key(None, "claude_inst") == "claude"
    assert provider_key("custom", "claude_default") == "claude"

    # Codex
    assert provider_key("codex") == "codex"
    assert provider_key("codexAgent") == "codex"
    assert provider_key("codex_a9e...") == "codex"
    assert provider_key("codex_uuid") == "codex"
    assert provider_key(None, "codex_inst") == "codex"

    # Unknown / None / empty
    assert provider_key("mystery") is None
    assert provider_key(None) is None
    assert provider_key("") is None
    assert provider_key("   ", "   ") is None
    assert provider_key("other", "other_inst") is None


def test_family_label():
    assert family_label("claude") == "Claude"
    assert family_label("codex") == "Codex"
    assert family_label("antigravity") == "Antigravity"
    assert family_label("mystery") == "Other"
    assert family_label(None) == "Other"
    assert family_label("") == "Other"


def test_provider_family_task_acceptance():
    # Done when requirements
    assert provider_family("claude") == "Claude"
    assert provider_family("claudeAgent") == "Claude"
    assert provider_family(None, "claude_inst") == "Claude"
    assert provider_family("antigravity_gemini_secondary") == "Antigravity"
    assert provider_family("codex_a9e…") == "Codex"
    assert provider_family("mystery") == "Other"
    assert provider_family("") == "Other"
    assert provider_family(None) == "Other"


def test_context_limits():
    # Verify _LIMITS table structure
    assert _LIMITS == [
        ("claude-opus-5-5", 1_000_000),
        ("gemini-", 1_048_576),
        ("gpt-6.1-sol", 475_000),
        ("gpt-6-astra", 258_400),
    ]

    # Done when requirements
    assert context_limit_for("gemini-3.8-flash-high") == 1_048_576
    assert context_limit_for("claude-sonnet-5-5") is None

    # Prefix matches
    assert context_limit_for("claude-opus-5-5") == 1_000_000
    assert context_limit_for("claude-opus-5-5-20261101") == 1_000_000
    assert context_limit_for("gemini-") == 1_048_576
    assert context_limit_for("gemini-2.5-pro") == 1_048_576
    assert context_limit_for("gpt-6.1-sol") == 475_000
    assert context_limit_for("gpt-6.1-sol-preview") == 475_000
    assert context_limit_for("gpt-6-astra") == 258_400
    assert context_limit_for("gpt-6-astra-2026") == 258_400

    # None for unknown, blank or <synthetic>
    assert context_limit_for(None) is None
    assert context_limit_for("") is None
    assert context_limit_for("   ") is None
    assert context_limit_for("<synthetic>") is None
    assert context_limit_for("<synthetic>-model") is None
    assert context_limit_for("unknown-model") is None
