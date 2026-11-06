"""Canonical provider keys and family label mappings."""

from __future__ import annotations

PROVIDER_KEYS: tuple[str, ...] = ("claude", "codex", "antigravity")

FAMILY_LABELS: dict[str, str] = {
    "claude": "Claude",
    "codex": "Codex",
    "antigravity": "Antigravity",
}


def provider_key(driver: str | None, provider_instance_id: str | None = None) -> str | None:
    """Resolve a driver and/or provider_instance_id to a canonical provider key.

    Preserves precedence: Antigravity is evaluated first, then Claude, then Codex.
    """
    drv = (driver or "").strip()
    inst = (provider_instance_id or "").strip()
    if drv == "antigravity" or inst.startswith("antigravity") or drv.startswith("antigravity"):
        return "antigravity"
    if drv == "claudeAgent" or drv.startswith("claude") or inst.startswith("claude"):
        return "claude"
    if drv.startswith("codex") or inst.startswith("codex"):
        return "codex"
    return None


def family_label(key: str | None) -> str:
    """Return the display family label for a canonical provider key, or 'Other'."""
    if key is None:
        return "Other"
    return FAMILY_LABELS.get(key, "Other")
