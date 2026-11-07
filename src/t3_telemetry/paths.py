"""Default locations for the store, T3, and each harness's session files.

Every path resolves in the same order: CLI flag, then environment variable,
then the built-in default. Nothing here touches the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def store_path(flag: Path | None = None) -> Path:
    """The telemetry store: ``--store``, ``T3_TELEMETRY_STORE``, ``~/.t3-telemetry/telemetry.sqlite``."""
    if flag is not None:
        return Path(flag).expanduser()
    return _env_path("T3_TELEMETRY_STORE") or Path.home() / ".t3-telemetry" / "telemetry.sqlite"


def t3_dir(flag: Path | None = None) -> Path:
    """T3 userdata: ``--t3-dir``, ``T3_TELEMETRY_T3_DIR``, ``$T3CODE_HOME/userdata``, ``~/.t3/userdata``."""
    if flag is not None:
        return Path(flag).expanduser()
    explicit = _env_path("T3_TELEMETRY_T3_DIR")
    if explicit is not None:
        return explicit
    home = _env_path("T3CODE_HOME") or Path.home() / ".t3"
    return home / "userdata"


def claude_home() -> Path:
    """Claude Code config: ``CLAUDE_CONFIG_DIR`` or ``~/.claude``."""
    return _env_path("CLAUDE_CONFIG_DIR") or Path.home() / ".claude"


def codex_home() -> Path:
    """Codex home: ``CODEX_HOME`` or ``~/.codex``."""
    return _env_path("CODEX_HOME") or Path.home() / ".codex"


def antigravity_cli_dir() -> Path:
    """Standalone Antigravity CLI data: ``ANTIGRAVITY_CLI_DIR`` or ``~/.gemini/antigravity-cli``."""
    return _env_path("ANTIGRAVITY_CLI_DIR") or Path.home() / ".gemini" / "antigravity-cli"


def t3_state_db(t3_userdata: Path) -> Path:
    """The ``statev2.sqlite`` file inside T3 userdata (or the path itself if it names a file)."""
    path = Path(t3_userdata)
    return path if path.suffix in (".sqlite", ".db") else path / "statev2.sqlite"
