"""Discover native telemetry and incrementally ingest it into the store."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Callable

from t3_telemetry.db_agy import read_agy_conversation
from t3_telemetry.db_t3 import T3StateSnapshot
from t3_telemetry.sources.antigravity import parse_antigravity_conversation
from t3_telemetry.sources.claude import parse_claude_file
from t3_telemetry.sources.codex import parse_codex_file
from t3_telemetry.store import TelemetryStore, utc_iso
from t3_telemetry.t3_mirror import mirror_t3_graph


@dataclass(slots=True)
class SourceRoots:
    claude_projects: Path
    codex_dirs: list[Path]
    antigravity_dirs: list[Path]
    t3_dir: Path


def default_roots(t3_dir: Path) -> SourceRoots:
    claude = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    codex = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    t3_dir = Path(t3_dir)
    agy = list((t3_dir / "providers" / "antigravity").glob("*/antigravity-acp/conversations"))
    agy.append(Path.home() / ".gemini/antigravity-cli/conversations")
    return SourceRoots(claude / "projects", [codex / "sessions", codex / "archived_sessions"], agy, t3_dir)


@dataclass(slots=True)
class SyncStats:
    files_seen: int = 0
    files_read: int = 0
    steps_written: int = 0
    errors: int = 0


class Collector:
    def __init__(self, store: TelemetryStore, roots: SourceRoots, reader: Callable = read_agy_conversation):
        self.store = store
        self.roots = roots
        self.reader = reader

    def sync(self, snapshot: T3StateSnapshot | None = None) -> SyncStats:
        stats = SyncStats()
        models = {s.session_key: s.model for s in self.store.sessions()}
        paths = [(p, "claude") for p in self.roots.claude_projects.glob("*/*.jsonl")]
        paths.extend((p, "claude") for p in self.roots.claude_projects.glob("*/*/subagents/agent-*.jsonl"))
        for root in self.roots.codex_dirs:
            paths.extend((p, "codex") for p in root.rglob("rollout-*.jsonl"))
        for root in self.roots.antigravity_dirs:
            paths.extend((p, "antigravity") for p in root.glob("*.db"))
        seen = set()
        for path, provider in paths:
            if path in seen:
                continue
            seen.add(path)
            stats.files_seen += 1
            try:
                stat = path.stat()
                size, mtime = stat.st_size, stat.st_mtime_ns
                if provider == "antigravity":
                    wal = Path(f"{path}-wal")
                    try:
                        wal_stat = wal.stat()
                    except FileNotFoundError:
                        pass
                    else:
                        size += wal_stat.st_size
                        mtime = max(mtime, wal_stat.st_mtime_ns)
                previous = self.store.get_file(path)
                if previous and previous[:2] == (size, mtime):
                    continue
                offset = previous[2] if previous and size >= previous[2] else 0
                if provider == "claude":
                    key = self.store.get_file_session_key(path)
                    result = parse_claude_file(path, offset, turn_id=self.store.last_turn_id(key) if offset else None)
                elif provider == "codex":
                    key = self.store.get_file_session_key(path) if offset else None
                    session_id = key.removeprefix("codex:") if key and key.startswith("codex:") else None
                    result = parse_codex_file(path, offset, session_id, model=models.get(key))
                else:
                    result = parse_antigravity_conversation(path.stem, self.roots.t3_dir, reader=self.reader)
                self.store.upsert_sessions(result.sessions)
                self.store.upsert_steps(result.steps)
                self.store.upsert_turns(result.turns)
                self.store.upsert_items(result.items)
                key = result.sessions[0].session_key if result.sessions else None
                self.store.set_file(path, provider, size, mtime, result.offset, key)
                models.update((s.session_key, s.model) for s in result.sessions if s.model is not None)
                stats.files_read += 1
                stats.steps_written += len(result.steps)
            except (OSError, sqlite3.Error, ValueError, json.JSONDecodeError):
                stats.errors += 1
        if snapshot is not None:
            mirror_t3_graph(self.store, snapshot)
        self.store.set_meta("last_sync_at", utc_iso(datetime.now(timezone.utc)))
        return stats
