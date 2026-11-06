"""Persistent store for native session telemetry."""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import Lock

SCHEMA_VERSION = 3
MAX_TEXT_CHARS = 2000
MAX_ITEM_CHARS = 300


@dataclass(slots=True)
class SessionRecord:
    session_key: str
    provider: str
    native_id: str
    parent_session_key: str | None = None
    title: str | None = None
    cwd: str | None = None
    model: str | None = None
    origin_hint: str = "unknown"
    first_ts: str | None = None
    last_ts: str | None = None


@dataclass(slots=True)
class StepRecord:
    session_key: str
    event_id: str
    ts: str
    model: str | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    tool_names: list[str] = field(default_factory=list)
    turn_id: str | None = None


@dataclass(slots=True)
class TurnRecord:
    session_key: str
    turn_id: str
    kind: str = "provider_turn"
    started_at: str | None = None
    completed_at: str | None = None
    prompt: str | None = None
    prompt_truncated: bool = False
    reply: str | None = None
    reply_truncated: bool = False
    context_limit: int | None = None


@dataclass(slots=True)
class ItemRecord:
    session_key: str
    item_id: str
    turn_id: str
    seq: int
    kind: str
    text: str = ""
    truncated: bool = False
    status: str = "completed"
    started_at: str | None = None
    completed_at: str | None = None
    tool_name: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    truncated_fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParseResult:
    sessions: list[SessionRecord]
    steps: list[StepRecord]
    offset: int
    turns: list[TurnRecord] = field(default_factory=list)
    items: list[ItemRecord] = field(default_factory=list)


def bounded(text: str | None, limit: int) -> tuple[str | None, bool]:
    """Return bounded text and whether it was truncated; blanks become None."""
    if text is None or not text.strip():
        return None, False
    return text[:limit], len(text) > limit


def utc_iso(value: str | datetime) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY, provider TEXT NOT NULL, size INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL, offset INTEGER NOT NULL, session_key TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
  session_key TEXT PRIMARY KEY, provider TEXT NOT NULL, native_id TEXT NOT NULL,
  parent_session_key TEXT, title TEXT, cwd TEXT, model TEXT,
  origin_hint TEXT NOT NULL DEFAULT 'unknown', first_ts TEXT, last_ts TEXT
);
CREATE TABLE IF NOT EXISTS steps (
  session_key TEXT NOT NULL, event_id TEXT NOT NULL, ts TEXT NOT NULL, model TEXT,
  input_tokens INTEGER NOT NULL DEFAULT 0, cache_read_tokens INTEGER NOT NULL DEFAULT 0,
  cache_creation_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
  thinking_tokens INTEGER NOT NULL DEFAULT 0, tool_names TEXT NOT NULL DEFAULT '[]', turn_id TEXT,
  PRIMARY KEY (session_key, event_id)
);
CREATE INDEX IF NOT EXISTS steps_ts ON steps(ts);
CREATE TABLE IF NOT EXISTS turns (
  session_key TEXT NOT NULL, turn_id TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'provider_turn',
  started_at TEXT, completed_at TEXT, prompt TEXT, prompt_truncated INTEGER NOT NULL DEFAULT 0,
  reply TEXT, reply_truncated INTEGER NOT NULL DEFAULT 0, context_limit INTEGER,
  PRIMARY KEY (session_key, turn_id)
);
CREATE TABLE IF NOT EXISTS turn_items (
  session_key TEXT NOT NULL, item_id TEXT NOT NULL, turn_id TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL, text TEXT NOT NULL DEFAULT '', truncated INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'completed', started_at TEXT, completed_at TEXT, tool_name TEXT,
  fields_json TEXT NOT NULL DEFAULT '{}', truncated_fields_json TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY (session_key, item_id)
);
CREATE INDEX IF NOT EXISTS turn_items_turn ON turn_items(session_key, turn_id, seq);
CREATE TABLE IF NOT EXISTS t3_threads (
  thread_id TEXT PRIMARY KEY, title TEXT, parent_thread_id TEXT, root_thread_id TEXT,
  driver TEXT, provider_instance_id TEXT, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS t3_bindings (
  session_key TEXT PRIMARY KEY, thread_id TEXT NOT NULL, provider_thread_id TEXT
);
"""


class TelemetryStore:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(_SCHEMA)
            row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            with self.conn:
                if row is not None and int(row[0]) < 2:
                    # v1 filed forked Codex subagent steps under the parent session; re-read Codex.
                    self.conn.execute("DELETE FROM steps WHERE session_key LIKE 'codex:%'")
                    self.conn.execute("DELETE FROM sessions WHERE provider='codex'")
                    self.conn.execute("DELETE FROM files WHERE provider='codex'")
                if row is not None and int(row[0]) < 3:
                    columns = {info[1] for info in self.conn.execute("PRAGMA table_info(steps)")}
                    if "turn_id" not in columns:
                        self.conn.execute("ALTER TABLE steps ADD COLUMN turn_id TEXT")
                    self.conn.execute("DELETE FROM files")
                self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def upsert_sessions(self, records: list[SessionRecord]) -> None:
        with self.lock, self.conn:
            self.conn.executemany("""
                INSERT INTO sessions VALUES (:session_key, :provider, :native_id,
                  :parent_session_key, :title, :cwd, :model, :origin_hint, :first_ts, :last_ts)
                ON CONFLICT(session_key) DO UPDATE SET
                  provider=COALESCE(excluded.provider, sessions.provider),
                  native_id=COALESCE(excluded.native_id, sessions.native_id),
                  parent_session_key=COALESCE(excluded.parent_session_key, sessions.parent_session_key),
                  title=COALESCE(excluded.title, sessions.title), cwd=COALESCE(excluded.cwd, sessions.cwd),
                  model=COALESCE(excluded.model, sessions.model),
                  origin_hint=CASE WHEN
                    (CASE excluded.origin_hint WHEN 't3' THEN 3 WHEN 'cli' THEN 2 WHEN 'sdk' THEN 1 ELSE 0 END) >
                    (CASE sessions.origin_hint WHEN 't3' THEN 3 WHEN 'cli' THEN 2 WHEN 'sdk' THEN 1 ELSE 0 END)
                    THEN excluded.origin_hint ELSE sessions.origin_hint END,
                  first_ts=CASE WHEN sessions.first_ts IS NULL THEN excluded.first_ts
                    WHEN excluded.first_ts IS NULL THEN sessions.first_ts ELSE MIN(sessions.first_ts, excluded.first_ts) END,
                  last_ts=CASE WHEN sessions.last_ts IS NULL THEN excluded.last_ts
                    WHEN excluded.last_ts IS NULL THEN sessions.last_ts ELSE MAX(sessions.last_ts, excluded.last_ts) END
            """, [asdict(record) for record in records])

    def upsert_steps(self, records: list[StepRecord]) -> None:
        with self.lock, self.conn:
            for record in records:
                row = self.conn.execute("SELECT tool_names FROM steps WHERE session_key=? AND event_id=?",
                                        (record.session_key, record.event_id)).fetchone()
                names = list(dict.fromkeys((json.loads(row[0]) if row else []) + record.tool_names))
                values = asdict(record)
                values["tool_names"] = json.dumps(names)
                self.conn.execute("""
                    INSERT INTO steps (session_key, event_id, ts, model, input_tokens, cache_read_tokens,
                      cache_creation_tokens, output_tokens, thinking_tokens, tool_names, turn_id)
                    VALUES (:session_key, :event_id, :ts, :model, :input_tokens,
                      :cache_read_tokens, :cache_creation_tokens, :output_tokens, :thinking_tokens, :tool_names, :turn_id)
                    ON CONFLICT(session_key, event_id) DO UPDATE SET
                      ts=excluded.ts, model=excluded.model, input_tokens=excluded.input_tokens,
                      cache_read_tokens=excluded.cache_read_tokens, cache_creation_tokens=excluded.cache_creation_tokens,
                      output_tokens=excluded.output_tokens, thinking_tokens=excluded.thinking_tokens,
                      tool_names=excluded.tool_names, turn_id=COALESCE(excluded.turn_id, steps.turn_id)
                """, values)

    def upsert_turns(self, records: list[TurnRecord]) -> None:
        with self.lock, self.conn:
            self.conn.executemany("""
                INSERT INTO turns VALUES (:session_key, :turn_id, :kind, :started_at, :completed_at,
                  :prompt, :prompt_truncated, :reply, :reply_truncated, :context_limit)
                ON CONFLICT(session_key, turn_id) DO UPDATE SET
                  kind=COALESCE(excluded.kind, turns.kind),
                  started_at=CASE WHEN turns.started_at IS NULL THEN excluded.started_at
                    WHEN excluded.started_at IS NULL THEN turns.started_at ELSE MIN(turns.started_at, excluded.started_at) END,
                  completed_at=CASE WHEN turns.completed_at IS NULL THEN excluded.completed_at
                    WHEN excluded.completed_at IS NULL THEN turns.completed_at ELSE MAX(turns.completed_at, excluded.completed_at) END,
                  prompt_truncated=CASE WHEN turns.prompt IS NULL THEN excluded.prompt_truncated ELSE turns.prompt_truncated END,
                  prompt=COALESCE(turns.prompt, excluded.prompt),
                  reply_truncated=CASE WHEN excluded.reply IS NOT NULL THEN excluded.reply_truncated ELSE turns.reply_truncated END,
                  reply=COALESCE(excluded.reply, turns.reply),
                  context_limit=COALESCE(excluded.context_limit, turns.context_limit)
            """, [asdict(record) for record in records])

    def upsert_items(self, records: list[ItemRecord]) -> None:
        values = []
        for record in records:
            item = asdict(record)
            item["fields_json"] = json.dumps(item.pop("fields"))
            item["truncated_fields_json"] = json.dumps(item.pop("truncated_fields"))
            values.append(item)
        with self.lock, self.conn:
            self.conn.executemany("""
                INSERT INTO turn_items VALUES (:session_key, :item_id, :turn_id, :seq, :kind, :text,
                  :truncated, :status, :started_at, :completed_at, :tool_name, :fields_json, :truncated_fields_json)
                ON CONFLICT(session_key, item_id) DO UPDATE SET
                  turn_id=COALESCE(turn_items.turn_id, excluded.turn_id), seq=COALESCE(turn_items.seq, excluded.seq),
                  kind=COALESCE(turn_items.kind, excluded.kind), started_at=COALESCE(turn_items.started_at, excluded.started_at),
                  tool_name=COALESCE(turn_items.tool_name, excluded.tool_name),
                  truncated=CASE WHEN excluded.text != '' THEN excluded.truncated ELSE turn_items.truncated END,
                  text=CASE WHEN excluded.text != '' THEN excluded.text ELSE turn_items.text END,
                  completed_at=COALESCE(excluded.completed_at, turn_items.completed_at),
                  status=CASE WHEN excluded.status='completed' AND turn_items.status='failed'
                    THEN turn_items.status ELSE excluded.status END,
                  fields_json=CASE WHEN excluded.fields_json != '{}' THEN excluded.fields_json ELSE turn_items.fields_json END,
                  truncated_fields_json=CASE WHEN excluded.truncated_fields_json != '[]'
                    THEN excluded.truncated_fields_json ELSE turn_items.truncated_fields_json END
            """, values)

    def get_meta(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def last_turn_id(self, session_key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT turn_id FROM turns WHERE session_key=? ORDER BY started_at DESC, turn_id DESC LIMIT 1", (session_key,)).fetchone()
            return row[0] if row else None

    def _rows_for(self, session_keys: list[str], select: str, order: tuple) -> list[sqlite3.Row]:
        if not session_keys:
            return []
        session_keys = list(dict.fromkeys(session_keys))
        rows = []
        with self.lock:
            for start in range(0, len(session_keys), 500):
                chunk = session_keys[start:start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows.extend(self.conn.execute(f"{select} WHERE {order[0]} IN ({placeholders})", chunk).fetchall())
        rows.sort(key=order[1])
        return rows

    def steps_for(self, session_keys: list[str]) -> list[sqlite3.Row]:
        return self._rows_for(session_keys, "SELECT steps.*, sessions.provider FROM steps JOIN sessions ON sessions.session_key=steps.session_key",
                              ("steps.session_key", lambda row: row["ts"]))

    def turns_for(self, session_keys: list[str]) -> list[sqlite3.Row]:
        return self._rows_for(session_keys, "SELECT * FROM turns",
                              ("session_key", lambda row: (row["started_at"] or "", row["turn_id"])))

    def items_for(self, session_keys: list[str]) -> list[sqlite3.Row]:
        return self._rows_for(session_keys, "SELECT * FROM turn_items",
                              ("session_key", lambda row: (row["session_key"], row["turn_id"], row["seq"])))

    def sessions_by_key(self) -> dict[str, SessionRecord]:
        return {session.session_key: session for session in self.sessions()}

    def get_file(self, path: Path) -> tuple[int, int, int] | None:
        with self.lock:
            row = self.conn.execute("SELECT size, mtime_ns, offset FROM files WHERE path=?", (str(path),)).fetchone()
            return tuple(row) if row else None

    def get_file_session_key(self, path: Path) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT session_key FROM files WHERE path=?", (str(path),)).fetchone()
            return row[0] if row else None

    def set_file(self, path: Path, provider: str, size: int, mtime_ns: int, offset: int,
                 session_key: str | None) -> None:
        with self.lock, self.conn:
            self.conn.execute("""INSERT INTO files VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET provider=excluded.provider, size=excluded.size,
                mtime_ns=excluded.mtime_ns, offset=excluded.offset, session_key=excluded.session_key""",
                (str(path), provider, size, mtime_ns, offset, session_key))

    def replace_t3_graph(self, threads: list[dict], bindings: list[dict]) -> None:
        with self.lock, self.conn:
            for thread in threads:
                columns = ("thread_id", "title", "parent_thread_id", "root_thread_id", "driver",
                           "provider_instance_id", "created_at", "updated_at")
                self.conn.execute("""INSERT INTO t3_threads VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET title=excluded.title,
                    parent_thread_id=excluded.parent_thread_id, root_thread_id=excluded.root_thread_id,
                    driver=excluded.driver, provider_instance_id=excluded.provider_instance_id,
                    created_at=excluded.created_at, updated_at=excluded.updated_at""",
                    tuple(thread.get(c) for c in columns))
            for binding in bindings:
                self.conn.execute("""INSERT INTO t3_bindings VALUES (?, ?, ?)
                    ON CONFLICT(session_key) DO UPDATE SET thread_id=excluded.thread_id,
                    provider_thread_id=excluded.provider_thread_id""",
                    (binding["session_key"], binding["thread_id"], binding.get("provider_thread_id")))

    def steps_between(self, start_utc: str, end_utc: str) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("""SELECT steps.*, sessions.provider FROM steps
                JOIN sessions ON sessions.session_key=steps.session_key
                WHERE steps.ts >= ? AND steps.ts < ? ORDER BY steps.ts""", (start_utc, end_utc)).fetchall()

    def all_steps(self) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute("""SELECT steps.*, sessions.provider FROM steps
                JOIN sessions ON sessions.session_key=steps.session_key ORDER BY steps.ts""").fetchall()

    def sessions(self) -> list[SessionRecord]:
        with self.lock:
            return [SessionRecord(**dict(row)) for row in self.conn.execute("SELECT * FROM sessions")]

    def bindings(self) -> dict[str, str]:
        with self.lock:
            return {row[0]: row[1] for row in self.conn.execute("SELECT session_key, thread_id FROM t3_bindings")}

    def t3_threads(self) -> dict[str, sqlite3.Row]:
        with self.lock:
            return {row["thread_id"]: row for row in self.conn.execute("SELECT * FROM t3_threads")}


def attribute_session(session_key: str, sessions: dict[str, SessionRecord], bindings: dict[str, str], threads: dict) -> tuple[str, bool]:
    """Attribute a native session to its app root, or its native session root."""
    path: list[str] = []
    current = session_key
    while current in sessions and current not in path:
        path.append(current)
        if current in bindings:
            thread_id = bindings[current]
            thread_path: list[str] = []
            while thread_id in threads and thread_id not in thread_path:
                thread_path.append(thread_id)
                thread = threads[thread_id]
                parent = thread["root_thread_id"] or thread["parent_thread_id"]
                if not parent or parent == thread_id:
                    break
                thread_id = parent
            else:
                if thread_id in thread_path:
                    thread_id = min(thread_path[thread_path.index(thread_id):])
            return thread_id, current != session_key or thread_id != bindings[current]
        parent = sessions[current].parent_session_key
        if not parent or parent not in sessions:
            break
        current = parent
    else:
        if current in path:
            current = min(path[path.index(current):])
    return f"session:{current}", current != session_key


def session_rows(store: TelemetryStore, limit: int = 50) -> list[dict]:
    """Group native sessions by their root app thread or native session."""
    from .providers import family_label, provider_key
    from .usage import PROVIDER_ORDER

    sessions = store.sessions_by_key()
    bindings, threads = store.bindings(), store.t3_threads()
    groups: dict[str, list[str]] = defaultdict(list)
    for key in sessions:
        row_id, _ = attribute_session(key, sessions, bindings, threads)
        groups[row_id].append(key)

    token_names = ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens", "thinking_tokens")
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {name: 0 for name in token_names})
    step_models: dict[str, str] = {}
    for step in store.all_steps():
        key = step["session_key"]
        for name in token_names:
            totals[key][name] += step[name]
        if step["model"]:
            step_models.setdefault(key, step["model"])

    rows = []
    for row_id, keys in groups.items():
        is_thread = not row_id.startswith("session:")
        if is_thread:
            root_keys = [k for k in keys if bindings.get(k) == row_id]
            root_key = (root_keys or [k for k in keys if k in bindings] or keys)[0]
            thread = threads.get(row_id)
            title = (thread["title"] if thread else None) or row_id
            origin, thread_id = "T3", row_id
            child_threads = {bindings[k] for k in keys if k in bindings and bindings[k] != row_id}
            native_children = {k for k in keys if k not in bindings and sessions[k].parent_session_key}
            subagents = len(child_threads) + len(native_children)
        else:
            root_key = row_id[len("session:"):]
            root = sessions[root_key]
            title = root.title or (f"{Path(root.cwd).name} · {root.provider}" if root.cwd else root_key)
            origin = {"t3": "T3", "cli": "CLI", "sdk": "SDK"}.get(root.origin_hint, "unknown")
            thread_id = None
            subagents = len(keys) - 1
        root = sessions[root_key]
        labels = {family_label(provider_key(sessions[k].provider)) for k in keys}
        counts = {name: sum(totals[k][name] for k in keys) for name in token_names}
        last_active = max((sessions[k].last_ts for k in keys if sessions[k].last_ts), default=None)
        rows.append({
            "session_key": root_key, "thread_id": thread_id, "origin": origin,
            "provider": ", ".join(label for label in PROVIDER_ORDER if label in labels),
            "title": title, "model": root.model or step_models.get(root_key),
            "last_active": last_active, "last_ts": last_active, "subagents": subagents,
            "tokens": sum(counts[name] for name in token_names[:4]), **counts,
        })
    rows.sort(key=lambda row: (row["last_active"] is not None, row["last_active"] or "", row["session_key"]), reverse=True)
    return rows[:limit] if limit is not None and limit > 0 else rows
