from datetime import datetime, timedelta, timezone
import json

from t3_telemetry.store import SessionRecord, StepRecord, TelemetryStore, utc_iso


def test_utc_iso():
    assert utc_iso("2026-11-03T22:08:01.113Z") == "2026-11-03T22:08:01.113000Z"
    assert utc_iso("2026-11-04T00:08:01.113+02:00") == "2026-11-03T22:08:01.113000Z"
    assert utc_iso(datetime(2026, 11, 3, 22, 8, 1)) == "2026-11-03T22:08:01.000000Z"
    assert utc_iso(datetime(2026, 11, 4, 0, 8, 1, tzinfo=timezone(timedelta(hours=2)))) == "2026-11-03T22:08:01.000000Z"


def test_sessions_merge_and_origin(tmp_path):
    store = TelemetryStore(tmp_path / "nested" / "store.db")
    key = "claude:1"
    store.upsert_sessions([SessionRecord(key, "claude", "1", title="old", origin_hint="sdk", first_ts="2026-02", last_ts="2026-03")])
    store.upsert_sessions([SessionRecord(key, "claude", "1", cwd="/work", origin_hint="cli", first_ts="2026-01", last_ts="2026-04")])
    store.upsert_sessions([SessionRecord(key, "claude", "1", title="new", origin_hint="sdk", first_ts="2026-02", last_ts="2026-03")])
    session = store.sessions()[0]
    assert (session.title, session.cwd, session.first_ts, session.last_ts, session.origin_hint) == ("new", "/work", "2026-01", "2026-04", "cli")
    store.upsert_sessions([SessionRecord(key, "claude", "1", origin_hint="t3")])
    store.upsert_sessions([SessionRecord(key, "claude", "1", origin_hint="unknown")])
    assert store.sessions()[0].origin_hint == "t3"
    store.close()


def test_steps_merge_range_and_file(tmp_path):
    store = TelemetryStore(tmp_path / "store.db")
    store.upsert_sessions([SessionRecord("codex:1", "codex", "1")])
    store.upsert_steps([StepRecord("codex:1", "a", "2026-01-01", model="old", input_tokens=1, tool_names=["read", "write"])])
    store.upsert_steps([StepRecord("codex:1", "a", "2026-01-01", model="new", input_tokens=2, tool_names=["write", "search"]),
                        StepRecord("codex:1", "b", "2026-01-02"), StepRecord("codex:1", "c", "2026-01-03")])
    rows = store.steps_between("2026-01-01", "2026-01-03")
    assert [r["event_id"] for r in rows] == ["a", "b"]
    assert (rows[0]["model"], rows[0]["input_tokens"], rows[0]["provider"]) == ("new", 2, "codex")
    assert json.loads(rows[0]["tool_names"]) == ["read", "write", "search"]
    path = tmp_path / "source"
    assert store.get_file(path) is None
    store.set_file(path, "codex", 10, 20, 5, "codex:1")
    assert store.get_file(path) == (10, 20, 5)
    store.close()


def test_graph_is_upsert_only(tmp_path):
    store = TelemetryStore(tmp_path / "store.db")
    store.replace_t3_graph([{"thread_id": "t", "title": "first"}], [{"session_key": "claude:1", "thread_id": "t"}])
    store.replace_t3_graph([{"thread_id": "t", "title": "updated"}, {"thread_id": "other"}], [])
    store.replace_t3_graph([], [])
    assert store.bindings() == {"claude:1": "t"}
    assert store.t3_threads()["t"]["title"] == "updated"
    assert "other" in store.t3_threads()
    store.close()


def test_session_rows_seeded_origin_descendants_and_cycles(tmp_path):
    from t3_telemetry.store import session_rows

    store = TelemetryStore(tmp_path / "store.db")
    store.upsert_sessions([
        SessionRecord("claude:root", "claude", "root", origin_hint="sdk", model="claude-3-5", first_ts="2026-11-01T00:00:00Z", last_ts="2026-11-04T12:00:00Z"),
        SessionRecord("claude:child1", "claude", "child1", parent_session_key="claude:root", origin_hint="sdk"),
        SessionRecord("claude:child2", "claude", "child2", parent_session_key="claude:child1", origin_hint="sdk"),
        SessionRecord("codex:standalone", "codex", "standalone", origin_hint="cli", cwd="/projects/my-app", first_ts="2026-11-02T00:00:00Z", last_ts="2026-11-05T12:00:00Z"),
        SessionRecord("antigravity:unbound", "antigravity", "ag1", title="AG Session", origin_hint="unknown", first_ts="2026-11-03T00:00:00Z", last_ts="2026-11-03T12:00:00Z"),
    ])
    store.replace_t3_graph([
        {"thread_id": "t1", "title": "T3 Bound Thread"},
    ], [
        {"session_key": "claude:root", "thread_id": "t1"},
    ])
    store.upsert_steps([
        StepRecord("claude:root", "step1", "2026-11-04T01:00:00Z", input_tokens=100, output_tokens=50),
        StepRecord("claude:child1", "step2", "2026-11-04T02:00:00Z", cache_read_tokens=20, output_tokens=30),
        StepRecord("claude:child2", "step3", "2026-11-04T03:00:00Z", input_tokens=10, cache_creation_tokens=5, thinking_tokens=8),
        StepRecord("codex:standalone", "step4", "2026-11-05T01:00:00Z", input_tokens=200, output_tokens=100),
    ])

    rows = session_rows(store)
    assert len(rows) == 3
    assert [r["session_key"] for r in rows] == ["codex:standalone", "claude:root", "antigravity:unbound"]

    # Check codex:standalone
    codex_row = rows[0]
    assert codex_row["origin"] == "cli"
    assert codex_row["thread_id"] is None
    assert codex_row["title"] == "my-app · codex"
    assert codex_row["subagents"] == 0
    assert codex_row["tokens"] == 300
    assert codex_row["input_tokens"] == 200
    assert codex_row["output_tokens"] == 100

    # Check claude:root (bound to T3: origin should be "T3" even though origin_hint is "sdk")
    claude_row = rows[1]
    assert claude_row["origin"] == "T3"
    assert claude_row["thread_id"] == "t1"
    assert claude_row["title"] == "T3 Bound Thread"
    assert claude_row["subagents"] == 2
    assert claude_row["tokens"] == 215
    assert claude_row["input_tokens"] == 110
    assert claude_row["cache_read_tokens"] == 20
    assert claude_row["cache_creation_tokens"] == 5
    assert claude_row["output_tokens"] == 80
    assert claude_row["thinking_tokens"] == 8

    # Check antigravity:unbound
    ag_row = rows[2]
    assert ag_row["origin"] == "unknown"
    assert ag_row["title"] == "AG Session"
    assert ag_row["subagents"] == 0
    assert ag_row["tokens"] == 0

    store.close()


def test_session_rows_limit_and_cycle_guard(tmp_path):
    from t3_telemetry.store import session_rows

    store = TelemetryStore(tmp_path / "limit.db")
    records = [
        SessionRecord(f"s:{i:03d}", "claude", f"{i}", last_ts=f"2026-11-01T{i:02d}:00:00Z" if i < 24 else f"2026-11-02T{i-24:02d}:00:00Z")
        for i in range(60)
    ]
    records.append(SessionRecord("c:1", "claude", "c1", parent_session_key="s:000"))
    records.append(SessionRecord("c:2", "claude", "c2", parent_session_key="c:1"))
    store.upsert_sessions(records)
    # Update to create cycle between c:1 and c:2
    store.conn.execute("UPDATE sessions SET parent_session_key='c:2' WHERE session_key='c:1'")

    rows = session_rows(store, limit=50)
    assert len(rows) == 50
    store.close()


def test_v1_store_drops_codex_rows_for_reread(tmp_path):
    path = tmp_path / "store.sqlite"
    store = TelemetryStore(path)
    store.upsert_sessions([SessionRecord("codex:a", "codex", "a"), SessionRecord("claude:b", "claude", "b")])
    store.upsert_steps([StepRecord("codex:a", "r1", "2026-11-03T00:00:00.000000Z"),
                        StepRecord("claude:b", "m1", "2026-11-03T00:00:00.000000Z")])
    store.set_file(tmp_path / "x.jsonl", "codex", 1, 1, 1, "codex:a")
    with store.conn:
        store.conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    store.close()
    store = TelemetryStore(path)
    assert [s.session_key for s in store.sessions()] == ["claude:b"]
    assert store.get_file(tmp_path / "x.jsonl") is None
    assert len(store.steps_between("2026", "2027")) == 1
    store.close()
