import json
from pathlib import Path

from t3_telemetry.collector import Collector, SourceRoots, default_roots
from t3_telemetry.models import ConversationTelemetry, ModelStep
from t3_telemetry.store import TelemetryStore


def line(obj):
    return json.dumps(obj) + "\n"


def assistant(event):
    return {"type": "assistant", "timestamp": "2026-11-03T12:00:00Z", "message": {"id": event, "usage": {"input_tokens": 3}}}


def test_incremental_discovery_and_codex_context(tmp_path):
    projects = tmp_path / "projects"
    main = projects / "project" / "parent.jsonl"
    sub = projects / "project" / "parent" / "subagents" / "agent-child.jsonl"
    codex = tmp_path / "codex" / "rollout-example.jsonl"
    for path in (main, sub, codex):
        path.parent.mkdir(parents=True, exist_ok=True)
    main.write_text(line(assistant("a")))
    sub.write_text(line(assistant("b")))
    codex.write_text(line({"type": "session_meta", "payload": {"id": "native"}}) + line({"type": "turn_context", "payload": {"model": "gpt-test"}}))
    store = TelemetryStore(tmp_path / "store.db")
    collector = Collector(store, SourceRoots(projects, [codex.parent], [], tmp_path))
    first = collector.sync()
    assert (first.files_seen, first.files_read, first.steps_written, first.errors) == (3, 3, 2, 0)
    assert collector.sync().files_read == 0
    with main.open("a") as f:
        f.write(line(assistant("c")))
    assert (s := collector.sync()).files_read == 1 and s.steps_written == 1
    with codex.open("a") as f:
        f.write(line({"type": "token_usage_record", "timestamp": "2026-11-03T12:00:00Z", "payload": {"response_id": "r", "usage": {"input_tokens": 5}}}))
    assert collector.sync().steps_written == 1
    rows = store.steps_between("2026", "2027")
    assert next(row for row in rows if row["event_id"] == "r")["model"] == "gpt-test"
    main.write_text(line(assistant("a")))
    assert collector.sync().files_read == 1
    assert len(store.steps_between("2026", "2027")) == 4
    assert store.get_file_session_key(codex) == "codex:native"
    store.close()


def test_antigravity_wal_and_retry(tmp_path):
    agy = tmp_path / "agy"
    agy.mkdir()
    bad = agy / "bad.db"
    good = agy / "good.db"
    bad.write_bytes(b"db")
    good.write_bytes(b"db")
    calls = []
    def reader(native_id, t3_dir):
        calls.append(native_id)
        if native_id == "bad" and calls.count("bad") == 1:
            raise ValueError("corrupt")
        return ConversationTelemetry(native_id=native_id, steps=[ModelStep(step_index=1, started_at="2026-11-03T00:00:00Z")])
    store = TelemetryStore(tmp_path / "store.db")
    collector = Collector(store, SourceRoots(tmp_path / "none", [], [agy], tmp_path), reader=reader)
    stats = collector.sync()
    assert (stats.files_seen, stats.files_read, stats.steps_written, stats.errors) == (2, 1, 1, 1)
    assert store.get_file(bad) is None
    assert collector.sync().files_read == 1
    assert collector.sync().files_read == 0
    Path(f"{good}-wal").write_bytes(b"wal")
    assert collector.sync().files_read == 1
    assert len(store.steps_between("2026", "2027")) == 2
    store.close()


def test_default_roots_respect_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    agy = tmp_path / "t3/providers/antigravity/instance/antigravity-acp/conversations"
    agy.mkdir(parents=True)
    roots = default_roots(tmp_path / "t3")
    assert roots.claude_projects == tmp_path / "claude/projects"
    assert roots.codex_dirs == [tmp_path / "codex/sessions", tmp_path / "codex/archived_sessions"]
    assert agy in roots.antigravity_dirs


def test_claude_turns_resuming_and_sync_time(tmp_path):
    projects = tmp_path / "projects"
    session_file = projects / "proj" / "session.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(
        line({"type": "user", "timestamp": "2026-11-03T12:00:00Z", "promptId": "turn-1", "message": {"content": "First"}})
        + line({"type": "assistant", "timestamp": "2026-11-03T12:00:01Z", "message": {
            "id": "m1", "usage": {"input_tokens": 1},
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
        }})
    )
    store = TelemetryStore(tmp_path / "store.db")
    collector = Collector(store, SourceRoots(projects, [], [], tmp_path))

    assert store.get_meta("last_sync_at") is None
    stats1 = collector.sync()
    first_sync_time = store.get_meta("last_sync_at")
    assert first_sync_time is not None
    assert stats1.steps_written == 1
    assert len(store.turns_for(["claude:session"])) == 1
    items = store.items_for(["claude:session"])
    assert len(items) == 1
    assert (items[0]["item_id"], items[0]["kind"], items[0]["text"]) == ("t1", "run", "ls")

    # Append second half to Claude file without starting a new turn
    with session_file.open("a") as f:
        f.write(
            line({"type": "user", "timestamp": "2026-11-03T12:00:02Z", "promptId": "turn-1",
                  "message": {"content": [{"type": "tool_result", "tool_use_id": "t1"}]}})
            + line({"type": "assistant", "timestamp": "2026-11-03T12:00:03Z", "message": {
                "id": "m2", "usage": {"input_tokens": 2},
                "content": [{"type": "text", "text": "Done"}],
            }})
        )

    stats2 = collector.sync()
    assert stats2.steps_written == 1
    steps = store.steps_for(["claude:session"])
    assert len(steps) == 2
    assert [s["event_id"] for s in steps] == ["m1", "m2"]
    # All steps of the second half end on the turn started in the first half
    assert [s["turn_id"] for s in steps] == ["turn-1", "turn-1"]

    items_after = store.items_for(["claude:session"])
    assert len(items_after) == 1
    assert items_after[0]["status"] == "completed"

    second_sync_time = store.get_meta("last_sync_at")
    assert second_sync_time is not None
    assert second_sync_time >= first_sync_time

    # Verify last_sync_at is updated even when a file errors
    bad_dir = projects / "proj" / "dir.jsonl"
    bad_dir.mkdir()
    stats3 = collector.sync()
    assert stats3.errors >= 1
    assert store.get_meta("last_sync_at") is not None
    store.close()

