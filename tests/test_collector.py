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
