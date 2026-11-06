from t3_telemetry.db_t3 import T3StateSnapshot
from t3_telemetry.models import AppThreadRecord, ProviderThreadRecord
from t3_telemetry.store import ItemRecord, SessionRecord, StepRecord, TelemetryStore, TurnRecord
from t3_telemetry.thread_view import build_thread_node


def setup_thread(tmp_path, provider="claude", status="idle"):
    store = TelemetryStore(tmp_path / "store.db")
    key = f"{provider}:native"
    store.upsert_sessions([SessionRecord(key, provider, "native", first_ts="2025-01-01T00:00:00Z", last_ts="2025-01-01T00:00:03Z")])
    store.replace_t3_graph([], [{"session_key": key, "thread_id": "thread", "provider_thread_id": "pt"}])
    app = AppThreadRecord("thread", None, "Title", provider, None, "model", None, "pt", None, None, None, None, None)
    pt = ProviderThreadRecord("pt", "thread", provider, "", "native", status)
    snapshot = T3StateSnapshot(threads={"thread": app}, provider_threads={"pt": pt}, provider_threads_by_thread={"thread": [pt]})
    return store, key, snapshot


def test_provider_turns_and_sync_state(tmp_path):
    store, key, snapshot = setup_thread(tmp_path, status="running")
    store.upsert_turns([TurnRecord(key, "a", started_at="2025-01-01T00:00:00Z", completed_at="2025-01-01T00:00:02Z", prompt="hello", reply="world"), TurnRecord(key, "b", started_at="2025-01-01T00:00:03Z")])
    store.upsert_steps([StepRecord(key, "1", "2025-01-01T00:00:00Z", "claude-opus-5-5", 10, 2, 3, 5, 1, turn_id="a"), StepRecord(key, "2", "2025-01-01T00:00:01Z", "claude-opus-5-5", 20, 4, 6, 10, 2, turn_id="a"), StepRecord(key, "3", "2025-01-01T00:00:03Z", "claude-opus-5-5", 30, 6, 9, 15, 3, turn_id="b")])
    store.upsert_items([ItemRecord(key, "i", "a", 1, "run", "command", started_at="2025-01-01T00:00:00Z", completed_at="2025-01-01T00:00:02Z")])
    assert build_thread_node("thread", snapshot, store).store_state == "syncing"
    store.set_meta("last_sync_at", "now")
    node = build_thread_node("thread", snapshot, store)
    assert node.store_state == "ready"
    assert [s.status for s in node.steps] == ["completed", "running"]
    assert node.steps[0].total_tokens == 60
    assert node.steps[0].response_tokens == 12
    assert [t.text for t in node.steps[0].context_texts] == ["hello", "world"]
    assert node.steps[0].timeline[0].duration_seconds == 2
    assert node.self_metrics.total_tokens == 120
    assert node.context_used_tokens == 60
    assert node.context_limit == 1_000_000
    assert node.context_limit_provenance == "measured"
    assert node.current_turn_metrics.turn_id == f"{key}|b"


def test_antigravity_tool_details_and_orphan(tmp_path):
    store, key, snapshot = setup_thread(tmp_path, "antigravity")
    store.upsert_turns([TurnRecord(key, "a", kind="model_step", started_at="2025-01-01T00:00:00Z")])
    store.upsert_steps([StepRecord(key, "1", "2025-01-01T00:00:00Z", output_tokens=2, turn_id="a"), StepRecord(key, "2", "2025-01-01T00:00:01Z", output_tokens=3)])
    store.upsert_items([ItemRecord(key, "i", "a", 1, "tool", tool_name="shell", fields={"command": "ls"}, truncated_fields=["command"])])
    node = build_thread_node("thread", snapshot, store)
    assert node.steps[0].tool_calls == ["shell"]
    assert node.steps[0].tool_details[0].fields == {"command": "ls"}
    assert node.steps[0].tool_details[0].truncated == ["command"]
    assert node.steps[1].activity_kind == "model_step"
    assert node.self_metrics.total_tokens == 5


def test_codex_recorded_limit_and_missing_session(tmp_path):
    store, key, snapshot = setup_thread(tmp_path, "codex")
    store.upsert_turns([TurnRecord(key, "a", started_at="2025-01-01T00:00:00Z", context_limit=12345)])
    store.upsert_steps([StepRecord(key, "1", "2025-01-01T00:00:00Z", input_tokens=12, turn_id="a")])
    store.set_meta("last_sync_at", "now")
    node = build_thread_node("thread", snapshot, store)
    assert node.context_limit == 12345
    assert node.context_limit_provenance == "measured"
    assert node.context_samples[0].limit == 12345
    snapshot.provider_threads_by_thread["thread"].append(ProviderThreadRecord("other", "thread", "codex", "", "missing", "idle"))
    assert build_thread_node("thread", snapshot, store).store_state == "syncing"
