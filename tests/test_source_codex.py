import json

import pytest

from t3_telemetry.sources.codex import parse_codex_file


def line(kind, payload, timestamp="2026-11-03T22:08:01Z", **extra):
    return (json.dumps({"type": kind, "payload": payload, "timestamp": timestamp, **extra}) + "\n").encode()


def test_usage_and_incremental_model_context(tmp_path):
    path = tmp_path / "rollout-test.jsonl"
    data = (
        line("session_meta", {"id": "session-id", "cwd": "/work", "originator": "codex-tui"})
        + line("turn_context", {"model": "gpt-x"})
        + line("token_usage_record", {"response_id": "r1", "usage": {
            "input_tokens": 100, "cached_input_tokens": 30, "cache_write_input_tokens": 20,
            "output_tokens": 10, "reasoning_output_tokens": 4}})
        + line("event_msg", {"type": "token_count", "usage": {"input_tokens": 999}})
        + line("token_usage_record", {"usage": {"input_tokens": 5, "cached_input_tokens": 8}}, ordinal=4)
        + b"invalid json\n"
    )
    appended = line("token_usage_record", {"response_id": "r3", "usage": {"input_tokens": 7}})
    path.write_bytes(data + appended[:12])
    result = parse_codex_file(path)
    assert result.offset == len(data)
    session = result.sessions[0]
    assert (session.session_key, session.native_id, session.cwd, session.origin_hint, session.model) == (
        "codex:session-id", "session-id", "/work", "cli", "gpt-x")
    assert session.first_ts == session.last_ts == "2026-11-03T22:08:01.000000Z"
    assert [s.event_id for s in result.steps] == ["r1", "ordinal:4"]
    first, second = result.steps
    assert (first.model, first.input_tokens, first.cache_read_tokens, first.cache_creation_tokens,
            first.output_tokens, first.thinking_tokens, first.tool_names) == ("gpt-x", 50, 30, 20, 10, 4, [])
    assert second.input_tokens == 0
    with path.open("ab") as stream:
        stream.write(appended[12:])
    result = parse_codex_file(path, result.offset, session_id="session-id", model=session.model)
    assert result.offset == path.stat().st_size
    assert result.sessions[0].session_key == "codex:session-id"
    assert [(s.event_id, s.model, s.input_tokens) for s in result.steps] == [("r3", "gpt-x", 7)]


@pytest.mark.parametrize(("originator", "expected"), [("T3 Code", "t3"), ("other", "unknown")])
def test_origin_and_filename_fallback(tmp_path, originator, expected):
    native_id = "01234567-89ab-cdef-0123-456789abcdef"
    path = tmp_path / f"rollout-date-{native_id}.jsonl"
    path.write_bytes(line("session_meta", {"originator": originator}))
    session = parse_codex_file(path).sessions[0]
    assert session.session_key == f"codex:{native_id}"
    assert session.origin_hint == expected


def test_forked_subagent_keeps_first_session_meta(tmp_path):
    path = tmp_path / "rollout-child.jsonl"
    path.write_bytes(
        line("session_meta", {"id": "child-id", "cwd": "/child", "originator": "codex-tui", "forked_from_id": "parent-id"})
        + line("session_meta", {"id": "parent-id", "cwd": "/parent", "originator": "T3 Code"})
        + line("token_usage_record", {"response_id": "r1", "usage": {"input_tokens": 5, "output_tokens": 1}})
    )
    result = parse_codex_file(path)
    assert result.sessions[0].session_key == "codex:child-id"
    assert result.sessions[0].cwd == "/child"
    assert result.sessions[0].origin_hint == "cli"
    assert [s.session_key for s in result.steps] == ["codex:child-id"]


def test_turns_items_and_steps(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(
        line("session_meta", {"id": "s"})
        + line("event_msg", {"type": "task_started", "turn_id": "t", "started_at": 1.5,
                             "model_context_window": 475000})
        + line("event_msg", {"type": "item_completed", "turn_id": "t", "item": {
            "id": "u", "type": "UserMessage", "content": [{"text": "hello"}, {"text": "world"}]}})
        + line("event_msg", {"type": "item_completed", "turn_id": "t", "started_at_ms": 2000,
                             "completed_at_ms": 2500, "item": {"id": "i", "type": "CommandExecution",
                                                               "command": ["sh", "ls"]}})
        + line("token_usage_record", {"turn_id": "t", "response_id": "r1", "usage": {}})
        + line("token_usage_record", {"turn_id": "t", "response_id": "r2", "usage": {}})
        + line("event_msg", {"type": "item_completed", "turn_id": "t", "item": {
            "id": "a", "type": "AgentMessage", "text": "done"}})
        + line("event_msg", {"type": "task_complete", "turn_id": "t", "completed_at": 3})
    )
    result = parse_codex_file(path)
    assert len(result.turns) == 1
    turn = result.turns[0]
    assert (turn.turn_id, turn.started_at, turn.completed_at, turn.context_limit, turn.prompt, turn.reply) == (
        "t", "1970-01-01T00:00:01.500000Z", "1970-01-01T00:00:03.000000Z", 475000,
        "hello\nworld", "done")
    assert [(i.kind, i.text, i.started_at, i.completed_at) for i in result.items] == [
        ("run", "ls", "1970-01-01T00:00:02.000000Z", "1970-01-01T00:00:02.500000Z")]
    assert [step.turn_id for step in result.steps] == ["t", "t"]


def test_fallback_turn_and_timestamp(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(
        line("session_meta", {"id": "s"})
        + line("token_usage_record", {"turn_id": "t", "response_id": "r", "usage": {}})
        + line("event_msg", {"type": "task_started", "turn_id": "other", "started_at": "bad"})
        + line("event_msg", {"type": "task_complete", "turn_id": "other"})
    )
    turns = {t.turn_id: t for t in parse_codex_file(path).turns}
    for turn in turns.values():
        assert turn.started_at == turn.completed_at == "2026-11-03T22:08:01.000000Z"


def test_records_before_session_meta_are_ignored(tmp_path):
    path = tmp_path / "rollout.jsonl"
    event = {"type": "item_completed", "turn_id": "t", "item": {"id": "i", "type": "Reasoning"}}
    path.write_bytes(line("event_msg", event) + line("session_meta", {"id": "s"}) + line("event_msg", event))
    result = parse_codex_file(path)
    assert len(result.items) == 1
    assert result.items[0].session_key == "codex:s"


def test_file_change_paths_from_dictionary(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_bytes(
        line("session_meta", {"id": "s"})
        + line("event_msg", {"type": "item_completed", "turn_id": "t", "item": {
            "id": "i", "type": "FileChange", "changes": {"first.py": {}, "second.py": {}}}})
    )
    item, = parse_codex_file(path).items
    assert (item.kind, item.text) == ("edit", "first.py (+1 more)")
