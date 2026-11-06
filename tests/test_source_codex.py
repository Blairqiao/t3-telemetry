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
