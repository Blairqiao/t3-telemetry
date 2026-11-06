import json

import pytest

from t3_telemetry.sources.claude import parse_claude_file


def line(value):
    return (json.dumps(value) + "\n").encode()


def assistant(message_id, timestamp, usage, content, model="model-a", **extra):
    return {"type": "assistant", "timestamp": timestamp, "message": {
        "id": message_id, "model": model, "usage": usage, "content": content,
    }, **extra}


def test_dedup_incremental_and_partial(tmp_path):
    path = tmp_path / "session.jsonl"
    first = assistant("m1", "2026-11-03T22:08:01.113Z", {"input_tokens": 1}, [
        {"type": "tool_use", "name": "Read"},
    ], entrypoint="sdk-ts", cwd="/work")
    second = assistant("m1", "2026-11-03T22:08:02Z", {
        "input_tokens": 2, "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 4, "output_tokens": 5,
        "output_tokens_details": {"thinking_tokens": 2},
    }, [{"type": "tool_use", "name": "Read"}, {"type": "tool_use", "name": "Edit"}],
        model="model-b", entrypoint="cli")
    next_line = line(assistant("m2", "2026-11-03T22:08:03Z", {}, "text", entrypoint="t3-code"))
    complete = line({"type": "ai-title", "aiTitle": "A title"}) + line(first) + b"{bad json}\n" + line([]) + line(second)
    path.write_bytes(complete + next_line[:15])

    result = parse_claude_file(path)
    assert result.offset == len(complete)
    assert len(result.sessions) == len(result.steps) == 1
    session, step = result.sessions[0], result.steps[0]
    assert (session.session_key, session.native_id, session.title, session.cwd) == (
        "claude:session", "session", "A title", "/work")
    assert (session.first_ts, session.last_ts, session.origin_hint, session.model) == (
        "2026-11-03T22:08:01.113000Z", "2026-11-03T22:08:02.000000Z", "cli", "model-b")
    assert (step.event_id, step.input_tokens, step.cache_read_tokens,
            step.cache_creation_tokens, step.output_tokens, step.thinking_tokens) == ("m1", 2, 3, 4, 5, 2)
    assert step.tool_names == ["Read", "Edit"]

    with path.open("ab") as stream:
        stream.write(next_line[15:])
    result = parse_claude_file(path, result.offset)
    assert result.offset == path.stat().st_size
    assert [step.event_id for step in result.steps] == ["m2"]
    assert result.steps[0].tool_names == []
    assert result.sessions[0].origin_hint == "t3"


def test_subagent_metadata_and_parent(tmp_path):
    path = tmp_path / "parent" / "subagents" / "agent-child.jsonl"
    path.parent.mkdir(parents=True)
    path.with_suffix(".meta.json").write_text(json.dumps({"description": "Investigate"}))
    path.write_bytes(line({"type": "ai-title", "aiTitle": "Ignored"}))
    session = parse_claude_file(path).sessions[0]
    assert (session.session_key, session.native_id, session.parent_session_key, session.title) == (
        "claude:parent/agent-child", "parent/agent-child", "claude:parent", "Investigate")


@pytest.mark.parametrize(("entrypoint", "expected"), [
    ("t3-code", "t3"), ("cli", "cli"), ("sdk-cli", "sdk"), ("other", "unknown"),
])
def test_origin_mapping(tmp_path, entrypoint, expected):
    path = tmp_path / "session.jsonl"
    path.write_bytes(line({"entrypoint": entrypoint}))
    assert parse_claude_file(path).sessions[0].origin_hint == expected


def test_origin_precedence_and_invalid_lines(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_bytes(b"\xff\n" + b"".join(line({"entrypoint": origin}) for origin in
                                      ("t3-code", "cli", "sdk-ts", "other")))
    result = parse_claude_file(path)
    assert result.sessions[0].origin_hint == "t3"
    assert result.offset == path.stat().st_size
