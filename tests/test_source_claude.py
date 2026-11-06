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


def test_turns_items_and_resuming(tmp_path):
    from t3_telemetry.store import TelemetryStore

    path = tmp_path / "session.jsonl"
    entries = [
        {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "promptId": "p1",
         "message": {"content": "First prompt"}},
        {"type": "user", "timestamp": "2026-01-01T00:00:01Z", "promptId": "meta",
         "isMeta": True, "message": {"content": "Internal"}},
        assistant("m1", "2026-01-01T00:00:02Z", {}, [
            {"type": "thinking", "thinking": "private"},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "t2", "name": "Edit", "input": {"file_path": "a.py"}},
        ]),
        {"type": "user", "timestamp": "2026-01-01T00:00:03Z", "promptId": "p1",
         "message": {"content": [{"type": "tool_result", "tool_use_id": "t1"},
                                 {"type": "tool_result", "tool_use_id": "t2", "is_error": True}]}},
        assistant("m2", "2026-01-01T00:00:04Z", {}, [{"type": "text", "text": "Done"}]),
        {"type": "user", "timestamp": "2026-01-01T00:00:05Z", "uuid": "u2",
         "message": {"content": [{"type": "text", "text": "Second"}, {"type": "text", "text": "prompt"}]}},
        assistant("m3", "2026-01-01T00:00:06Z", {}, [{"type": "text", "text": "Reply"}]),
    ]
    chunks = [line(entry) for entry in entries]
    path.write_bytes(b"".join(chunks))
    whole = parse_claude_file(path)
    split = sum(map(len, chunks[:3]))
    path.write_bytes(b"".join(chunks[:3]))
    first = parse_claude_file(path)
    path.write_bytes(b"".join(chunks))
    second = parse_claude_file(path, first.offset, turn_id="p1")
    assert first.offset == split
    assert [step.turn_id for step in whole.steps] == ["p1", "p1", "uuid:u2"]

    stores = [TelemetryStore(tmp_path / f"{name}.db") for name in ("whole", "split")]
    for store, results in zip(stores, ([whole], [first, second])):
        for result in results:
            store.upsert_sessions(result.sessions)
            store.upsert_steps(result.steps)
            store.upsert_turns(result.turns)
            store.upsert_items(result.items)
    turns = [dict(row) for row in stores[0].turns_for(["claude:session"])]
    assert [(t["turn_id"], t["prompt"], t["reply"]) for t in turns] == [
        ("p1", "First prompt", "Done"), ("uuid:u2", "Second\nprompt", "Reply")]
    items = [dict(row) for row in stores[0].items_for(["claude:session"])]
    assert [(i["kind"], i["text"], i["status"]) for i in items] == [
        ("think", "", "completed"), ("run", "ls", "completed"), ("edit", "a.py", "failed")]
    assert turns == [dict(row) for row in stores[1].turns_for(["claude:session"])]
    # Sequence values encode the starting offset, but preserve the same order.
    resumed_items = [dict(row) for row in stores[1].items_for(["claude:session"])]
    assert [{k: v for k, v in i.items() if k != "seq"} for i in items] == [
        {k: v for k, v in i.items() if k != "seq"} for i in resumed_items]
    for store in stores:
        store.close()
