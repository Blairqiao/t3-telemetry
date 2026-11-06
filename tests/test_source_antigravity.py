from inspect import signature
from pathlib import Path

from t3_telemetry.db_agy import read_agy_conversation
from t3_telemetry.models import ConversationTelemetry, ModelStep, RecordedText, ToolCallDetail
from t3_telemetry.sources.antigravity import parse_antigravity_conversation


def test_fake_reader_two_steps_one_without_started_at_and_parent_id(tmp_path: Path):
    step1 = ModelStep(
        step_index=1,
        started_at="2026-11-03T22:08:01.113Z",
        input_tokens=100,
        cache_read_tokens=50,
        cache_creation_tokens=25,
        output_tokens=30,
        thinking_tokens=15,
        tool_calls=["view_file", "run_command"],
    )
    step2 = ModelStep(
        step_index=2,
        started_at=None,
        input_tokens=200,
        output_tokens=40,
    )

    conv = ConversationTelemetry(
        native_id="conv-child-1",
        steps=[step1, step2],
        parent_native_ids=["conv-parent-0"],
        model_name="gemini-1.5-pro",
    )

    def fake_reader(native_id: str, t3_dir: Path) -> ConversationTelemetry:
        assert native_id == "conv-child-1"
        assert t3_dir == tmp_path
        return conv

    result = parse_antigravity_conversation("conv-child-1", tmp_path, reader=fake_reader)

    assert result.offset == 0
    assert len(result.sessions) == 1
    assert len(result.steps) == 1
    assert len(result.turns) == 1
    assert len(result.items) == 0

    session = result.sessions[0]
    assert session.session_key == "antigravity:conv-child-1"
    assert session.provider == "antigravity"
    assert session.native_id == "conv-child-1"
    assert session.parent_session_key == "antigravity:conv-parent-0"
    assert session.model == "gemini-1.5-pro"
    assert session.origin_hint == "unknown"
    assert session.first_ts == "2026-11-03T22:08:01.113000Z"
    assert session.last_ts == "2026-11-03T22:08:01.113000Z"

    step = result.steps[0]
    assert step.session_key == "antigravity:conv-child-1"
    assert step.event_id == "step:1"
    assert step.turn_id == "step:1"
    assert step.ts == "2026-11-03T22:08:01.113000Z"
    assert step.model == "gemini-1.5-pro"
    assert step.input_tokens == 100
    assert step.cache_read_tokens == 50
    assert step.cache_creation_tokens == 25
    assert step.output_tokens == 30
    assert step.thinking_tokens == 15
    assert step.tool_names == ["view_file", "run_command"]

    turn = result.turns[0]
    assert turn.session_key == "antigravity:conv-child-1"
    assert turn.turn_id == "step:1"
    assert turn.kind == "model_step"
    assert turn.started_at == "2026-11-03T22:08:01.113000Z"
    assert turn.completed_at is None
    assert turn.prompt is None
    assert turn.reply is None


def test_no_parent_id(tmp_path: Path):
    conv = ConversationTelemetry(
        native_id="conv-root",
        steps=[],
        parent_native_ids=[],
        model_name=None,
    )

    result = parse_antigravity_conversation("conv-root", tmp_path, reader=lambda n, d: conv)

    assert len(result.sessions) == 1
    session = result.sessions[0]
    assert session.parent_session_key is None
    assert session.model is None
    assert session.first_ts is None
    assert session.last_ts is None
    assert result.steps == []
    assert result.turns == []
    assert result.items == []
    assert result.offset == 0


def test_multiple_steps_min_max_timestamps(tmp_path: Path):
    step_early = ModelStep(
        step_index=1,
        started_at="2026-11-03T10:00:00Z",
        input_tokens=10,
        output_tokens=5,
    )
    step_late = ModelStep(
        step_index=2,
        started_at="2026-11-03T12:00:00Z",
        input_tokens=20,
        output_tokens=10,
    )
    # Passed in reverse order to ensure min/max does not assume ordered steps
    conv = ConversationTelemetry(
        native_id="conv-multi",
        steps=[step_late, step_early],
        model_name="model-x",
    )

    result = parse_antigravity_conversation("conv-multi", tmp_path, reader=lambda n, d: conv)

    assert len(result.steps) == 2
    assert len(result.turns) == 2
    assert result.steps[0].turn_id == "step:2"
    assert result.steps[1].turn_id == "step:1"
    session = result.sessions[0]
    assert session.first_ts == "2026-11-03T10:00:00.000000Z"
    assert session.last_ts == "2026-11-03T12:00:00.000000Z"


def test_default_reader_is_read_agy_conversation():
    sig = signature(parse_antigravity_conversation)
    assert sig.parameters["reader"].default is read_agy_conversation


def test_stub_reader_two_steps_one_with_two_tool_details(tmp_path: Path):
    step1 = ModelStep(
        step_index=1,
        started_at="2026-11-03T22:08:01.113Z",
        completed_at="2026-11-03T22:08:05.500Z",
        input_tokens=100,
        output_tokens=30,
        context_texts=[
            RecordedText(role="user", text="First user prompt", truncated=False),
            RecordedText(role="user", text="Second user prompt", truncated=True),
            RecordedText(role="assistant", text="First assistant reply", truncated=True),
            RecordedText(role="assistant", text="Final assistant reply", truncated=False),
        ],
        tool_details=[
            ToolCallDetail(
                name="run_command",
                fields={"command": "uv run pytest", "cwd": "/repo"},
                truncated=[],
            ),
            ToolCallDetail(
                name="view_file",
                fields={"path": "/repo/src/main.py"},
                truncated=["path"],
            ),
        ],
    )
    step2 = ModelStep(
        step_index=2,
        started_at="2026-11-03T22:08:10.000Z",
        completed_at=None,
        input_tokens=200,
        output_tokens=40,
        context_texts=[
            RecordedText(role="user", text="Another prompt", truncated=True),
        ],
        tool_details=[],
    )

    conv = ConversationTelemetry(
        native_id="conv-stub-1",
        steps=[step1, step2],
        model_name="gemini-1.5-pro",
    )

    result = parse_antigravity_conversation(
        "conv-stub-1", tmp_path, reader=lambda native_id, t3_dir: conv
    )

    assert len(result.turns) == 2
    assert len(result.items) == 2

    # Verify turns
    turn1, turn2 = result.turns
    assert turn1.session_key == "antigravity:conv-stub-1"
    assert turn1.turn_id == "step:1"
    assert turn1.kind == "model_step"
    assert turn1.started_at == "2026-11-03T22:08:01.113000Z"
    assert turn1.completed_at == "2026-11-03T22:08:05.500000Z"
    # First user prompt wins
    assert turn1.prompt == "First user prompt"
    assert turn1.prompt_truncated is False
    # Last assistant reply wins
    assert turn1.reply == "Final assistant reply"
    assert turn1.reply_truncated is False

    assert turn2.session_key == "antigravity:conv-stub-1"
    assert turn2.turn_id == "step:2"
    assert turn2.kind == "model_step"
    assert turn2.started_at == "2026-11-03T22:08:10.000000Z"
    assert turn2.completed_at is None
    assert turn2.prompt == "Another prompt"
    assert turn2.prompt_truncated is True
    assert turn2.reply is None
    assert turn2.reply_truncated is False

    # Verify items
    item0, item1 = result.items
    assert item0.session_key == "antigravity:conv-stub-1"
    assert item0.item_id == "step:1:tool:0"
    assert item0.turn_id == "step:1"
    assert item0.seq == 0
    assert item0.kind == "tool"
    assert item0.tool_name == "run_command"
    assert item0.fields == {"command": "uv run pytest", "cwd": "/repo"}
    assert item0.truncated_fields == []
    assert "uv run pytest" in item0.text

    assert item1.session_key == "antigravity:conv-stub-1"
    assert item1.item_id == "step:1:tool:1"
    assert item1.turn_id == "step:1"
    assert item1.seq == 1
    assert item1.kind == "tool"
    assert item1.tool_name == "view_file"
    assert item1.fields == {"path": "/repo/src/main.py"}
    assert item1.truncated_fields == ["path"]
    assert "/repo/src/main.py" in item1.text
