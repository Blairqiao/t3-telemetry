from inspect import signature
from pathlib import Path

from t3_telemetry.db_agy import read_agy_conversation
from t3_telemetry.models import ConversationTelemetry, ModelStep
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
    assert step.ts == "2026-11-03T22:08:01.113000Z"
    assert step.model == "gemini-1.5-pro"
    assert step.input_tokens == 100
    assert step.cache_read_tokens == 50
    assert step.cache_creation_tokens == 25
    assert step.output_tokens == 30
    assert step.thinking_tokens == 15
    assert step.tool_names == ["view_file", "run_command"]


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
    session = result.sessions[0]
    assert session.first_ts == "2026-11-03T10:00:00.000000Z"
    assert session.last_ts == "2026-11-03T12:00:00.000000Z"


def test_default_reader_is_read_agy_conversation():
    sig = signature(parse_antigravity_conversation)
    assert sig.parameters["reader"].default is read_agy_conversation
