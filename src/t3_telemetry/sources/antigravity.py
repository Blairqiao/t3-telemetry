"""Antigravity source adapter for native conversation telemetry."""

from pathlib import Path
from typing import Callable

from t3_telemetry.db_agy import read_agy_conversation
from t3_telemetry.models import ConversationTelemetry
from t3_telemetry.store import ParseResult, SessionRecord, StepRecord, utc_iso


def parse_antigravity_conversation(
    native_id: str,
    t3_dir: Path,
    reader: Callable[..., ConversationTelemetry] = read_agy_conversation,
) -> ParseResult:
    conv = reader(native_id, t3_dir)
    session_key = f"antigravity:{native_id}"
    parent_session_key = (
        f"antigravity:{conv.parent_native_ids[0]}"
        if conv.parent_native_ids
        else None
    )

    steps: list[StepRecord] = []
    for step in conv.steps:
        if not step.started_at:
            continue
        step_rec = StepRecord(
            session_key=session_key,
            event_id=f"step:{step.step_index}",
            ts=utc_iso(step.started_at),
            model=getattr(step, "model", None) or conv.model_name,
            input_tokens=step.input_tokens,
            cache_read_tokens=step.cache_read_tokens,
            cache_creation_tokens=step.cache_creation_tokens,
            output_tokens=step.output_tokens,
            thinking_tokens=step.thinking_tokens,
            tool_names=list(step.tool_calls) if step.tool_calls else [],
        )
        steps.append(step_rec)

    first_ts = min((s.ts for s in steps), default=None)
    last_ts = max((s.ts for s in steps), default=None)

    session = SessionRecord(
        session_key=session_key,
        provider="antigravity",
        native_id=native_id,
        parent_session_key=parent_session_key,
        model=conv.model_name,
        origin_hint="unknown",
        first_ts=first_ts,
        last_ts=last_ts,
    )

    return ParseResult(sessions=[session], steps=steps, offset=0)
