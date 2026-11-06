"""Antigravity source adapter for native conversation telemetry."""

from pathlib import Path
from typing import Callable

from t3_telemetry.activity import describe_tool
from t3_telemetry.db_agy import read_agy_conversation
from t3_telemetry.models import ConversationTelemetry
from t3_telemetry.store import (
    MAX_ITEM_CHARS,
    ItemRecord,
    ParseResult,
    SessionRecord,
    StepRecord,
    TurnRecord,
    bounded,
    utc_iso,
)


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
    turns: list[TurnRecord] = []
    items: list[ItemRecord] = []
    for step in conv.steps:
        if not step.started_at:
            continue
        turn_id = f"step:{step.step_index}"
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
            turn_id=turn_id,
        )
        steps.append(step_rec)

        prompt = None
        prompt_truncated = False
        reply = None
        reply_truncated = False
        user_rec = next((r for r in (step.context_texts or []) if r.role == "user"), None)
        if user_rec is not None:
            prompt = user_rec.text
            prompt_truncated = user_rec.truncated
        asst_rec = next((r for r in reversed(step.context_texts or []) if r.role == "assistant"), None)
        if asst_rec is not None:
            reply = asst_rec.text
            reply_truncated = asst_rec.truncated

        turns.append(
            TurnRecord(
                session_key=session_key,
                turn_id=turn_id,
                kind="model_step",
                started_at=utc_iso(step.started_at),
                completed_at=utc_iso(step.completed_at) if step.completed_at else None,
                prompt=prompt,
                prompt_truncated=prompt_truncated,
                reply=reply,
                reply_truncated=reply_truncated,
            )
        )

        for i, detail in enumerate(step.tool_details or []):
            item_text, item_truncated = bounded(describe_tool(detail), MAX_ITEM_CHARS)
            items.append(
                ItemRecord(
                    session_key=session_key,
                    item_id=f"step:{step.step_index}:tool:{i}",
                    turn_id=turn_id,
                    seq=i,
                    kind="tool",
                    text=item_text or "",
                    truncated=item_truncated,
                    tool_name=detail.name,
                    fields=detail.fields,
                    truncated_fields=detail.truncated,
                )
            )

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

    return ParseResult(sessions=[session], steps=steps, offset=0, turns=turns, items=items)
