"""Read token usage and metadata from Codex rollout JSONL files."""

from datetime import datetime, timezone
from pathlib import Path

from t3_telemetry.sources._jsonl import read_jsonl
from t3_telemetry.store import (
    MAX_ITEM_CHARS, MAX_TEXT_CHARS, ItemRecord, ParseResult, SessionRecord,
    StepRecord, TurnRecord, bounded, utc_iso,
)


def parse_codex_file(
    path: Path, offset: int = 0, session_id: str | None = None, *, model: str | None = None
) -> ParseResult:
    """Parse complete new records; pass the saved model when resuming at an offset."""
    path = Path(path)
    native_id = session_id or path.stem[-36:]
    session = SessionRecord(session_key=f"codex:{native_id}", provider="codex", native_id=native_id)
    steps: list[StepRecord] = []
    turns: dict[str, TurnRecord] = {}
    items: list[ItemRecord] = []
    seq_base = offset * 1000

    def turn(turn_id: str) -> TurnRecord:
        if turn_id not in turns:
            turns[turn_id] = TurnRecord(session_key=session.session_key, turn_id=turn_id)
        return turns[turn_id]

    def epoch(value: object, fallback: str | None = None, *, milliseconds: bool = False) -> str | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return utc_iso(datetime.fromtimestamp(value / (1000 if milliseconds else 1), timezone.utc))
            except (OverflowError, OSError, ValueError):
                pass
        return fallback
    # Forked subagent rollouts repeat the parent's session_meta after their own; only the first is this file's.
    meta_seen = offset > 0
    for offset, entry in read_jsonl(path, offset):
        if entry is None:
            continue
        ts = None
        if isinstance(entry.get("timestamp"), str):
            try:
                ts = utc_iso(entry["timestamp"])
            except ValueError:
                pass
        if ts:
            session.first_ts = min(session.first_ts, ts) if session.first_ts else ts
            session.last_ts = max(session.last_ts, ts) if session.last_ts else ts
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = entry.get("type")
        if kind == "session_meta" and not meta_seen:
            meta_seen = True
            if isinstance(payload.get("id"), str) and payload["id"]:
                session.native_id = payload["id"]
                session.session_key = f"codex:{session.native_id}"
            if isinstance(payload.get("cwd"), str):
                session.cwd = payload["cwd"]
            originator = payload.get("originator")
            if isinstance(originator, str):
                if "T3" in originator:
                    session.origin_hint = "t3"
                elif originator.startswith("codex"):
                    session.origin_hint = "cli"
        elif not meta_seen:
            continue
        elif kind == "turn_context":
            if isinstance(payload.get("model"), str):
                model = payload["model"]
                session.model = model
        elif kind == "event_msg":
            event_type = payload.get("type")
            turn_id = payload.get("turn_id")
            if not isinstance(turn_id, str) or not turn_id:
                continue
            if event_type == "task_started":
                record = turn(turn_id)
                started = epoch(payload.get("started_at"), ts)
                if started is not None:
                    record.started_at = min(record.started_at, started) if record.started_at else started
                record.context_limit = payload.get("model_context_window")
            elif event_type == "task_complete":
                record = turn(turn_id)
                completed = epoch(payload.get("completed_at"), ts)
                if completed is not None:
                    record.completed_at = max(record.completed_at, completed) if record.completed_at else completed
            elif event_type == "item_completed":
                item = payload.get("item")
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type in ("UserMessage", "AgentMessage"):
                    content = item.get("content", item.get("text", ""))
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = "\n".join(part.get("text", "") for part in content
                                         if isinstance(part, dict) and isinstance(part.get("text"), str))
                    else:
                        text = ""
                    value, truncated = bounded(text, MAX_TEXT_CHARS)
                    record = turn(turn_id)
                    if item_type == "UserMessage" and record.prompt is None:
                        record.prompt, record.prompt_truncated = value, truncated
                    elif item_type == "AgentMessage" and value is not None:
                        record.reply, record.reply_truncated = value, truncated
                    continue
                item_id = item.get("id")
                if not isinstance(item_id, str) or not isinstance(item_type, str):
                    continue
                text = ""
                if item_type == "CommandExecution":
                    item_kind = "run"
                    command = item.get("command", "")
                    text = str(command[-1]) if isinstance(command, list) and command else str(command) if not isinstance(command, list) else ""
                elif item_type == "FileChange":
                    item_kind = "edit"
                    changes = item.get("changes") or []
                    if isinstance(changes, dict) and changes:
                        text = str(next(iter(changes)))
                        if len(changes) > 1:
                            text += f" (+{len(changes) - 1} more)"
                    elif isinstance(changes, list) and changes:
                        first = changes[0]
                        text = str(first.get("path", "")) if isinstance(first, dict) else str(first)
                        if len(changes) > 1:
                            text += f" (+{len(changes) - 1} more)"
                elif item_type == "McpToolCall":
                    item_kind, text = "tool", f"{item.get('server', '')}.{item.get('tool', '')}"
                elif item_type == "Extension":
                    extension_kind = item.get("kind", "")
                    item_kind, text = ("search", str(item.get("query", ""))) if extension_kind == "web.search" else ("tool", str(extension_kind))
                elif item_type == "Reasoning":
                    item_kind = "think"
                elif item_type == "SubAgentActivity":
                    item_kind, text = "subagent", f"{item.get('kind', '')} {item.get('agent_path', '')}"
                elif item_type == "CollabAgentToolCall":
                    item_kind, text = "tool", str(item.get("tool", ""))
                elif item_type == "ImageView":
                    item_kind, text = "tool", f"View image {item.get('path', '')}"
                elif item_type == "ContextCompaction":
                    item_kind = "compact"
                else:
                    item_kind, text = "tool", item_type
                text, truncated = bounded(text, MAX_ITEM_CHARS)
                items.append(ItemRecord(
                    session_key=session.session_key, item_id=item_id, turn_id=turn_id,
                    seq=seq_base + len(items), kind=item_kind, text=text or "", truncated=truncated,
                    started_at=epoch(payload.get("started_at_ms"), milliseconds=True),
                    completed_at=epoch(payload.get("completed_at_ms"), milliseconds=True),
                ))
        elif kind == "token_usage_record" and ts is not None:
            usage = payload.get("usage")
            event_id = payload.get("response_id")
            if not event_id and entry.get("ordinal") is not None:
                event_id = f"ordinal:{entry['ordinal']}"
            if not isinstance(usage, dict) or not isinstance(event_id, str):
                continue
            cached = usage.get("cached_input_tokens") or 0
            cache_write = usage.get("cache_write_input_tokens") or 0
            turn_id = payload.get("turn_id")
            if isinstance(turn_id, str) and turn_id:
                if turn_id not in turns:
                    turns[turn_id] = TurnRecord(session_key=session.session_key, turn_id=turn_id,
                                                started_at=ts, completed_at=ts)
            else:
                turn_id = None
            steps.append(StepRecord(
                session_key=session.session_key, event_id=event_id, ts=ts, model=model,
                input_tokens=max(0, (usage.get("input_tokens") or 0) - cached - cache_write),
                cache_read_tokens=cached, cache_creation_tokens=cache_write,
                output_tokens=usage.get("output_tokens") or 0,
                thinking_tokens=usage.get("reasoning_output_tokens") or 0, turn_id=turn_id,
            ))
            session.model = model
    return ParseResult(sessions=[session], steps=steps, offset=offset, turns=list(turns.values()), items=items)
