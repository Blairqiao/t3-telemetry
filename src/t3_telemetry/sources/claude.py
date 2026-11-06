"""Read token usage and session metadata from Claude JSONL transcripts."""

import json
from pathlib import Path

from t3_telemetry.sources._jsonl import read_jsonl
from t3_telemetry.store import (
    MAX_ITEM_CHARS, MAX_TEXT_CHARS, ItemRecord, ParseResult, SessionRecord,
    StepRecord, TurnRecord, bounded, utc_iso,
)


_ORIGIN_RANK = {"unknown": 0, "sdk": 1, "cli": 2, "t3": 3}


def _prompt_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Tool results are responses to prior calls, not new user prompts.
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            return None
        return "\n".join(b["text"] for b in content if isinstance(b, dict)
                         and b.get("type") == "text" and isinstance(b.get("text"), str))
    return None


def _tool_item(name: str, inputs: object) -> tuple[str, str]:
    data = inputs if isinstance(inputs, dict) else {}
    if name == "Bash":
        kind, value = "run", data.get("command")
    elif name in {"Edit", "Write", "MultiEdit", "NotebookEdit"}:
        kind, value = "edit", data.get("file_path") or data.get("notebook_path")
    elif name in {"Grep", "Glob"}:
        kind, value = "search", data.get("pattern")
    elif name == "WebSearch":
        kind, value = "search", data.get("query")
    elif name == "WebFetch":
        kind, value = "tool", data.get("url")
    elif name in {"Agent", "Task"}:
        kind, value = "subagent", data.get("description")
    elif name == "Read":
        kind, value = "tool", f"Read {data.get('file_path') or ''}"
    else:
        kind, value = "tool", name
    return kind, value if isinstance(value, str) else ""


def parse_claude_file(path: Path, offset: int = 0, *, turn_id: str | None = None) -> ParseResult:
    path = Path(path)
    subagent = path.parent.name == "subagents" and path.stem.startswith("agent-")
    native_id = f"{path.parent.parent.name}/{path.stem}" if subagent else path.stem
    session_key = f"claude:{native_id}"
    session = SessionRecord(
        session_key=session_key,
        provider="claude",
        native_id=native_id,
        parent_session_key=f"claude:{path.parent.parent.name}" if subagent else None,
    )
    if subagent:
        try:
            meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
            if isinstance(meta, dict) and isinstance(meta.get("description"), str):
                session.title = meta["description"]
        except (OSError, UnicodeError, ValueError):
            pass

    steps: dict[str, StepRecord] = {}
    turns: list[TurnRecord] = []
    items: list[ItemRecord] = []
    current_turn = turn_id
    seq_base = offset * 1000
    for offset, entry in read_jsonl(path, offset):
        if entry is None:
            continue

        if not subagent and entry.get("type") == "ai-title" and isinstance(entry.get("aiTitle"), str):
            session.title = entry["aiTitle"]
        if isinstance(entry.get("cwd"), str):
            session.cwd = entry["cwd"]
        ts = None
        if isinstance(entry.get("timestamp"), str):
            try:
                ts = utc_iso(entry["timestamp"])
            except ValueError:
                pass
        if ts is not None:
            session.first_ts = min(session.first_ts, ts) if session.first_ts else ts
            session.last_ts = max(session.last_ts, ts) if session.last_ts else ts
        entrypoint = entry.get("entrypoint")
        origin = "unknown"
        if entrypoint == "t3-code":
            origin = "t3"
        elif entrypoint == "cli":
            origin = "cli"
        elif isinstance(entrypoint, str) and entrypoint.startswith("sdk"):
            origin = "sdk"
        if _ORIGIN_RANK[origin] > _ORIGIN_RANK[session.origin_hint]:
            session.origin_hint = origin

        message = entry.get("message")
        entry_type = entry.get("type")
        content = message.get("content") if isinstance(message, dict) else None
        if entry_type == "user" and not entry.get("isMeta"):
            prompt, truncated = bounded(_prompt_text(content), MAX_TEXT_CHARS)
            prompt_id = entry.get("promptId") or (f"uuid:{entry['uuid']}" if entry.get("uuid") else None)
            if prompt is not None and isinstance(prompt_id, str) and prompt_id != current_turn:
                current_turn = prompt_id
                turns.append(TurnRecord(session_key, current_turn, started_at=ts,
                                        completed_at=ts, prompt=prompt, prompt_truncated=truncated))
        is_tool_result = (entry_type == "user" and isinstance(content, list)
                          and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content))
        if current_turn and ts and (entry_type == "assistant" or is_tool_result):
            turns.append(TurnRecord(session_key, current_turn, completed_at=ts))
        if current_turn and isinstance(content, list):
            message_id = message.get("id") if isinstance(message, dict) else None
            for index, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if entry_type == "assistant" and block_type == "text" and isinstance(block.get("text"), str):
                    reply, truncated = bounded(block["text"], MAX_TEXT_CHARS)
                    if reply is not None:
                        turns.append(TurnRecord(session_key, current_turn, reply=reply, reply_truncated=truncated))
                elif entry_type == "assistant" and block_type == "tool_use" and isinstance(block.get("id"), str):
                    name = block.get("name") if isinstance(block.get("name"), str) else ""
                    kind, text = _tool_item(name, block.get("input"))
                    text, truncated = bounded(text, MAX_ITEM_CHARS)
                    items.append(ItemRecord(session_key, block["id"], current_turn, seq_base + len(items),
                                            kind, text or "", truncated, "running", started_at=ts))
                elif entry_type == "assistant" and block_type == "thinking" and isinstance(message_id, str):
                    items.append(ItemRecord(session_key, f"{message_id}:think:{index}", current_turn,
                                            seq_base + len(items), "think", started_at=ts, completed_at=ts))
                elif entry_type == "user" and block_type == "tool_result" and isinstance(block.get("tool_use_id"), str):
                    items.append(ItemRecord(session_key, block["tool_use_id"], current_turn, 0, "tool",
                                            status="failed" if block.get("is_error") else "completed", completed_at=ts))
        if entry_type != "assistant" or not isinstance(message, dict):
            continue
        usage = message.get("usage")
        event_id = message.get("id")
        if not isinstance(usage, dict) or not isinstance(event_id, str) or ts is None:
            continue
        tools = list(steps[event_id].tool_names) if event_id in steps else []
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name")
                    if isinstance(name, str) and name not in tools:
                        tools.append(name)
        details = usage.get("output_tokens_details")
        if not isinstance(details, dict):
            details = {}
        step = StepRecord(
            session_key=session_key, event_id=event_id, ts=ts,
            model=message.get("model"),
            input_tokens=usage.get("input_tokens") or 0,
            cache_read_tokens=usage.get("cache_read_input_tokens") or 0,
            cache_creation_tokens=usage.get("cache_creation_input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            thinking_tokens=details.get("thinking_tokens") or 0,
            tool_names=tools, turn_id=current_turn,
        )
        steps[event_id] = step
        session.model = step.model

    return ParseResult(sessions=[session], steps=list(steps.values()), offset=offset, turns=turns, items=items)
