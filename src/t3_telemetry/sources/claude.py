"""Read token usage and session metadata from Claude JSONL transcripts."""

import json
from pathlib import Path

from t3_telemetry.sources._jsonl import read_jsonl
from t3_telemetry.store import ParseResult, SessionRecord, StepRecord, utc_iso


_ORIGIN_RANK = {"unknown": 0, "sdk": 1, "cli": 2, "t3": 3}


def parse_claude_file(path: Path, offset: int = 0) -> ParseResult:
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
        if entry.get("type") != "assistant" or not isinstance(message, dict):
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
            tool_names=tools,
        )
        steps[event_id] = step
        session.model = step.model

    return ParseResult(sessions=[session], steps=list(steps.values()), offset=offset)
