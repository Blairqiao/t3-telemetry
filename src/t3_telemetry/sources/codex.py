"""Read token usage and metadata from Codex rollout JSONL files."""

from pathlib import Path

from t3_telemetry.sources._jsonl import read_jsonl
from t3_telemetry.store import ParseResult, SessionRecord, StepRecord, utc_iso


def parse_codex_file(
    path: Path, offset: int = 0, session_id: str | None = None, *, model: str | None = None
) -> ParseResult:
    """Parse complete new records; pass the saved model when resuming at an offset."""
    path = Path(path)
    native_id = session_id or path.stem[-36:]
    session = SessionRecord(session_key=f"codex:{native_id}", provider="codex", native_id=native_id)
    steps: list[StepRecord] = []
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
        elif kind == "turn_context":
            if isinstance(payload.get("model"), str):
                model = payload["model"]
                session.model = model
        elif kind == "token_usage_record" and ts is not None:
            usage = payload.get("usage")
            event_id = payload.get("response_id")
            if not event_id and entry.get("ordinal") is not None:
                event_id = f"ordinal:{entry['ordinal']}"
            if not isinstance(usage, dict) or not isinstance(event_id, str):
                continue
            cached = usage.get("cached_input_tokens") or 0
            cache_write = usage.get("cache_write_input_tokens") or 0
            steps.append(StepRecord(
                session_key=session.session_key, event_id=event_id, ts=ts, model=model,
                input_tokens=max(0, (usage.get("input_tokens") or 0) - cached - cache_write),
                cache_read_tokens=cached, cache_creation_tokens=cache_write,
                output_tokens=usage.get("output_tokens") or 0,
                thinking_tokens=usage.get("reasoning_output_tokens") or 0,
            ))
            session.model = model
    return ParseResult(sessions=[session], steps=steps, offset=offset)
