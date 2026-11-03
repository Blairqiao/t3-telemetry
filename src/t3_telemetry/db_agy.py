"""Antigravity conversation database and transcript reader.

Reads native SQLite .db conversations and enriches steps with tool calls
and subagent linkages from transcript.jsonl.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from t3_telemetry.models import ConversationTelemetry, ModelStep
from t3_telemetry.protobuf import decode_step_metadata

UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
CONV_ID_PATTERN = re.compile(r"\"conversationId\"\s*:\s*\"([0-9a-fA-F-]{36})\"")
CONV_ID_FALLBACK = re.compile(r"conversationId[\"\s:]+([0-9a-fA-F-]{36})")


def find_agy_db_path(native_id: str, t3_dir: Path | str) -> Path | None:
    """Finds the Antigravity conversation .db for a given native ID.

    Searches:
    1. <t3_dir>/providers/antigravity/*/antigravity-acp/conversations/<native_id>.db
    2. <t3_dir>/conversations/<native_id>.db
    3. ~/.gemini/antigravity-cli/conversations/<native_id>.db
    """
    base = Path(t3_dir).expanduser()
    db_filename = f"{native_id}.db"

    # 1. Provider nested path
    matches = list(base.glob(f"providers/antigravity/*/antigravity-acp/conversations/{db_filename}"))
    if matches:
        return matches[0]

    # 2. Flat path
    flat = base / "conversations" / db_filename
    if flat.exists():
        return flat

    # 3. Gemini CLI fallback
    gemini_db = Path("~/.gemini/antigravity-cli/conversations").expanduser() / db_filename
    if gemini_db.exists():
        return gemini_db

    return None


def find_agy_transcript_path(native_id: str, t3_dir: Path | str) -> Path | None:
    """Finds the Antigravity transcript.jsonl for a given native ID.

    Searches:
    1. <t3_dir>/providers/antigravity/*/antigravity-acp/brain/<native_id>/.system_generated/logs/transcript.jsonl
    2. <t3_dir>/brain/<native_id>/.system_generated/logs/transcript.jsonl
    3. <t3_dir>/transcripts/<native_id>.jsonl
    4. ~/.gemini/antigravity-cli/brain/<native_id>/.system_generated/logs/transcript.jsonl
    """
    base = Path(t3_dir).expanduser()

    # 1. Provider nested brain path
    matches = list(base.glob(f"providers/antigravity/*/antigravity-acp/brain/{native_id}/.system_generated/logs/transcript.jsonl"))
    if matches:
        return matches[0]

    # 2. Flat brain path
    flat_brain = base / "brain" / native_id / ".system_generated" / "logs" / "transcript.jsonl"
    if flat_brain.exists():
        return flat_brain

    # 3. Flat transcripts fixture path
    flat_transcript = base / "transcripts" / f"{native_id}.jsonl"
    if flat_transcript.exists():
        return flat_transcript

    # 4. Gemini CLI fallback
    gemini_transcript = (
        Path("~/.gemini/antigravity-cli/brain").expanduser()
        / native_id
        / ".system_generated"
        / "logs"
        / "transcript.jsonl"
    )
    if gemini_transcript.exists():
        return gemini_transcript

    return None


def read_transcript_data(
    transcript_path: Path | None,
) -> tuple[list[ModelStep], dict[int, list[str]], list[str]]:
    """Reads transcript.jsonl line by line.

    Returns:
        (fallback_steps, tool_calls_by_step, child_native_ids)
    """
    if transcript_path is None or not transcript_path.exists():
        return [], {}, []

    fallback_steps: list[ModelStep] = []
    tool_calls_by_step: dict[int, list[str]] = {}
    child_native_ids: list[str] = []

    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                # Search for subagent conversation IDs
                for m in CONV_ID_PATTERN.finditer(line_str):
                    cid = m.group(1)
                    if cid not in child_native_ids:
                        child_native_ids.append(cid)
                for m in CONV_ID_FALLBACK.finditer(line_str):
                    cid = m.group(1)
                    if cid not in child_native_ids:
                        child_native_ids.append(cid)

                try:
                    obj = json.loads(line_str)
                except Exception:
                    continue

                if not isinstance(obj, dict):
                    continue

                step_index = obj.get("step_index")
                tool_calls_raw = obj.get("tool_calls")
                tool_names: list[str] = []
                if isinstance(tool_calls_raw, list):
                    for tc in tool_calls_raw:
                        if isinstance(tc, dict) and "name" in tc:
                            tool_names.append(str(tc["name"]))

                if step_index is not None and tool_names:
                    try:
                        tool_calls_by_step[int(step_index)] = tool_names
                    except (ValueError, TypeError):
                        pass

                if obj.get("type") == "PLANNER_RESPONSE":
                    try:
                        idx_val = int(step_index) if step_index is not None else (len(fallback_steps) + 1)
                    except (ValueError, TypeError):
                        idx_val = len(fallback_steps) + 1

                    inp = int(obj.get("input_tokens") or 0)
                    out = int(obj.get("output_tokens") or 0)
                    cache = int(obj.get("cache_read_tokens") or 0)
                    created_at = obj.get("created_at")
                    if created_at is not None:
                        created_at = str(created_at)

                    fallback_step = ModelStep(
                        step_index=idx_val,
                        source="transcript",
                        status="completed",
                        input_tokens=inp,
                        cache_read_tokens=cache,
                        output_tokens=out,
                        response_tokens=out,
                        started_at=created_at,
                        completed_at=created_at,
                        tool_calls=tool_names,
                    )
                    fallback_steps.append(fallback_step)

    except OSError:
        pass

    return fallback_steps, tool_calls_by_step, child_native_ids


def read_agy_conversation(
    native_id: str,
    t3_dir: Path | str,
    source_label: str = "main",
) -> ConversationTelemetry:
    """Reads Antigravity conversation telemetry for native_id.

    Attempts to read SQLite <nativeId>.db with mode=ro URI connection and
    enriches steps with tool call names from transcript.jsonl. If DB is absent
    or corrupt, gracefully falls back to transcript.jsonl PLANNER_RESPONSE
    steps.
    """
    base = Path(t3_dir).expanduser()
    db_path = find_agy_db_path(native_id, base)
    transcript_path = find_agy_transcript_path(native_id, base)

    fallback_steps, tool_calls_by_step, child_ids = read_transcript_data(transcript_path)

    if db_path is not None and db_path.exists():
        try:
            uri = f"file:{db_path.resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=1.0)
            cursor = conn.cursor()

            # Query parent_references if table exists
            parent_ids: list[str] = []
            try:
                cursor.execute("SELECT data FROM parent_references")
                for (data_blob,) in cursor.fetchall():
                    if data_blob:
                        text = (
                            data_blob.decode("latin-1", errors="ignore")
                            if isinstance(data_blob, bytes)
                            else str(data_blob)
                        )
                        for u in UUID_PATTERN.findall(text):
                            if u not in parent_ids and u != native_id:
                                parent_ids.append(u)
            except sqlite3.OperationalError:
                pass

            # Query inference steps (step_type = 15)
            cursor.execute(
                "SELECT idx, step_type, status, metadata FROM steps WHERE step_type = 15 ORDER BY idx ASC"
            )
            rows = cursor.fetchall()
            conn.close()

            steps: list[ModelStep] = []
            model_name: str | None = None

            for ordinal, (idx, _st, _status, meta_blob) in enumerate(rows):
                meta = decode_step_metadata(meta_blob)
                if model_name is None and meta.get("model_name"):
                    model_name = meta["model_name"]

                # Match tool calls by step_index == idx first, or by ordinal PLANNER_RESPONSE
                tools = tool_calls_by_step.get(idx)
                if tools is None:
                    if ordinal < len(fallback_steps):
                        tools = fallback_steps[ordinal].tool_calls
                    else:
                        tools = []

                step = ModelStep(
                    step_index=idx,
                    source=source_label,
                    status="completed",
                    input_tokens=meta["input_tokens"],
                    cache_read_tokens=meta["cache_read_tokens"],
                    cache_creation_tokens=meta["cache_creation_tokens"],
                    thinking_tokens=meta["thinking_tokens"],
                    response_tokens=meta["response_tokens"],
                    output_tokens=meta["output_tokens"],
                    tool_calls=tools,
                    started_at=meta["started_at"],
                    completed_at=meta["completed_at"],
                    duration_seconds=meta["duration_seconds"],
                )
                steps.append(step)

            return ConversationTelemetry(
                native_id=native_id,
                steps=steps,
                child_native_ids=child_ids,
                parent_native_ids=parent_ids,
                model_name=model_name,
                source="db",
            )
        except sqlite3.Error:
            # Fall back to transcript
            pass

    # Fallback to transcript.jsonl if DB absent or errored
    if fallback_steps:
        return ConversationTelemetry(
            native_id=native_id,
            steps=fallback_steps,
            child_native_ids=child_ids,
            parent_native_ids=[],
            model_name=None,
            source="transcript",
        )

    return ConversationTelemetry(
        native_id=native_id,
        steps=[],
        child_native_ids=[],
        parent_native_ids=[],
        model_name=None,
        source="missing",
    )
