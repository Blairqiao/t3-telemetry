"""Antigravity conversation database and transcript reader.

Reads native SQLite .db conversations and enriches steps with tool calls
and subagent linkages from transcript.jsonl.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from t3_telemetry.models import ConversationTelemetry, ModelStep, NativeLaunchRecord, ToolCallDetail
from t3_telemetry.paths import antigravity_cli_dir
from t3_telemetry.protobuf import decode_step_metadata

UUID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
CONV_ID_PATTERN = re.compile(r"\"conversationId\"\s*:\s*\"([0-9a-fA-F-]{36})\"")
CONV_ID_FALLBACK = re.compile(r"conversationId[\"\s:]+([0-9a-fA-F-]{36})")

# Recorded argument keys we surface, by normalized field. Content bodies
# (file contents, replacements, prompts' attachments) are never retained.
RECOGNIZED_ARG_KEYS: dict[str, tuple[str, ...]] = {
    "command": ("CommandLine", "command", "cmd"),
    "cwd": ("Cwd", "cwd"),
    "path": ("AbsolutePath", "TargetFile", "Path", "path", "file_path", "DirectoryPath", "SearchPath", "SearchDirectory"),
    "query": ("Query", "query", "Pattern", "pattern"),
    "task": ("TaskId", "Description", "description", "task", "title", "TypeName"),
    "recipient": ("Recipient",),
    "summary": ("toolSummary",),
}
MAX_FIELD_CHARS = 500
MAX_TOOL_DETAILS_PER_STEP = 20
LAUNCH_TOOL_NAMES = ("invoke_subagent",)


# Transcript lines that never carry a tool result, so they take no result slot.
NON_RESULT_LINE_TYPES = frozenset({"CHECKPOINT", "EPHEMERAL_MESSAGE"})


def _decode_scalar(val: object) -> object:
    """Antigravity JSON-encodes each argument value (``"\\"pwd\\""``, ``"1"``); unwrap encoded strings only."""
    if not isinstance(val, str):
        return val
    try:
        decoded = json.loads(val)
    except ValueError:
        return val
    return decoded if isinstance(decoded, str) else val


def parse_tool_call_detail(tc: object) -> ToolCallDetail | None:
    """Keep the name plus recognized scalar argument fields of one transcript tool call.

    ``args`` (or ``arguments``/``input``) may be a dict or a JSON-object string.
    Anything else yields a name-only detail. Values are bounded, never interpreted.
    """
    if not isinstance(tc, dict) or "name" not in tc:
        return None
    raw = tc.get("args")
    if raw is None:
        raw = tc.get("arguments", tc.get("input"))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    detail = ToolCallDetail(name=str(tc["name"]))
    if not isinstance(raw, dict):
        return detail
    for norm, keys in RECOGNIZED_ARG_KEYS.items():
        for key in keys:
            val = _decode_scalar(raw.get(key))
            if isinstance(val, bool) or not isinstance(val, (str, int, float)):
                continue
            text = str(val).strip()
            if not text:
                continue
            if len(text) > MAX_FIELD_CHARS:
                text = text[:MAX_FIELD_CHARS]
                detail.truncated.append(norm)
            detail.fields[norm] = text
            break
    return detail


def find_agy_db_path(native_id: str, t3_dir: Path | str) -> Path | None:
    """Finds the Antigravity conversation .db for a given native ID.

    Searches:
    1. <t3_dir>/providers/antigravity/*/antigravity-acp/conversations/<native_id>.db
    2. <t3_dir>/conversations/<native_id>.db
    3. <antigravity CLI dir>/conversations/<native_id>.db (see paths.antigravity_cli_dir)
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
    gemini_db = antigravity_cli_dir() / "conversations" / db_filename
    if gemini_db.exists():
        return gemini_db

    return None


def find_agy_transcript_path(native_id: str, t3_dir: Path | str) -> Path | None:
    """Finds the Antigravity transcript.jsonl for a given native ID.

    Searches:
    1. <t3_dir>/providers/antigravity/*/antigravity-acp/brain/<native_id>/.system_generated/logs/transcript.jsonl
    2. <t3_dir>/brain/<native_id>/.system_generated/logs/transcript.jsonl
    3. <t3_dir>/transcripts/<native_id>.jsonl
    4. <antigravity CLI dir>/brain/<native_id>/.system_generated/logs/transcript.jsonl
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
        antigravity_cli_dir() / "brain"
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
    steps, by_step, _details, children = read_transcript_details(transcript_path)
    return steps, by_step, children


def read_transcript_details(
    transcript_path: Path | None,
) -> tuple[list[ModelStep], dict[int, list[str]], dict[int, list[ToolCallDetail]], list[str]]:
    """Like ``read_transcript_data`` but also returns recognized tool-call details by step index."""
    steps, by_step, details, children, _launches = read_transcript_launches(transcript_path)
    return steps, by_step, details, children


def read_transcript_launches(
    transcript_path: Path | None,
) -> tuple[
    list[ModelStep], dict[int, list[str]], dict[int, list[ToolCallDetail]], list[str], list[NativeLaunchRecord]
]:
    """Full transcript scan, adding one launch record per discovered child conversation.

    The transcript records no tool-call ID, so each ``invoke_subagent`` call is
    paired by position: a response with N tool calls owns its next N
    non-``PLANNER_RESPONSE`` lines, and the line at the invoke's index is its result.
    A result without a new ``conversationId`` drops the call: no launch is
    recorded for a failed invoke. A child ID found elsewhere is kept as a
    discovery-only launch (no step, time, or task).
    """
    if transcript_path is None or not transcript_path.exists():
        return [], {}, {}, [], []

    fallback_steps: list[ModelStep] = []
    tool_calls_by_step: dict[int, list[str]] = {}
    tool_details_by_step: dict[int, list[ToolCallDetail]] = {}
    child_native_ids: list[str] = []
    launches: list[NativeLaunchRecord] = []
    pending_calls: list[NativeLaunchRecord | None] = []  # one slot per tool call; None = not an invoke

    try:
        with open(transcript_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                # Search for subagent conversation IDs
                new_ids: list[str] = []
                for pattern in (CONV_ID_PATTERN, CONV_ID_FALLBACK):
                    for m in pattern.finditer(line_str):
                        cid = m.group(1)
                        if cid not in child_native_ids:
                            child_native_ids.append(cid)
                            new_ids.append(cid)

                try:
                    obj = json.loads(line_str)
                except Exception:
                    obj = None

                # Positional pairing: a PLANNER_RESPONSE with N tool calls owns its
                # next N non-PLANNER lines, in tool-call order (the transcript has no
                # tool-call ID). Only the slot of an invoke call is read as that
                # invoke's result. A launch result usually embeds JSON in
                # ``content``, escaped on the raw line; only that result line is read
                # this way, so other text that mentions IDs (command output) cannot
                # invent a child. A result with no ID drops its call (failed launch).
                # A new PLANNER_RESPONSE drops any calls still unmatched.
                is_planner = isinstance(obj, dict) and obj.get("type") == "PLANNER_RESPONSE"
                owner: NativeLaunchRecord | None = None
                is_bookkeeping = isinstance(obj, dict) and obj.get("type") in NON_RESULT_LINE_TYPES
                if is_planner:
                    pending_calls = []
                elif pending_calls and not is_bookkeeping:
                    owner = pending_calls.pop(0)
                if owner is not None and isinstance(obj, dict):
                    content = obj.get("content")
                    if isinstance(content, str):
                        for m in CONV_ID_PATTERN.finditer(content):
                            cid = m.group(1)
                            if cid not in child_native_ids:
                                child_native_ids.append(cid)
                                new_ids.append(cid)

                for i, cid in enumerate(new_ids):
                    rec = owner if (i == 0 and owner is not None) else NativeLaunchRecord(child_native_id=cid)
                    rec.child_native_id = cid
                    launches.append(rec)

                if not isinstance(obj, dict):
                    continue

                step_index = obj.get("step_index")
                tool_calls_raw = obj.get("tool_calls")
                tool_names: list[str] = []
                tool_details: list[ToolCallDetail] = []
                if isinstance(tool_calls_raw, list):
                    for tc in tool_calls_raw:
                        detail = parse_tool_call_detail(tc)
                        slot: NativeLaunchRecord | None = None
                        if detail is not None:
                            tool_names.append(detail.name)
                            if len(tool_details) < MAX_TOOL_DETAILS_PER_STEP:
                                tool_details.append(detail)
                            if detail.name in LAUNCH_TOOL_NAMES:
                                slot = _launch_call(detail, step_index, obj.get("created_at"))
                        if is_planner:
                            pending_calls.append(slot)

                if step_index is not None and tool_names:
                    try:
                        tool_calls_by_step[int(step_index)] = tool_names
                        tool_details_by_step[int(step_index)] = tool_details
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
                        tool_details=tool_details,
                    )
                    fallback_steps.append(fallback_step)

    except OSError:
        pass

    return fallback_steps, tool_calls_by_step, tool_details_by_step, child_native_ids, launches


def _launch_call(detail: ToolCallDetail, step_index: object, created_at: object) -> NativeLaunchRecord:
    try:
        idx = int(step_index) if step_index is not None else None
    except (ValueError, TypeError):
        idx = None
    return NativeLaunchRecord(
        child_native_id="",
        step_index=idx,
        launched_at=str(created_at) if created_at is not None else None,
        task=detail.fields.get("task"),
        tool_name=detail.name,
    )


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

    fallback_steps, tool_calls_by_step, tool_details_by_step, child_ids, launches = read_transcript_launches(
        transcript_path
    )

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
                details = tool_details_by_step.get(idx, [])
                if tools is None:
                    if ordinal < len(fallback_steps):
                        tools = fallback_steps[ordinal].tool_calls
                        details = fallback_steps[ordinal].tool_details
                    else:
                        tools = []
                        details = []

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
                    tool_details=list(details),
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
                launches=launches,
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
            launches=launches,
        )

    return ConversationTelemetry(
        native_id=native_id,
        steps=[],
        child_native_ids=[],
        parent_native_ids=[],
        model_name=None,
        source="missing",
    )
