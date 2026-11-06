"""T3 statev2.sqlite reader for orchestration projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from t3_telemetry.models import (
    AppThreadRecord,
    ProviderThreadRecord,
    ProviderTurnRecord,
    SubagentLinkRecord,
)

# Bounds for message text associated with provider turns.
CONTEXT_ID_CHUNK = 500  # well under SQLite's default 999 bound-variable limit


def _usage_counts(usage: dict) -> tuple[int, int, int, int, int]:
    cache_read = int(usage.get("cachedInputTokens") or 0)
    cache_creation = int(usage.get("cacheCreationTokens") or 0)
    uncached = max(0, int(usage.get("inputTokens") or 0) - cache_read - cache_creation)
    output = int(usage.get("outputTokens") or 0)
    thinking = int(usage.get("reasoningTokens") or usage.get("reasoningOutputTokens") or 0)
    return uncached, cache_read, cache_creation, output, thinking


def connect_readonly_sqlite(db_path: Path | str) -> sqlite3.Connection:
    """Open a read-only SQLite connection using URI mode."""
    resolved = Path(db_path).resolve()
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, timeout=1.0)


def parse_iso_duration(started_at: str | None, completed_at: str | None) -> float:
    """Calculate duration in seconds between two ISO-8601 timestamps."""
    if not started_at or not completed_at:
        return 0.0
    try:
        s_str = started_at.replace("Z", "+00:00")
        c_str = completed_at.replace("Z", "+00:00")
        dt_start = datetime.fromisoformat(s_str)
        dt_end = datetime.fromisoformat(c_str)
        diff = (dt_end - dt_start).total_seconds()
        return max(0.0, float(diff))
    except Exception:
        return 0.0


@dataclass(slots=True)
class T3StateSnapshot:
    """In-memory snapshot of orchestration projections from statev2.sqlite."""

    threads: dict[str, AppThreadRecord] = field(default_factory=dict)
    provider_threads: dict[str, ProviderThreadRecord] = field(default_factory=dict)
    provider_turns_by_thread: dict[str, list[ProviderTurnRecord]] = field(default_factory=dict)
    subagents_by_parent: dict[str, list[SubagentLinkRecord]] = field(default_factory=dict)
    provider_threads_by_thread: dict[str, list[ProviderThreadRecord]] = field(default_factory=dict)

    def get_provider_thread(self, thread_id: str) -> ProviderThreadRecord | None:
        """Get the active provider thread for an app thread, or latest available."""
        app_thread = self.threads.get(thread_id)
        if app_thread and app_thread.active_provider_thread_id:
            pt = self.provider_threads.get(app_thread.active_provider_thread_id)
            if pt is not None:
                return pt
        pts = self.provider_threads_by_thread.get(thread_id, [])
        if pts:
            return pts[-1]
        return None


def _extract_thinking_level(options: list[dict[str, Any]] | None) -> str | None:
    if not options or not isinstance(options, list):
        return None
    for opt in options:
        if isinstance(opt, dict) and opt.get("id") in ("thinking_level", "reasoningEffort", "effort"):
            val = opt.get("value")
            return str(val) if val is not None else None
    return None


def read_t3_state(t3_dir: Path | str) -> T3StateSnapshot:
    """Read statev2.sqlite from t3_dir into T3StateSnapshot."""
    base_path = Path(t3_dir)
    if base_path.is_file() and (base_path.name == "statev2.sqlite" or base_path.suffix in (".sqlite", ".db")):
        db_path = base_path
    else:
        db_path = base_path / "statev2.sqlite"

    if not db_path.is_file():
        return T3StateSnapshot()

    snapshot = T3StateSnapshot()

    conn = connect_readonly_sqlite(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        # Check existing tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}

        # 1. orchestration_v2_projection_threads
        if "orchestration_v2_projection_threads" in tables:
            cur.execute("""
                SELECT thread_id, project_id, title, default_provider, runtime_mode,
                       interaction_mode, active_provider_thread_id, created_at, updated_at,
                       archived_at, deleted_at, payload_json, provider_instance_id
                FROM orchestration_v2_projection_threads
                WHERE deleted_at IS NULL
            """)
            for row in cur.fetchall():
                thread_id = row["thread_id"]
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    payload = {}

                model_selection = payload.get("modelSelection") or {}
                model = model_selection.get("model")
                options = model_selection.get("options")
                thinking_level = _extract_thinking_level(options)

                lineage = payload.get("lineage") or {}
                parent_thread_id = lineage.get("parentThreadId")
                root_thread_id = lineage.get("rootThreadId")
                relationship_to_parent = lineage.get("relationshipToParent")

                app_record = AppThreadRecord(
                    thread_id=thread_id,
                    project_id=row["project_id"],
                    title=row["title"] or "",
                    default_provider=row["default_provider"] or "",
                    provider_instance_id=row["provider_instance_id"] or payload.get("providerInstanceId"),
                    model=model,
                    thinking_level=thinking_level,
                    active_provider_thread_id=row["active_provider_thread_id"],
                    parent_thread_id=parent_thread_id,
                    root_thread_id=root_thread_id,
                    relationship_to_parent=relationship_to_parent,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    archived_at=row["archived_at"],
                    deleted_at=row["deleted_at"],
                )
                snapshot.threads[thread_id] = app_record

        # 2. orchestration_v2_projection_provider_threads
        if "orchestration_v2_projection_provider_threads" in tables:
            cur.execute("""
                SELECT provider_thread_id, thread_id, owner_node_id, provider, status,
                       updated_at, payload_json, driver, provider_instance_id
                FROM orchestration_v2_projection_provider_threads
            """)
            for row in cur.fetchall():
                pt_id = row["provider_thread_id"]
                th_id = row["thread_id"]
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    payload = {}

                driver = row["driver"] or payload.get("driver") or row["provider"] or ""
                prov_inst = row["provider_instance_id"] or payload.get("providerInstanceId") or ""
                native_ref = payload.get("nativeThreadRef") or {}
                native_id = native_ref.get("nativeId")

                context_usage = payload.get("contextUsage") or {}
                used_tokens = context_usage.get("usedTokens")
                max_tokens = context_usage.get("maxTokens")

                pt_record = ProviderThreadRecord(
                    provider_thread_id=pt_id,
                    thread_id=th_id,
                    driver=driver,
                    provider_instance_id=prov_inst,
                    native_id=native_id,
                    status=row["status"] or "idle",
                    used_tokens=int(used_tokens) if used_tokens is not None else None,
                    max_tokens=int(max_tokens) if max_tokens is not None else None,
                    updated_at=row["updated_at"],
                )
                snapshot.provider_threads[pt_id] = pt_record
                if th_id:
                    snapshot.provider_threads_by_thread.setdefault(th_id, []).append(pt_record)

        # 3. orchestration_v2_projection_provider_turns
        if "orchestration_v2_projection_provider_turns" in tables:
            cur.execute("""
                SELECT provider_turn_id, thread_id, provider_thread_id, node_id, ordinal,
                       status, started_at, completed_at, payload_json
                FROM orchestration_v2_projection_provider_turns
                ORDER BY ordinal ASC
            """)
            for row in cur.fetchall():
                th_id = row["thread_id"]
                started_at = row["started_at"]
                completed_at = row["completed_at"]
                duration = parse_iso_duration(started_at, completed_at)

                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    payload = {}

                native_ref = payload.get("nativeTurnRef") or {}
                driver = native_ref.get("driver") or ""
                native_turn_id = native_ref.get("nativeId")

                token_usage = payload.get("tokenUsage") or {}
                used_tokens = token_usage.get("usedTokens")
                max_tokens = token_usage.get("maxTokens")

                turn_token_usage = payload.get("turnTokenUsage")
                usage_provenance = "unavailable"
                usage_status = "unavailable"
                input_tokens = 0
                cache_read = 0
                cache_creation = 0
                output_tokens = 0
                thinking = 0
                response_tokens = 0

                if turn_token_usage is not None and isinstance(turn_token_usage, dict):
                    raw_status = turn_token_usage.get("usageStatus", "unavailable")
                    if raw_status != "unavailable":
                        usage_status = raw_status
                        usage_provenance = "exact"
                        input_tokens, cache_read, cache_creation, output_tokens, thinking = _usage_counts(turn_token_usage)
                        response_tokens = max(0, output_tokens - thinking)
                    else:
                        # turnTokenUsage explicitly unavailable, check fallback for non-antigravity
                        if driver != "antigravity" and (token_usage.get("inputTokens") or token_usage.get("outputTokens")):
                            input_tokens, cache_read, cache_creation, output_tokens, thinking = _usage_counts(token_usage)
                            response_tokens = max(0, output_tokens - thinking)
                            usage_status = "complete"
                            usage_provenance = "cumulative_fallback"
                        else:
                            usage_status = "unavailable"
                            usage_provenance = "unavailable"
                else:
                    # turnTokenUsage absent, fallback to tokenUsage if non-antigravity
                    if driver != "antigravity" and (token_usage.get("inputTokens") or token_usage.get("outputTokens")):
                        input_tokens, cache_read, cache_creation, output_tokens, thinking = _usage_counts(token_usage)
                        response_tokens = max(0, output_tokens - thinking)
                        usage_status = "complete"
                        usage_provenance = "cumulative_fallback"
                    else:
                        usage_status = "unavailable"
                        usage_provenance = "unavailable"

                turn_record = ProviderTurnRecord(
                    provider_turn_id=row["provider_turn_id"],
                    thread_id=th_id,
                    provider_thread_id=row["provider_thread_id"],
                    driver=driver,
                    native_turn_id=native_turn_id,
                    ordinal=row["ordinal"],
                    status=row["status"] or "completed",
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                    used_tokens=int(used_tokens) if used_tokens is not None else None,
                    max_tokens=int(max_tokens) if max_tokens is not None else None,
                    input_tokens=input_tokens,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                    thinking_tokens=thinking,
                    response_tokens=response_tokens,
                    output_tokens=output_tokens,
                    usage_status=usage_status,
                    usage_provenance=usage_provenance,
                    node_id=row["node_id"],
                )
                snapshot.provider_turns_by_thread.setdefault(th_id, []).append(turn_record)

        # 4. orchestration_v2_projection_subagents
        if "orchestration_v2_projection_subagents" in tables:
            cur.execute("""
                SELECT subagent_id, thread_id, parent_node_id, provider, provider_thread_id,
                       child_thread_id, origin, status, started_at, completed_at, updated_at,
                       payload_json, driver, provider_instance_id
                FROM orchestration_v2_projection_subagents
            """)
            for row in cur.fetchall():
                parent_th_id = row["thread_id"]
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except Exception:
                    payload = {}

                child_th_id = row["child_thread_id"] or payload.get("childThreadId")
                title = payload.get("title") or ""
                model = payload.get("model") or (payload.get("modelSelection") or {}).get("model")
                driver = row["driver"] or payload.get("driver") or row["provider"] or ""
                status = row["status"] or payload.get("status") or "completed"
                origin = row["origin"] or payload.get("origin") or "app_owned"
                prov_inst = row["provider_instance_id"] or payload.get("providerInstanceId")

                started_at = row["started_at"] or payload.get("startedAt")
                completed_at = row["completed_at"] or payload.get("completedAt")
                duration = parse_iso_duration(started_at, completed_at)

                sub_record = SubagentLinkRecord(
                    subagent_id=row["subagent_id"],
                    parent_thread_id=parent_th_id,
                    child_thread_id=child_th_id,
                    driver=driver,
                    provider_instance_id=prov_inst,
                    model=model,
                    title=title,
                    status=status,
                    origin=origin,
                    run_id=payload.get("runId"),
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_seconds=duration,
                )
                snapshot.subagents_by_parent.setdefault(parent_th_id, []).append(sub_record)

    finally:
        conn.close()

    return snapshot
