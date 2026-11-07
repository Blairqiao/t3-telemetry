"""Mirror T3 threads and provider session bindings into TelemetryStore."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import unquote

from t3_telemetry.providers import provider_key
from t3_telemetry.store import SessionRecord, StepRecord, utc_iso

if TYPE_CHECKING:
    from t3_telemetry.db_t3 import T3StateSnapshot
    from t3_telemetry.store import TelemetryStore


def mirror_t3_graph(store: TelemetryStore, snapshot: T3StateSnapshot) -> None:
    """Mirror snapshot app threads and provider session bindings into store."""
    raw_threads = snapshot.threads.values() if isinstance(snapshot.threads, dict) else snapshot.threads
    threads = [
        {
            "thread_id": t.thread_id,
            "title": t.title,
            "parent_thread_id": t.parent_thread_id,
            "root_thread_id": t.root_thread_id,
            "driver": t.default_provider,
            "provider_instance_id": t.provider_instance_id,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
        }
        for t in raw_threads
    ]

    raw_provider_threads = (
        snapshot.provider_threads.values()
        if isinstance(snapshot.provider_threads, dict)
        else snapshot.provider_threads
    )
    bindings: list[dict[str, str | None]] = []
    fallback_sessions: list[SessionRecord] = []
    fallback_steps: list[StepRecord] = []
    for pt in raw_provider_threads:
        if not pt.thread_id:
            continue
        provider = provider_key(pt.driver, pt.provider_instance_id)
        if provider is None:
            # No parser for this harness: keep its usage visible as "Other"
            # from the turn totals T3 itself recorded.
            session, steps = _fallback_session(snapshot, pt)
            if steps:
                fallback_sessions.append(session)
                fallback_steps.extend(steps)
                bindings.append({"session_key": session.session_key, "thread_id": pt.thread_id,
                                 "provider_thread_id": pt.provider_thread_id})
            continue
        if not pt.native_id or pt.native_id.startswith("pending:"):
            continue
        bindings.append({
            "session_key": f"{provider}:{pt.native_id}",
            "thread_id": pt.thread_id,
            "provider_thread_id": pt.provider_thread_id,
        })

    session_keys = set(store.sessions_by_key())
    for parent_thread_id, links in snapshot.subagents_by_parent.items():
        for link in links:
            if (
                link.origin != "provider_native"
                or provider_key(link.driver, link.provider_instance_id) != "claude"
                or not link.child_thread_id
            ):
                continue
            unquoted_id = unquote(link.subagent_id)
            if "task:" not in unquoted_id:
                continue
            agent_id = unquoted_id.rsplit("task:", 1)[1]
            candidates = [
                pt for pt in snapshot.provider_threads.values()
                if pt.thread_id == parent_thread_id
                and provider_key(pt.driver, pt.provider_instance_id) == "claude"
                and pt.native_id and not pt.native_id.startswith("pending:")
            ]
            # Native child IDs can encode their provider thread ID. Prefer that
            # direct reference over an inference from the available sessions.
            child_id = unquote(link.child_thread_id)
            referenced = [pt for pt in candidates if re.search(r"(?:^|:)" + re.escape(pt.provider_thread_id) + r"(?:$|:)", child_id)]
            matching = referenced or [
                pt for pt in candidates
                if f"claude:{pt.native_id}/agent-{agent_id}" in session_keys
            ]
            if not matching and len(candidates) == 1:
                matching = candidates
            if len(matching) != 1:
                continue
            parent_pt = matching[0]
            bindings.append({
                "session_key": f"claude:{parent_pt.native_id}/agent-{agent_id}",
                "thread_id": link.child_thread_id,
                "provider_thread_id": None,
            })

    if fallback_sessions:
        store.upsert_sessions(fallback_sessions)
        store.upsert_steps(fallback_steps)
    store.replace_t3_graph(threads, bindings)


FALLBACK_PROVIDER = "other"


def _fallback_session(snapshot: T3StateSnapshot, pt) -> tuple[SessionRecord, list[StepRecord]]:
    """Build an ``other:`` session from T3's recorded turn usage for an unparsed harness."""
    app = snapshot.threads.get(pt.thread_id)
    model = getattr(app, "model", None) if app else None
    key = f"{FALLBACK_PROVIDER}:{pt.provider_thread_id}"
    steps = []
    for turn in snapshot.provider_turns_by_thread.get(pt.thread_id, []):
        if turn.provider_thread_id != pt.provider_thread_id:
            continue
        ts = turn.completed_at or turn.started_at
        total = turn.input_tokens + turn.cache_read_tokens + turn.cache_creation_tokens + turn.output_tokens
        if not ts or total <= 0:
            continue
        steps.append(StepRecord(
            session_key=key, event_id=turn.provider_turn_id, ts=utc_iso(ts), model=model,
            input_tokens=turn.input_tokens, cache_read_tokens=turn.cache_read_tokens,
            cache_creation_tokens=turn.cache_creation_tokens, output_tokens=turn.output_tokens,
            thinking_tokens=turn.thinking_tokens,
        ))
    steps.sort(key=lambda step: step.ts)
    session = SessionRecord(
        session_key=key, provider=FALLBACK_PROVIDER, native_id=pt.native_id or pt.provider_thread_id,
        title=app.title if app else None, model=model, origin_hint="t3",
        first_ts=steps[0].ts if steps else None, last_ts=steps[-1].ts if steps else None,
    )
    return session, steps
