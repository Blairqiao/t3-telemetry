"""Build thread telemetry from the persisted native-session store."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json

from t3_telemetry.aggregator import build_subagent_launches, snapshot_thread_status
from t3_telemetry.context_limits import context_limit_for
from t3_telemetry.db_t3 import T3StateSnapshot, parse_iso_duration
from t3_telemetry.models import (
    ContextSample, ModelStep, RecordedText, SubagentLaunch, ThreadTelemetryNode,
    TokenMetricsSummary, ToolCallDetail, TurnItem, TurnMetrics,
)
from t3_telemetry.providers import provider_key
from t3_telemetry.store import TelemetryStore


def _used(row) -> int:
    return sum(row[name] for name in ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens"))


def _own_data(keys, ctx, running):
    """Build steps, context and metrics for the node's bound sessions."""
    store, sessions = ctx["store"], ctx["sessions"]
    rows = store.steps_for(keys)
    turns = store.turns_for(keys)
    items = store.items_for(keys)
    grouped_steps = defaultdict(list)
    grouped_items = defaultdict(list)
    for row in rows:
        if row["turn_id"] is not None:
            grouped_steps[(row["session_key"], row["turn_id"])].append(row)
    for item in items:
        grouped_items[(item["session_key"], item["turn_id"])].append(item)

    # Orphan records retain their usage even when no enclosing turn was recorded.
    entries = [(turn["started_at"] or "", turn["session_key"], turn["turn_id"], turn, None) for turn in turns]
    entries.extend((row["ts"] or "", row["session_key"], row["event_id"], None, row)
                   for row in rows if row["turn_id"] is None)
    entries.sort(key=lambda entry: entry[:3])
    newest_turn = (turns[-1]["session_key"], turns[-1]["turn_id"]) if turns else None
    steps = []
    actual_turn_steps = []
    for index, (_, session_key, identity, turn, orphan) in enumerate(entries, 1):
        turn_rows = grouped_steps[(session_key, identity)] if turn is not None else [orphan]
        turn_items = grouped_items[(session_key, identity)] if turn is not None else []
        kind = turn["kind"] if turn is not None else "model_step"
        turn_key = f"{session_key}|{identity}" if turn is not None else None
        started = turn["started_at"] if turn is not None else orphan["ts"]
        completed = turn["completed_at"] if turn is not None else None
        totals = {name: sum(row[name] for row in turn_rows) for name in
                  ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens", "thinking_tokens")}
        texts = []
        if turn is not None:
            for role, column in (("user", "prompt"), ("assistant", "reply")):
                if turn[column]:
                    texts.append(RecordedText(role, turn[column], bool(turn[column + "_truncated"])))
        timeline = [TurnItem(item["kind"], item["text"], bool(item["truncated"]), item["status"],
                             item["started_at"], item["completed_at"],
                             parse_iso_duration(item["started_at"], item["completed_at"]))
                    for item in turn_items] if kind == "provider_turn" else []
        details = [ToolCallDetail(item["tool_name"] or item["text"], json.loads(item["fields_json"]),
                                  json.loads(item["truncated_fields_json"]))
                   for item in turn_items] if kind == "model_step" else []
        step = ModelStep(step_index=index, activity_kind=kind, provider_turn_id=turn_key,
                         turn_key=turn_key, status="running" if running and turn is not None and
                         (session_key, identity) == newest_turn else "completed",
                         **totals, response_tokens=totals["output_tokens"] - totals["thinking_tokens"],
                         model=turn_rows[-1]["model"] if turn_rows else None,
                         provider=sessions[session_key].provider, started_at=started, completed_at=completed,
                         duration_seconds=parse_iso_duration(started, completed),
                         usage_status="complete" if turn_rows else "unavailable", usage_provenance="exact",
                         context_used_tokens=_used(turn_rows[-1]) if turn_rows else None,
                         context_texts=texts, timeline=timeline, tool_details=details,
                         tool_calls=[detail.name for detail in details])
        steps.append(step)
        if turn is not None:
            actual_turn_steps.append(step)

    recent_key = max(keys, key=lambda key: (sessions[key].last_ts or "", key)) if keys else None
    recent_rows = [row for row in rows if row["session_key"] == recent_key]
    recent_turns = [turn for turn in turns if turn["session_key"] == recent_key and turn["context_limit"] is not None]
    measured_limit = recent_turns[-1]["context_limit"] if recent_turns else None
    limit = measured_limit if measured_limit is not None else context_limit_for(recent_rows[-1]["model"] if recent_rows else None)
    samples = [ContextSample(source_id=recent_key, used_tokens=_used(row), timestamp=row["ts"],
                             limit=limit, provenance="measured") for row in recent_rows]
    metrics = TokenMetricsSummary()
    for step in steps:
        metrics.add_step(step)
    current = None
    if actual_turn_steps:
        step = actual_turn_steps[-1]
        current = TurnMetrics(ordinal=step.step_index, turn_id=step.turn_key, status=step.status,
                              availability="available" if step.usage_status == "complete" else "unavailable",
                              provenance="exact", activity_kind=step.activity_kind,
                              input_tokens=step.input_tokens, cache_read_tokens=step.cache_read_tokens,
                              cache_creation_tokens=step.cache_creation_tokens, output_tokens=step.output_tokens,
                              thinking_tokens=step.thinking_tokens, response_tokens=step.response_tokens,
                              duration_seconds=step.duration_seconds, step_count=1,
                              provider_turn_count=int(step.activity_kind == "provider_turn"),
                              model_step_count=int(step.activity_kind == "model_step"))
    return steps, {
        "current_turn_metrics": current,
        "turn_metrics_availability": current.availability if current else "unavailable",
        "context_samples": samples,
        "context_used_tokens": samples[-1].used_tokens if samples else 0,
        "context_limit": limit,
        "context_limit_provenance": "measured" if limit is not None else "default",
        "context_provenance": "measured" if samples else "unknown",
        "latest_step_label": steps[-1].row_label if steps else None,
    }, metrics


def build_thread_node(thread_id: str, snapshot: T3StateSnapshot, store: TelemetryStore,
                      *, _visited: set[str] | None = None) -> ThreadTelemetryNode:
    """Build a telemetry tree, sharing store indexes across its descendants."""
    ctx = {"store": store, "sessions": store.sessions_by_key(), "bindings": store.bindings(),
           "syncing": store.get_meta("last_sync_at") is None}
    return _build_thread_node(thread_id, snapshot, ctx, _visited if _visited is not None else set())


def _build_thread_node(thread_id, snapshot, ctx, visited):
    if thread_id in visited:
        return ThreadTelemetryNode(thread_id=thread_id, title=thread_id, driver="",
                                   provider_instance_id=None, model="unknown", status="completed")
    visited.add(thread_id)
    sessions, bindings = ctx["sessions"], ctx["bindings"]
    keys = [key for key, owner in bindings.items() if owner == thread_id and key in sessions]
    keys.sort(key=lambda key: (sessions[key].first_ts or "", key))
    app = snapshot.threads.get(thread_id)
    pt = snapshot.get_provider_thread(thread_id)
    status = snapshot_thread_status(snapshot, thread_id)
    steps, fields, metrics = _own_data(keys, ctx, status == "running")
    syncing = ctx["syncing"]
    for provider_thread in snapshot.provider_threads_by_thread.get(thread_id, []):
        native_id = provider_thread.native_id
        if native_id and not native_id.startswith("pending:"):
            key = f"{provider_key(provider_thread.driver, provider_thread.provider_instance_id)}:{native_id}"
            if key not in sessions:
                syncing = True
    node = ThreadTelemetryNode(
        thread_id=thread_id, title=app.title if app else thread_id,
        driver=app.default_provider if app else "", provider_instance_id=app.provider_instance_id if app else None,
        model=(app.model or "unknown") if app else "unknown", thinking_level=app.thinking_level if app else None,
        updated_at=app.updated_at if app else None, status=status, native_id=pt.native_id if pt else None,
        active_provider_thread_id=pt.provider_thread_id if pt else None,
        provider=provider_key(app.default_provider, app.provider_instance_id) if app else (sessions[keys[0]].provider if keys else None),
        store_state="syncing" if syncing else "ready", steps=steps, combined_steps=list(steps),
        self_metrics=metrics, rollup_metrics=replace(metrics), **fields,
    )
    links = snapshot.subagents_by_parent.get(thread_id, [])
    processed_children = set()
    for link in links:
        child_id = link.child_thread_id
        if not child_id or child_id in visited or child_id in processed_children:
            continue
        processed_children.add(child_id)
        child = _build_thread_node(child_id, snapshot, ctx, visited)
        if link.title and (not child.title or child.title == child_id):
            child.title = link.title
        if link.model and child.model in ("unknown", "", None):
            child.model = link.model
        if link.status == "running":
            child.status = "running"
        if link.duration_seconds > 0 and child.self_metrics.duration_seconds == 0.0:
            child.self_metrics.duration_seconds = link.duration_seconds
            child.rollup_metrics.duration_seconds = max(child.rollup_metrics.duration_seconds, link.duration_seconds)
        node.subagents.append(child)

    for candidate in snapshot.threads.values():
        child_id = candidate.thread_id
        if candidate.parent_thread_id == thread_id and child_id not in processed_children and child_id not in visited:
            processed_children.add(child_id)
            node.subagents.append(_build_thread_node(child_id, snapshot, ctx, visited))

    native_children = []
    for key, session in sessions.items():
        if session.parent_session_key not in keys or key in bindings or session.native_id in visited:
            continue
        visited.add(session.native_id)
        child_steps, child_fields, child_metrics = _own_data([key], ctx, False)
        child = ThreadTelemetryNode(
            thread_id=session.native_id, title=session.title or f"Native subagent {session.native_id[:8]}",
            driver=node.driver, provider_instance_id=node.provider_instance_id,
            provider=session.provider, model=session.model or node.model, status="completed",
            native_id=session.native_id, active_provider_thread_id=session.native_id,
            store_state=node.store_state, steps=child_steps, combined_steps=list(child_steps),
            self_metrics=child_metrics, rollup_metrics=replace(child_metrics), **child_fields,
        )
        node.subagents.append(child)
        native_children.append((session, child))

    node.launches = build_subagent_launches(node, links, snapshot, None)
    for session, child in native_children:
        if not any(launch.child_thread_id == child.thread_id for launch in node.launches):
            node.launches.append(SubagentLaunch(
                launch_id=f"native:{session.native_id}", title=child.title, origin="native", status="completed",
                child_thread_id=session.native_id, child_native_id=session.native_id, child_present=True,
                launched_at=session.first_ts, launch_time_source="child session start", sources=["C"],
            ))

    node.rollup_metrics = TokenMetricsSummary()
    node.rollup_metrics.add_summary(node.self_metrics)
    for child in node.subagents:
        node.rollup_metrics.add_summary(child.rollup_metrics)
    total = len(node.subagents) + sum(child.rollup_metrics.total_subagents for child in node.subagents)
    active = sum((child.status == "running") + child.rollup_metrics.active_subagents for child in node.subagents)
    node.self_metrics.total_subagents = node.rollup_metrics.total_subagents = total
    node.self_metrics.active_subagents = node.rollup_metrics.active_subagents = active
    node.combined_steps = sorted(
        [*node.steps, *(step for child in node.subagents for step in child.combined_steps)],
        key=lambda step: (step.started_at or "", step.step_index),
    )
    return node
