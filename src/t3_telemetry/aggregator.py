"""Subagent and delegation tree aggregator for T3 Code & Antigravity."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from t3_telemetry.db_t3 import T3StateSnapshot, read_t3_state
from t3_telemetry.models import (
    ConversationTelemetry,
    ModelStep,
    ThreadTelemetryNode,
    TokenMetricsSummary,
)


def build_thread_telemetry(
    thread_id: str,
    t3_dir: Path | str,
    snapshot: T3StateSnapshot | None = None,
    agy_reader: Callable[[str, Path, str], ConversationTelemetry] | None = None,
    _visited: set[str] | None = None,
    source_label: str = "main",
) -> ThreadTelemetryNode:
    """Build a ThreadTelemetryNode for thread_id, linking subagents and aggregating metrics."""
    if _visited is None:
        _visited = set()

    if thread_id in _visited:
        return ThreadTelemetryNode(
            thread_id=thread_id,
            title=thread_id,
            driver="",
            provider_instance_id=None,
            model="unknown",
            status="completed",
        )
    _visited.add(thread_id)

    if snapshot is None:
        snapshot = read_t3_state(t3_dir)

    app_thread = snapshot.threads.get(thread_id)
    pt = snapshot.get_provider_thread(thread_id)
    turns = snapshot.provider_turns_by_thread.get(thread_id, [])

    # Basic metadata resolution
    if app_thread:
        title = app_thread.title
        driver = app_thread.default_provider
        provider_instance_id = app_thread.provider_instance_id
        model = app_thread.model or "unknown"
        thinking_level = app_thread.thinking_level
        updated_at = app_thread.updated_at
    else:
        title = thread_id
        driver = pt.driver if pt else ""
        provider_instance_id = pt.provider_instance_id if pt else None
        model = "unknown"
        thinking_level = None
        updated_at = pt.updated_at if pt else None

    if pt:
        if not driver:
            driver = pt.driver
        if not provider_instance_id:
            provider_instance_id = pt.provider_instance_id
        native_id = pt.native_id
        base_status = pt.status
    else:
        native_id = None
        base_status = "idle"

    subagent_links = snapshot.subagents_by_parent.get(thread_id, [])

    # Determine status
    is_running = (
        base_status == "running"
        or any(t.status == "running" for t in turns)
        or any(s.status == "running" for s in subagent_links)
    )
    status = "running" if is_running else (base_status if base_status in ("idle", "completed", "failed", "cancelled") else "completed")

    # Step extraction
    steps: list[ModelStep] = []
    agy_telemetry: ConversationTelemetry | None = None

    is_antigravity = driver == "antigravity" or (provider_instance_id and provider_instance_id.startswith("antigravity"))

    if is_antigravity:
        # Resolve Antigravity native ID
        if not native_id or native_id.startswith("pending:"):
            for t in turns:
                if t.native_turn_id and ":turn:" in t.native_turn_id:
                    native_id = t.native_turn_id.split(":turn:")[0]
                    break

        if native_id and native_id not in _visited:
            _visited.add(native_id)
            if agy_reader is not None:
                agy_reader_fn = agy_reader
            else:
                try:
                    from t3_telemetry.db_agy import read_agy_conversation
                    agy_reader_fn = read_agy_conversation
                except ImportError:
                    agy_reader_fn = None

            if agy_reader_fn is not None:
                try:
                    agy_telemetry = agy_reader_fn(native_id, Path(t3_dir), source_label)
                except Exception:
                    agy_telemetry = None

        if agy_telemetry:
            steps = list(agy_telemetry.steps)
            if (model == "unknown" or not model) and agy_telemetry.model_name:
                model = agy_telemetry.model_name
    else:
        # Claude, Codex, etc.
        for turn in turns:
            if turn.usage_status != "unavailable" or turn.input_tokens > 0 or turn.output_tokens > 0 or turn.status == "running":
                step = ModelStep(
                    step_index=turn.ordinal,
                    source=source_label,
                    status=turn.status,
                    input_tokens=turn.input_tokens,
                    cache_read_tokens=turn.cache_read_tokens,
                    cache_creation_tokens=turn.cache_creation_tokens,
                    thinking_tokens=turn.thinking_tokens,
                    response_tokens=turn.response_tokens,
                    output_tokens=turn.output_tokens,
                    tool_calls=[f"turn #{turn.ordinal}"],
                    started_at=turn.started_at,
                    completed_at=turn.completed_at,
                    duration_seconds=turn.duration_seconds,
                )
                steps.append(step)

    # Context window usage
    context_used_tokens = pt.used_tokens if pt and pt.used_tokens is not None else 0
    context_max_tokens = pt.max_tokens if pt and pt.max_tokens is not None else 1_048_576

    for turn in turns:
        if turn.used_tokens is not None and turn.used_tokens > context_used_tokens:
            context_used_tokens = turn.used_tokens
        if turn.max_tokens is not None and turn.max_tokens > 0:
            context_max_tokens = turn.max_tokens

    if is_antigravity and agy_telemetry:
        if context_used_tokens == 0 or agy_telemetry.latest_context_tokens > context_used_tokens:
            context_used_tokens = agy_telemetry.latest_context_tokens

    if context_max_tokens is None or context_max_tokens <= 0:
        context_max_tokens = 1_048_576

    # Self metrics
    self_metrics = TokenMetricsSummary()
    for s in steps:
        self_metrics.add_step(s)

    node = ThreadTelemetryNode(
        thread_id=thread_id,
        title=title,
        driver=driver,
        provider_instance_id=provider_instance_id,
        model=model,
        thinking_level=thinking_level,
        native_id=native_id,
        status=status,
        updated_at=updated_at,
        context_used_tokens=context_used_tokens,
        context_max_tokens=context_max_tokens,
        self_metrics=self_metrics,
        steps=steps,
    )

    # Discover and build subagents
    processed_children: set[str] = set()
    child_counter = 1

    # Source A: orchestration_v2_projection_subagents
    for sub_link in subagent_links:
        child_id = sub_link.child_thread_id
        if child_id and child_id not in _visited and child_id not in processed_children:
            processed_children.add(child_id)
            child_label = f"sub-{child_counter}"
            child_counter += 1
            child_node = build_thread_telemetry(
                child_id,
                t3_dir,
                snapshot=snapshot,
                agy_reader=agy_reader,
                _visited=_visited,
                source_label=child_label,
            )
            # Inherit link metadata if child node lacks it
            if sub_link.title and (not child_node.title or child_node.title == child_id):
                child_node.title = sub_link.title
            if sub_link.model and child_node.model in ("unknown", "", None):
                child_node.model = sub_link.model
            if sub_link.status and sub_link.status == "running":
                child_node.status = "running"
            if sub_link.duration_seconds > 0 and child_node.self_metrics.duration_seconds == 0.0:
                child_node.self_metrics.duration_seconds = sub_link.duration_seconds
                child_node.rollup_metrics.duration_seconds = max(
                    child_node.rollup_metrics.duration_seconds, sub_link.duration_seconds
                )
            node.subagents.append(child_node)

    # Source B: AppThreadRecord with parent_thread_id == thread_id
    for candidate_th in snapshot.threads.values():
        if (
            candidate_th.parent_thread_id == thread_id
            and candidate_th.thread_id not in processed_children
            and candidate_th.thread_id not in _visited
        ):
            processed_children.add(candidate_th.thread_id)
            child_label = f"sub-{child_counter}"
            child_counter += 1
            child_node = build_thread_telemetry(
                candidate_th.thread_id,
                t3_dir,
                snapshot=snapshot,
                agy_reader=agy_reader,
                _visited=_visited,
                source_label=child_label,
            )
            node.subagents.append(child_node)

    # Source C: Native Antigravity child conversations
    if agy_telemetry and agy_telemetry.child_native_ids:
        existing_native_ids = {c.native_id for c in node.subagents if c.native_id}
        for child_native_id in agy_telemetry.child_native_ids:
            if child_native_id in existing_native_ids or child_native_id in _visited:
                continue

            child_label = f"sub-{child_counter}"
            child_counter += 1
            _visited.add(child_native_id)

            if agy_reader is not None:
                agy_reader_fn = agy_reader
            else:
                try:
                    from t3_telemetry.db_agy import read_agy_conversation
                    agy_reader_fn = read_agy_conversation
                except ImportError:
                    agy_reader_fn = None

            child_conv = None
            if agy_reader_fn is not None:
                try:
                    child_conv = agy_reader_fn(child_native_id, Path(t3_dir), child_label)
                except Exception:
                    child_conv = None

            child_steps = list(child_conv.steps) if child_conv else []
            c_self = TokenMetricsSummary()
            for cs in child_steps:
                c_self.add_step(cs)

            c_rollup = TokenMetricsSummary()
            c_rollup.add_summary(c_self)

            child_node = ThreadTelemetryNode(
                thread_id=child_native_id,
                title=f"Native subagent {child_native_id[:8]}",
                driver="antigravity",
                provider_instance_id=provider_instance_id,
                model=child_conv.model_name or model if child_conv else model,
                native_id=child_native_id,
                status="completed",
                self_metrics=c_self,
                rollup_metrics=c_rollup,
                steps=child_steps,
                combined_steps=list(child_steps),
            )
            node.subagents.append(child_node)

    # Compute rollup metrics
    node.rollup_metrics = TokenMetricsSummary()
    node.rollup_metrics.add_summary(node.self_metrics)
    for child in node.subagents:
        node.rollup_metrics.add_summary(child.rollup_metrics)

    # Direct + transitive subagents count
    total_subagents = len(node.subagents) + sum(c.rollup_metrics.total_subagents for c in node.subagents)
    active_subagents = sum(
        (1 if c.status == "running" else 0) + c.rollup_metrics.active_subagents
        for c in node.subagents
    )
    node.self_metrics.total_subagents = total_subagents
    node.self_metrics.active_subagents = active_subagents
    node.rollup_metrics.total_subagents = total_subagents
    node.rollup_metrics.active_subagents = active_subagents

    # Populate combined_steps
    all_steps = list(node.steps)

    def _collect_descendants(n: ThreadTelemetryNode) -> None:
        for c in n.subagents:
            all_steps.extend(c.steps)
            _collect_descendants(c)

    _collect_descendants(node)
    node.combined_steps = sorted(all_steps, key=lambda s: (s.started_at or "", s.step_index))

    return node


def list_root_threads(
    t3_dir: Path | str,
    limit: int = 20,
    agy_reader: Callable[[str, Path, str], ConversationTelemetry] | None = None,
) -> list[ThreadTelemetryNode]:
    """List root threads sorted by updated_at descending."""
    snapshot = read_t3_state(t3_dir)
    root_candidates = [
        t for t in snapshot.threads.values()
        if t.parent_thread_id is None and t.deleted_at is None
    ]
    # Sort descending by updated_at or created_at
    root_candidates.sort(key=lambda t: t.updated_at or t.created_at or "", reverse=True)
    selected = root_candidates[:limit]

    nodes = [
        build_thread_telemetry(t.thread_id, t3_dir, snapshot=snapshot, agy_reader=agy_reader)
        for t in selected
    ]
    return nodes


def resolve_thread(
    t3_dir: Path | str,
    query: str | None = None,
    agy_reader: Callable[[str, Path, str], ConversationTelemetry] | None = None,
) -> ThreadTelemetryNode | None:
    """Resolve a thread by query (or latest active root thread if query is None)."""
    snapshot = read_t3_state(t3_dir)
    if not snapshot.threads:
        return None

    if query is None or query.strip() == "":
        root_candidates = [
            t for t in snapshot.threads.values()
            if t.parent_thread_id is None and t.deleted_at is None
        ]
        if not root_candidates:
            root_candidates = [t for t in snapshot.threads.values() if t.deleted_at is None]
        if not root_candidates:
            return None

        root_candidates.sort(key=lambda t: t.updated_at or t.created_at or "", reverse=True)
        return build_thread_telemetry(root_candidates[0].thread_id, t3_dir, snapshot=snapshot, agy_reader=agy_reader)

    q = query.strip().lower()

    # 1. Exact thread_id
    for t in snapshot.threads.values():
        if t.thread_id == query.strip():
            return build_thread_telemetry(t.thread_id, t3_dir, snapshot=snapshot, agy_reader=agy_reader)

    # 2. Prefix thread_id
    matches = [t for t in snapshot.threads.values() if t.thread_id.lower().startswith(q)]
    if matches:
        matches.sort(key=lambda t: t.updated_at or t.created_at or "", reverse=True)
        return build_thread_telemetry(matches[0].thread_id, t3_dir, snapshot=snapshot, agy_reader=agy_reader)

    # 3. Provider native_id match
    for pt in snapshot.provider_threads.values():
        if pt.native_id and pt.native_id.lower().startswith(q) and pt.thread_id in snapshot.threads:
            return build_thread_telemetry(pt.thread_id, t3_dir, snapshot=snapshot, agy_reader=agy_reader)

    # 4. Title substring (case-insensitive)
    title_matches = [t for t in snapshot.threads.values() if q in (t.title or "").lower()]
    if title_matches:
        title_matches.sort(key=lambda t: t.updated_at or t.created_at or "", reverse=True)
        return build_thread_telemetry(title_matches[0].thread_id, t3_dir, snapshot=snapshot, agy_reader=agy_reader)

    return None
