"""Subagent and delegation tree aggregator for T3 Code & Antigravity."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from t3_telemetry.db_t3 import T3StateSnapshot, read_t3_state
from datetime import datetime

from t3_telemetry.models import (
    ContextSample,
    ConversationTelemetry,
    ModelStep,
    NativeLaunchRecord,
    SubagentLaunch,
    ThreadTelemetryNode,
    TokenMetricsSummary,
    TurnMetrics,
)


def parse_iso_timestamp(ts: str | None) -> float | None:
    """Parse ISO-8601 timestamp string into epoch float timestamp, or None."""
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def extract_turn_metrics(
    driver: str,
    status: str,
    turns: list,
) -> tuple[TurnMetrics, str]:
    """Extract current or latest completed provider turn metrics.

    Returns (turn_metrics, availability), where availability is 'available', 'partial', or 'unavailable'.
    """
    if driver == "antigravity":
        return TurnMetrics(availability="unavailable", provenance="unavailable"), "unavailable"

    if status == "running":
        running_turns = [t for t in turns if t.status == "running"]
        target = running_turns[-1] if running_turns else (turns[-1] if turns else None)
    else:
        completed_turns = [t for t in turns if t.status in ("completed", "idle")]
        target = completed_turns[-1] if completed_turns else (turns[-1] if turns else None)

    if target is None:
        return TurnMetrics(availability="unavailable", provenance="unavailable"), "unavailable"

    if target.usage_provenance == "cumulative_fallback":
        return (
            TurnMetrics(
                ordinal=target.ordinal,
                turn_id=target.provider_turn_id,
                status=target.status,
                availability="unavailable",
                provenance="cumulative_fallback",
                duration_seconds=target.duration_seconds,
            ),
            "unavailable",
        )

    if target.usage_provenance == "exact":
        if target.usage_status == "complete":
            return (
                TurnMetrics(
                    ordinal=target.ordinal,
                    turn_id=target.provider_turn_id,
                    status=target.status,
                    availability="available",
                    provenance="exact",
                    input_tokens=target.input_tokens,
                    cache_read_tokens=target.cache_read_tokens,
                    cache_creation_tokens=target.cache_creation_tokens,
                    thinking_tokens=target.thinking_tokens,
                    response_tokens=target.response_tokens,
                    output_tokens=target.output_tokens,
                    duration_seconds=target.duration_seconds,
                    step_count=1,
                    provider_turn_count=1,
                ),
                "available",
            )
        elif target.usage_status == "partial":
            return (
                TurnMetrics(
                    ordinal=target.ordinal,
                    turn_id=target.provider_turn_id,
                    status=target.status,
                    availability="partial",
                    provenance="exact",
                    input_tokens=target.input_tokens,
                    cache_read_tokens=target.cache_read_tokens,
                    cache_creation_tokens=target.cache_creation_tokens,
                    thinking_tokens=target.thinking_tokens,
                    response_tokens=target.response_tokens,
                    output_tokens=target.output_tokens,
                    duration_seconds=target.duration_seconds,
                    step_count=1,
                    provider_turn_count=1,
                ),
                "partial",
            )
        else:
            return (
                TurnMetrics(
                    ordinal=target.ordinal,
                    turn_id=target.provider_turn_id,
                    status=target.status,
                    availability="unavailable",
                    provenance="exact",
                    duration_seconds=target.duration_seconds,
                ),
                "unavailable",
            )

    return (
        TurnMetrics(
            ordinal=target.ordinal,
            turn_id=target.provider_turn_id,
            status=target.status,
            availability="unavailable",
            provenance="unavailable",
            duration_seconds=target.duration_seconds,
        ),
        "unavailable",
    )


def extract_context_telemetry(
    *,
    turns: list[ProviderTurnRecord] | None = None,
    active_pt_id: str | None = None,
    pt: ProviderThreadRecord | None = None,
    is_antigravity: bool = False,
    agy_steps: list[ModelStep] | None = None,
    native_id: str | None = None,
) -> tuple[int, int, int | None, str, str, bool, list[ContextSample], str | None]:
    """Reconstruct context history and determine freshest applicable reading.

    Returns:
        (context_used_tokens, context_max_tokens, context_limit, context_provenance,
         context_limit_provenance, context_uncertainty, context_samples, latest_step_label)
    """
    turns = turns or []
    agy_steps = agy_steps or []

    # 1. Filter provider turns to active provider thread
    if active_pt_id:
        turns_for_active_pt = [t for t in turns if t.provider_thread_id == active_pt_id]
    else:
        turns_for_active_pt = list(turns)

    pt_for_active = pt if (pt and (not active_pt_id or pt.provider_thread_id == active_pt_id)) else None

    # 2. Collect measured samples
    measured_samples: list[ContextSample] = []
    for turn in turns_for_active_pt:
        if turn.used_tokens is not None:
            limit = turn.max_tokens if (turn.max_tokens is not None and turn.max_tokens > 0) else None
            ts = turn.completed_at or turn.started_at
            measured_samples.append(
                ContextSample(
                    source_id=turn.provider_turn_id,
                    used_tokens=turn.used_tokens,
                    timestamp=ts,
                    limit=limit,
                    provenance="measured",
                    provider_thread_id=turn.provider_thread_id,
                )
            )

    if not measured_samples and pt_for_active and pt_for_active.used_tokens is not None:
        limit = pt_for_active.max_tokens if (pt_for_active.max_tokens is not None and pt_for_active.max_tokens > 0) else None
        measured_samples.append(
            ContextSample(
                source_id=f"pt:{pt_for_active.provider_thread_id}",
                used_tokens=pt_for_active.used_tokens,
                timestamp=pt_for_active.updated_at,
                limit=limit,
                provenance="measured",
                provider_thread_id=pt_for_active.provider_thread_id,
            )
        )

    # 3. Collect estimated samples from Antigravity
    estimated_samples: list[ContextSample] = []
    if is_antigravity and agy_steps:
        for s in agy_steps:
            inf_tok = s.input_tokens + s.cache_read_tokens + s.output_tokens
            ts = s.completed_at or s.started_at
            source_id = f"{native_id}:step:{s.step_index}" if native_id else f"step:{s.step_index}"
            estimated_samples.append(
                ContextSample(
                    source_id=source_id,
                    used_tokens=inf_tok,
                    timestamp=ts,
                    limit=None,
                    provenance="estimated",
                    provider_thread_id=active_pt_id,
                )
            )

    # 4. Determine freshest applicable sample and recency uncertainty
    latest_measured: ContextSample | None = None
    uncertainty = False

    if measured_samples:
        ts_values = [parse_iso_timestamp(s.timestamp) for s in measured_samples]
        if all(t is not None for t in ts_values):
            measured_samples.sort(key=lambda s: parse_iso_timestamp(s.timestamp) or 0.0)
            latest_measured = measured_samples[-1]
            ts_set = {s.timestamp for s in measured_samples}
            uncertainty = (len(ts_set) < len(measured_samples))
        else:
            latest_measured = measured_samples[-1]
            uncertainty = True

    latest_estimated: ContextSample | None = None
    if estimated_samples:
        latest_estimated = estimated_samples[-1]

    # Rule: Measured provider context wins over rough Antigravity estimate unless provably stale
    context_uncertainty = False
    if latest_measured is not None and latest_estimated is not None:
        t_meas = parse_iso_timestamp(latest_measured.timestamp)
        t_est = parse_iso_timestamp(latest_estimated.timestamp)
        if t_meas is not None and t_est is not None and t_est > t_meas:
            selected_sample = latest_estimated
        else:
            selected_sample = latest_measured
            context_uncertainty = uncertainty
    elif latest_measured is not None:
        selected_sample = latest_measured
        context_uncertainty = uncertainty
    elif latest_estimated is not None:
        selected_sample = latest_estimated
    else:
        selected_sample = None

    # 5. Populate context fields and samples
    all_samples: list[ContextSample] = []
    if measured_samples and estimated_samples:
        all_samples = list(measured_samples) + list(estimated_samples)
        all_samples.sort(key=lambda s: parse_iso_timestamp(s.timestamp) or 0.0)
    elif measured_samples:
        all_samples = list(measured_samples)
    elif estimated_samples:
        all_samples = list(estimated_samples)

    context_samples = all_samples[-50:]

    if selected_sample is not None:
        context_used_tokens = selected_sample.used_tokens
        context_provenance = selected_sample.provenance
        if selected_sample.limit is not None and selected_sample.limit > 0:
            context_limit = selected_sample.limit
            context_max_tokens = selected_sample.limit
            context_limit_provenance = "measured"
        else:
            context_limit = None
            context_max_tokens = 1_048_576
            context_limit_provenance = "default"
    else:
        context_used_tokens = 0
        context_provenance = "unknown"
        context_limit = None
        context_max_tokens = 1_048_576
        context_limit_provenance = "default"

    latest_step_label = None
    if is_antigravity and agy_steps:
        latest_step_label = f"Model step #{agy_steps[-1].step_index}"

    return (
        context_used_tokens,
        context_max_tokens,
        context_limit,
        context_provenance,
        context_limit_provenance,
        context_uncertainty,
        context_samples,
        latest_step_label,
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

    # Identify active provider thread
    active_pt_id = None
    if app_thread and app_thread.active_provider_thread_id:
        active_pt_id = app_thread.active_provider_thread_id
    elif pt and pt.provider_thread_id:
        active_pt_id = pt.provider_thread_id
    elif turns:
        active_pt_id = turns[-1].provider_thread_id

    # If pt is not for active_pt_id, lookup matching pt
    if active_pt_id and (not pt or pt.provider_thread_id != active_pt_id):
        pt_candidate = snapshot.provider_threads.get(active_pt_id)
        if pt_candidate:
            pt = pt_candidate

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
                    activity_kind="provider_turn",
                    provider_turn_id=turn.provider_turn_id,
                    input_tokens=turn.input_tokens,
                    cache_read_tokens=turn.cache_read_tokens,
                    cache_creation_tokens=turn.cache_creation_tokens,
                    thinking_tokens=turn.thinking_tokens,
                    response_tokens=turn.response_tokens,
                    output_tokens=turn.output_tokens,
                    tool_calls=[],
                    started_at=turn.started_at,
                    completed_at=turn.completed_at,
                    duration_seconds=turn.duration_seconds,
                    context_texts=list(snapshot.turn_context.get(turn.provider_turn_id, [])),
                    usage_status=turn.usage_status,
                    usage_provenance=turn.usage_provenance,
                )
                steps.append(step)

    (
        context_used_tokens,
        context_max_tokens,
        context_limit,
        context_provenance,
        context_limit_provenance,
        context_uncertainty,
        context_samples,
        latest_step_label,
    ) = extract_context_telemetry(
        turns=turns,
        active_pt_id=active_pt_id,
        pt=pt,
        is_antigravity=is_antigravity,
        agy_steps=agy_telemetry.steps if (is_antigravity and agy_telemetry) else None,
        native_id=native_id,
    )

    # Self metrics
    self_metrics = TokenMetricsSummary()
    for s in steps:
        self_metrics.add_step(s)

    # Current / Latest provider turn metrics
    current_turn_metrics, turn_metrics_availability = extract_turn_metrics(driver, status, turns)



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
        active_provider_thread_id=active_pt_id,
        context_used_tokens=context_used_tokens,
        context_max_tokens=context_max_tokens,
        context_limit=context_limit,
        context_provenance=context_provenance,
        context_limit_provenance=context_limit_provenance,
        context_uncertainty=context_uncertainty,
        context_samples=context_samples,
        self_metrics=self_metrics,
        current_turn_metrics=current_turn_metrics,
        turn_metrics_availability=turn_metrics_availability,
        latest_step_label=latest_step_label,
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

            (
                c_used_tokens,
                c_max_tokens,
                c_limit,
                c_provenance,
                c_limit_provenance,
                c_uncertainty,
                c_context_samples,
                c_latest_step_label,
            ) = extract_context_telemetry(
                is_antigravity=True,
                agy_steps=child_steps,
                native_id=child_native_id,
                active_pt_id=child_native_id,
            )

            child_node = ThreadTelemetryNode(
                thread_id=child_native_id,
                title=f"Native subagent {child_native_id[:8]}",
                driver="antigravity",
                provider_instance_id=provider_instance_id,
                model=child_conv.model_name or model if child_conv else model,
                native_id=child_native_id,
                status="completed",
                active_provider_thread_id=child_native_id,
                context_used_tokens=c_used_tokens,
                context_max_tokens=c_max_tokens,
                context_limit=c_limit,
                context_provenance=c_provenance,
                context_limit_provenance=c_limit_provenance,
                context_uncertainty=c_uncertainty,
                context_samples=c_context_samples,
                latest_step_label=c_latest_step_label,
                self_metrics=c_self,
                rollup_metrics=c_rollup,
                steps=child_steps,
                combined_steps=list(child_steps),
            )
            node.subagents.append(child_node)

    node.launches = build_subagent_launches(node, subagent_links, snapshot, agy_telemetry)

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


def build_subagent_launches(
    node: ThreadTelemetryNode,
    subagent_links: list,
    snapshot: T3StateSnapshot,
    agy_telemetry: ConversationTelemetry | None,
) -> list[SubagentLaunch]:
    """One launch per known invocation of an immediate child, merged across Sources A/B/C.

    Identity: Source A uses ``subagent_id``, so repeated links to one child stay
    distinct. Source C uses the child conversation ID. Source B lineage has no
    invocation ID, so it falls back to the parent/child pair (``child:<id>``) and
    merges into any A or C launch that already targets that child. Times are only
    copied from records; nothing is inferred.
    """
    present = {c.thread_id for c in node.subagents}
    native_to_thread = {c.native_id: c.thread_id for c in node.subagents if c.native_id}
    status_by_id = {c.thread_id: c.status for c in node.subagents}
    launches: list[SubagentLaunch] = []
    by_id: dict[str, SubagentLaunch] = {}
    by_target: dict[str, SubagentLaunch] = {}

    def add(launch: SubagentLaunch) -> None:
        launch.child_present = bool(launch.child_thread_id and launch.child_thread_id in present)
        by_id[launch.launch_id] = launch
        if launch.child_thread_id:
            by_target.setdefault(launch.child_thread_id, launch)
        launches.append(launch)

    # Source A: app-owned or provider-native subagent records.
    for link_index, link in enumerate(subagent_links):
        launch_id = (
            f"subagent:{link.subagent_id}"
            if link.subagent_id
            else f"child:{link.child_thread_id or f'{node.thread_id}:unlinked:{link_index}'}"
        )
        if launch_id in by_id:
            continue  # same record read twice
        add(
            SubagentLaunch(
                launch_id=launch_id,
                title=link.title or "",
                origin=link.origin or "app_owned",
                status=link.status or "unknown",
                child_thread_id=link.child_thread_id,
                launched_at=link.started_at,
                launch_time_source="subagent record start" if link.started_at else None,
                sources=["A"],
            )
        )

    # Source B: thread lineage (discovery only).
    for th in snapshot.threads.values():
        if th.parent_thread_id != node.thread_id or th.thread_id == node.thread_id:
            continue
        existing = by_target.get(th.thread_id)
        if existing is not None:
            if "B" not in existing.sources:
                existing.sources.append("B")
            continue
        add(
            SubagentLaunch(
                launch_id=f"child:{th.thread_id}",
                title=th.title or "",
                origin="lineage",
                status=status_by_id.get(th.thread_id, "unknown"),
                child_thread_id=th.thread_id,
                launched_at=th.created_at,
                launch_time_source="child thread created" if th.created_at else None,
                sources=["B"],
            )
        )

    # Source C: native Antigravity transcript launches.
    if agy_telemetry is not None:
        records = list(agy_telemetry.launches)
        recorded = {r.child_native_id for r in records}
        records += [
            NativeLaunchRecord(child_native_id=cid) for cid in agy_telemetry.child_native_ids if cid not in recorded
        ]
        for rec in records:
            target = native_to_thread.get(rec.child_native_id, rec.child_native_id)
            existing = by_id.get(f"native:{rec.child_native_id}") or by_target.get(target)
            if existing is not None:
                if "C" not in existing.sources:
                    existing.sources.append("C")
                existing.child_native_id = existing.child_native_id or rec.child_native_id
                if existing.parent_step_index is None:
                    existing.parent_step_index = rec.step_index
                    existing.tool_name = rec.tool_name
                # A recorded tool-call time beats a lineage proxy (child thread creation).
                if rec.launched_at and (
                    existing.launched_at is None or existing.launch_time_source == "child thread created"
                ):
                    existing.launched_at = rec.launched_at
                    existing.launch_time_source = "transcript tool call"
                if not existing.title and rec.task:
                    existing.title = rec.task
                continue
            add(
                SubagentLaunch(
                    launch_id=f"native:{rec.child_native_id}",
                    title=rec.task or "",
                    origin="native",
                    status=status_by_id.get(target, "unknown"),
                    child_thread_id=target,
                    child_native_id=rec.child_native_id,
                    launched_at=rec.launched_at,
                    launch_time_source="transcript tool call" if rec.launched_at else None,
                    parent_step_index=rec.step_index,
                    tool_name=rec.tool_name,
                    sources=["C"],
                )
            )
    return launches


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
    return resolve_thread_in_snapshot(t3_dir, snapshot, query, agy_reader)


def resolve_thread_id_in_snapshot(snapshot: T3StateSnapshot, query: str | None = None) -> str | None:
    """Resolve a thread ID purely from a snapshot without building telemetry."""
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
        return root_candidates[0].thread_id

    q = query.strip().lower()

    # 1. Exact thread_id
    for t in snapshot.threads.values():
        if t.thread_id == query.strip():
            return t.thread_id

    # 2. Prefix thread_id
    matches = [t for t in snapshot.threads.values() if t.thread_id.lower().startswith(q)]
    if matches:
        matches.sort(key=lambda t: t.updated_at or t.created_at or "", reverse=True)
        return matches[0].thread_id

    # 3. Provider native_id match
    for pt in snapshot.provider_threads.values():
        if pt.native_id and pt.native_id.lower().startswith(q) and pt.thread_id in snapshot.threads:
            return pt.thread_id

    # 4. Title substring (case-insensitive)
    title_matches = [t for t in snapshot.threads.values() if q in (t.title or "").lower()]
    if title_matches:
        title_matches.sort(key=lambda t: t.updated_at or t.created_at or "", reverse=True)
        return title_matches[0].thread_id

    return None


def resolve_thread_in_snapshot(
    t3_dir: Path | str,
    snapshot: T3StateSnapshot,
    query: str | None = None,
    agy_reader: Callable[[str, Path, str], ConversationTelemetry] | None = None,
) -> ThreadTelemetryNode | None:
    """Resolve a query using an already loaded snapshot and return built node."""
    thread_id = resolve_thread_id_in_snapshot(snapshot, query)
    if thread_id is None:
        return None
    return build_thread_telemetry(thread_id, t3_dir, snapshot=snapshot, agy_reader=agy_reader)
