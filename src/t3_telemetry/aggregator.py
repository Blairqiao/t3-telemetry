"""Subagent and delegation tree aggregator for T3 Code & Antigravity."""

from __future__ import annotations

from datetime import datetime

from t3_telemetry.db_t3 import T3StateSnapshot
from t3_telemetry.models import (
    ConversationTelemetry,
    NativeLaunchRecord,
    SubagentLaunch,
    ThreadTelemetryNode,
)


def snapshot_thread_status(snapshot: T3StateSnapshot, thread_id: str) -> str:
    """Resolve status using only the already loaded snapshot."""
    app_thread = snapshot.threads.get(thread_id)
    pt = snapshot.get_provider_thread(thread_id)
    turns = snapshot.provider_turns_by_thread.get(thread_id, [])
    active_id = (app_thread.active_provider_thread_id if app_thread else None) or (pt.provider_thread_id if pt else None) or (turns[-1].provider_thread_id if turns else None)
    if active_id and (not pt or pt.provider_thread_id != active_id):
        pt = snapshot.provider_threads.get(active_id) or pt
    base_status = pt.status if pt else "idle"
    if base_status == "running" or any(t.status == "running" for t in turns) or any(s.status == "running" for s in snapshot.subagents_by_parent.get(thread_id, [])):
        return "running"
    return base_status if base_status in ("idle", "completed", "failed", "cancelled") else "completed"


def running_root_ids(snapshot: T3StateSnapshot) -> list[str]:
    return [t.thread_id for t in snapshot.threads.values() if t.parent_thread_id is None and t.deleted_at is None and snapshot_thread_status(snapshot, t.thread_id) == "running"]


def parse_iso_timestamp(ts: str | None) -> float | None:
    """Parse ISO-8601 timestamp string into epoch float timestamp, or None."""
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


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
