"""Presentation-level activity entries built from an agent's recorded steps.

An entry is the unit the activity feed selects and the detail screen inspects.
Identity never depends on list position or display text, so selection survives
polling, resizing, and the ``--max-steps`` window sliding forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from datetime import datetime

from t3_telemetry.models import ModelStep, SubagentLaunch, ThreadTelemetryNode, ToolCallDetail

SCOPE_PROVIDER_TURN = "provider turn"
SCOPE_INFERENCE_STEP = "inference step"
SCOPE_LAUNCH = "subagent launch"
KIND_LAUNCH = "subagent_launch"
UNDATED_LABEL = "Undated"

SCOPE_NOTES = {
    SCOPE_PROVIDER_TURN: (
        "Provider turn: totals cover the whole turn, including every tool call inside it. "
        "They are not per-tool usage."
    ),
    SCOPE_INFERENCE_STEP: (
        "Inference step: totals cover one model call. Tools it requested are not billed separately here."
    ),
    SCOPE_LAUNCH: (
        "Subagent launch: a marker, not a counted step. The child's tokens and time are in its own "
        "overview and in rollup totals, never in this agent's own totals."
    ),
}


@dataclass(frozen=True)
class ActivityEntry:
    """One selectable row of the activity feed.

    ``detail`` is the backing record; later features can attach richer detail
    without changing the feed. ``child_thread_id`` is the optional target when
    an entry stands for a launched subagent.
    """

    entry_id: str
    title: str
    kind: str
    status: str
    total_tokens: int | None
    duration_seconds: float
    accounting_scope: str
    detail: ModelStep | None = None
    child_thread_id: str | None = None
    usage_status: str = "complete"
    usage_provenance: str = "exact"
    launch: SubagentLaunch | None = None

    @property
    def is_launch(self) -> bool:
        return self.launch is not None

    @property
    def scope_note(self) -> str:
        return SCOPE_NOTES.get(self.accounting_scope, "")


# C0 controls (except tab/newline), DEL, and C1 controls, which can drive a terminal.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")

TOOL_VERBS = {
    "run_command": "Run command",
    "view_file": "Read file",
    "read_file": "Read file",
    "write_to_file": "Write file",
    "write_file": "Write file",
    "replace_file_content": "Edit file",
    "multi_replace_file_content": "Edit file",
    "edit_file": "Edit file",
    "list_dir": "List directory",
    "grep_search": "Search",
    "find_by_name": "Find files",
    "search_web": "Search web",
    "invoke_subagent": "Launch subagent",
    "send_message": "Send message",
    "manage_task": "Manage task",
}
# Order in which recognized fields describe a call; first present wins.
PRIMARY_FIELDS = ("command", "path", "query", "task", "recipient", "summary")
DETAIL_FIELD_LABELS = (
    ("command", "Command"),
    ("cwd", "Working dir"),
    ("path", "Path"),
    ("query", "Query"),
    ("task", "Task"),
    ("recipient", "Recipient"),
    ("summary", "Recorded summary"),
)


def neutralize(text: str, *, single_line: bool = False) -> str:
    """Make recorded text safe to print: control characters become visible ``\\xNN`` escapes."""
    out = _CONTROL_CHARS.sub(lambda m: f"\\x{ord(m.group()):02x}", text)
    if single_line:
        out = _WHITESPACE.sub(" ", out).strip()
    return out


def describe_tool(detail: ToolCallDetail) -> str:
    """Deterministic one-line description from recorded fields only."""
    verb = TOOL_VERBS.get(detail.name, detail.name)
    for key in PRIMARY_FIELDS:
        value = detail.fields.get(key)
        if value:
            suffix = "…" if key in detail.truncated else ""
            return f"{neutralize(verb, single_line=True)} · {neutralize(value, single_line=True)}{suffix}"
    # Name only: show the recorded name itself rather than a guessed verb.
    return f"{neutralize(detail.name, single_line=True)} · details unavailable"


def _step_description(step: ModelStep) -> str | None:
    details = step.tool_details or [ToolCallDetail(name=n) for n in step.tool_calls]
    if details:
        desc = describe_tool(details[0])
        extra = max(len(step.tool_calls), len(details)) - 1
        return f"{desc} (+{extra} more)" if extra > 0 else desc
    for rec in step.context_texts:
        text = neutralize(rec.text, single_line=True)
        if text:
            label = "Prompt" if rec.role == "user" else "Reply"
            return f"{label} · {text}"
    return None


def _native_identity(step: ModelStep) -> str:
    if step.provider_turn_id:
        return f"turn:{step.provider_turn_id}"
    return f"{step.activity_kind}:{step.source}:{step.step_index}"


def _title(step: ModelStep) -> str:
    desc = _step_description(step)
    ident = step.row_label.rsplit(" ", 1)[-1]  # compact "#N"; full identity is in the detail view
    if desc:
        return f"{desc} · {ident}"
    if step.activity_kind != "provider_turn":
        return f"{step.action_summary} · {ident}"  # e.g. "think + reply"
    return f"{step.row_label} · action details unavailable"


def _total_tokens(step: ModelStep) -> int | None:
    """None when usage is unavailable (never shown as zero) or not recorded yet on a running turn."""
    if step.usage_status == "unavailable":
        return None
    if (
        step.activity_kind == "provider_turn"
        and step.status == "running"
        and step.total_tokens == 0
        and step.thinking_tokens == 0
        and step.cache_creation_tokens == 0
    ):
        return None
    return step.total_tokens


def selectable_steps(node: ThreadTelemetryNode, max_steps: int) -> list[ModelStep]:
    """The node's own steps inside the ``--max-steps`` window (0 means all)."""
    steps = node.steps
    if max_steps and len(steps) > max_steps:
        return steps[-max_steps:]
    return steps


def _step_entries(node: ThreadTelemetryNode, steps: list[ModelStep]) -> list[ActivityEntry]:
    prefix = f"{node.thread_id}|{node.active_provider_thread_id or '-'}"
    seen: dict[str, int] = {}
    entries: list[ActivityEntry] = []
    for step in steps:
        # Turn IDs are already unique, so omit the provider thread: it may appear on a later poll.
        base = f"{node.thread_id}|{_native_identity(step)}" if step.provider_turn_id else f"{prefix}|{_native_identity(step)}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        entry_id = base if count == 0 else f"{base}#{count}"
        is_turn = step.activity_kind == "provider_turn"
        entries.append(
            ActivityEntry(
                entry_id=entry_id,
                title=_title(step),
                kind=step.activity_kind,
                status=step.status,
                total_tokens=_total_tokens(step),
                duration_seconds=step.duration_seconds,
                accounting_scope=SCOPE_PROVIDER_TURN if is_turn else SCOPE_INFERENCE_STEP,
                detail=step,
                usage_status=step.usage_status,
                usage_provenance=step.usage_provenance,
            )
        )
    return entries


def _epoch(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def launch_entry(node: ThreadTelemetryNode, launch: SubagentLaunch, undated: bool) -> ActivityEntry:
    """A navigable marker for one invocation; it carries no tokens and no duration."""
    task = neutralize(launch.title, single_line=True) or "task not recorded"
    target = neutralize(launch.child_native_id or launch.child_thread_id or "unknown child", single_line=True)
    title = f"Subagent launched · {task} · → {target[:8]}"
    if undated:
        title = f"{UNDATED_LABEL} · {title}"
    return ActivityEntry(
        entry_id=f"{node.thread_id}|launch:{launch.launch_id}",
        title=title,
        kind=KIND_LAUNCH,
        status=launch.status,
        total_tokens=None,
        duration_seconds=0.0,
        accounting_scope=SCOPE_LAUNCH,
        child_thread_id=launch.child_thread_id if launch.child_present else None,
        usage_status="unavailable",
        usage_provenance="unavailable",
        launch=launch,
    )


def build_activity_entries(node: ThreadTelemetryNode, max_steps: int = 20) -> list[ActivityEntry]:
    """Adapt the node's own steps (never combined subagent steps) into stable entries.

    ``--max-steps`` windows counted steps only; launch markers for immediate
    children are merged in afterwards and never evict normal activity.
    Placement: a marker sits right after the retained parent step that recorded
    its tool call; otherwise by recorded launch time (after steps with the same
    time, ties keep discovery order). Markers with neither form an explicitly
    labeled undated group at the top. All known launch markers are kept;
    ``max_steps`` limits counted activities only.
    """
    steps = selectable_steps(node, max_steps)
    step_entries = _step_entries(node, steps)
    if not node.launches:
        return step_entries

    retained_index = {s.step_index: i for i, s in enumerate(steps) if s.activity_kind != "provider_turn"}
    after_step: dict[int, list[SubagentLaunch]] = {}
    dated: list[tuple[float, int, SubagentLaunch]] = []
    undated: list[SubagentLaunch] = []
    for seq, launch in enumerate(node.launches):
        if launch.parent_step_index is not None and launch.parent_step_index in retained_index:
            after_step.setdefault(retained_index[launch.parent_step_index], []).append(launch)
            continue
        ts = _epoch(launch.launched_at)
        if ts is None:
            undated.append(launch)
        else:
            dated.append((ts, seq, launch))
    dated.sort(key=lambda item: (item[0], item[1]))
    out = [launch_entry(node, launch, undated=True) for launch in undated]
    pending = list(dated)
    for i, (step, entry) in enumerate(zip(steps, step_entries)):
        step_ts = _epoch(step.started_at or step.completed_at)
        while step_ts is not None and pending and pending[0][0] < step_ts:
            out.append(launch_entry(node, pending.pop(0)[2], undated=False))
        out.append(entry)
        out.extend(launch_entry(node, launch, undated=False) for launch in after_step.get(i, []))
    out.extend(launch_entry(node, item[2], undated=False) for item in pending)
    return out


def launch_targets(entries: list[ActivityEntry]) -> dict[str, str]:
    """Launch entry ID -> child node ID for launches that can be opened."""
    return {e.entry_id: e.child_thread_id for e in entries if e.is_launch and e.child_thread_id}
