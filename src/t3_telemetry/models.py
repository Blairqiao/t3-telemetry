"""Shared domain dataclasses for T3 Code and Antigravity telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCallDetail:
    """Recognized fields from one recorded tool call.

    ``fields`` uses normalized keys (``command``, ``cwd``, ``path``, ``query``,
    ``task``, ``summary``, ``recipient``) holding bounded recorded text.
    ``truncated`` lists keys whose text was cut at the retention bound.
    """

    name: str
    fields: dict[str, str] = field(default_factory=dict)
    truncated: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RecordedText:
    """Bounded message text explicitly associated with an activity (role is ``user`` or ``assistant``)."""

    role: str
    text: str
    truncated: bool = False


@dataclass(slots=True)
class TurnItem:
    """One recorded action inside a provider turn."""

    kind: str
    text: str = ""
    truncated: bool = False
    status: str = "completed"
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float = 0.0


@dataclass(slots=True)
class ModelStep:
    """A single inference step (Antigravity step_type=15 or Claude/Codex provider turn)."""

    step_index: int
    source: str = "main"
    status: str = "completed"
    activity_kind: str = "model_step"
    provider_turn_id: str | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thinking_tokens: int = 0
    response_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[str] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float = 0.0
    spawned_subagent_ids: list[str] = field(default_factory=list)
    # Optional recorded context; tool_calls stays the list of names.
    tool_details: list[ToolCallDetail] = field(default_factory=list)
    context_texts: list[RecordedText] = field(default_factory=list)
    # "complete", "partial" or "unavailable"; provenance "exact", "cumulative_fallback", "unavailable".
    usage_status: str = "complete"
    usage_provenance: str = "exact"
    context_used_tokens: int | None = None
    context_max_tokens: int | None = None
    timeline: list[TurnItem] = field(default_factory=list)
    model: str | None = None
    provider: str | None = None
    turn_key: str | None = None

    @property
    def kind(self) -> str:
        return self.activity_kind

    @property
    def new_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens

    @property
    def context_read_tokens(self) -> int:
        return self.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens + self.output_tokens

    @property
    def row_label(self) -> str:
        if self.activity_kind == "provider_turn":
            return f"Provider turn #{self.step_index}"
        return f"Model step #{self.step_index}"

    @property
    def action_summary(self) -> str:
        if self.tool_calls:
            return ", ".join(self.tool_calls)
        if self.activity_kind == "provider_turn":
            return "unknown"
        if self.thinking_tokens > 0 and self.response_tokens > 0:
            return "think + reply"
        if self.thinking_tokens > 0:
            return "think"
        return "reply"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.activity_kind
        data["total_tokens"] = self.total_tokens
        data["new_input_tokens"] = self.new_input_tokens
        data["context_read_tokens"] = self.context_read_tokens
        data["action_summary"] = self.action_summary
        data["row_label"] = self.row_label
        return data


@dataclass(slots=True)
class NativeLaunchRecord:
    """An ``invoke_subagent`` call in a parent transcript paired with the child conversation it started.

    ``step_index``/``launched_at``/``task`` are None when the child ID was found
    without a matching recorded call (discovery only).
    """

    child_native_id: str
    step_index: int | None = None
    launched_at: str | None = None
    task: str | None = None
    tool_name: str | None = None


@dataclass(slots=True)
class SubagentLaunch:
    """One known invocation of an immediate child, for the parent's activity feed.

    A launch is a marker, never a counted step: it carries no tokens or duration.
    ``launch_id`` is the invocation identity: ``subagent:<subagent_id>`` (Source A),
    ``native:<conversation id>`` (Source C), or ``child:<child thread id>`` when only
    a parent/child link is known (Source B lineage). ``sources`` lists every source
    that discovered this same invocation.
    """

    launch_id: str
    title: str
    origin: str
    status: str
    child_thread_id: str | None = None
    child_native_id: str | None = None
    child_present: bool = False
    launched_at: str | None = None
    launch_time_source: str | None = None
    parent_step_index: int | None = None
    tool_name: str | None = None
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConversationTelemetry:
    """Decoded telemetry for a single Antigravity conversation (<nativeId>.db + transcript.jsonl)."""

    native_id: str
    steps: list[ModelStep] = field(default_factory=list)
    child_native_ids: list[str] = field(default_factory=list)
    parent_native_ids: list[str] = field(default_factory=list)
    model_name: str | None = None
    source: str = "db"
    launches: list[NativeLaunchRecord] = field(default_factory=list)

    @property
    def latest_context_tokens(self) -> int:
        if not self.steps:
            return 0
        last = self.steps[-1]
        return last.input_tokens + last.cache_read_tokens + last.output_tokens


@dataclass(slots=True)
class ProviderTurnRecord:
    """A single turn recorded in orchestration_v2_projection_provider_turns."""

    provider_turn_id: str
    thread_id: str
    provider_thread_id: str
    driver: str
    native_turn_id: str | None
    ordinal: int
    status: str
    started_at: str | None
    completed_at: str | None
    duration_seconds: float = 0.0
    used_tokens: int | None = None
    max_tokens: int | None = None
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thinking_tokens: int = 0
    response_tokens: int = 0
    output_tokens: int = 0
    usage_status: str = "unavailable"
    usage_provenance: str = "unavailable"
    node_id: str | None = None


@dataclass(slots=True)
class ProviderThreadRecord:
    """A driver-specific thread in orchestration_v2_projection_provider_threads."""

    provider_thread_id: str
    thread_id: str
    driver: str
    provider_instance_id: str
    native_id: str | None
    status: str
    used_tokens: int | None = None
    max_tokens: int | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class SubagentLinkRecord:
    """A parent-to-child link in orchestration_v2_projection_subagents."""

    subagent_id: str
    parent_thread_id: str
    child_thread_id: str | None
    driver: str
    provider_instance_id: str | None
    model: str | None
    title: str
    status: str
    origin: str = "app_owned"
    run_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float = 0.0


@dataclass(slots=True)
class AppThreadRecord:
    """An application thread in orchestration_v2_projection_threads."""

    thread_id: str
    project_id: str | None
    title: str
    default_provider: str
    provider_instance_id: str | None
    model: str | None
    thinking_level: str | None
    active_provider_thread_id: str | None
    parent_thread_id: str | None
    root_thread_id: str | None
    relationship_to_parent: str | None
    created_at: str | None
    updated_at: str | None
    archived_at: str | None = None
    deleted_at: str | None = None


@dataclass(slots=True)
class TokenMetricsSummary:
    """Aggregated token and execution metrics (used for both Self Metrics and Rollup Totals)."""

    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    thinking_tokens: int = 0
    response_tokens: int = 0
    output_tokens: int = 0
    step_count: int = 0
    provider_turn_count: int = 0
    model_step_count: int = 0
    active_subagents: int = 0
    total_subagents: int = 0
    duration_seconds: float = 0.0

    @property
    def new_input_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_tokens

    @property
    def context_read_tokens(self) -> int:
        return self.cache_read_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens + self.output_tokens

    @property
    def cache_hit_pct(self) -> float:
        denom = self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens
        if denom <= 0:
            return 0.0
        return (self.cache_read_tokens / denom) * 100.0

    @property
    def thinking_pct(self) -> float:
        if self.output_tokens <= 0:
            return 0.0
        return (self.thinking_tokens / self.output_tokens) * 100.0

    @property
    def steps_display(self) -> str:
        if self.provider_turn_count > 0 and self.model_step_count > 0:
            turn_str = "1 turn" if self.provider_turn_count == 1 else f"{self.provider_turn_count} turns"
            step_str = "1 step" if self.model_step_count == 1 else f"{self.model_step_count} steps"
            return f"{turn_str}, {step_str}"
        if self.provider_turn_count > 0:
            return "1 turn" if self.provider_turn_count == 1 else f"{self.provider_turn_count} turns"
        if self.model_step_count > 0:
            return "1 step" if self.model_step_count == 1 else f"{self.model_step_count} steps"
        if self.step_count > 0:
            return "1 step" if self.step_count == 1 else f"{self.step_count} steps"
        return "0"

    def add_step(self, step: ModelStep) -> None:
        self.input_tokens += step.input_tokens
        self.cache_read_tokens += step.cache_read_tokens
        self.cache_creation_tokens += step.cache_creation_tokens
        self.thinking_tokens += step.thinking_tokens
        self.response_tokens += step.response_tokens
        self.output_tokens += step.output_tokens
        self.step_count += 1
        if step.activity_kind == "provider_turn":
            self.provider_turn_count += 1
        else:
            self.model_step_count += 1
        self.duration_seconds += step.duration_seconds

    def add_summary(self, other: TokenMetricsSummary) -> None:
        self.input_tokens += other.input_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.thinking_tokens += other.thinking_tokens
        self.response_tokens += other.response_tokens
        self.output_tokens += other.output_tokens
        self.step_count += other.step_count
        self.provider_turn_count += other.provider_turn_count
        self.model_step_count += other.model_step_count
        self.duration_seconds += other.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "thinking_tokens": self.thinking_tokens,
            "response_tokens": self.response_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "new_input_tokens": self.new_input_tokens,
            "context_read_tokens": self.context_read_tokens,
            "cache_hit_pct": round(self.cache_hit_pct, 2),
            "thinking_pct": round(self.thinking_pct, 2),
            "step_count": self.step_count,
            "provider_turn_count": self.provider_turn_count,
            "model_step_count": self.model_step_count,
            "active_subagents": self.active_subagents,
            "total_subagents": self.total_subagents,
            "duration_seconds": round(self.duration_seconds, 3),
        }


@dataclass(slots=True)
class TurnMetrics(TokenMetricsSummary):
    """Metrics for a single provider turn."""

    ordinal: int | None = None
    turn_id: str | None = None
    status: str = "completed"
    availability: str = "unavailable"
    provenance: str = "exact"
    activity_kind: str = "provider_turn"

    def to_dict(self) -> dict[str, Any]:
        data = TokenMetricsSummary.to_dict(self)
        data["ordinal"] = self.ordinal
        data["turn_id"] = self.turn_id
        data["status"] = self.status
        data["availability"] = self.availability
        data["provenance"] = self.provenance
        data["activity_kind"] = self.activity_kind
        return data


@dataclass(slots=True)
class ContextSample:
    """A context occupancy reading at a point in time."""

    source_id: str
    used_tokens: int
    timestamp: str | None = None
    limit: int | None = None
    provenance: str = "measured"  # "measured" or "estimated"
    provider_thread_id: str | None = None
    observed_at: str | None = None
    timestamp_kind: str = "event"  # "event" or "observation"

    @property
    def source_identity(self) -> str:
        return self.source_id

    @property
    def max_tokens(self) -> int | None:
        return self.limit

    def to_dict(self) -> dict[str, Any]:
        data = {
            "source_id": self.source_id,
            "used_tokens": self.used_tokens,
            "timestamp": self.timestamp,
            "limit": self.limit,
            "provenance": self.provenance,
            "provider_thread_id": self.provider_thread_id,
        }
        if self.observed_at is not None:
            data["observed_at"] = self.observed_at
        if self.timestamp_kind != "event":
            data["timestamp_kind"] = self.timestamp_kind
        return data


@dataclass(slots=True)
class ThreadTelemetryNode:
    """Full telemetry tree node for an app thread or native subagent."""

    thread_id: str
    title: str
    driver: str
    provider_instance_id: str | None
    model: str
    thinking_level: str | None = None
    native_id: str | None = None
    status: str = "idle"
    updated_at: str | None = None
    active_provider_thread_id: str | None = None
    context_used_tokens: int = 0
    context_max_tokens: int = 1048576
    context_limit: int | None = None
    context_provenance: str = "unknown"
    context_limit_provenance: str = "default"
    context_uncertainty: bool = False
    context_samples: list[ContextSample] = field(default_factory=list)
    self_metrics: TokenMetricsSummary = field(default_factory=TokenMetricsSummary)
    rollup_metrics: TokenMetricsSummary = field(default_factory=TokenMetricsSummary)
    current_turn_metrics: TurnMetrics | None = None
    turn_metrics_availability: str = "unavailable"
    latest_step_label: str | None = None
    steps: list[ModelStep] = field(default_factory=list)
    combined_steps: list[ModelStep] = field(default_factory=list)
    subagents: list[ThreadTelemetryNode] = field(default_factory=list)
    launches: list[SubagentLaunch] = field(default_factory=list)
    provider: str | None = None
    store_state: str = "ready"

    @property
    def provider_thread_id(self) -> str | None:
        return self.active_provider_thread_id

    @property
    def observed_peak_tokens(self) -> int:
        vals = [self.context_used_tokens] + [s.used_tokens for s in self.context_samples]
        return max(vals) if vals else 0

    @property
    def turn_metrics(self) -> TurnMetrics | None:
        return self.current_turn_metrics

    @property
    def context_pct(self) -> float:
        lim = self.context_limit if self.context_limit is not None else self.context_max_tokens
        if lim is None or lim <= 0:
            return 0.0
        return min(100.0, (self.context_used_tokens / lim) * 100.0)

    def to_dict(self) -> dict[str, Any]:
        from .usage import provider_breakdown

        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "driver": self.driver,
            "provider": self.provider,
            "store_state": self.store_state,
            "by_provider": provider_breakdown(self),
            "provider_instance_id": self.provider_instance_id,
            "model": self.model,
            "thinking_level": self.thinking_level,
            "native_id": self.native_id,
            "active_provider_thread_id": self.active_provider_thread_id,
            "status": self.status,
            "updated_at": self.updated_at,
            "context_used_tokens": self.context_used_tokens,
            "context_max_tokens": self.context_max_tokens,
            "context_limit": self.context_limit,
            "context_provenance": self.context_provenance,
            "context_limit_provenance": self.context_limit_provenance,
            "context_uncertainty": self.context_uncertainty,
            "context_pct": round(self.context_pct, 2),
            "context_samples": [s.to_dict() for s in self.context_samples],
            "self_metrics": self.self_metrics.to_dict(),
            "rollup_metrics": self.rollup_metrics.to_dict(),
            "current_turn_metrics": self.current_turn_metrics.to_dict() if self.current_turn_metrics else None,
            "turn_metrics_availability": self.turn_metrics_availability,
            "latest_step_label": self.latest_step_label,
            "steps": [s.to_dict() for s in self.steps],
            "subagents": [c.to_dict() for c in self.subagents],
            "launches": [launch.to_dict() for launch in self.launches],
        }
