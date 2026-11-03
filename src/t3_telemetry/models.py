"""Shared domain dataclasses for T3 Code and Antigravity telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelStep:
    """A single inference step (Antigravity step_type=15 or Claude/Codex provider turn)."""

    step_index: int
    source: str = "main"
    status: str = "completed"
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

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.output_tokens

    @property
    def action_summary(self) -> str:
        if self.tool_calls:
            return ", ".join(self.tool_calls)
        if self.thinking_tokens > 0 and self.response_tokens > 0:
            return "think + reply"
        if self.thinking_tokens > 0:
            return "think"
        return "reply"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        data["action_summary"] = self.action_summary
        return data


@dataclass(slots=True)
class ConversationTelemetry:
    """Decoded telemetry for a single Antigravity conversation (<nativeId>.db + transcript.jsonl)."""

    native_id: str
    steps: list[ModelStep] = field(default_factory=list)
    child_native_ids: list[str] = field(default_factory=list)
    parent_native_ids: list[str] = field(default_factory=list)
    model_name: str | None = None
    source: str = "db"

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
    active_subagents: int = 0
    total_subagents: int = 0
    duration_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.output_tokens

    @property
    def cache_hit_pct(self) -> float:
        denom = self.input_tokens + self.cache_read_tokens
        if denom <= 0:
            return 0.0
        return (self.cache_read_tokens / denom) * 100.0

    @property
    def thinking_pct(self) -> float:
        if self.output_tokens <= 0:
            return 0.0
        return (self.thinking_tokens / self.output_tokens) * 100.0

    def add_step(self, step: ModelStep) -> None:
        self.input_tokens += step.input_tokens
        self.cache_read_tokens += step.cache_read_tokens
        self.cache_creation_tokens += step.cache_creation_tokens
        self.thinking_tokens += step.thinking_tokens
        self.response_tokens += step.response_tokens
        self.output_tokens += step.output_tokens
        self.step_count += 1
        self.duration_seconds += step.duration_seconds

    def add_summary(self, other: TokenMetricsSummary) -> None:
        self.input_tokens += other.input_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.thinking_tokens += other.thinking_tokens
        self.response_tokens += other.response_tokens
        self.output_tokens += other.output_tokens
        self.step_count += other.step_count
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
            "cache_hit_pct": round(self.cache_hit_pct, 2),
            "thinking_pct": round(self.thinking_pct, 2),
            "step_count": self.step_count,
            "active_subagents": self.active_subagents,
            "total_subagents": self.total_subagents,
            "duration_seconds": round(self.duration_seconds, 3),
        }


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
    context_used_tokens: int = 0
    context_max_tokens: int = 1048576
    self_metrics: TokenMetricsSummary = field(default_factory=TokenMetricsSummary)
    rollup_metrics: TokenMetricsSummary = field(default_factory=TokenMetricsSummary)
    steps: list[ModelStep] = field(default_factory=list)
    combined_steps: list[ModelStep] = field(default_factory=list)
    subagents: list[ThreadTelemetryNode] = field(default_factory=list)

    @property
    def context_pct(self) -> float:
        if self.context_max_tokens <= 0:
            return 0.0
        return min(100.0, (self.context_used_tokens / self.context_max_tokens) * 100.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "driver": self.driver,
            "provider_instance_id": self.provider_instance_id,
            "model": self.model,
            "thinking_level": self.thinking_level,
            "native_id": self.native_id,
            "status": self.status,
            "updated_at": self.updated_at,
            "context_used_tokens": self.context_used_tokens,
            "context_max_tokens": self.context_max_tokens,
            "context_pct": round(self.context_pct, 2),
            "self_metrics": self.self_metrics.to_dict(),
            "rollup_metrics": self.rollup_metrics.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "subagents": [c.to_dict() for c in self.subagents],
        }
