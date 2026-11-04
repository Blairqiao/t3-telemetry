"""In-memory bounded LRU context history store for live agent telemetry."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Hashable, Sequence

from t3_telemetry.models import ContextSample, ThreadTelemetryNode


def make_agent_key(
    node_or_id: ThreadTelemetryNode | str,
    active_pt_id: str | None = None,
) -> tuple[str, str | None]:
    """Derive stable agent key combining app/native identity and provider-thread identity."""
    if isinstance(node_or_id, ThreadTelemetryNode):
        ident = node_or_id.thread_id
        pt_id = node_or_id.active_provider_thread_id
        return (ident, pt_id)
    return (node_or_id, active_pt_id)


def clone_sample(s: ContextSample) -> ContextSample:
    """Create an independent immutable copy of a ContextSample."""
    return ContextSample(
        source_id=s.source_id,
        used_tokens=s.used_tokens,
        timestamp=s.timestamp,
        limit=s.limit,
        provenance=s.provenance,
        provider_thread_id=s.provider_thread_id,
        observed_at=getattr(s, "observed_at", None),
        timestamp_kind=getattr(s, "timestamp_kind", "event"),
    )


def _samples_equivalent(s1: ContextSample, s2: ContextSample) -> bool:
    """Check if two samples have identical context telemetry values."""
    return (
        s1.used_tokens == s2.used_tokens
        and s1.limit == s2.limit
        and s1.provenance == s2.provenance
        and s1.provider_thread_id == s2.provider_thread_id
    )


class ContextHistoryStore:
    """In-memory bounded LRU store for context occupancy samples.

    Retains at most `max_samples_per_agent` (default 256) for each of at most
    `max_agents` (default 64) recently viewed agents.
    """

    def __init__(self, max_samples_per_agent: int = 256, max_agents: int = 64) -> None:
        self.max_samples_per_agent = max_samples_per_agent
        self.max_agents = max_agents
        self._agents: OrderedDict[Hashable, list[ContextSample]] = OrderedDict()
        self._previous_sources: dict[Hashable, set[str]] = {}

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_key: Hashable) -> bool:
        return agent_key in self._agents

    def get(self, agent_key: Hashable) -> list[ContextSample]:
        """Retrieve stored samples for agent_key, marking it recently viewed."""
        if agent_key not in self._agents:
            return []
        self._agents.move_to_end(agent_key)
        return [clone_sample(s) for s in self._agents[agent_key]]

    def touch(self, agent_key: Hashable) -> None:
        """Mark agent_key as recently viewed in LRU order."""
        if agent_key in self._agents:
            self._agents.move_to_end(agent_key)

    def clear(self) -> None:
        """Clear all stored context history."""
        self._agents.clear()
        self._previous_sources.clear()

    def merge(
        self,
        agent_key: Hashable,
        historical_samples: Sequence[ContextSample],
        observed_sample: ContextSample | None = None,
    ) -> list[ContextSample]:
        """Merge historical records and live observations for agent_key.

        - Preserves immutable copies so inputs and store are isolated.
        - Updates mutable in-flight historical records by stable source ID.
        - Preserves graph shape and peak when provider-thread snapshot source ID
          stays same while observed values change.
        - Deduplicates unchanged live observations even if observation timestamp differs.
        - Preserves observed context drops.
        - Robust chronology handling missing timestamps without sorting to epoch 0.
        - Bounded to `max_samples_per_agent` and `max_agents` LRU.
        """
        # 1. LRU tracking
        if agent_key in self._agents:
            self._agents.move_to_end(agent_key)
            existing = self._agents[agent_key]
        else:
            if len(self._agents) >= self.max_agents:
                evicted_key, _ = self._agents.popitem(last=False)
                self._previous_sources.pop(evicted_key, None)
            existing = []
            self._agents[agent_key] = existing

        # Remember the source snapshot, rather than every ID ever observed. Repeated
        # upstream records must not reenter the window after sample eviction.
        previous_sources = self._previous_sources.get(agent_key, set())
        self._previous_sources[agent_key] = {s.source_id for s in historical_samples}

        # 2. Merge historical samples
        for raw_h in historical_samples:
            h = clone_sample(raw_h)
            match_idx = None
            for idx, item in enumerate(existing):
                if item.source_id == h.source_id:
                    match_idx = idx
                    break

            if match_idx is not None:
                existing_item = existing[match_idx]
                if h.source_id.startswith("pt:") and existing_item.used_tokens != h.used_tokens:
                    # Provider-thread snapshot changed value: preserve previous reading
                    # as a historical observation point, and append the new reading
                    existing_item.source_id = f"{h.source_id}:obs:{existing_item.timestamp or match_idx}"
                    existing_item.timestamp_kind = "observation"
                    existing.append(h)
                else:
                    # In-flight mutable upsert (e.g. running turn completed or same value)
                    existing_item.used_tokens = h.used_tokens
                    if h.timestamp is not None:
                        existing_item.timestamp = h.timestamp
                    existing_item.limit = h.limit
                    existing_item.provenance = h.provenance
                    existing_item.provider_thread_id = h.provider_thread_id
            elif h.source_id not in previous_sources:
                existing.append(h)

        # 3. Merge live observed sample
        if observed_sample is not None:
            # Do not fabricate a zero measurement if context is unknown
            if observed_sample.provenance != "unknown" and not (
                observed_sample.used_tokens == 0 and observed_sample.provenance == "unknown"
            ):
                obs = clone_sample(observed_sample)
                if not getattr(obs, "observed_at", None) and getattr(obs, "timestamp_kind", "") == "observation":
                    obs.observed_at = datetime.now(timezone.utc).isoformat()

                if existing:
                    latest = existing[-1]
                    if not _samples_equivalent(latest, obs):
                        existing.append(obs)
                else:
                    existing.append(obs)

        # 4. Enforce sample bound
        if len(existing) > self.max_samples_per_agent:
            existing[:] = existing[-self.max_samples_per_agent:]

        # 5. Return immutable copies
        return [clone_sample(s) for s in existing]
