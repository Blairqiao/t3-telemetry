"""Pure calendar-based aggregation of telemetry usage."""
import sqlite3
import threading
import time as clock_time
from pathlib import Path
from typing import Callable

from .aggregator import snapshot_thread_status
from .db_t3 import T3StateSnapshot

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone

from .models import ModelStep, ThreadTelemetryNode
from .prices import CostBreakdown, model_key, step_cost
from .providers import family_label, provider_key
from .store import TelemetryStore, attribute_session, utc_iso

PERIODS = ("today", "week", "month")
PERIOD_LABELS = {"today": "Today", "week": "This week", "month": "This month"}
PROVIDER_ORDER = ("Claude", "Codex", "Antigravity", "Other")
LARGE_CONTEXT_TOKENS = 200_000
SCOPE_ORDER: tuple[str, ...] = ("Main", "Subagents")


@dataclass(slots=True)
class TokenBreakdown:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens + self.output_tokens

    def add_counts(self, input: int, cache_read: int, cache_creation: int, output: int) -> None:
        self.input_tokens += input
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_creation
        self.output_tokens += output

    def add_step(self, step: ModelStep) -> None:
        for name in ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens"):
            setattr(self, name, getattr(self, name) + getattr(step, name))


@dataclass(slots=True)
class ModelUsage:
    tokens: TokenBreakdown = field(default_factory=TokenBreakdown)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    steps: int = 0
    thinking_tokens: int = 0  # already included in tokens.output_tokens


def cache_hit_pct(tokens: TokenBreakdown) -> float | None:
    """cache_read / (input + cache_read + cache_creation) * 100; None when that prompt total is 0."""
    total = tokens.input_tokens + tokens.cache_read_tokens + tokens.cache_creation_tokens
    return tokens.cache_read_tokens / total * 100 if total else None


@dataclass(slots=True)
class UsageStats:
    tokens: TokenBreakdown = field(default_factory=TokenBreakdown)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    subagent_tokens: TokenBreakdown = field(default_factory=TokenBreakdown)
    subagent_cost: CostBreakdown = field(default_factory=CostBreakdown)
    provider_tokens: dict[str, TokenBreakdown] = field(default_factory=dict)
    provider_cost: dict[str, CostBreakdown] = field(default_factory=dict)
    model_tokens: dict[str, int] = field(default_factory=dict)
    model_cost: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    large_context_steps: int = 0
    thinking_tokens: int = 0
    provider_thinking: dict[str, int] = field(default_factory=dict)
    # family label ("Claude", "Codex", "Antigravity", "Other") -> model name -> usage
    provider_models: dict[str, dict[str, ModelUsage]] = field(default_factory=dict)

    def add(self, model: str | None, when: datetime | None, input: int, cache_read: int,
            cache_write: int, output: int, subagent: bool, *, family: str = "Other", thinking: int = 0) -> None:
        self.tokens.add_counts(input, cache_read, cache_write, output)
        c = step_cost(model, when, input, cache_read, cache_write, output)
        self.cost.add(c)
        self.provider_tokens.setdefault(family, TokenBreakdown()).add_counts(input, cache_read, cache_write, output)
        self.provider_cost.setdefault(family, CostBreakdown()).add(c)
        if subagent:
            self.subagent_tokens.add_counts(input, cache_read, cache_write, output)
            self.subagent_cost.add(c)
        name = model_key(model) or (model if model and not model.startswith("<") else "unknown")
        self.model_tokens[name] = self.model_tokens.get(name, 0) + input + cache_read + cache_write + output
        self.model_cost[name] = self.model_cost.get(name, 0.0) + c.total
        self.thinking_tokens += thinking
        self.provider_thinking[family] = self.provider_thinking.get(family, 0) + thinking
        mu = self.provider_models.setdefault(family, {}).setdefault(name, ModelUsage())
        mu.tokens.add_counts(input, cache_read, cache_write, output)
        mu.cost.add(c)
        mu.steps += 1
        mu.thinking_tokens += thinking
        self.steps += 1
        if input + cache_read + cache_write > LARGE_CONTEXT_TOKENS:
            self.large_context_steps += 1


def stats_for_node(node: ThreadTelemetryNode) -> UsageStats:
    stats = UsageStats()
    seen = set()
    stack = [node]
    while stack:
        n = stack.pop()
        stack.extend(reversed(n.subagents))
        for step in n.steps:
            identity = step.turn_key or (n.thread_id, step.source, step.provider_turn_id, step.step_index)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                started_at = datetime.fromisoformat(step.started_at)
            except (TypeError, ValueError):
                started_at = None
            stats.add(step.model or n.model, started_at, step.input_tokens, step.cache_read_tokens,
                      step.cache_creation_tokens, step.output_tokens, subagent=n is not node,
                      family=family_label(step.provider or n.provider or provider_key(n.driver, n.provider_instance_id)),
                      thinking=step.thinking_tokens)
    return stats


def provider_breakdown(node: ThreadTelemetryNode) -> dict[str, dict[str, int]]:
    """Token counts for each provider across the node and its descendants."""
    result = {}
    for family, counts in stats_for_node(node).provider_tokens.items():
        if counts.total > 0:
            result[family.lower()] = {name: getattr(counts, name) for name in (
                "input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens"
            )} | {"total_tokens": counts.total}
    return result


def _step_time(step: ModelStep) -> datetime | None:
    try:
        return datetime.fromisoformat(step.started_at) if step.started_at else None
    except (TypeError, ValueError):
        return None


def step_usd(model: str | None, step: ModelStep) -> CostBreakdown:
    """Estimated list-price cost of one recorded step."""
    return step_cost(model, _step_time(step), step.input_tokens, step.cache_read_tokens,
                     step.cache_creation_tokens, step.output_tokens)


def node_cost(node: ThreadTelemetryNode, rollup: bool = True) -> CostBreakdown:
    """Estimated cost of an agent's own steps, or of the agent and all its subagents."""
    if rollup:
        return stats_for_node(node).cost
    total = CostBreakdown()
    for step in node.steps:
        total.add(step_usd(node.model, step))
    return total


def turn_cost(node: ThreadTelemetryNode) -> CostBreakdown | None:
    """Cost of the steps recorded under the current/latest provider turn; None when unmatched."""
    turn = node.current_turn_metrics
    if turn is None or not turn.turn_id:
        return None
    steps = [s for s in node.steps if s.provider_turn_id == turn.turn_id]
    if not steps:
        return None
    total = CostBreakdown()
    for step in steps:
        total.add(step_usd(node.model, step))
    return total


@dataclass(slots=True)
class PeriodWindow:
    key: str
    label: str
    start: datetime
    end: datetime
    bucket: str
    bucket_starts: list[datetime]


@dataclass(slots=True)
class ThreadUsage:
    thread_id: str
    title: str
    status: str
    providers: list[str]
    tokens: TokenBreakdown
    cost: CostBreakdown = field(default_factory=CostBreakdown)

    def value(self, measure: str) -> float:
        return self.cost.total if measure == "usd" else self.tokens.total


@dataclass(slots=True)
class PeriodUsage:
    window: PeriodWindow
    total: TokenBreakdown
    by_provider: dict[str, TokenBreakdown]
    buckets: list[dict[str, int]]
    threads: list[ThreadUsage]
    stats: UsageStats = field(default_factory=UsageStats)
    scope_buckets: list[dict[str, int]] = field(default_factory=list)
    cost_by_provider: dict[str, CostBreakdown] = field(default_factory=dict)
    cost_buckets: list[dict[str, float]] = field(default_factory=list)
    scope_cost_buckets: list[dict[str, float]] = field(default_factory=list)

    def total_value(self, measure: str) -> float:
        return self.stats.cost.total if measure == "usd" else self.total.total

    def top(self, measure: str = "tokens", n: int = 10) -> list[ThreadUsage]:
        """Largest threads by the active measure (``threads`` stays in token order)."""
        if measure != "usd":
            return self.threads[:n]
        return sorted(self.threads, key=lambda row: (-row.cost.total, -row.tokens.total, row.title))[:n]

    def series(self, split: str, measure: str) -> list[dict[str, float]]:
        if split == "scope":
            return self.scope_cost_buckets if measure == "usd" else self.scope_buckets
        return self.cost_buckets if measure == "usd" else self.buckets


@dataclass(slots=True)
class UsageReport:
    generated_at: datetime
    periods: dict[str, PeriodUsage]
    # Lifetime totals per root thread (or unbound session), across every recorded step.
    all_time: dict[str, ThreadUsage] = field(default_factory=dict)

    def root_total(self, period: str, thread_id: str, measure: str = "tokens") -> float:
        return next((row.value(measure) for row in self.periods[period].threads if row.thread_id == thread_id), 0)

    def lifetime(self, thread_id: str, measure: str = "tokens") -> float | None:
        row = self.all_time.get(thread_id)
        return row.value(measure) if row is not None else None


def _new_period(w: PeriodWindow) -> PeriodUsage:
    n = len(w.bucket_starts)
    return PeriodUsage(w, TokenBreakdown(), {}, [{} for _ in range(n)], [],
                       scope_buckets=[{} for _ in range(n)], cost_buckets=[{} for _ in range(n)],
                       scope_cost_buckets=[{} for _ in range(n)])


def _finish_period(usage: PeriodUsage) -> None:
    keep = [f for f in PROVIDER_ORDER if f in usage.by_provider and usage.by_provider[f].total > 0]
    usage.by_provider = {f: usage.by_provider[f] for f in keep}
    usage.cost_by_provider = {f: usage.cost_by_provider.get(f, CostBreakdown()) for f in keep}
    usage.buckets = [{f: b[f] for f in PROVIDER_ORDER if b.get(f, 0) != 0} for b in usage.buckets]
    usage.cost_buckets = [{f: b[f] for f in PROVIDER_ORDER if b.get(f, 0) != 0} for b in usage.cost_buckets]
    usage.scope_buckets = [{s: b[s] for s in SCOPE_ORDER if b.get(s, 0)} for b in usage.scope_buckets]
    usage.scope_cost_buckets = [{s: b[s] for s in SCOPE_ORDER if b.get(s, 0)} for b in usage.scope_cost_buckets]
    usage.threads.sort(key=lambda row: (-row.tokens.total, row.title))


def _add_to_period(usage: PeriodUsage, idx: int, family: str, scope: str,
                   counts: tuple[int, int, int, int], cost: CostBreakdown) -> None:
    usage.total.add_counts(*counts)
    usage.by_provider.setdefault(family, TokenBreakdown()).add_counts(*counts)
    usage.cost_by_provider.setdefault(family, CostBreakdown()).add(cost)
    tokens = sum(counts)
    bucket = usage.buckets[idx]
    bucket[family] = bucket.get(family, 0) + tokens
    cbucket = usage.cost_buckets[idx]
    cbucket[family] = cbucket.get(family, 0.0) + cost.total
    sbucket = usage.scope_buckets[idx]
    sbucket[scope] = sbucket.get(scope, 0) + tokens
    scbucket = usage.scope_cost_buckets[idx]
    scbucket[scope] = scbucket.get(scope, 0.0) + cost.total


def provider_family(driver: str | None, provider_instance_id: str | None = None) -> str:
    return family_label(provider_key(driver, provider_instance_id))


def period_window(key: str, now: datetime) -> PeriodWindow:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    zone = now.tzinfo
    day = now.astimezone(zone).date()
    if key == "today":
        first, last = day, day + timedelta(days=1)
    elif key == "week":
        first = day - timedelta(days=day.weekday())
        last = first + timedelta(days=7)
    elif key == "month":
        first = day.replace(day=1)
        last = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    else:
        raise ValueError(f"unknown period: {key}")
    midnight = lambda d: datetime.combine(d, time.min, tzinfo=zone)
    start, end = midnight(first), midnight(last)
    if key == "today":
        cursor, end_utc = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        starts = []
        while cursor < end_utc:
            starts.append(cursor.astimezone(zone))
            cursor += timedelta(hours=1)
    else:
        starts = [midnight(first + timedelta(days=i)) for i in range((last - first).days)]
    return PeriodWindow(key, PERIOD_LABELS[key], start, end, "hour" if key == "today" else "day", starts)





def build_store_usage_report(store: TelemetryStore, now: datetime,
                             statuses: dict[str, str] | None = None) -> UsageReport:
    windows = {key: period_window(key, now) for key in PERIODS}
    periods = {key: _new_period(w) for key, w in windows.items()}
    utc_starts = {key: [d.astimezone(timezone.utc) for d in w.bucket_starts] for key, w in windows.items()}
    sessions = {s.session_key: s for s in store.sessions()}
    bindings, threads = store.bindings(), store.t3_threads()
    statuses = statuses or {}

    rows = {key: {} for key in PERIODS}
    all_time: dict[str, tuple[ThreadUsage, set[str]]] = {}
    start = min(w.start.astimezone(timezone.utc) for w in windows.values())
    end = max(w.end.astimezone(timezone.utc) for w in windows.values())
    period_steps = store.all_steps() if hasattr(store, "all_steps") else store.steps_between(utc_iso(start), utc_iso(end))
    for step in period_steps:
        t = datetime.fromisoformat(step["ts"].replace("Z", "+00:00")).astimezone(timezone.utc)
        family = provider_family(step["provider"])
        row_id, is_subagent = attribute_session(step["session_key"], sessions, bindings, threads)
        if row_id.startswith("session:"):
            root_key = row_id[len("session:"):]
            session = sessions[root_key]
            title = session.title or (f"{Path(session.cwd).name} · {session.provider}" if session.cwd else root_key)
            status = "idle"
        else:
            thread = threads.get(row_id)
            title = (thread["title"] if thread else None) or row_id
            status = statuses.get(row_id, "idle")
        counts = tuple(step[name] for name in ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens"))
        model = step["model"] or sessions[step["session_key"]].model if step["session_key"] in sessions else step["model"]
        cost = step_cost(model, t, *counts)
        life, life_families = all_time.setdefault(row_id, (ThreadUsage(row_id, title, status, [], TokenBreakdown()), set()))
        life.tokens.add_counts(*counts)
        life.cost.add(cost)
        if sum(counts):
            life_families.add(family)
        for key, usage in periods.items():
            w = usage.window
            if not w.start.astimezone(timezone.utc) <= t < w.end.astimezone(timezone.utc):
                continue
            usage.stats.add(model, t, *counts, subagent=is_subagent, family=family,
                            thinking=step["thinking_tokens"] if "thinking_tokens" in step.keys() else 0)
            _add_to_period(usage, bisect_right(utc_starts[key], t) - 1, family,
                           "Subagents" if is_subagent else "Main", counts, cost)
            row, families = rows[key].setdefault(row_id, (ThreadUsage(row_id, title, status, [], TokenBreakdown()), set()))
            row.tokens.add_counts(*counts)
            row.cost.add(cost)
            if sum(counts):
                families.add(family)
    for key, usage in periods.items():
        for row, families in rows[key].values():
            if row.tokens.total > 0:
                row.providers = [f for f in PROVIDER_ORDER if f in families]
                usage.threads.append(row)
        _finish_period(usage)
    lifetime = {}
    for row_id, (row, families) in all_time.items():
        row.providers = [f for f in PROVIDER_ORDER if f in families]
        lifetime[row_id] = row
    return UsageReport(now.astimezone(now.tzinfo), periods, lifetime)


def read_store_usage_report(store: TelemetryStore, snapshot: T3StateSnapshot,
                            now: datetime | None = None) -> UsageReport:
    statuses = {t.thread_id: snapshot_thread_status(snapshot, t.thread_id)
                for t in snapshot.threads.values() if t.parent_thread_id is None and t.deleted_at is None}
    return build_store_usage_report(store, now or datetime.now().astimezone(), statuses)


def _default_usage_reader(*_args: object, **_kwargs: object) -> UsageReport:
    raise ValueError("UsageCache needs a reader")


class UsageCache:
    """Rebuild the usage report at most every ``max_age`` seconds.

    A failed read is also held for ``max_age`` so a broken database is not re-read on
    every poll. With ``background=True``, refreshes after the first successful report
    run on a daemon thread and ``get`` returns the previous report until it finishes.
    """

    def __init__(self, max_age: float = 5.0, clock: Callable[[], float] = clock_time.monotonic,
                 reader: Callable[..., UsageReport] | None = None,
                 background: bool = False) -> None:
        self.max_age = max_age
        self.clock = clock
        self.reader = reader or _default_usage_reader
        self.background = background
        self.report: UsageReport | None = None
        self.built_at: float | None = None  # time of the last attempt, successful or not
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def get(self, t3_dir: Path, snapshot: T3StateSnapshot, force: bool = False) -> UsageReport | None:
        with self._lock:
            stale = force or self.built_at is None or self.clock() - self.built_at >= self.max_age
            have_report = self.report is not None
        if stale:
            if self.background and have_report:
                self._start_worker(t3_dir, snapshot)
            else:
                self._refresh(t3_dir, snapshot)
        with self._lock:
            return self.report

    def _refresh(self, t3_dir: Path, snapshot: T3StateSnapshot) -> None:
        try:
            report = self.reader(t3_dir, snapshot)
        except (sqlite3.Error, OSError):
            report = None
        with self._lock:
            self.built_at = self.clock()
            if report is not None:
                self.report = report

    def _start_worker(self, t3_dir: Path, snapshot: T3StateSnapshot) -> None:
        if self._worker is not None and self._worker.is_alive():
            return

        def run() -> None:
            try:
                self._refresh(t3_dir, snapshot)
            except Exception:
                # A traceback on stderr would corrupt the live screen; keep the old report.
                with self._lock:
                    self.built_at = self.clock()

        self._worker = threading.Thread(target=run, name="usage-refresh", daemon=True)
        self._worker.start()
