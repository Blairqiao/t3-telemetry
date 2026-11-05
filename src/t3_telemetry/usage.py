"""Pure calendar-based aggregation of telemetry usage."""
import sqlite3
import threading
import time as clock_time
from pathlib import Path
from typing import Callable

from .aggregator import build_thread_telemetry
from .db_t3 import T3StateSnapshot

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from .models import ModelStep, ThreadTelemetryNode

PERIODS = ("today", "week", "month")
PERIOD_LABELS = {"today": "Today", "week": "This week", "month": "This month"}
PROVIDER_ORDER = ("Claude", "Codex", "Antigravity", "Other")


@dataclass(slots=True)
class TokenBreakdown:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens + self.output_tokens

    def add_step(self, step: ModelStep) -> None:
        for name in ("input_tokens", "cache_read_tokens", "cache_creation_tokens", "output_tokens"):
            setattr(self, name, getattr(self, name) + getattr(step, name))


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


@dataclass(slots=True)
class PeriodUsage:
    window: PeriodWindow
    total: TokenBreakdown
    by_provider: dict[str, TokenBreakdown]
    buckets: list[dict[str, int]]
    threads: list[ThreadUsage]


@dataclass(slots=True)
class UsageReport:
    generated_at: datetime
    periods: dict[str, PeriodUsage]

    def root_total(self, period: str, thread_id: str) -> int:
        return next((row.tokens.total for row in self.periods[period].threads if row.thread_id == thread_id), 0)


def provider_family(driver: str | None, provider_instance_id: str | None = None) -> str:
    value = (driver or provider_instance_id or "").lower()
    return next((name for name in PROVIDER_ORDER[:-1] if value.startswith(name.lower())), "Other")


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


def build_usage_report(roots: list[ThreadTelemetryNode], now: datetime) -> UsageReport:
    windows = {key: period_window(key, now) for key in PERIODS}
    periods = {key: PeriodUsage(w, TokenBreakdown(), {}, [{} for _ in w.bucket_starts], []) for key, w in windows.items()}
    utc_starts = {key: [d.astimezone(timezone.utc) for d in w.bucket_starts] for key, w in windows.items()}
    seen = set()
    for root in roots:
        totals = {key: TokenBreakdown() for key in PERIODS}
        families = {key: {} for key in PERIODS}
        stack = [root]
        while stack:
            node = stack.pop()
            stack.extend(reversed(node.subagents))
            for step in node.steps:
                identity = (node.thread_id, step.source, step.provider_turn_id, step.step_index)
                if identity in seen:
                    continue
                seen.add(identity)
                try:
                    t = datetime.fromisoformat(step.started_at).astimezone(timezone.utc)
                except (TypeError, ValueError):
                    continue
                family = provider_family(node.driver, node.provider_instance_id)
                for key, usage in periods.items():
                    w = usage.window
                    if not w.start.astimezone(timezone.utc) <= t < w.end.astimezone(timezone.utc):
                        continue
                    usage.total.add_step(step)
                    usage.by_provider.setdefault(family, TokenBreakdown()).add_step(step)
                    totals[key].add_step(step)
                    families[key].setdefault(family, TokenBreakdown()).add_step(step)
                    bucket = usage.buckets[bisect_right(utc_starts[key], t) - 1]
                    step_tokens = TokenBreakdown()
                    step_tokens.add_step(step)
                    bucket[family] = bucket.get(family, 0) + step_tokens.total
        for key, usage in periods.items():
            if totals[key].total > 0:
                usage.threads.append(ThreadUsage(root.thread_id, root.title, root.status, [f for f in PROVIDER_ORDER if f in families[key] and families[key][f].total > 0], totals[key]))
    for usage in periods.values():
        usage.by_provider = {f: usage.by_provider[f] for f in PROVIDER_ORDER if f in usage.by_provider and usage.by_provider[f].total > 0}
        usage.buckets = [{f: b[f] for f in PROVIDER_ORDER if b.get(f, 0) != 0} for b in usage.buckets]
        usage.threads.sort(key=lambda row: (-row.tokens.total, row.title))
    return UsageReport(now.astimezone(now.tzinfo), periods)


def read_usage_report(t3_dir: Path | str, snapshot: T3StateSnapshot, now: datetime | None = None) -> UsageReport:
    roots = [t for t in snapshot.threads.values() if t.parent_thread_id is None and t.deleted_at is None]
    trees = [build_thread_telemetry(t.thread_id, t3_dir, snapshot=snapshot) for t in roots]
    return build_usage_report(trees, now or datetime.now().astimezone())


class UsageCache:
    """Rebuild the usage report at most every ``max_age`` seconds.

    A failed read is also held for ``max_age`` so a broken database is not re-read on
    every poll. With ``background=True``, refreshes after the first successful report
    run on a daemon thread and ``get`` returns the previous report until it finishes.
    """

    def __init__(self, max_age: float = 5.0, clock: Callable[[], float] = clock_time.monotonic,
                 reader: Callable[[Path, T3StateSnapshot], UsageReport] | None = None,
                 background: bool = False) -> None:
        self.max_age = max_age
        self.clock = clock
        self.reader = reader or read_usage_report
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
