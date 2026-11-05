from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from t3_telemetry.models import ModelStep, ThreadTelemetryNode
from t3_telemetry.usage import build_usage_report, period_window, provider_family

ZONE = ZoneInfo("America/Los_Angeles")
NOW = datetime(2026, 11, 4, 20, tzinfo=ZONE)


def node(id, title, driver, steps=(), children=()):
    return ThreadTelemetryNode(thread_id=id, title=title, driver=driver, provider_instance_id=None,
                               model="m", steps=list(steps), subagents=list(children))


def step(index, date, tokens=10):
    return ModelStep(step_index=index, started_at=date, input_tokens=tokens)


def test_windows():
    week = period_window("week", NOW)
    assert week.start == datetime(2026, 11, 2, tzinfo=ZONE)
    assert len(week.bucket_starts) == 7
    dec = period_window("month", datetime(2026, 12, 15, tzinfo=ZONE))
    assert dec.end == datetime(2027, 1, 1, tzinfo=ZONE)
    assert len(period_window("month", datetime(2027, 2, 1, tzinfo=ZONE)).bucket_starts) == 28
    assert len(period_window("today", datetime(2026, 11, 1, 12, tzinfo=ZONE)).bucket_starts) == 25
    with pytest.raises(ValueError):
        period_window("today", datetime(2026, 1, 1))


def test_families():
    assert [provider_family(x) for x in ("claudeAgent", "codex_uuid", "antigravity_gemini_secondary", "")] == ["Claude", "Codex", "Antigravity", "Other"]
    assert provider_family(None, "codex_x") == "Codex"


def test_aggregation():
    child_step = step(1, "2026-11-05T02:00:00Z", 20)
    child = node("child", "Child", "antigravity_gemini_secondary", [child_step, child_step])
    root = node("root", "Root", "claude", [step(1, "2026-11-04T08:00:00Z"), step(2, None), step(3, "bad"), step(4, "2026-11-05T08:00:00Z"), step(5, "2026-11-01T19:00:00Z")], [child, child])
    report = build_usage_report([root, node("zero", "Zero", "codex")], NOW)
    today = report.periods["today"]
    assert today.total.total == 30
    assert today.buckets[18] == {"Antigravity": 20}
    assert today.buckets[0] == {"Claude": 10}
    assert today.threads[0].tokens.total == 30
    assert today.threads[0].providers == ["Claude", "Antigravity"]
    assert {k: v.total for k, v in today.by_provider.items()} == {"Claude": 10, "Antigravity": 20}
    assert report.periods["week"].total.total == 40  # end of today is within week
    assert report.periods["month"].total.total == 50
    assert report.root_total("today", "missing") == 0
    assert len(today.threads) == 1


def test_sort_and_token_fields():
    s = ModelStep(step_index=1, started_at="2026-11-05T02:00:00Z", input_tokens=1, cache_read_tokens=2, cache_creation_tokens=3, output_tokens=4)
    roots = [node("z", "Z", "codex", [s]), node("a", "A", "codex", [s]), node("big", "Big", "codex", [step(1, s.started_at, 11)])]
    rows = build_usage_report(roots, NOW).periods["today"].threads
    assert [r.title for r in rows] == ["Big", "A", "Z"]
    assert rows[1].tokens.total == 10


def test_usage_cache_ttl_force_and_failure(tmp_path):
    import sqlite3
    from t3_telemetry.usage import UsageCache

    now = [0.0]
    calls = []
    report = object()

    def reader(path, snapshot):
        calls.append((path, snapshot))
        if len(calls) == 4:
            raise sqlite3.OperationalError("unavailable")
        return report

    cache = UsageCache(max_age=5, clock=lambda: now[0], reader=reader)
    assert cache.get(tmp_path, None) is report
    assert cache.get(tmp_path, None) is report
    assert len(calls) == 1
    now[0] = 5
    assert cache.get(tmp_path, None) is report
    assert cache.get(tmp_path, None, force=True) is report
    assert cache.get(tmp_path, None, force=True) is report
    assert len(calls) == 4
    empty = UsageCache(reader=lambda *_: (_ for _ in ()).throw(OSError()))
    assert empty.get(tmp_path, None) is None


def test_usage_cache_backs_off_after_failure(tmp_path):
    from t3_telemetry.usage import UsageCache

    now = [0.0]
    calls = []

    def reader(*_):
        calls.append(now[0])
        raise OSError("locked")

    cache = UsageCache(max_age=5, clock=lambda: now[0], reader=reader)
    assert cache.get(tmp_path, None) is None
    now[0] = 1
    assert cache.get(tmp_path, None) is None
    assert calls == [0.0]
    now[0] = 5
    cache.get(tmp_path, None)
    assert calls == [0.0, 5]


def test_usage_cache_background_refresh_keeps_old_report(tmp_path):
    import threading
    from t3_telemetry.usage import UsageCache

    now = [0.0]
    release = threading.Event()
    reports = iter(["first", "second"])

    def reader(*_):
        report = next(reports)
        if report == "second":
            release.wait(5)
        return report

    cache = UsageCache(max_age=5, clock=lambda: now[0], reader=reader, background=True)
    assert cache.get(tmp_path, None) == "first"  # no report yet: synchronous
    now[0] = 6
    assert cache.get(tmp_path, None) == "first"  # refresh runs in the background
    assert cache.get(tmp_path, None) == "first"  # one worker at a time
    release.set()
    cache._worker.join(5)
    assert cache.get(tmp_path, None) == "second"


def test_usage_cache_background_worker_swallows_unexpected_errors(tmp_path):
    from t3_telemetry.usage import UsageCache

    now = [0.0]
    calls = []

    def reader(*_):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("bad row")
        return "first"

    cache = UsageCache(max_age=5, clock=lambda: now[0], reader=reader, background=True)
    cache.get(tmp_path, None)
    now[0] = 6
    cache.get(tmp_path, None)
    cache._worker.join(5)
    assert cache.get(tmp_path, None) == "first"
    assert len(calls) == 2  # the failure also waits max_age before retrying
