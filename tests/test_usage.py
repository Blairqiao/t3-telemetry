from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from t3_telemetry.models import ModelStep, ThreadTelemetryNode
from t3_telemetry.usage import build_usage_report, period_window, provider_family, stats_for_node

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
    assert today.scope_buckets[18] == {"Subagents": 20}
    assert today.scope_buckets[0] == {"Main": 10}
    assert today.stats.subagent_tokens.total == 20
    for usage in report.periods.values():
        assert sum(sum(b.values()) for b in usage.scope_buckets) == sum(sum(b.values()) for b in usage.buckets)
        assert usage.stats.tokens.total == usage.total.total
    assert today.threads[0].tokens.total == 30
    assert today.threads[0].providers == ["Claude", "Antigravity"]
    assert {k: v.total for k, v in today.by_provider.items()} == {"Claude": 10, "Antigravity": 20}
    assert report.periods["week"].total.total == 40  # end of today is within week
    assert report.periods["month"].total.total == 50
    assert report.root_total("today", "missing") == 0
    assert len(today.threads) == 1


def test_stats_for_node():
    child = node("child", "Child", "codex", [ModelStep(step_index=1, started_at=None, input_tokens=1_000_000)])
    child.model = "gpt-6.1-sol"
    root = node("root", "Root", "claude", [ModelStep(step_index=1, started_at="bad", output_tokens=1_000_000)], [child, child])
    root.model = "claude-opus-5-5"
    stats = stats_for_node(root)
    assert stats.tokens.total == 2_000_000
    assert stats.subagent_tokens.total == 1_000_000
    assert stats.steps == 2
    assert stats.model_tokens == {"claude-opus-5-5": 1_000_000, "gpt-6.1-sol": 1_000_000}
    assert stats.cost.total == pytest.approx(24.0)
    assert stats.subagent_cost.total == pytest.approx(4.0)


def test_stats_large_context_and_unpriced():
    root = node("root", "Root", "claude", [ModelStep(step_index=1, started_at=None, input_tokens=60_000, cache_read_tokens=150_000)])
    stats = stats_for_node(root)
    assert stats.large_context_steps == 1
    assert stats.cost.unpriced_tokens == 210_000


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


def test_store_report_attribution_and_boundaries(tmp_path):
    from t3_telemetry.db_t3 import T3StateSnapshot
    from t3_telemetry.store import SessionRecord, StepRecord, TelemetryStore, utc_iso
    from t3_telemetry.usage import build_store_usage_report, read_store_usage_report

    store = TelemetryStore(tmp_path / 'usage.sqlite')
    store.upsert_sessions([
        SessionRecord('claude:main', 'claude', 'main'),
        SessionRecord('claude:child', 'claude', 'child', parent_session_key='claude:main'),
        SessionRecord('codex:bound', 'codex', 'bound'),
        SessionRecord('codex:free', 'codex', 'free', cwd='/work/project'),
    ])
    store.replace_t3_graph([
        {'thread_id': 'root', 'title': 'Root'},
        {'thread_id': 'child', 'title': 'Child', 'parent_thread_id': 'root'},
    ], [
        {'session_key': 'claude:main', 'thread_id': 'root'},
        {'session_key': 'codex:bound', 'thread_id': 'child'},
    ])
    store.upsert_steps([
        StepRecord('claude:main', 'a', utc_iso('2026-11-04T08:00:00Z'), input_tokens=1, cache_read_tokens=2),
        StepRecord('claude:child', 'b', utc_iso('2026-11-05T02:00:00Z'), cache_creation_tokens=3),
        StepRecord('codex:bound', 'c', utc_iso('2026-11-05T02:00:00Z'), output_tokens=4),
        StepRecord('codex:free', 'd', utc_iso('2026-11-05T02:00:00Z'), input_tokens=5),
        StepRecord('codex:free', 'end', utc_iso('2026-11-05T08:00:00Z'), input_tokens=6),
    ])
    report = build_store_usage_report(store, NOW, {'root': 'running'})
    today = report.periods['today']
    assert today.total.total == 15
    assert [(r.thread_id, r.title, r.status, r.tokens.total, r.providers) for r in today.threads] == [
        ('root', 'Root', 'running', 10, ['Claude', 'Codex']),
        ('session:codex:free', 'project · codex', 'idle', 5, ['Codex']),
    ]
    assert today.buckets[0] == {'Claude': 3}
    assert today.buckets[18] == {'Claude': 3, 'Codex': 9}
    assert report.periods['week'].total.total == report.periods['month'].total.total == 21
    assert read_store_usage_report(store, T3StateSnapshot(), NOW).root_total('today', 'root') == 10
    store.close()


def test_store_week_spans_month_and_cycles(tmp_path):
    from t3_telemetry.store import SessionRecord, StepRecord, TelemetryStore, utc_iso
    from t3_telemetry.usage import build_store_usage_report

    store = TelemetryStore(tmp_path / 'usage.sqlite')
    store.upsert_sessions([
        SessionRecord('codex:a', 'codex', 'a', parent_session_key='codex:b'),
        SessionRecord('codex:b', 'codex', 'b', parent_session_key='codex:a'),
        SessionRecord('claude:c', 'claude', 'c'),
    ])
    store.replace_t3_graph([
        {'thread_id': 'x', 'parent_thread_id': 'y'},
        {'thread_id': 'y', 'parent_thread_id': 'x'},
    ], [{'session_key': 'claude:c', 'thread_id': 'x'}])
    store.upsert_steps([
        StepRecord('codex:a', 'a', utc_iso('2026-10-30T12:00:00Z'), input_tokens=2),
        StepRecord('codex:b', 'b', utc_iso('2026-11-01T12:00:00Z'), input_tokens=3),
        StepRecord('claude:c', 'c', utc_iso('2026-11-01T12:00:00Z'), input_tokens=4),
    ])
    report = build_store_usage_report(store, datetime(2026, 11, 1, 12, tzinfo=ZONE))
    assert report.periods['week'].total.total == 9
    assert report.periods['month'].total.total == 7
    assert report.root_total('week', 'session:codex:a') == 5
    assert report.root_total('week', 'x') == 4
    store.close()


def test_store_report_scope_and_session_model(tmp_path):
    from t3_telemetry.store import SessionRecord, StepRecord, TelemetryStore, utc_iso
    from t3_telemetry.usage import build_store_usage_report

    store = TelemetryStore(tmp_path / 'usage.sqlite')
    store.upsert_sessions([
        SessionRecord('main', 'codex', 'main'),
        SessionRecord('child', 'codex', 'child', model='gpt-6.1-sol', parent_session_key='main'),
        SessionRecord('bound', 'codex', 'bound'),
    ])
    store.replace_t3_graph([
        {'thread_id': 'root'},
        {'thread_id': 'child-thread', 'parent_thread_id': 'root'},
    ], [{'session_key': 'bound', 'thread_id': 'child-thread'}])
    store.upsert_steps([
        StepRecord('main', 'a', utc_iso('2026-11-05T02:00:00Z'), input_tokens=10),
        StepRecord('child', 'b', utc_iso('2026-11-05T02:00:00Z'), model=None, input_tokens=1_000_000),
        StepRecord('bound', 'c', utc_iso('2026-11-05T02:00:00Z'), output_tokens=20),
    ])
    report = build_store_usage_report(store, NOW)
    today = report.periods['today']
    assert today.scope_buckets[18] == {'Main': 10, 'Subagents': 1_000_020}
    assert today.stats.subagent_tokens.total == 1_000_020
    assert today.stats.model_cost['gpt-6.1-sol'] == pytest.approx(4.0)
    for usage in report.periods.values():
        assert sum(sum(b.values()) for b in usage.scope_buckets) == sum(sum(b.values()) for b in usage.buckets)
        assert usage.stats.tokens.total == usage.total.total
    store.close()


def priced(index, date, model_tokens=1_000_000):
    return ModelStep(step_index=index, started_at=date, input_tokens=model_tokens)


def test_cost_series_lifetime_and_top_order():
    # claude-opus-5-5 input is $4/M; claude-sonnet-5-5 is $2/M (neither has a long-context tier).
    cheap = ThreadTelemetryNode(thread_id="cheap", title="Cheap", driver="codex", provider_instance_id=None,
                                model="claude-sonnet-5-5", steps=[priced(1, "2026-11-05T02:00:00Z", 3_000_000)])
    child = ThreadTelemetryNode(thread_id="kid", title="Kid", driver="claude", provider_instance_id=None,
                                model="claude-opus-5-5", steps=[priced(1, "2026-11-05T02:00:00Z")])
    pricey = ThreadTelemetryNode(thread_id="pricey", title="Pricey", driver="claude", provider_instance_id=None,
                                 model="claude-opus-5-5", steps=[priced(1, "2026-11-05T02:00:00Z"), priced(2, "2026-10-01T02:00:00Z")],
                                 subagents=[child])
    report = build_usage_report([cheap, pricey], NOW)
    today = report.periods["today"]
    assert today.stats.cost.total == pytest.approx(14.0)
    assert sum(sum(b.values()) for b in today.cost_buckets) == pytest.approx(14.0)
    assert sum(sum(b.values()) for b in today.scope_cost_buckets) == pytest.approx(14.0)
    assert today.scope_cost_buckets[18] == {"Main": pytest.approx(10.0), "Subagents": pytest.approx(4.0)}
    assert {k: round(v.total, 2) for k, v in today.cost_by_provider.items()} == {"Claude": 8.0, "Codex": 6.0}
    assert today.series("provider", "usd") is today.cost_buckets
    assert today.series("scope", "tokens") is today.scope_buckets
    # Token order puts Cheap (3M) ahead of Pricey (2M); dollar order flips it.
    assert [t.thread_id for t in today.top("tokens")] == ["cheap", "pricey"]
    assert [t.thread_id for t in today.top("usd")] == ["pricey", "cheap"]
    assert today.total_value("usd") == pytest.approx(14.0)
    assert report.root_total("today", "pricey", "usd") == pytest.approx(8.0)
    # Lifetime totals include steps outside every period window.
    assert report.lifetime("pricey") == 3_000_000
    assert report.lifetime("pricey", "usd") == pytest.approx(12.0)
    assert report.lifetime("missing") is None


def test_node_and_turn_cost():
    from t3_telemetry.models import TurnMetrics
    from t3_telemetry.usage import node_cost, turn_cost

    child = ThreadTelemetryNode(thread_id="kid", title="Kid", driver="claude", provider_instance_id=None,
                                model="claude-opus-5-5", steps=[priced(1, "2026-11-05T02:00:00Z")])
    s1 = priced(1, "2026-11-05T02:00:00Z")
    s1.provider_turn_id = "t1"
    s2 = priced(2, "2026-11-05T03:00:00Z", 500_000)
    s2.provider_turn_id = "t2"
    root = ThreadTelemetryNode(thread_id="root", title="Root", driver="claude", provider_instance_id=None,
                               model="claude-opus-5-5", steps=[s1, s2], subagents=[child])
    assert node_cost(root, rollup=False).total == pytest.approx(6.0)
    assert node_cost(root).total == pytest.approx(10.0)
    assert turn_cost(root) is None
    root.current_turn_metrics = TurnMetrics(turn_id="t2")
    assert turn_cost(root).total == pytest.approx(2.0)
    root.current_turn_metrics = TurnMetrics(turn_id="other")
    assert turn_cost(root) is None


def test_store_report_costs_and_lifetime(tmp_path):
    from t3_telemetry.store import SessionRecord, StepRecord, TelemetryStore, utc_iso
    from t3_telemetry.usage import build_store_usage_report

    store = TelemetryStore(tmp_path / 'usage.sqlite')
    store.upsert_sessions([SessionRecord('claude:main', 'claude', 'main', model='claude-opus-5-5'),
                           SessionRecord('codex:free', 'codex', 'free', model='mystery-model')])
    store.replace_t3_graph([{'thread_id': 'root', 'title': 'Root'}], [{'session_key': 'claude:main', 'thread_id': 'root'}])
    store.upsert_steps([
        StepRecord('claude:main', 'a', utc_iso('2026-11-05T02:00:00Z'), input_tokens=1_000_000),
        StepRecord('claude:main', 'old', utc_iso('2025-01-05T02:00:00Z'), input_tokens=1_000_000),
        StepRecord('codex:free', 'u', utc_iso('2026-11-05T02:00:00Z'), input_tokens=7),
    ])
    report = build_store_usage_report(store, NOW)
    today = report.periods['today']
    assert today.cost_buckets[18] == {'Claude': pytest.approx(4.0)}
    assert [(t.thread_id, round(t.cost.total, 2)) for t in today.top('usd')] == [('root', 4.0), ('session:codex:free', 0.0)]
    assert report.lifetime('root') == 2_000_000
    assert report.lifetime('root', 'usd') == pytest.approx(8.0)
    assert report.all_time['session:codex:free'].cost.unpriced_tokens == 7
    store.close()


def test_provider_split_and_serialization():
    from t3_telemetry.models import TurnMetrics
    from t3_telemetry.usage import provider_breakdown

    root_step = ModelStep(step_index=1, started_at=None, input_tokens=10, cache_read_tokens=2,
                          provider="claude", turn_key="session|turn", model="gpt-6.1-sol")
    child_step = ModelStep(step_index=1, started_at=None, cache_creation_tokens=3, output_tokens=4,
                           provider="codex", turn_key="child|turn")
    child = node("child", "Child", "codex", [child_step, child_step])
    root = node("root", "Root", "claude", [root_step], [child])
    root.rollup_metrics.input_tokens = 10
    root.rollup_metrics.cache_read_tokens = 2
    root.rollup_metrics.cache_creation_tokens = 3
    root.rollup_metrics.output_tokens = 4
    root.current_turn_metrics = TurnMetrics(turn_id="turn")
    stats = stats_for_node(root)
    assert {k: v.total for k, v in stats.provider_tokens.items()} == {"Claude": 12, "Codex": 7}
    assert stats.model_tokens["gpt-6.1-sol"] == 12
    split = provider_breakdown(root)
    assert split["claude"] == {"input_tokens": 10, "cache_read_tokens": 2,
                                "cache_creation_tokens": 0, "output_tokens": 0, "total_tokens": 12}
    assert split["codex"]["total_tokens"] == 7
    data = root.to_dict()
    assert data["by_provider"] == split
    assert sum(v["total_tokens"] for v in split.values()) == data["rollup_metrics"]["total_tokens"]
    assert data["current_turn_metrics"]["turn_id"] == "turn"
