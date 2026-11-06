import sys
import io
import json
import sqlite3
from pathlib import Path
import pytest
from rich.console import Console

from t3_telemetry.models import (
    ModelStep,
    ThreadTelemetryNode,
    TokenMetricsSummary,
)
from t3_telemetry.ui import (
    format_tokens,
    format_duration,
    render_context_bar,
    build_dashboard_renderable,
    build_threads_list_table,
    build_sessions_table,
)
from t3_telemetry.cli import _browser_frame, build_parser, main, run_live_loop
from t3_telemetry.tui import NavigationState, ScreenState, TerminalAdapter


def test_format_tokens():
    assert format_tokens(None) == "0"
    assert format_tokens(0) == "0"
    assert format_tokens(-5) == "0"
    assert format_tokens(500) == "500"
    assert format_tokens(999) == "999"
    assert format_tokens(1000) == "1.0k"
    assert format_tokens(1540) == "1.5k"
    assert format_tokens(999_999) == "1000.0k"
    assert format_tokens(1_000_000) == "1.00M"
    assert format_tokens(1_048_576) == "1.05M"


def test_format_duration():
    assert format_duration(None) == "0.0s"
    assert format_duration(0.0) == "0.0s"
    assert format_duration(-1.0) == "0.0s"
    assert format_duration(5.23) == "5.2s"
    assert format_duration(59.94) == "59.9s"
    assert format_duration(60.0) == "1m00s"
    assert format_duration(75.5) == "1m15s"
    assert format_duration(3665.0) == "61m05s"


def test_render_context_bar():
    # Green (< 60%)
    bar_green = render_context_bar(300_000, 1_000_000, width=20)
    assert "green" in bar_green
    assert "300.0k / 1.00M" in bar_green
    assert "30.0%" in bar_green

    # Yellow (60% - 85%)
    bar_yellow = render_context_bar(700_000, 1_000_000, width=20)
    assert "yellow" in bar_yellow
    assert "700.0k / 1.00M" in bar_yellow
    assert "70.0%" in bar_yellow

    # Red (> 85%)
    bar_red = render_context_bar(900_000, 1_000_000, width=20)
    assert "red" in bar_red
    assert "900.0k / 1.00M" in bar_red
    assert "90.0%" in bar_red

    # Zero max tokens edge case
    bar_zero = render_context_bar(100, 0, width=20)
    assert "0.0%" in bar_zero


def test_build_dashboard_renderable_self_only():
    step1 = ModelStep(
        step_index=1,
        source="main",
        tool_calls=["read_file"],
        input_tokens=1000,
        cache_read_tokens=500,
        thinking_tokens=200,
        response_tokens=100,
        output_tokens=300,
        duration_seconds=1.2,
    )
    self_summary = TokenMetricsSummary(
        input_tokens=1000,
        cache_read_tokens=500,
        thinking_tokens=200,
        response_tokens=100,
        output_tokens=300,
        step_count=1,
        duration_seconds=1.2,
    )
    node = ThreadTelemetryNode(
        thread_id="th-1234567890abcdef",
        title="Main Thread Working On Telemetry",
        driver="antigravity",
        provider_instance_id="antigravity_gemini",
        model="gemini-3.8-flash",
        thinking_level="high",
        status="running",
        context_used_tokens=1500,
        context_max_tokens=1048576,
        self_metrics=self_summary,
        rollup_metrics=self_summary,
        steps=[step1],
        combined_steps=[step1],
    )

    renderable = build_dashboard_renderable(node, max_steps=10)

    # Render to narrow 60-col console
    buf_narrow = io.StringIO()
    console_narrow = Console(file=buf_narrow, width=60, color_system=None)
    console_narrow.print(renderable)
    output_narrow = buf_narrow.getvalue()
    assert "Main Thread" in output_narrow
    assert "th-12345" in output_narrow
    assert "RUNNING" in output_narrow
    assert "Whole chat" in output_narrow

    # Render to standard wide console
    buf_wide = io.StringIO()
    console_wide = Console(file=buf_wide, width=120, color_system=None)
    console_wide.print(renderable)
    output_wide = buf_wide.getvalue()
    assert "read_file" in output_wide
    assert "Whole chat · this agent" in output_wide
    assert "Current turn (running)" in output_wide
    assert "Chat + subagents" in output_wide


def test_build_dashboard_renderable_with_subagents():
    child_step = ModelStep(
        step_index=1,
        source="sub-1",
        tool_calls=["pytest"],
        input_tokens=2000,
        cache_read_tokens=1000,
        thinking_tokens=300,
        response_tokens=150,
        output_tokens=450,
        duration_seconds=2.5,
    )
    child_summary = TokenMetricsSummary(
        input_tokens=2000,
        cache_read_tokens=1000,
        thinking_tokens=300,
        response_tokens=150,
        output_tokens=450,
        step_count=1,
        duration_seconds=2.5,
    )
    child_node = ThreadTelemetryNode(
        thread_id="child-subagent-1",
        title="Research Subagent",
        driver="antigravity",
        provider_instance_id="antigravity_gemini",
        model="gemini-3.8-flash",
        status="completed",
        self_metrics=child_summary,
        rollup_metrics=child_summary,
        steps=[child_step],
        combined_steps=[child_step],
    )

    parent_summary = TokenMetricsSummary(
        input_tokens=1000,
        cache_read_tokens=500,
        thinking_tokens=100,
        response_tokens=50,
        output_tokens=150,
        step_count=1,
        duration_seconds=1.0,
    )
    rollup_summary = TokenMetricsSummary(
        input_tokens=3000,
        cache_read_tokens=1500,
        thinking_tokens=400,
        response_tokens=200,
        output_tokens=600,
        step_count=2,
        total_subagents=1,
        duration_seconds=3.5,
    )
    parent_step = ModelStep(
        step_index=1,
        source="main",
        tool_calls=["delegate_task"],
        input_tokens=1000,
        cache_read_tokens=500,
        thinking_tokens=100,
        response_tokens=50,
        output_tokens=150,
        duration_seconds=1.0,
    )
    parent_node = ThreadTelemetryNode(
        thread_id="parent-thread-id",
        title="Root Orchestrator",
        driver="codex",
        provider_instance_id="codex",
        model="gpt-5-codex",
        status="idle",
        self_metrics=parent_summary,
        rollup_metrics=rollup_summary,
        steps=[parent_step],
        combined_steps=[parent_step, child_step],
        subagents=[child_node],
    )

    renderable = build_dashboard_renderable(parent_node, max_steps=10)
    buf = io.StringIO()
    console = Console(file=buf, width=120, color_system=None)
    console.print(renderable)
    output = buf.getvalue()

    assert "Subagents & Delegated Runs" in output
    assert "Research Subagent" in output
    assert "sub-1" in output
    assert "gpt-5-codex" in output
    assert "IDLE" in output


def test_build_threads_list_table():
    summary = TokenMetricsSummary(
        input_tokens=5000,
        cache_read_tokens=2500,
        thinking_tokens=600,
        response_tokens=300,
        output_tokens=900,
        step_count=5,
        total_subagents=2,
        duration_seconds=15.0,
    )
    node = ThreadTelemetryNode(
        thread_id="th-abcdef123456",
        title="Feature Thread",
        driver="claude",
        provider_instance_id="claude",
        model="claude-opus-5-5",
        status="running",
        context_used_tokens=12000,
        context_max_tokens=200000,
        self_metrics=summary,
        rollup_metrics=summary,
    )

    table = build_threads_list_table([node])
    buf = io.StringIO()
    console = Console(file=buf, width=120, color_system=None)
    console.print(table)
    output = buf.getvalue()

    assert "Recent T3 Code App Threads" in output
    assert "th-abcde" in output
    assert "Feature" in output and "Thread" in output
    assert "claude" in output
    assert "5.0k" in output


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE orchestration_v2_projection_threads (
        thread_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        title TEXT NOT NULL,
        default_provider TEXT NOT NULL,
        runtime_mode TEXT NOT NULL,
        interaction_mode TEXT NOT NULL,
        active_provider_thread_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        archived_at TEXT,
        deleted_at TEXT,
        payload_json TEXT NOT NULL,
        provider_instance_id TEXT
    );

    CREATE TABLE orchestration_v2_projection_provider_threads (
        provider_thread_id TEXT PRIMARY KEY,
        thread_id TEXT,
        owner_node_id TEXT,
        provider TEXT NOT NULL,
        provider_session_id TEXT,
        status TEXT NOT NULL,
        first_run_ordinal INTEGER,
        last_run_ordinal INTEGER,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        driver TEXT,
        provider_instance_id TEXT
    );

    CREATE TABLE orchestration_v2_projection_provider_turns (
        provider_turn_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        provider_thread_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        run_attempt_id TEXT,
        ordinal INTEGER NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE orchestration_v2_projection_subagents (
        subagent_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        run_id TEXT,
        parent_node_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        provider_thread_id TEXT,
        child_thread_id TEXT,
        origin TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        driver TEXT,
        provider_instance_id TEXT,
        completion_delivery_state TEXT
    );
    """)


@pytest.fixture
def synthetic_t3_dir(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = sqlite3.connect(db_file)
    create_schema(conn)

    t1_payload = {"modelSelection": {"model": "gemini-3.8-flash", "thinkingLevel": "high"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, archived_at, deleted_at, payload_json, provider_instance_id)
    VALUES ('th-root-alpha', 'proj-1', 'Alpha Implementation', 'antigravity', 'full-access', 'default',
            'pt-alpha', '2026-11-03T10:00:00Z', '2026-11-03T10:30:00Z', NULL, NULL, ?, 'antigravity_gemini')
    """, (json.dumps(t1_payload),))

    pt_alpha = {"contextUsage": {"usedTokens": 45000, "maxTokens": 1048576}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES ('pt-alpha', 'th-root-alpha', 'node-1', 'antigravity', 'idle', '2026-11-03T10:30:00Z',
            ?, 'antigravity', 'antigravity_gemini')
    """, (json.dumps(pt_alpha),))

    t2_payload = {"modelSelection": {"model": "claude-opus-5-5"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, archived_at, deleted_at, payload_json, provider_instance_id)
    VALUES ('th-root-beta', 'proj-1', 'Beta Architecture Design', 'claude', 'full-access', 'default',
            'pt-beta', '2026-11-03T09:00:00Z', '2026-11-03T09:30:00Z', NULL, NULL, ?, 'claude')
    """, (json.dumps(t2_payload),))

    conn.commit()
    conn.close()
    return tmp_path


def test_cli_list(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["--list", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Recent T3 Code App Threads" in captured
    assert "th-root-" in captured
    assert "45.0k" in captured


def test_cli_list_json(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["--list", "--json", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["thread_id"] == "th-root-alpha"


def test_cli_once_default_latest(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["--once", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Alpha Implementation" in captured
    assert "gemini-3.8-flash" in captured


def test_cli_once_query_by_id_prefix(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["th-root-b", "--once", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Beta Architecture Design" in captured
    assert "claude-opus-5-5" in captured


def test_cli_once_query_by_title_substring(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["architecture", "--once", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Beta Architecture Design" in captured


def test_cli_once_not_found(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["nonexistent", "--once", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 1
    err = capsys.readouterr().err
    assert "No thread found" in err


def test_cli_json_found(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["th-root-alpha", "--json", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert data["thread_id"] == "th-root-alpha"
    assert data["title"] == "Alpha Implementation"
    assert "rollup_metrics" in data


def test_cli_json_not_found(synthetic_t3_dir: Path, capsys: pytest.CaptureFixture):
    code = main(["unknown-id", "--json", "--t3-dir", str(synthetic_t3_dir)])
    assert code == 1
    err = capsys.readouterr().err
    assert "No thread found" in err


def test_cli_live_loop_max_iterations(synthetic_t3_dir: Path):
    parser = build_parser()
    args = parser.parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir), "--no-follow", "--interval", "0.01"])
    buf = io.StringIO()
    test_console = Console(file=buf, width=120, color_system=None)
    # Live loop with _max_iterations=2 must exit cleanly with code 0
    code = run_live_loop(args, console=test_console, _max_iterations=2)
    assert code == 0


# ==============================================================================
# Comprehensive Event Loop, Viewport, and CLI Behavior Tests
# ==============================================================================

class CaptureLive:
    def __init__(self, *args, **kwargs):
        self.updates = []
        self.console = kwargs.get("console")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, renderable, refresh=True):
        self.updates.append(renderable)


class ScriptedAdapter(TerminalAdapter):
    def __init__(self, actions, initial_size=(24, 80)):
        self.actions = list(actions)
        self.current_size = initial_size
        self.began = False
        self.ended = False

    def begin(self):
        self.began = True

    def end(self):
        self.ended = True

    def size(self):
        return self.current_size

    def read_key(self, timeout):
        if not self.actions:
            return "q"
        action = self.actions.pop(0)
        if isinstance(action, tuple):
            self.current_size, key = action
            return key
        if isinstance(action, BaseException):
            raise action
        return action


def test_cli_invalid_intervals_fail_cleanly(capsys: pytest.CaptureFixture):
    for bad in ["0", "-1", "nan", "inf"]:
        with pytest.raises(SystemExit) as exc_info:
            main(["--interval", bad])
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "--interval must be a finite positive number" in err


def test_event_loop_input_repaint_promptly_without_extra_db_poll(
    synthetic_t3_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    import t3_telemetry.cli as cli_mod

    poll_count = 0
    orig_read_t3_state = cli_mod.read_t3_state

    def counted_read_t3_state(p):
        nonlocal poll_count
        poll_count += 1
        return orig_read_t3_state(p)

    monkeypatch.setattr(cli_mod, "read_t3_state", counted_read_t3_state)

    args = build_parser().parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir), "--interval", "10.0"])
    adapter = ScriptedAdapter(["DOWN", "q"])
    capture = CaptureLive()

    t = 0.0

    def clock():
        nonlocal t
        t += 0.01
        return t

    code = run_live_loop(args, _adapter=adapter, _clock=clock, _live_cls=lambda **kw: capture)
    assert code == 0
    # Must have polled DB only once for the scheduled poll, not for the input key
    assert poll_count == 1
    # Must have rendered initial frame and repainted on DOWN key
    assert len(capture.updates) >= 2
    assert adapter.ended is True


def test_event_loop_resize_repaint_promptly_without_extra_db_poll(
    synthetic_t3_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    import t3_telemetry.cli as cli_mod

    poll_count = 0
    orig_read_t3_state = cli_mod.read_t3_state

    def counted_read_t3_state(p):
        nonlocal poll_count
        poll_count += 1
        return orig_read_t3_state(p)

    monkeypatch.setattr(cli_mod, "read_t3_state", counted_read_t3_state)

    args = build_parser().parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir), "--interval", "10.0"])
    adapter = ScriptedAdapter([((30, 100), None), "q"], initial_size=(24, 80))
    capture = CaptureLive()

    t = 0.0

    def clock():
        nonlocal t
        t += 0.01
        return t

    code = run_live_loop(args, _adapter=adapter, _clock=clock, _live_cls=lambda **kw: capture)
    assert code == 0
    assert poll_count == 1
    # Live update was triggered for the resize promptly
    assert len(capture.updates) >= 2


def test_event_loop_one_snapshot_per_scheduled_poll_including_query(
    synthetic_t3_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    import t3_telemetry.cli as cli_mod

    poll_count = 0
    orig_read_t3_state = cli_mod.read_t3_state

    def counted_read_t3_state(p):
        nonlocal poll_count
        poll_count += 1
        return orig_read_t3_state(p)

    monkeypatch.setattr(cli_mod, "read_t3_state", counted_read_t3_state)

    args = build_parser().parse_args(["--no-store", "Beta", "--t3-dir", str(synthetic_t3_dir), "--interval", "0.5"])
    capture = CaptureLive()

    t = 0.0

    def clock():
        nonlocal t
        t += 0.6
        return t

    code = run_live_loop(
        args, _adapter=ScriptedAdapter([None, None]), _clock=clock, _max_iterations=2, _live_cls=lambda **kw: capture
    )
    assert code == 0
    # Exactly one snapshot per scheduled poll, and query resolved from that snapshot
    assert poll_count == 2


def test_event_loop_same_root_build_reuse(
    synthetic_t3_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    import t3_telemetry.cli as cli_mod

    build_counts = {}
    orig_build = cli_mod.build_thread_telemetry

    def counted_build(tid, t3_dir, **kwargs):
        build_counts[tid] = build_counts.get(tid, 0) + 1
        return orig_build(tid, t3_dir, **kwargs)

    monkeypatch.setattr(cli_mod, "build_thread_telemetry", counted_build)

    args = build_parser().parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir), "--interval", "10.0"])
    adapter = ScriptedAdapter(["q"])
    run_live_loop(args, _adapter=adapter, _max_iterations=1, _live_cls=lambda **kw: CaptureLive())

    # th-root-alpha is both summary and selected root, so it must be built once per poll
    assert build_counts.get("th-root-alpha") == 1


def test_event_loop_unchanged_poll_no_redraw(synthetic_t3_dir: Path):
    args = build_parser().parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir), "--interval", "1.0"])
    capture = CaptureLive()

    t = 0.0

    def clock():
        nonlocal t
        t += 1.5
        return t

    adapter = ScriptedAdapter([None, None])
    code = run_live_loop(
        args, _adapter=adapter, _clock=clock, _max_iterations=2, _live_cls=lambda **kw: capture
    )
    assert code == 0
    # Redrawn on poll 1, but poll 2 has identical data so no redundant redraw
    assert len(capture.updates) == 1


def test_event_loop_selected_follow_pin_independence(synthetic_t3_dir: Path):
    db_file = synthetic_t3_dir / "statev2.sqlite"
    args = build_parser().parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir), "--interval", "1.0"])
    capture = CaptureLive()

    t = 0.0

    class FollowAdapter(TerminalAdapter):
        def __init__(self):
            self.calls = 0

        def read_key(self, timeout):
            nonlocal t
            self.calls += 1
            if self.calls == 1:
                conn = sqlite3.connect(db_file)
                conn.execute(
                    "UPDATE orchestration_v2_projection_threads SET updated_at = '2026-11-03T11:00:00Z' WHERE thread_id = 'th-root-beta'"
                )
                conn.commit()
                conn.close()
                t += 2.0
                return None
            return "q"

    run_live_loop(
        args, _adapter=FollowAdapter(), _clock=lambda: t, _max_iterations=2, _live_cls=lambda **kw: capture
    )
    assert len(capture.updates) >= 2
    frame2 = capture.updates[-1]
    buf = io.StringIO()
    Console(file=buf, width=80, height=24).print(frame2)
    out = buf.getvalue()
    assert "Beta Architecture Design" in out


def test_event_loop_pinning_with_no_follow(synthetic_t3_dir: Path):
    db_file = synthetic_t3_dir / "statev2.sqlite"
    args = build_parser().parse_args(
        ["--no-store", "--t3-dir", str(synthetic_t3_dir), "--no-follow", "--interval", "1.0"]
    )
    capture = CaptureLive()

    t = 0.0

    class PinAdapter(TerminalAdapter):
        def __init__(self):
            self.calls = 0

        def read_key(self, timeout):
            nonlocal t
            self.calls += 1
            if self.calls == 1:
                conn = sqlite3.connect(db_file)
                conn.execute(
                    "UPDATE orchestration_v2_projection_threads SET updated_at = '2026-11-03T11:00:00Z' WHERE thread_id = 'th-root-beta'"
                )
                conn.commit()
                conn.close()
                t += 2.0
                return None
            return "q"

    run_live_loop(
        args, _adapter=PinAdapter(), _clock=lambda: t, _max_iterations=2, _live_cls=lambda **kw: capture
    )
    frame2 = capture.updates[-1]
    buf = io.StringIO()
    Console(file=buf, width=80, height=24).print(frame2)
    out = buf.getvalue()
    # Pinned to Alpha, so Alpha stays in summary header
    assert "Alpha Implementation" in out


def test_event_loop_cleanup_on_interrupt_and_error(synthetic_t3_dir: Path):
    args = build_parser().parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir), "--interval", "1.0"])

    adapter_interrupt = ScriptedAdapter([KeyboardInterrupt()])
    code = run_live_loop(args, _adapter=adapter_interrupt, _live_cls=lambda **kw: CaptureLive())
    assert code == 0
    assert adapter_interrupt.ended is True

    adapter_error = ScriptedAdapter([RuntimeError("terminal read failure")])
    with pytest.raises(RuntimeError, match="terminal read failure"):
        run_live_loop(args, _adapter=adapter_error, _live_cls=lambda **kw: CaptureLive())
    assert adapter_error.ended is True


def test_noninteractive_fallback(synthetic_t3_dir: Path, monkeypatch: pytest.MonkeyPatch):
    args = build_parser().parse_args(["--no-store", "--t3-dir", str(synthetic_t3_dir)])
    buf = io.StringIO()
    fake_console = Console(file=buf, force_terminal=False, color_system=None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    code = run_live_loop(args, console=fake_console)
    assert code == 0
    out = buf.getvalue()
    assert "Alpha Implementation" in out


def test_activity_isolation_parent_child_grandchild():
    step_gc = ModelStep(step_index=1, tool_calls=["act_gamma"], input_tokens=100)
    grandchild = ThreadTelemetryNode(
        thread_id="th-grandchild",
        title="Grandchild Node",
        driver="antigravity",
        provider_instance_id=None,
        model="gemini-flash",
        steps=[step_gc],
    )

    step_child = ModelStep(step_index=1, tool_calls=["act_beta"], input_tokens=200)
    child = ThreadTelemetryNode(
        thread_id="th-child",
        title="Child Node",
        driver="antigravity",
        provider_instance_id=None,
        model="gemini-flash",
        steps=[step_child],
        subagents=[grandchild],
    )

    step_parent = ModelStep(step_index=1, tool_calls=["act_alpha"], input_tokens=500)
    parent = ThreadTelemetryNode(
        thread_id="th-parent",
        title="Parent Root Node",
        driver="codex",
        provider_instance_id=None,
        model="gpt-5-codex",
        steps=[step_parent],
        combined_steps=[step_parent, step_child, step_gc],
        subagents=[child],
    )

    nav = NavigationState()
    nav.reconcile(["th-parent"])
    nav.enter()

    # 1. Render Parent View: Only parent's own steps appear, NOT child or grandchild steps
    layout_parent = _browser_frame([], node=parent, summary=parent, nav=nav, height=30, width=100)
    buf_p = io.StringIO()
    Console(file=buf_p, width=100, height=30).print(layout_parent)
    out_p = buf_p.getvalue()
    assert "act_alpha" in out_p
    assert "act_beta" not in out_p
    assert "act_gamma" not in out_p

    # 2. Render Child View: Only child's own steps appear
    nav.stack.append(ScreenState(node_id="th-child", title="Child Node"))
    layout_child = _browser_frame([], node=child, summary=None, nav=nav, height=30, width=100)
    buf_c = io.StringIO()
    Console(file=buf_c, width=100, height=30).print(layout_child)
    out_c = buf_c.getvalue()
    assert "act_beta" in out_c
    assert "act_alpha" not in out_c
    assert "act_gamma" not in out_c

    # 3. Render Grandchild View: Only grandchild's own steps appear
    nav.stack.append(ScreenState(node_id="th-grandchild", title="Grandchild Node"))
    layout_gc = _browser_frame([], node=grandchild, summary=None, nav=nav, height=30, width=100)
    buf_gc = io.StringIO()
    Console(file=buf_gc, width=100, height=30).print(layout_gc)
    out_gc = buf_gc.getvalue()
    assert "act_gamma" in out_gc
    assert "act_alpha" not in out_gc
    assert "act_beta" not in out_gc


def test_missing_child_telemetry_renders_usable_screen():
    empty_child = ThreadTelemetryNode(
        thread_id="th-empty-child",
        title="Empty Native Child",
        driver="antigravity",
        provider_instance_id=None,
        model="gemini-flash",
        status="completed",
        steps=[],
    )

    nav = NavigationState()
    nav.stack = [ScreenState(node_id="th-empty-child", title=empty_child.title)]

    layout = _browser_frame([], node=empty_child, summary=empty_child, nav=nav, height=24, width=80)
    buf = io.StringIO()
    Console(file=buf, width=80, height=24).print(layout)
    out = buf.getvalue()

    assert "Empty Native Child" in out
    assert "No recorded activity for this agent." in out
    assert "COMPLETED" in out


def test_query_sets_selected_id_so_escape_lands_on_queried_chat(synthetic_t3_dir: Path):
    args = build_parser().parse_args(["--no-store",
        "--t3-dir", str(synthetic_t3_dir),
        "Beta",
    ])
    capture = CaptureLive()

    class EscAdapter(TerminalAdapter):
        def __init__(self):
            self.calls = 0

        def read_key(self, timeout):
            self.calls += 1
            if self.calls == 1:
                return "ESC"
            return "q"

    adapter = EscAdapter()
    run_live_loop(args, _adapter=adapter, _max_iterations=2, _live_cls=lambda **kw: capture)

    # After ESC, user is back on the list view and selection is on the queried chat th-root-beta
    last_frame = capture.updates[-1]
    buf = io.StringIO()
    Console(file=buf, width=80, height=24).print(last_frame)
    out = buf.getvalue()
    # The selection marker ▶ must be next to Beta Architecture Design
    assert any("▶" in line and "Beta Architecture" in line for line in out.splitlines())


def test_snapshot_list_and_subagent_table_title_status_colors_no_status_column():
    running_node = ThreadTelemetryNode(
        thread_id="th-run-1234",
        title="Running Thread",
        driver="codex",
        provider_instance_id=None,
        model="gpt-5-codex",
        status="running",
    )
    failed_node = ThreadTelemetryNode(
        thread_id="th-fail-1234",
        title="Failed Thread",
        driver="claude",
        provider_instance_id=None,
        model="claude-opus-5-5",
        status="failed",
    )

    # 1. Snapshot list table: No dedicated Status column
    table = build_threads_list_table([running_node, failed_node])
    col_names = [col.header for col in table.columns]
    assert "Status" not in col_names

    # 2. Subagents table in dashboard renderable: No dedicated Status column
    parent = ThreadTelemetryNode(
        thread_id="th-parent",
        title="Parent Thread",
        driver="codex",
        provider_instance_id=None,
        model="gpt-5-codex",
        status="idle",
        subagents=[running_node, failed_node],
    )
    renderable = build_dashboard_renderable(parent)
    buf = io.StringIO()
    Console(file=buf, width=120).print(renderable)
    out = buf.getvalue()

    assert "Running Thread" in out
    assert "Failed Thread" in out


def test_live_renderer_scopes_running_with_exact_turn():
    """Live renderer _browser_frame displays the 3 scopes with Current turn (running) when running."""
    from t3_telemetry.models import TurnMetrics

    turn = TurnMetrics(
        ordinal=2,
        turn_id="pturn-2",
        status="running",
        availability="available",
        input_tokens=1500,
        output_tokens=300,
        duration_seconds=2.0,
        step_count=1,
        provider_turn_count=1,
    )
    self_sum = TokenMetricsSummary(
        input_tokens=4000,
        output_tokens=800,
        step_count=2,
        provider_turn_count=2,
        duration_seconds=5.0,
    )
    step1 = ModelStep(
        step_index=1,
        activity_kind="provider_turn",
        provider_turn_id="pturn-1",
        input_tokens=2500,
        output_tokens=500,
        duration_seconds=3.0,
        tool_calls=[],
    )
    step2 = ModelStep(
        step_index=2,
        activity_kind="provider_turn",
        provider_turn_id="pturn-2",
        input_tokens=1500,
        output_tokens=300,
        duration_seconds=2.0,
        tool_calls=[],
    )
    node = ThreadTelemetryNode(
        thread_id="th-running-prov",
        title="Running Provider Thread",
        driver="claudeAgent",
        provider_instance_id="claude_inst",
        model="claude-3-7-sonnet",
        status="running",
        self_metrics=self_sum,
        rollup_metrics=self_sum,
        current_turn_metrics=turn,
        turn_metrics_availability="available",
        steps=[step1, step2],
    )

    nav = NavigationState()
    nav.reconcile(["th-running-prov"])
    nav.enter()

    layout = _browser_frame([], node=node, summary=node, nav=nav, height=28, width=100)
    buf = io.StringIO()
    Console(file=buf, width=100, height=28).print(layout)
    out = buf.getvalue()

    # Scope 1: Current turn (running)
    assert "Current turn (running)" in out
    # Scope 2: Whole chat · this agent
    assert "Whole chat · this agent" in out
    # Scope 3: Chat + subagents
    assert "Chat + subagents" in out

    # Activity stream shows explicit Provider turn #N and unknown tool when none provided
    assert "Provider turn #1" in out
    assert "Provider turn #2" in out
    assert "unknown" in out


def test_live_renderer_scopes_idle_with_unavailable_turn():
    """Live renderer _browser_frame displays Latest completed turn (idle) as unavailable for Antigravity."""
    self_sum = TokenMetricsSummary(
        input_tokens=2000,
        output_tokens=400,
        step_count=2,
        model_step_count=2,
        duration_seconds=3.0,
    )
    step1 = ModelStep(
        step_index=1,
        activity_kind="model_step",
        input_tokens=1000,
        output_tokens=200,
        tool_calls=["read_file"],
    )
    step2 = ModelStep(
        step_index=2,
        activity_kind="model_step",
        input_tokens=1000,
        output_tokens=200,
        tool_calls=[],
        thinking_tokens=100,
        response_tokens=100,
    )
    node = ThreadTelemetryNode(
        thread_id="th-idle-agy",
        title="Idle Antigravity Chat",
        driver="antigravity",
        provider_instance_id="agy_gemini",
        model="gemini-flash",
        status="idle",
        self_metrics=self_sum,
        rollup_metrics=self_sum,
        turn_metrics_availability="unavailable",
        latest_step_label="Model step #2",
        steps=[step1, step2],
    )

    nav = NavigationState()
    nav.reconcile(["th-idle-agy"])
    nav.enter()

    layout = _browser_frame([], node=node, summary=node, nav=nav, height=28, width=100)
    buf = io.StringIO()
    Console(file=buf, width=100, height=28).print(layout)
    out = buf.getvalue()

    # Scope 1: Latest completed turn (idle) with unavailable
    assert "Latest completed turn (idle)" in out
    assert "unavailable" in out
    # Scope 2: Whole chat · this agent
    assert "Whole chat · this agent" in out
    assert "2 steps" in out
    # Scope 3: Chat + subagents
    assert "Chat + subagents" in out

    # Steps have compact #N identity and correct action summary
    assert "· #1" in out
    assert "read_file" in out
    assert "· #2" in out
    assert "think + reply" in out


def test_render_context_graph_states_and_transitions():
    """Context graph renders honest states: multi-sample drops, single point, no-data, and gaps."""
    from t3_telemetry.models import ContextSample, ThreadTelemetryNode, TokenMetricsSummary
    from t3_telemetry.ui import render_context_graph

    # 1. Multi-point with context drop and known limit
    samples = [
        ContextSample(source_id="turn-1", used_tokens=75000, limit=200000, timestamp="2026-11-03T10:01:00Z", provenance="measured"),
        ContextSample(source_id="turn-2", used_tokens=25000, limit=200000, timestamp="2026-11-03T10:02:15Z", provenance="measured"),
    ]
    node = ThreadTelemetryNode(
        thread_id="th-graph",
        title="Graph Node",
        driver="claudeAgent",
        provider_instance_id="claude",
        model="claude-3-7-sonnet",
        context_used_tokens=25000,
        context_limit=200000,
        context_provenance="measured",
        context_samples=samples,
    )

    chart = render_context_graph(node, width=60, as_panel=True)
    buf = io.StringIO()
    Console(file=buf, width=60, color_system=None).print(chart)
    out = buf.getvalue()

    assert "Measured" in out
    assert "Current: 25.0k" in out
    assert "Peak: 75.0k" in out
    assert "Limit: 200.0k" in out
    assert "Context drop" in out
    assert "10:01:00" in out
    assert "10:02:15" in out

    # 2. Single reading honest display
    node_single = ThreadTelemetryNode(
        thread_id="th-single",
        title="Single Point",
        driver="claudeAgent",
        provider_instance_id="claude",
        model="claude-3-7-sonnet",
        context_used_tokens=45000,
        context_limit=200000,
        context_provenance="measured",
        context_samples=[samples[0]],
    )
    chart_single = render_context_graph(node_single, width=60, as_panel=False)
    buf_single = io.StringIO()
    Console(file=buf_single, width=60, color_system=None).print(chart_single)
    out_single = buf_single.getvalue()
    assert "single reading" in out_single
    assert "█" in out_single  # single reading drawn as a bar on the auto axis

    # 3. No data honest display
    node_empty = ThreadTelemetryNode(
        thread_id="th-empty",
        title="Empty",
        driver="antigravity",
        provider_instance_id="agy",
        model="gemini",
        context_used_tokens=0,
        context_limit=None,
        context_provenance="unknown",
        context_samples=[],
    )
    chart_empty = render_context_graph(node_empty, width=60, as_panel=False)
    buf_empty = io.StringIO()
    Console(file=buf_empty, width=60, color_system=None).print(chart_empty)
    out_empty = buf_empty.getvalue()
    assert "No context history recorded" in out_empty
    assert "Limit: unknown" in out_empty

    # 4. Provenance gap and uncertainty label
    gap_samples = [
        ContextSample(source_id="turn-1", used_tokens=50000, limit=200000, timestamp="2026-11-03T10:01:00Z", provenance="measured"),
        ContextSample(source_id="step-1", used_tokens=15000, limit=None, timestamp="2026-11-03T10:05:00Z", provenance="estimated"),
    ]
    node_gap = ThreadTelemetryNode(
        thread_id="th-gap",
        title="Gap Node",
        driver="antigravity",
        provider_instance_id="agy",
        model="gemini",
        context_used_tokens=15000,
        context_limit=None,
        context_provenance="measured",
        context_uncertainty=True,
        context_samples=gap_samples,
    )
    chart_gap = render_context_graph(node_gap, width=60, as_panel=False)
    buf_gap = io.StringIO()
    Console(file=buf_gap, width=60, color_system=None).print(chart_gap)
    out_gap = buf_gap.getvalue()
    assert "Measured (uncertain recency)" in out_gap
    assert "┊" in out_gap


def test_build_dashboard_renderable_includes_context_graph():
    """Snapshot dashboard renderable includes Context History chart."""
    from t3_telemetry.models import ContextSample, ThreadTelemetryNode, TokenMetricsSummary
    from t3_telemetry.ui import build_dashboard_renderable

    sample = ContextSample(source_id="t-1", used_tokens=12000, limit=128000, timestamp="2026-11-03T10:01:00Z", provenance="measured")
    node = ThreadTelemetryNode(
        thread_id="th-dash-graph",
        title="Dash Graph",
        driver="codex",
        provider_instance_id="codex-1",
        model="gpt-5-codex",
        status="running",
        context_used_tokens=12000,
        context_limit=128000,
        context_limit_provenance="measured",
        context_samples=[sample],
    )

    renderable = build_dashboard_renderable(node, max_steps=10)
    buf = io.StringIO()
    Console(file=buf, width=100, color_system=None).print(renderable)
    out = buf.getvalue()
    assert "Context History" in out
    assert "12.0k" in out


def test_browser_frame_60x24_usability():
    """Terminal browser frame remains fully legible and functional at strict 60x24 geometry."""
    from t3_telemetry.models import ContextSample, ModelStep, ThreadTelemetryNode, TokenMetricsSummary, TurnMetrics
    from t3_telemetry.tui import NavigationState
    from t3_telemetry.cli import _browser_frame

    child = ThreadTelemetryNode(
        thread_id="child-th-1",
        title="Worker Subagent",
        driver="codex",
        provider_instance_id="codex",
        model="gpt-5-codex",
        status="running",
    )
    step1 = ModelStep(step_index=1, source="main", tool_calls=["search"], duration_seconds=1.2)
    sample = ContextSample(source_id="t-1", used_tokens=15000, limit=200000, timestamp="2026-11-03T10:01:00Z", provenance="measured")

    parent = ThreadTelemetryNode(
        thread_id="parent-60x24",
        title="Parent Orchestrator",
        driver="claudeAgent",
        provider_instance_id="claude",
        model="claude-3-7-sonnet",
        status="running",
        context_used_tokens=15000,
        context_limit=200000,
        context_provenance="measured",
        context_samples=[sample],
        current_turn_metrics=TurnMetrics(ordinal=1, status="running", availability="available", step_count=1, input_tokens=5000, output_tokens=1000),
        subagents=[child],
        steps=[step1],
    )

    nav = NavigationState()
    nav.reconcile(["parent-60x24"])
    nav.enter()

    # Construct frame at strict 60x24
    layout = _browser_frame([], node=parent, summary=parent, nav=nav, height=24, width=60)
    buf = io.StringIO()
    Console(file=buf, width=60, height=24, color_system=None).print(layout)
    out = buf.getvalue()

    # Verify all components render cleanly within 60x24
    assert "Parent Orchestrator" in out
    assert "RUNNING" in out
    assert "Current turn (running)" in out
    assert "Whole chat · this agent" in out
    assert "Subagents" in out
    assert "Worker Subage" in out
    assert "Activity" in out
    assert "Context:" in out


def test_scope_label_completed_and_failed_polish():
    """Scope label displays Latest completed turn (without idle) for failed and completed statuses."""
    from t3_telemetry.models import ThreadTelemetryNode, TokenMetricsSummary, TurnMetrics
    from t3_telemetry.ui import build_scopes_summary

    sm = TokenMetricsSummary(step_count=1, input_tokens=1000, output_tokens=200)

    # 1. Completed status
    node_comp = ThreadTelemetryNode(
        thread_id="t-comp",
        title="Completed Thread",
        driver="codex",
        provider_instance_id="codex",
        model="gpt-5-codex",
        status="completed",
        self_metrics=sm,
        rollup_metrics=sm,
        current_turn_metrics=TurnMetrics(ordinal=1, status="completed", availability="available", input_tokens=1000, output_tokens=200),
    )
    lines_comp = build_scopes_summary(node_comp, width=60)
    comp_text = " ".join(line.plain for line in lines_comp)
    assert "Latest completed turn:" in comp_text
    assert "(idle)" not in comp_text

    # 2. Failed status
    node_failed = ThreadTelemetryNode(
        thread_id="t-fail",
        title="Failed Thread",
        driver="codex",
        provider_instance_id="codex",
        model="gpt-5-codex",
        status="failed",
        self_metrics=sm,
        rollup_metrics=sm,
        current_turn_metrics=TurnMetrics(ordinal=1, status="failed", availability="available", input_tokens=1000, output_tokens=200),
    )
    lines_fail = build_scopes_summary(node_failed, width=60)
    fail_text = " ".join(line.plain for line in lines_fail)
    assert "Latest completed turn:" in fail_text
    assert "(idle)" not in fail_text

    # 3. Idle status preserves (idle)
    node_idle = ThreadTelemetryNode(
        thread_id="t-idle",
        title="Idle Thread",
        driver="codex",
        provider_instance_id="codex",
        model="gpt-5-codex",
        status="idle",
        self_metrics=sm,
        rollup_metrics=sm,
    )
    lines_idle = build_scopes_summary(node_idle, width=60)
    idle_text = " ".join(line.plain for line in lines_idle)
    assert "Latest completed turn (idle):" in idle_text



def test_browser_frame_60x24_repro_parent_claude_child_worker_scopes_context_activity_visible():
    """Repro from Task 4 review round 1:

    parent running claude, child Worker, 5 own tool steps, two measured ContextSample
    25000/45000 limit 200000 time 10:01/10:02. At Console 60x24, Recent Activity and
    a useful activity row must be visible, subagent visible and selectable, and 120x40
    shows both tables cleanly.
    """
    from t3_telemetry.models import ThreadTelemetryNode, ContextSample, ModelStep
    from t3_telemetry.tui import NavigationState
    from t3_telemetry.cli import _browser_frame

    parent = ThreadTelemetryNode(
        thread_id="p-repro-claude",
        title="Parent Orchestrator Claude",
        driver="claude",
        provider_instance_id="claude-inst",
        model="claude-3-5-sonnet-20241022",
        status="running",
        context_used_tokens=45000,
        context_limit=200000,
        context_provenance="measured",
        context_samples=[
            ContextSample(source_id="s1", used_tokens=25000, timestamp="2026-11-03T10:01:00Z", limit=200000, provenance="measured"),
            ContextSample(source_id="s2", used_tokens=45000, timestamp="2026-11-03T10:02:00Z", limit=200000, provenance="measured"),
        ],
        steps=[
            ModelStep(step_index=i, source="claude", status="completed", tool_calls=[f"tool_{i}"], input_tokens=100, output_tokens=50)
            for i in range(1, 6)
        ],
    )
    for s in parent.steps:
        parent.self_metrics.add_step(s)
    parent.rollup_metrics.add_summary(parent.self_metrics)

    child = ThreadTelemetryNode(
        thread_id="c-repro-worker",
        title="Worker Subagent",
        driver="claude",
        provider_instance_id="claude-inst",
        model="claude-3-5-sonnet-20241022",
        status="running",
        steps=[ModelStep(step_index=1, source="claude", status="completed", tool_calls=["child_tool"], input_tokens=50, output_tokens=20)],
    )
    for s in child.steps:
        child.self_metrics.add_step(s)
    child.rollup_metrics.add_summary(child.self_metrics)
    parent.subagents.append(child)
    parent.rollup_metrics.add_summary(child.rollup_metrics)
    parent.rollup_metrics.total_subagents = 1

    nav = NavigationState()
    nav.reconcile(["p-repro-claude"])
    nav.enter()
    nav.current_screen.child_ids = [child.thread_id]
    nav.current_screen.selected_child_id = child.thread_id
    nav.current_screen.focus = "activity"

    # 1. 60x24 activity focus: Recent Activity and useful activity row are guaranteed visible
    layout = _browser_frame([parent], node=parent, summary=parent, nav=nav, height=24, width=60)
    buf = io.StringIO()
    Console(file=buf, width=60, height=24, force_terminal=True, color_system=None).print(layout)
    lines_act = buf.getvalue().splitlines()
    assert len(lines_act) == 24
    out_act = chr(10).join(lines_act)
    assert "Activity" in out_act
    assert "Subagents" in out_act
    assert "Worker Subagent" in out_act
    # Default selection is the most recent entry
    assert any("tool_5" in line or "Model step #5" in line for line in lines_act)

    # 2. Activity selection works: selecting an earlier entry brings it into view
    nav.current_screen.selected_activity_id = nav.current_screen.activity_ids[2]
    layout_scrolled = _browser_frame([parent], node=parent, summary=parent, nav=nav, height=24, width=60)
    buf_scrolled = io.StringIO()
    Console(file=buf_scrolled, width=60, height=24, force_terminal=True, color_system=None).print(layout_scrolled)
    lines_scrolled = buf_scrolled.getvalue().splitlines()
    assert len(lines_scrolled) == 24
    assert any("tool_3" in line or "Model step #3" in line for line in lines_scrolled)
    nav.current_screen.selected_activity_id = nav.current_screen.activity_ids[-1]

    # 3. 60x24 subagents focus: selected child is visible with marker ▶ and Enter usable
    nav.tab()
    assert nav.current_screen.focus == "subagents"
    layout_sub = _browser_frame([parent], node=parent, summary=parent, nav=nav, height=24, width=60)
    buf_sub = io.StringIO()
    Console(file=buf_sub, width=60, height=24, force_terminal=True, color_system=None).print(layout_sub)
    lines_sub = buf_sub.getvalue().splitlines()
    assert len(lines_sub) == 24
    out_sub = chr(10).join(lines_sub)
    assert any("▶" in line and "Worker Subagent" in line for line in lines_sub)
    assert "Activity" in out_sub

    # 4. 120x40 geometry: both subagents and activity tables fully visible
    layout_120 = _browser_frame([parent], node=parent, summary=parent, nav=nav, height=40, width=120)
    buf_120 = io.StringIO()
    Console(file=buf_120, width=120, height=40, force_terminal=True, color_system=None).print(layout_120)
    lines_120 = buf_120.getvalue().splitlines()
    assert len(lines_120) == 40
    out_120 = chr(10).join(lines_120)
    assert "Subagents" in out_120
    assert "Activity" in out_120
    assert "Worker Subagent" in out_120
    assert "tool_1" in out_120 or "Model step #1" in out_120


def test_browser_frame_content_accessibility_with_long_model_and_title():
    """Content accessibility with very long title and model name at 60x24 and 120x40 geometry."""
    from t3_telemetry.models import ThreadTelemetryNode, ContextSample, ModelStep
    from t3_telemetry.tui import NavigationState
    from t3_telemetry.cli import _browser_frame

    long_title = "Parent Orchestrator Thread With Exceptionally Long Title For Testing Line Wrapping"
    long_model = "claude-3-5-sonnet-20241022-extended-long-name-identifier-preview"
    node = ThreadTelemetryNode(
        thread_id="t-long-info",
        title=long_title,
        driver="claudeAgent",
        provider_instance_id="claude-inst",
        model=long_model,
        status="running",
        context_used_tokens=55000,
        context_limit=200000,
        context_provenance="measured",
        context_samples=[
            ContextSample(source_id="s1", used_tokens=30000, timestamp="2026-11-03T10:01:00Z", limit=200000, provenance="measured"),
            ContextSample(source_id="s2", used_tokens=55000, timestamp="2026-11-03T10:02:00Z", limit=200000, provenance="measured"),
        ],
        steps=[ModelStep(step_index=1, tool_calls=["execute_task"], input_tokens=500, output_tokens=200)],
    )

    nav = NavigationState()
    nav.reconcile(["t-long-info"])
    nav.enter()

    # Strict 60x24 layout
    layout_60 = _browser_frame([node], node=node, summary=node, nav=nav, height=24, width=60)
    buf_60 = io.StringIO()
    Console(file=buf_60, width=60, height=24, force_terminal=True, color_system=None).print(layout_60)
    lines_60 = buf_60.getvalue().splitlines()
    assert len(lines_60) == 24
    out_60 = chr(10).join(lines_60)
    assert "Activity" in out_60
    assert "execute_task" in out_60 or "Model step #1" in out_60

    # 120x40 layout
    layout_120 = _browser_frame([node], node=node, summary=node, nav=nav, height=40, width=120)
    buf_120 = io.StringIO()
    Console(file=buf_120, width=120, height=40, force_terminal=True, color_system=None).print(layout_120)
    lines_120 = buf_120.getvalue().splitlines()
    assert len(lines_120) == 40
    out_120 = chr(10).join(lines_120)
    assert "Activity" in out_120
    assert "execute_task" in out_120 or "Model step #1" in out_120


def _graph_node(values, limit=None, provenances=None, timestamps=True, current=None):
    from t3_telemetry.models import ContextSample, ThreadTelemetryNode

    samples = [
        ContextSample(
            source_id=f"s{i}",
            used_tokens=v,
            limit=limit,
            timestamp=(f"2026-11-03T10:{i // 60:02d}:{i % 60:02d}Z" if timestamps else None),
            provenance=(provenances[i] if provenances else "measured"),
        )
        for i, v in enumerate(values)
    ]
    return ThreadTelemetryNode(
        thread_id="th-axis",
        title="Axis",
        driver="codex",
        provider_instance_id="c",
        model="m",
        context_used_tokens=values[-1] if current is None and values else (current or 0),
        context_limit=limit,
        context_provenance="measured",
        context_samples=samples,
    )


def _plot(lines, rows):
    """Return the plot rows (top first) located by their axis glyph."""
    plot = [ln.plain for ln in lines if "┤" in ln.plain or "│" in ln.plain]
    assert len(plot) == rows
    return plot


def _column_heights(plot):
    """Count filled cells per column, from the axis glyph onward."""
    start = plot[0].index("┤") + 1
    width = max(len(r) for r in plot)
    heights = []
    for col in range(start, width):
        cells = [r[col] if col < len(r) else " " for r in plot]
        if any(c not in " " for c in cells):
            heights.append(sum(1 for c in cells if c == "█"))
    return heights


def test_nice_ceiling_rounding():
    from t3_telemetry.ui import nice_ceiling

    assert nice_ceiling(0) == 1000  # all-zero data still gets a nonzero axis
    assert nice_ceiling(1) == 1
    assert nice_ceiling(3) == 5
    assert nice_ceiling(80_000) == 100_000
    assert nice_ceiling(40_000) == 50_000
    assert nice_ceiling(100_000) == 100_000
    assert nice_ceiling(150_001) == 200_000
    assert nice_ceiling(1_200_000) == 2_000_000


def test_context_graph_example_scale_and_geometry():
    """25k, 80k, 40k under a 200k limit: axis tops at 100k, limit shown separately."""
    from t3_telemetry.ui import context_graph_lines

    node = _graph_node([25_000, 80_000, 40_000], limit=200_000)
    for width, rows in ((56, 3), (116, 5)):
        lines = context_graph_lines(node, width, rows)
        assert all(ln.cell_len <= width for ln in lines)
        plot = _plot(lines, rows)
        assert plot[0].lstrip().startswith("100k┤")
        assert plot[-1].lstrip().startswith("0┤")
        assert plot[rows // 2].lstrip().startswith("50k┤")
        text = "\n".join(ln.plain for ln in lines)
        assert "Limit: 200.0k" in text and "Current: 40.0k" in text and "Peak: 80.0k" in text
    # 5 rows of 20k: 25k -> 1 full row, 80k -> 4, 40k -> 2.
    assert _column_heights(_plot(context_graph_lines(node, 116, 5), 5)) == [1, 4, 2]


def test_context_graph_low_usage_visible_under_large_limit():
    from t3_telemetry.ui import context_graph_lines

    node = _graph_node([300, 500], limit=1_000_000)
    plot = _plot(context_graph_lines(node, 60, 3), 3)
    assert plot[0].lstrip().startswith("500┤")
    assert _column_heights(plot)[-1] == 3  # latest reading fills the auto axis


def test_context_graph_zero_single_and_unknown_limit():
    from t3_telemetry.ui import context_graph_lines

    zero = context_graph_lines(_graph_node([0, 0]), 60, 3)
    plot = _plot(zero, 3)
    assert plot[0].lstrip().startswith("1k┤")
    assert "Limit: unknown" in "\n".join(ln.plain for ln in zero)

    single = context_graph_lines(_graph_node([12_000], limit=128_000), 60, 3)
    plot = _plot(single, 3)
    assert plot[0].lstrip().startswith("20k┤")
    assert _column_heights(plot) == [1]  # 12k of 20k: one full row of three plus a partial
    assert "at 10:00:00" in "\n".join(ln.plain for ln in single)


def test_bucket_context_samples_preserves_spikes_drops_and_latest():
    from t3_telemetry.models import ContextSample
    from t3_telemetry.ui import bucket_context_samples

    values = [1000 * i for i in range(256)]
    values[100] = 900_000  # spike
    values[200] = 5  # drop
    samples = [ContextSample(source_id=str(i), used_tokens=v) for i, v in enumerate(values)]
    buckets = bucket_context_samples(samples, 50)
    assert len(buckets) == 50
    assert sum(b.count for b in buckets) == 256
    assert [b.first_index for b in buckets] == sorted(b.first_index for b in buckets)
    assert max(b.high for b in buckets) == 900_000
    assert min(b.low for b in buckets) == 0
    assert any(b.has_drop and b.low == 5 for b in buckets)
    assert buckets[-1].latest == values[-1]
    assert bucket_context_samples(samples, 300)[-1].count == 1  # no bucketing when it fits


def test_context_graph_full_history_mixed_provenance_bounded():
    from t3_telemetry.ui import context_graph_lines, render_context_graph

    values = [(i * 700) % 90_000 for i in range(256)]
    provs = ["measured"] * 200 + ["estimated"] * 56
    node = _graph_node(values, provenances=provs, timestamps=False)
    for width in (40, 56, 116):
        lines = context_graph_lines(node, width, 3)
        assert all(ln.cell_len <= width for ln in lines)
        text = "\n".join(ln.plain for ln in lines)
        assert "█ latest ░ col peak" in text
        assert "↓ drop" in text and "┊ source switch" in text
        assert "#1 … #256 (no times)" in text
    # Exact height budget is honored, and rendering stays within console width.
    for height in (8, 12):
        buf = io.StringIO()
        Console(file=buf, width=60, color_system=None).print(render_context_graph(node, width=60, as_panel=False, height=height))
        out = buf.getvalue().splitlines()
        assert len(out) == height
        assert all(len(ln) <= 60 for ln in out)


def test_context_graph_range_fills_only_half_covered_cells():
    from t3_telemetry.ui import ContextBucket, RANGE_CHAR, _bar_cell

    def bucket(high):
        return ContextBucket(0, 2, 10_000, high, 10_000, frozenset({"measured"}), "measured", False, False)

    # 3 rows on a 100k axis: the middle row spans ~33k-67k.
    assert _bar_cell(bucket(34_000), 1, 3, 100_000)[0] == " "
    assert _bar_cell(bucket(60_000), 1, 3, 100_000)[0] == RANGE_CHAR
    assert _bar_cell(bucket(60_000), 2, 3, 100_000)[0] == " "


def test_context_graph_tight_heights_exact_and_keep_scale():
    from t3_telemetry.ui import context_graph_lines

    nodes = [
        _graph_node([25_000, 80_000, 40_000], limit=200_000),
        _graph_node([(i * 700) % 90_000 for i in range(256)], provenances=["measured"] * 200 + ["estimated"] * 56),
    ]
    for node in nodes:
        for width in (40, 60):
            for height in range(3, 9):
                lines = context_graph_lines(node, width, 3, height)
                assert len(lines) == height
                assert all(ln.cell_len <= width for ln in lines)
                text = "\n".join(ln.plain for ln in lines)
                assert "not limit" in text
                assert "100k┤" in text
                if width >= 60:
                    assert "Limit:" in text


def test_context_graph_single_row_labels_ceiling_and_narrow_width():
    from t3_telemetry.ui import context_graph_lines

    node = _graph_node([25_000, 80_000, 40_000], limit=200_000)
    plot = _plot(context_graph_lines(node, 60, 1), 1)
    assert plot[0].lstrip().startswith("100k┤")
    for width in (8, 12, 16):
        lines = context_graph_lines(node, width, 3)
        assert all(ln.cell_len <= width for ln in lines)


def test_usage_graph_helpers_and_rendering():
    from datetime import datetime, timedelta, timezone
    from io import StringIO

    from rich.console import Console

    from t3_telemetry.ui import merge_buckets, stack_column, usage_graph_lines
    from t3_telemetry.usage import PeriodUsage, TokenBreakdown, period_window

    now = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    starts = [now + timedelta(days=i) for i in range(31)]
    merged = merge_buckets([{"Claude": 1, "Codex": 2}] * 31, starts, 27)
    assert len(merged) <= 27
    assert merged[0][0] == starts[0]
    assert sum(values["Claude"] for _, values in merged) == 31
    assert sum(values["Codex"] for _, values in merged) == 62
    assert stack_column({"Claude": 75, "Codex": 25}, 100, 4) == ["Claude", "Claude", "Claude", "Codex"]
    assert stack_column({"Codex": 1}, 100, 4) == ["Codex", None, None, None]
    assert stack_column({}, 100, 4) == [None] * 4

    for key in ("week", "month"):
        window = period_window(key, now)
        period = PeriodUsage(window, TokenBreakdown(input_tokens=10), {"Claude": TokenBreakdown(input_tokens=10)},
                             [{"Claude": 10} for _ in window.bucket_starts], [])
        lines = usage_graph_lines(period, 60, 5, now)
        assert len(lines) == 7
        output = StringIO()
        console = Console(file=output, width=60, color_system=None)
        for line in lines:
            console.print(line)
        rendered = output.getvalue().splitlines()
        assert len(rendered) == 7
        assert all(len(line) <= 60 for line in rendered)
        assert "·" in rendered[-2]

    # At 80 columns the month gets 1-cell bars; day labels must stay separated.
    window = period_window("month", now)
    period = PeriodUsage(window, TokenBreakdown(), {}, [{} for _ in window.bucket_starts], [])
    x_labels = usage_graph_lines(period, 80, 4, now)[-1].plain.split()
    assert all(label.isdigit() and int(label) <= 31 for label in x_labels)

    from t3_telemetry.usage import UsageStats
    w_week = period_window("week", now)
    stats = UsageStats()
    stats.add("claude-sonnet", now, 35, 0, 0, 0, subagent=False)
    stats.add("claude-sonnet", now, 35, 0, 0, 0, subagent=True)
    p = PeriodUsage(
        w_week,
        TokenBreakdown(input_tokens=70),
        {},
        [{} for _ in w_week.bucket_starts],
        [],
        stats=stats,
        scope_buckets=[{"Main": 5, "Subagents": 5} for _ in w_week.bucket_starts],
    )
    scope_lines = usage_graph_lines(p, 60, 5, now, split="scope")
    assert len(scope_lines) == 7
    assert "Main" in scope_lines[0].plain and "Subagents" in scope_lines[0].plain
    assert any("▓" in line.plain for line in scope_lines[1:-1])


def test_breakdown_lines_and_kpis():
    from datetime import datetime, timezone
    from io import StringIO

    from rich.console import Console

    from t3_telemetry.insights import Kpi, headline
    from t3_telemetry.ui import KIND_MEANINGS, breakdown_lines, format_usd, kind_legend_lines, kpi_lines
    from t3_telemetry.usage import UsageStats

    assert format_usd(12.5) == "$12.50"
    assert len(KIND_MEANINGS) == 4

    now = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    stats = UsageStats()
    stats.add("claude-sonnet-4-5", now, input=1000, cache_read=2000, cache_write=500, output=300, subagent=False)

    for w in (58, 78, 118):
        for measure in ("tokens", "usd"):
            lines = breakdown_lines(stats, measure, w, "Main", kpis_on=True, legend="detailed")
            output = StringIO()
            console = Console(file=output, width=w, color_system=None)
            for line in lines:
                console.print(line)
            rendered = output.getvalue().splitlines()
            assert all(len(line) <= w for line in rendered)
            assert any("Cache read" in line for line in rendered)
            h = headline(stats, measure, "Main")
            expected_h = h if len(h) <= w else h[:w - 1] + "…"
            assert any(expected_h in line for line in rendered)
            if w == 58:
                assert not any("·" in line for line in rendered)
            if w == 118:
                assert any("·" in line for line in rendered)

    # kpi_lines with 5 items at width 58 keeps 4 (58 // 4 = 14; 58 // 5 = 11 drops one)
    items5 = [Kpi(f"v{i}", f"label{i}") for i in range(5)]
    klines = kpi_lines(items5, 58)
    values = klines[0].plain.split()
    assert len(values) == 4


def test_sync_creates_store_with_empty_roots(tmp_path, monkeypatch, capsys):
    from t3_telemetry.collector import SourceRoots
    monkeypatch.setattr("t3_telemetry.cli.default_roots", lambda d: SourceRoots(tmp_path / "empty", [], [], d))
    path = tmp_path / "store.sqlite"
    assert main(["--sync", "--store", str(path), "--t3-dir", str(tmp_path)]) == 0
    assert path.exists()
    assert "files_seen=0 files_read=0 steps_written=0 errors=0" in capsys.readouterr().out


def test_build_sessions_table():
    rows = [
        {
            "session_key": "claude:1",
            "thread_id": "th-1",
            "origin": "T3",
            "provider": "claude",
            "title": "T3 Main Session",
            "model": "claude-3-5-sonnet",
            "last_active": "2026-11-04T12:00:00Z",
            "subagents": 2,
            "tokens": 1540,
            "status": "running",
        },
        {
            "session_key": "codex:2",
            "thread_id": None,
            "origin": "cli",
            "provider": "codex",
            "title": "Standalone Session",
            "model": "gpt-4o",
            "last_active": "2026-11-03T10:00:00Z",
            "subagents": 0,
            "tokens": 500,
        },
    ]
    table = build_sessions_table(rows)
    assert table.title == "Recent Sessions"
    assert len(table.columns) == 7
    col_names = [str(col.header) for col in table.columns]
    assert col_names == ["Origin", "Provider", "Title", "Model", "Last active", "Subagents", "Tokens"]

    console = Console(file=io.StringIO(), color_system="truecolor", width=120)
    console.print(table)
    rendered = console.file.getvalue()
    assert "T3 Main Session" in rendered
    assert "Standalone Session" in rendered
    assert "1.5k" in rendered
    assert "500" in rendered


def test_cli_sessions_table_and_json(tmp_path, monkeypatch, capsys):
    from t3_telemetry.collector import SourceRoots
    from t3_telemetry.store import SessionRecord, StepRecord, TelemetryStore

    monkeypatch.setattr("t3_telemetry.cli.default_roots", lambda d: SourceRoots(tmp_path / "empty", [], [], d))
    db_path = tmp_path / "store.sqlite"
    store = TelemetryStore(db_path)
    store.upsert_sessions([
        SessionRecord("claude:main", "claude", "main", origin_hint="sdk", model="sonnet", first_ts="2026-11-01", last_ts="2026-11-04T12:00:00Z"),
        SessionRecord("claude:child", "claude", "child", parent_session_key="claude:main"),
        SessionRecord("codex:free", "codex", "free", origin_hint="cli", cwd="/work/repo", first_ts="2026-11-02", last_ts="2026-11-05T12:00:00Z"),
    ])
    store.replace_t3_graph([
        {"thread_id": "th-root", "title": "Bound Root Thread"},
    ], [
        {"session_key": "claude:main", "thread_id": "th-root"},
    ])
    store.upsert_steps([
        StepRecord("claude:main", "s1", "2026-11-04T10:00:00Z", input_tokens=100, output_tokens=50),
        StepRecord("claude:child", "s2", "2026-11-04T11:00:00Z", input_tokens=20, output_tokens=30),
        StepRecord("codex:free", "s3", "2026-11-05T10:00:00Z", input_tokens=500, output_tokens=500),
    ])
    store.close()

    # 1. Test table mode (--sessions)
    rc = main(["--sessions", "--store", str(db_path), "--t3-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "files_seen=" not in out
    assert "Recent Sessions" in out
    assert "Bound" in out
    assert "repo" in out
    assert "T3" in out
    assert "cli" in out

    # 2. Test JSON mode (--sessions --json)
    rc_json = main(["--sessions", "--json", "--store", str(db_path), "--t3-dir", str(tmp_path)])
    assert rc_json == 0
    json_out = capsys.readouterr().out
    assert "files_seen=" not in json_out
    parsed = json.loads(json_out)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    # Most recent first: codex:free (2026-11-05), then claude:main (2026-11-04)
    assert parsed[0]["session_key"] == "codex:free"
    assert parsed[0]["origin"] == "cli"
    assert parsed[0]["tokens"] == 1000

    assert parsed[1]["session_key"] == "claude:main"
    assert parsed[1]["origin"] == "T3"
    assert parsed[1]["thread_id"] == "th-root"
    assert parsed[1]["subagents"] == 1
    assert parsed[1]["tokens"] == 200


def test_kind_bar_allocation():
    from t3_telemetry.prices import KINDS
    from t3_telemetry.ui import allocate_cells, kind_bar_line

    values = {"cache_read": 57, "cache_write": 21, "output": 16, "input": 0.3}
    cells = allocate_cells(values, 100, KINDS)
    assert sum(cells.values()) == 100
    assert cells["input"] == 1
    assert allocate_cells({}, 100, KINDS) == dict.fromkeys(KINDS, 0)
    assert allocate_cells(values, 0, KINDS) == dict.fromkeys(KINDS, 0)
    assert sum(allocate_cells(values, 3, KINDS).values()) == 3
    assert allocate_cells(dict.fromkeys(KINDS, 1), 3, KINDS) == dict(zip(KINDS, (1, 1, 1, 0)))
    for sample in (values, dict.fromkeys(KINDS, 1), dict(zip(KINDS, (1, 13, 37, 101)))):
        for width in range(1, 201):
            assert sum(allocate_cells(sample, width, KINDS).values()) == width
            assert kind_bar_line(sample, width).cell_len == width
    empty = kind_bar_line({}, 10)
    assert empty.plain == "░" * 10
    assert empty.style == "dim"


def test_overview_120x40_shows_usage_breakdown():
    from t3_telemetry.cli import _browser_frame
    from t3_telemetry.models import ModelStep, ThreadTelemetryNode
    from t3_telemetry.tui import NavigationState

    node = ThreadTelemetryNode(
        thread_id="priced-thread", title="Priced chat", driver="claude", provider_instance_id="claude-inst",
        model="claude-sonnet-5-5", status="running",
        steps=[ModelStep(step_index=1, input_tokens=100, cache_read_tokens=900, output_tokens=100)],
    )
    nav = NavigationState()
    nav.reconcile([node.thread_id])
    nav.enter()
    layout = _browser_frame([node], node=node, summary=node, nav=nav, height=40, width=120)
    buf = io.StringIO()
    Console(file=buf, width=120, height=40, force_terminal=True, color_system=None).print(layout)
    output = buf.getvalue()
    assert "of this chat's tokens" in output
    assert "Cache read" in output
