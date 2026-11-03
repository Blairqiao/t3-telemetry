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
)
from t3_telemetry.cli import build_parser, main, run_live_loop


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
    assert "Self" in output_narrow

    # Render to standard wide console
    buf_wide = io.StringIO()
    console_wide = Console(file=buf_wide, width=120, color_system=None)
    console_wide.print(renderable)
    output_wide = buf_wide.getvalue()
    assert "read_file" in output_wide
    assert "Self" in output_wide


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
    args = parser.parse_args(["--t3-dir", str(synthetic_t3_dir), "--no-follow", "--interval", "0.01"])
    buf = io.StringIO()
    test_console = Console(file=buf, width=120, color_system=None)
    # Live loop with _max_iterations=2 must exit cleanly with code 0
    code = run_live_loop(args, console=test_console, _max_iterations=2)
    assert code == 0
