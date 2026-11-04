"""Tests for terminal adapter, navigation state, and viewport rendering."""

from __future__ import annotations

import io
import os
import pty
import termios
import time

from rich.console import Console
from rich.text import Text

from t3_telemetry.cli import _browser_frame
from t3_telemetry.models import AppThreadRecord
from t3_telemetry.terminal import PosixTerminalAdapter
from t3_telemetry.tui import NavigationState, TelemetryBrowser, TerminalAdapter, Viewport


def test_navigation_keeps_selection_by_id_on_reorder_and_removal():
    nav = NavigationState()
    nav.reconcile(["a", "b", "c"])
    nav.move(1)
    nav.reconcile(["c", "b", "a"])
    assert nav.selected_id == "b"
    nav.reconcile(["c", "a"])
    assert nav.selected_id in nav.ids
    nav.reconcile([])
    assert nav.selected_id is None


def test_browser_key_navigation():
    class Adapter:
        def draw(self, value):
            pass

    browser = TelemetryBrowser(Adapter())
    browser.navigation.reconcile(["a", "b"])
    browser.handle_key("DOWN")
    browser.handle_key("ENTER")
    assert browser.navigation.detail_id == "b"
    browser.handle_key("ESC")
    assert browser.navigation.detail_id is None


def test_selection_retains_nearest_position_when_selected_id_removed():
    nav = NavigationState()
    nav.reconcile(["a", "b", "c"])
    nav.move(2)
    nav.reconcile(["a", "b"])
    assert nav.selected_id == "b"


def test_escape_after_query_does_not_implicitly_reopen():
    nav = NavigationState()
    nav.reconcile(["a"])
    nav.detail_id = "a"
    browser = TelemetryBrowser(type("Adapter", (), {"draw": lambda *_: None})(), nav)
    browser.handle_key("ESC")
    assert nav.detail_id is None


def test_escape_sequence_adapter_reads_up_from_pty():
    master, slave = pty.openpty()

    class Stream:
        def fileno(self):
            return slave

        def read(self, n):
            return os.read(slave, n).decode()

    adapter = PosixTerminalAdapter(Stream())
    try:
        adapter.begin()
        os.write(master, b"[A")
        assert adapter.read_key(0.2) == "UP"
    finally:
        adapter.end()
        os.close(master)
        os.close(slave)


def test_real_pty_split_arrow_delayed():
    master, slave = pty.openpty()

    class Stream:
        def fileno(self):
            return slave

    adapter = PosixTerminalAdapter(Stream())
    try:
        adapter.begin()
        # 1. Send only partial ESC byte
        os.write(master, b"")
        res1 = adapter.read_key(0.01)
        # Should not prematurely consume ESC
        assert res1 is None
        # 2. Delayed remainder arrives
        os.write(master, b"[A")
        res2 = adapter.read_key(0.06)
        assert res2 == "UP"
    finally:
        adapter.end()
        os.close(master)
        os.close(slave)


def test_real_pty_standalone_escape_after_ambiguity_window():
    master, slave = pty.openpty()

    class Stream:
        def fileno(self):
            return slave

    adapter = PosixTerminalAdapter(Stream())
    try:
        adapter.begin()
        os.write(master, b"")
        res1 = adapter.read_key(0.01)
        assert res1 is None
        # Wait beyond the 50ms ambiguity window
        time.sleep(0.06)
        res2 = adapter.read_key(0.02)
        assert res2 == "ESC"
    finally:
        adapter.end()
        os.close(master)
        os.close(slave)


def test_real_pty_multiple_queued_keys_stream():
    master, slave = pty.openpty()

    class Stream:
        def fileno(self):
            return slave

    adapter = PosixTerminalAdapter(Stream())
    try:
        adapter.begin()
        # Queue text and escape sequence together
        os.write(master, b"jk" + bytes([13]) + b"[B")
        time.sleep(0.02)
        assert adapter.read_key(0.05) == "j"
        assert adapter.read_key(0.05) == "k"
        assert adapter.read_key(0.05) == "ENTER"
        assert adapter.read_key(0.05) == "DOWN"
    finally:
        adapter.end()
        os.close(master)
        os.close(slave)


def test_real_pty_eof_graceful():
    master, slave = pty.openpty()

    class Stream:
        def fileno(self):
            return slave

    adapter = PosixTerminalAdapter(Stream())
    try:
        adapter.begin()
        os.close(master)
        assert adapter.read_key(0.05) is None
    finally:
        adapter.end()
        os.close(slave)


def test_real_pty_cbreak_restoration():
    master, slave = pty.openpty()

    class Stream:
        def fileno(self):
            return slave

    adapter = PosixTerminalAdapter(Stream())
    try:
        orig = termios.tcgetattr(slave)
        adapter.begin()
        current = termios.tcgetattr(slave)
        # In cbreak, ICANON and ECHO are cleared
        assert (current[3] & (termios.ECHO | termios.ICANON)) == 0
        adapter.end()
        restored = termios.tcgetattr(slave)
        # Verify canonical modes and control chars restored
        assert (restored[3] & (termios.ECHO | termios.ICANON)) == (orig[3] & (termios.ECHO | termios.ICANON))
        assert restored[6] == orig[6]
    finally:
        os.close(master)
        os.close(slave)


def test_viewport_clipping_and_scrolling():
    console = Console(width=40, height=20, color_system=None)
    content = Text(chr(10).join(f"Row {i}" for i in range(30)))

    # Height 5, scroll 0
    vp0 = Viewport(content, height=5, scroll=0)
    buf0 = io.StringIO()
    c0 = Console(file=buf0, width=40, height=10)
    c0.print(vp0)
    lines0 = [l for l in buf0.getvalue().splitlines() if l.strip()]
    assert len(lines0) == 5
    assert "Row 0" in lines0[0]
    assert "Row 4" in lines0[-1]

    # Height 5, scroll 10
    vp10 = Viewport(content, height=5, scroll=10)
    buf10 = io.StringIO()
    c10 = Console(file=buf10, width=40, height=10)
    c10.print(vp10)
    lines10 = [l for l in buf10.getvalue().splitlines() if l.strip()]
    assert len(lines10) == 5
    assert "Row 10" in lines10[0]
    assert "Row 14" in lines10[-1]


def test_list_viewport_row_budget_60x24_select_30th_chat():
    roots = [
        AppThreadRecord(
            thread_id=f"th-{i:03d}",
            project_id=None,
            title=f"Chat {i:03d} - Very Long Thread Title That Exceeds Normal Terminal Width Extensively",
            default_provider="t3",
            provider_instance_id=None,
            model="gemini-3.8-flash",
            thinking_level=None,
            active_provider_thread_id=None,
            parent_thread_id=None,
            root_thread_id=None,
            relationship_to_parent=None,
            created_at=None,
            updated_at=None,
        )
        for i in range(50)
    ]

    nav = NavigationState()
    nav.reconcile([r.thread_id for r in roots])
    # Select the 30th chat (index 29)
    nav.selected_id = roots[29].thread_id

    layout = _browser_frame(roots, node=None, summary=None, nav=nav, height=24, width=60)
    buf = io.StringIO()
    console = Console(file=buf, width=60, height=24, force_terminal=True, color_system=None)
    console.print(layout)
    output = buf.getvalue()
    lines = output.splitlines()

    # The layout must not overflow 24 lines
    assert len(lines) == 24
    # The 30th chat must be visible with selection marker ▶
    assert any("▶" in line and "029" in line for line in lines)
    # Long titles must be ellipsized to stay on one row
    assert any("…" in line for line in lines)


def test_breadcrumb_parent_child_grandchild_navigation_and_restoration():
    nav = NavigationState()
    nav.reconcile(["root-1"])
    assert nav.selected_id == "root-1"
    assert nav.detail_id is None

    # 1. Enter root thread -> depth 1
    nav.enter()
    assert nav.detail_id == "root-1"
    assert len(nav.stack) == 1
    screen1 = nav.current_screen
    screen1.title = "Root Thread"
    screen1.child_ids = ["child-1", "child-2"]
    screen1.selected_child_id = "child-1"
    screen1.focus = "activity"
    screen1.activity_scroll = 3
    screen1.max_activity_scroll = 10

    # 2. Tab to switch focus to subagents
    nav.tab()
    assert screen1.focus == "subagents"
    nav.move(1)
    assert screen1.selected_child_id == "child-2"

    # 3. Enter child-2 -> depth 2
    nav.enter()
    assert nav.detail_id == "child-2"
    assert len(nav.stack) == 2
    screen2 = nav.current_screen
    screen2.title = "Child 2"
    screen2.child_ids = ["gc-1"]
    screen2.selected_child_id = "gc-1"
    screen2.focus = "activity"
    screen2.activity_scroll = 5
    screen2.max_activity_scroll = 10

    # 4. Tab and enter gc-1 -> depth 3
    nav.tab()
    assert screen2.focus == "subagents"
    nav.enter()
    assert nav.detail_id == "gc-1"
    assert len(nav.stack) == 3
    screen3 = nav.current_screen
    screen3.title = "Grandchild 1"

    # 5. Escape back to depth 2: exact state restored
    nav.escape()
    assert nav.detail_id == "child-2"
    assert len(nav.stack) == 2
    restored2 = nav.current_screen
    assert restored2.focus == "subagents"
    assert restored2.selected_child_id == "gc-1"
    assert restored2.activity_scroll == 5

    # 6. Escape back to depth 1: exact state restored
    nav.escape()
    assert nav.detail_id == "root-1"
    assert len(nav.stack) == 1
    restored1 = nav.current_screen
    assert restored1.focus == "subagents"
    assert restored1.selected_child_id == "child-2"
    assert restored1.activity_scroll == 3

    # 7. Escape back to root list: depth 0
    nav.escape()
    assert nav.detail_id is None
    assert len(nav.stack) == 0
    assert nav.selected_id == "root-1"


def test_disappearance_reconciliation_to_nearest_available_ancestor():
    nav = NavigationState()
    nav.reconcile(["root-1"])
    nav.enter()
    nav.current_screen.title = "Root Thread"
    nav.current_screen.child_ids = ["child-1"]
    nav.current_screen.selected_child_id = "child-1"

    nav.enter()
    nav.current_screen.title = "Child 1"
    nav.current_screen.child_ids = ["gc-1"]
    nav.current_screen.selected_child_id = "gc-1"

    nav.enter()
    nav.current_screen.title = "Grandchild 1"
    assert len(nav.stack) == 3

    # Grandchild disappears: reconcile to nearest available ancestor (Child 1)
    available_tree = {"root-1", "child-1"}
    titles = {"root-1": "Root Thread", "child-1": "Child 1"}
    nav.reconcile_tree(available_tree, titles)

    assert len(nav.stack) == 2
    assert nav.detail_id == "child-1"
    assert nav.notice is not None
    assert "Grandchild 1" in nav.notice
    assert "Child 1" in nav.notice

    # Child disappears: reconcile to Root
    nav.reconcile_tree({"root-1"}, {"root-1": "Root Thread"})
    assert len(nav.stack) == 1
    assert nav.detail_id == "root-1"
    assert "Child 1" in nav.notice
    assert "Root Thread" in nav.notice

    # Root disappears: reconcile back to list
    nav.reconcile([])
    assert len(nav.stack) == 0
    assert nav.detail_id is None
    assert "no longer available" in nav.notice


def test_logical_scroll_clamping_prevents_overscroll():
    nav = NavigationState()
    nav.reconcile(["root-1"])
    nav.enter()
    screen = nav.current_screen
    screen.focus = "activity"
    screen.max_activity_scroll = 5
    screen.activity_scroll = 0

    # Overscroll downwards by 20 moves
    for _ in range(20):
        nav.move(1)

    # Must be clamped strictly at 5, NOT 20
    assert screen.activity_scroll == 5

    # A single Up move immediately decrements to 4
    nav.move(-1)
    assert screen.activity_scroll == 4


def test_detail_viewport_row_budget_60x24_with_subagents():
    from t3_telemetry.models import ThreadTelemetryNode, TokenMetricsSummary, ModelStep

    child = ThreadTelemetryNode(
        thread_id="child-1",
        title="Immediate Subagent 1",
        driver="antigravity",
        provider_instance_id=None,
        model="gemini-3.8-flash",
        status="running",
        self_metrics=TokenMetricsSummary(input_tokens=1000, step_count=2, duration_seconds=1.0),
        rollup_metrics=TokenMetricsSummary(input_tokens=1000, step_count=2, duration_seconds=1.0),
        steps=[ModelStep(step_index=1, tool_calls=["search"], input_tokens=1000, duration_seconds=1.0)],
    )

    root = ThreadTelemetryNode(
        thread_id="th-root",
        title="Root Thread With Long Name Exceeding Width",
        driver="codex",
        provider_instance_id=None,
        model="gpt-5-codex",
        status="running",
        self_metrics=TokenMetricsSummary(input_tokens=2000, step_count=5, duration_seconds=3.0),
        rollup_metrics=TokenMetricsSummary(input_tokens=3000, step_count=7, duration_seconds=4.0, total_subagents=1),
        steps=[
            ModelStep(step_index=i, tool_calls=[f"step_{i}"], input_tokens=400, duration_seconds=0.6)
            for i in range(1, 6)
        ],
        subagents=[child],
    )

    nav = NavigationState()
    nav.reconcile(["th-root"])
    nav.enter()
    nav.current_screen.title = root.title
    nav.current_screen.child_ids = [child.thread_id]
    nav.current_screen.selected_child_id = child.thread_id
    nav.current_screen.focus = "subagents"

    layout = _browser_frame([], node=root, summary=root, nav=nav, height=24, width=60)
    buf = io.StringIO()
    console = Console(file=buf, width=60, height=24, force_terminal=True, color_system=None)
    console.print(layout)
    lines = buf.getvalue().splitlines()

    assert len(lines) == 24
    # Immediate subagent must be visible with selection marker ▶
    assert any("▶" in line and "Immediate Subagent" in line for line in lines)
    # Status column must not exist as dedicated column; textual status in header
    assert any("RUNNING" in line for line in lines)


# --- Task 5: Live loop context history integration, navigation retention & summary isolation ---

from pathlib import Path
import json
import sqlite3
import pytest
from t3_telemetry.cli import build_parser, run_live_loop
from t3_telemetry.context_history import ContextHistoryStore, make_agent_key
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


def setup_test_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    return conn


class CaptureLive:
    def __init__(self):
        self.updates = []

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


def test_live_loop_context_history_retention_across_rebuilds(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES ('th-root', 'proj', 'Root Thread', 'claudeAgent', 'full-access', 'default',
            'pt-root', '2026-11-03T10:00:00Z', '2026-11-03T10:00:00Z', NULL,
            ?, 'claude_inst')
    """, (json.dumps(th_payload),))

    pt_payload = {
        "nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-1"},
        "contextUsage": {"usedTokens": 10000, "maxTokens": 200000}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES ('pt-root', 'th-root', 'th-root', 'claudeAgent', 'running', '2026-11-03T10:00:00Z', ?, 'claudeAgent', 'claude_inst')
    """, (json.dumps(pt_payload),))
    conn.commit()

    history_store = ContextHistoryStore()
    args = build_parser().parse_args(["--t3-dir", str(tmp_path), "--interval", "0.01"])
    adapter = ScriptedAdapter(["q"])

    # First run: 1 poll -> records 10,000 tokens
    run_live_loop(args, _adapter=adapter, _max_iterations=1, _live_cls=lambda **kw: CaptureLive(), _history_store=history_store)
    agent_key = ("th-root", "pt-root")
    samples1 = history_store.get(agent_key)
    assert len(samples1) == 1
    assert samples1[0].used_tokens == 10000

    # Simulate provider thread updating context to 25,000 tokens in database
    pt_payload2 = {
        "nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-1"},
        "contextUsage": {"usedTokens": 25000, "maxTokens": 200000}
    }
    conn.execute("""
    UPDATE orchestration_v2_projection_provider_threads
    SET payload_json = ?, updated_at = '2026-11-03T10:01:00Z'
    WHERE provider_thread_id = 'pt-root'
    """, (json.dumps(pt_payload2),))
    conn.commit()
    conn.close()

    # Second run: another poll with same persistent history_store across snapshot/cache rebuild
    adapter2 = ScriptedAdapter(["q"])
    run_live_loop(args, _adapter=adapter2, _max_iterations=1, _live_cls=lambda **kw: CaptureLive(), _history_store=history_store)

    samples2 = history_store.get(agent_key)
    # Both 10,000 and 25,000 are retained in history store across rebuilds!
    assert len(samples2) == 2
    assert samples2[0].used_tokens == 10000
    assert samples2[1].used_tokens == 25000


def test_navigation_retention_reopening_screen_preserves_context_history(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    for idx, (tid, ptid, tok) in enumerate([("th-a", "pt-a", 12000), ("th-b", "pt-b", 34000)]):
        th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
        conn.execute("""
        INSERT INTO orchestration_v2_projection_threads
        (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
         active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
        VALUES (?, 'proj', ?, 'claudeAgent', 'full-access', 'default', ?,
                '2026-11-03T10:00:00Z', ?, NULL, ?, 'claude_inst')
        """, (tid, f"Thread {tid.upper()}", ptid, f"2026-11-03T10:0{idx}:00Z", json.dumps(th_payload)))

        pt_payload = {
            "nativeThreadRef": {"driver": "claudeAgent", "nativeId": f"c-{tid}"},
            "contextUsage": {"usedTokens": tok, "maxTokens": 200000}
        }
        conn.execute("""
        INSERT INTO orchestration_v2_projection_provider_threads
        (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
        VALUES (?, ?, ?, 'claudeAgent', 'running', ?, ?, 'claudeAgent', 'claude_inst')
        """, (ptid, tid, tid, f"2026-11-03T10:0{idx}:00Z", json.dumps(pt_payload)))
    conn.commit()
    conn.close()

    history_store = ContextHistoryStore()
    args = build_parser().parse_args(["--t3-dir", str(tmp_path), "--interval", "10.0"])

    # Navigation actions:
    # 1. ENTER (open Thread A detail)
    # 2. ESC (back to list)
    # 3. DOWN (select Thread B)
    # 4. ENTER (open Thread B detail)
    # 5. ESC (back to list)
    # 6. UP (select Thread A)
    # 7. ENTER (reopen Thread A detail)
    # 8. q (quit)
    actions = ["ENTER", "ESC", "DOWN", "ENTER", "ESC", "UP", "ENTER", "q"]
    adapter = ScriptedAdapter(actions)
    capture = CaptureLive()

    run_live_loop(args, _adapter=adapter, _live_cls=lambda **kw: capture, _history_store=history_store)

    # Both threads have their context history in the persistent history store
    key_a = ("th-a", "pt-a")
    key_b = ("th-b", "pt-b")
    assert len(history_store.get(key_a)) >= 1
    assert history_store.get(key_a)[0].used_tokens == 12000
    assert len(history_store.get(key_b)) >= 1
    assert history_store.get(key_b)[0].used_tokens == 34000


def test_summary_thread_switch_isolation_does_not_contaminate_selected_graph(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    # Thread S (summary): 50,000 tokens
    # Thread A (selected): 10,000 tokens
    for tid, ptid, tok, updated in [
        ("th-summary", "pt-summary", 50000, "2026-11-03T10:10:00Z"),
        ("th-selected", "pt-selected", 10000, "2026-11-03T10:00:00Z"),
    ]:
        th_payload = {"modelSelection": {"model": "gpt-5-codex"}, "lineage": {"parentThreadId": None}}
        conn.execute("""
        INSERT INTO orchestration_v2_projection_threads
        (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
         active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
        VALUES (?, 'proj', ?, 'codex', 'full-access', 'default', ?,
                '2026-11-03T10:00:00Z', ?, NULL, ?, 'codex_inst')
        """, (tid, f"Thread {tid}", ptid, updated, json.dumps(th_payload)))

        pt_payload = {
            "nativeThreadRef": {"driver": "codex", "nativeId": f"c-{tid}"},
            "contextUsage": {"usedTokens": tok, "maxTokens": 200000}
        }
        conn.execute("""
        INSERT INTO orchestration_v2_projection_provider_threads
        (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
        VALUES (?, ?, ?, 'codex', 'running', ?, ?, 'codex', 'codex_inst')
        """, (ptid, tid, tid, updated, json.dumps(pt_payload)))
    conn.commit()
    conn.close()

    history_store = ContextHistoryStore()
    # Query th-selected so it is selected, while th-summary is latest (roots[0]) as summary
    args = build_parser().parse_args(["th-selected", "--t3-dir", str(tmp_path), "--interval", "10.0"])
    adapter = ScriptedAdapter(["q"])

    run_live_loop(args, _adapter=adapter, _max_iterations=1, _live_cls=lambda **kw: CaptureLive(), _history_store=history_store)

    key_sel = ("th-selected", "pt-selected")
    key_sum = ("th-summary", "pt-summary")

    samples_sel = history_store.get(key_sel)
    samples_sum = history_store.get(key_sum)

    # Thread selected has ONLY 10,000 tokens reading; summary 50,000 does NOT contaminate it!
    assert len(samples_sel) == 1
    assert samples_sel[0].used_tokens == 10000

    assert len(samples_sum) == 1
    assert samples_sum[0].used_tokens == 50000


def test_avoid_timestamp_only_observation_causing_redraw_every_unchanged_poll(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES ('th-1', 'proj', 'Thread 1', 'claudeAgent', 'full-access', 'default',
            'pt-1', '2026-11-03T10:00:00Z', '2026-11-03T10:00:00Z', NULL,
            ?, 'claude_inst')
    """, (json.dumps(th_payload),))

    pt_payload = {
        "nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-1"},
        "contextUsage": {"usedTokens": 15000, "maxTokens": 200000}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES ('pt-1', 'th-1', 'th-1', 'claudeAgent', 'running', '2026-11-03T10:00:00Z', ?, 'claudeAgent', 'claude_inst')
    """, (json.dumps(pt_payload),))
    conn.commit()
    conn.close()

    args = build_parser().parse_args(["--t3-dir", str(tmp_path), "--interval", "0.01"])
    adapter = ScriptedAdapter(["q"])
    capture = CaptureLive()

    t = 0.0

    def clock():
        nonlocal t
        t += 0.05
        return t

    # Run for 3 poll iterations where data in database is unchanged
    code = run_live_loop(args, _adapter=adapter, _clock=clock, _max_iterations=3, _live_cls=lambda **kw: capture)
    assert code == 0
    # Initial render drew the frame once. Because context observations in poll 2 and 3 were unchanged,
    # signature did not change, avoiding timestamp-only observations causing redraw every unchanged poll!
    assert len(capture.updates) == 1


def test_graph_correction_repaints_with_unchanged_count_and_peak(monkeypatch):
    from t3_telemetry.models import ThreadTelemetryNode, ContextSample
    from types import SimpleNamespace
    from t3_telemetry import cli

    root = SimpleNamespace(thread_id="root", title="Root", model="model",
                           parent_thread_id=None, deleted_at=None,
                           updated_at="2026-11-03T10:00:00Z", created_at=None)
    snapshot = SimpleNamespace(threads={"root": root},
                               get_provider_thread=lambda _: None)
    monkeypatch.setattr(cli, "read_t3_state", lambda _: snapshot)
    builds = 0

    def build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return ThreadTelemetryNode(
            thread_id="root", title="Root", driver="codex",
            provider_instance_id="codex", model="model", context_used_tokens=10000,
            context_samples=[ContextSample(source_id="first", used_tokens=1000),
                             ContextSample(source_id="middle", used_tokens=5000 if builds == 1 else 8000),
                             ContextSample(source_id="last", used_tokens=10000)],
        )

    monkeypatch.setattr(cli, "build_thread_telemetry", build)
    tick = 0.0

    def clock():
        nonlocal tick
        tick += 0.05
        return tick

    capture = CaptureLive()
    adapter = ScriptedAdapter([None, None, None])
    args = build_parser().parse_args(["--interval", "0.01"])
    assert run_live_loop(args, _adapter=adapter, _clock=clock, _max_iterations=3,
                         _live_cls=lambda **kw: capture) == 0
    assert builds == 3
    assert len(capture.updates) == 2


@pytest.mark.parametrize("change", ["metrics", "tool", "child", "limit", "provenance", "turn"])
def test_live_values_repaint_without_row_count_changes(monkeypatch, change):
    from types import SimpleNamespace
    from t3_telemetry import cli
    from t3_telemetry.models import ThreadTelemetryNode, ModelStep, TurnMetrics

    root = SimpleNamespace(thread_id="root", title="Root", model="model",
                           parent_thread_id=None, deleted_at=None,
                           updated_at="2026-11-03T10:00:00Z", created_at=None)
    snapshot = SimpleNamespace(threads={"root": root}, get_provider_thread=lambda _: None)
    monkeypatch.setattr(cli, "read_t3_state", lambda _: snapshot)
    builds = 0

    def build(*args, **kwargs):
        nonlocal builds
        builds += 1
        node = ThreadTelemetryNode(
            thread_id="root", title="Root", driver="codex", provider_instance_id="codex",
            model="model", steps=[ModelStep(step_index=1, tool_calls=["read_file"])],
            current_turn_metrics=TurnMetrics(),
            subagents=[ThreadTelemetryNode(thread_id="child", title="Child", driver="codex",
                                          provider_instance_id="codex", model="model", status="running")],
        )
        if builds > 1:
            if change == "metrics":
                node.self_metrics.input_tokens = 2500
            elif change == "tool":
                node.steps[0].tool_calls = ["run_command"]
            elif change == "child":
                node.subagents[0].status = "completed"
            elif change == "limit":
                node.context_limit = 200000
            elif change == "provenance":
                node.context_uncertainty = True
            elif change == "turn":
                node.current_turn_metrics.output_tokens = 200
        return node

    monkeypatch.setattr(cli, "build_thread_telemetry", build)
    tick = 0.0

    def clock():
        nonlocal tick
        tick += 0.05
        return tick

    capture = CaptureLive()
    args = build_parser().parse_args(["--interval", "0.01"])
    assert run_live_loop(args, _adapter=ScriptedAdapter([None, None, None]), _clock=clock,
                         _max_iterations=3, _live_cls=lambda **kw: capture) == 0
    assert builds == 3
    assert len(capture.updates) == 2
