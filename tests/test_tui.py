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
    screen1.activity_offset = 3

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
    screen2.activity_offset = 5

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
    assert restored2.activity_offset == 5

    # 6. Escape back to depth 1: exact state restored
    nav.escape()
    assert nav.detail_id == "root-1"
    assert len(nav.stack) == 1
    restored1 = nav.current_screen
    assert restored1.focus == "subagents"
    assert restored1.selected_child_id == "child-2"
    assert restored1.activity_offset == 3

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
    nav.current_screen.focus = "subagents"

    nav.enter()
    nav.current_screen.title = "Child 1"
    nav.current_screen.child_ids = ["gc-1"]
    nav.current_screen.selected_child_id = "gc-1"
    nav.current_screen.focus = "subagents"

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
    screen.detail_activity_id = "a1"
    screen.max_detail_scroll = 5
    screen.detail_scroll = 0

    # Overscroll downwards by 20 moves
    for _ in range(20):
        nav.move(1)

    # Must be clamped strictly at 5, NOT 20
    assert screen.detail_scroll == 5

    # A single Up move immediately decrements to 4
    nav.move(-1)
    assert screen.detail_scroll == 4


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
    monkeypatch.setattr(cli, "UsageCache", lambda: _EmptyUsageCache())

    root = SimpleNamespace(active_provider_thread_id=None, thread_id="root", title="Root", model="model",
                           parent_thread_id=None, deleted_at=None,
                           updated_at="2026-11-03T10:00:00Z", created_at=None)
    snapshot = SimpleNamespace(provider_turns_by_thread={}, subagents_by_parent={}, provider_threads={}, threads={"root": root},
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


@pytest.mark.parametrize("change", ["metrics", "tool", "child", "limit", "provenance", "turn", "tool_detail", "context"])
def test_live_values_repaint_without_row_count_changes(monkeypatch, change):
    from types import SimpleNamespace
    from t3_telemetry import cli
    monkeypatch.setattr(cli, "UsageCache", lambda: _EmptyUsageCache())
    from t3_telemetry.models import ThreadTelemetryNode, ModelStep, TurnMetrics, ToolCallDetail, RecordedText

    root = SimpleNamespace(active_provider_thread_id=None, thread_id="root", title="Root", model="model",
                           parent_thread_id=None, deleted_at=None,
                           updated_at="2026-11-03T10:00:00Z", created_at=None)
    snapshot = SimpleNamespace(provider_turns_by_thread={}, subagents_by_parent={}, provider_threads={}, threads={"root": root}, get_provider_thread=lambda _: None)
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
            elif change == "tool_detail":
                node.steps[0].tool_details = [ToolCallDetail(name="read_file", fields={"path": "src/b.py"})]
            elif change == "context":
                node.steps[0].context_texts = [RecordedText(role="user", text="new prompt")]
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


# --- Activity selection, detail screen, and list visibility ---


def _activity_step(i, **kw):
    from t3_telemetry.models import ModelStep

    return ModelStep(step_index=i, source="main", tool_calls=[f"tool_{i}"], input_tokens=100,
                     output_tokens=50, duration_seconds=1.0, **kw)


def _activity_node(n_steps=12, n_children=5, steps=None):
    from t3_telemetry.models import ThreadTelemetryNode

    kids = [
        ThreadTelemetryNode(thread_id=f"c{i}", title=f"Child {i}", driver="claude",
                            provider_instance_id="x", model="m", status="running")
        for i in range(1, n_children + 1)
    ]
    return ThreadTelemetryNode(
        thread_id="th", title="Parent", driver="claude", provider_instance_id="x", model="m",
        status="running", steps=steps if steps is not None else [_activity_step(i) for i in range(1, n_steps + 1)],
        subagents=kids,
    )


def _open_overview(node):
    nav = NavigationState()
    nav.reconcile([node.thread_id])
    nav.enter()
    nav.current_screen.child_ids = [c.thread_id for c in node.subagents]
    nav.current_screen.selected_child_id = node.subagents[0].thread_id if node.subagents else None
    return nav


def _render_lines(node, nav, height, width, max_steps=50):
    buf = io.StringIO()
    layout = _browser_frame([node], node=node, summary=node, nav=nav, height=height, width=width, max_steps=max_steps)
    Console(file=buf, width=width, height=height, color_system=None).print(layout)
    lines = buf.getvalue().splitlines()
    assert len(lines) == height
    return lines


@pytest.mark.parametrize("height,width", [(30, 100), (24, 60), (40, 120)])
def test_child_selection_is_visible_after_every_key_across_all_children(height, width):
    node = _activity_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, height, width)
    browser.handle_key("TAB")
    assert nav.current_screen.focus == "subagents"

    visible_rows_seen = set()
    for expected in range(1, 6):
        lines = _render_lines(node, nav, height, width)
        assert any("▶" in line and f"Child {expected}" in line for line in lines), (expected, lines)
        visible_rows_seen.add(sum(1 for line in lines if "Child " in line and " m " in line))
        browser.handle_key("DOWN")
    # Tall sizes expand the list to show all five rows.
    if height >= 30:
        assert visible_rows_seen == {5}
    for expected in range(5, 0, -1):
        browser.handle_key("UP") if expected < 5 else None
        lines = _render_lines(node, nav, height, width)
        assert any("▶" in line and f"Child {expected}" in line for line in lines)


@pytest.mark.parametrize("height,width", [(30, 100), (24, 60)])
def test_activity_selection_is_visible_and_range_label_tracks_window(height, width):
    node = _activity_node(n_steps=12, n_children=0)
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    lines = _render_lines(node, nav, height, width)
    assert any("▶" in line and "tool_12" in line for line in lines)  # defaults to the latest entry
    assert any("of 12" in line and "Activity" in line for line in lines)
    for step in range(11, 0, -1):
        browser.handle_key("k")
        lines = _render_lines(node, nav, height, width)
        assert any("▶" in line and f"tool_{step}" in line and f"#{step}" in line for line in lines), step
    for step in range(2, 13):
        browser.handle_key("j")
        lines = _render_lines(node, nav, height, width)
        assert any("▶" in line and f"tool_{step}" in line for line in lines), step


def test_enter_dispatches_strictly_by_focus():
    node = _activity_node(n_steps=3, n_children=2)
    nav = _open_overview(node)
    _render_lines(node, nav, 30, 100)
    screen = nav.current_screen
    assert screen.focus == "activity" and screen.selected_child_id == "c1"

    nav.enter()  # activity focus must not open the selected child
    assert nav.detail_id == "th" and len(nav.stack) == 1
    assert screen.detail_activity_id == screen.selected_activity_id
    nav.enter()  # ordinary activity detail ignores Enter
    assert len(nav.stack) == 1 and screen.detail_activity_id is not None
    nav.tab()  # and Tab
    assert screen.focus == "activity"
    nav.escape()
    assert screen.detail_activity_id is None and len(nav.stack) == 1

    nav.tab()
    nav.enter()
    assert nav.detail_id == "c1" and len(nav.stack) == 2


def test_activity_detail_opens_selected_entry_scrolls_and_escape_restores_overview():
    steps = [_activity_step(i) for i in range(1, 13)]
    steps[4].tool_calls = [f"long_tool_name_{k}" for k in range(40)]
    node = _activity_node(steps=steps, n_children=3)
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    screen = nav.current_screen
    _render_lines(node, nav, 24, 60)
    browser.handle_key("TAB")
    browser.handle_key("DOWN")
    browser.handle_key("TAB")  # back to activity
    for _ in range(7):
        browser.handle_key("UP")  # select step 5
    before = _render_lines(node, nav, 24, 60)
    saved = (screen.focus, screen.selected_activity_id, screen.activity_offset,
             screen.selected_child_id, screen.child_scroll)

    browser.handle_key("ENTER")
    assert len(nav.stack) == 1 and nav.detail_id == "th"
    lines = _render_lines(node, nav, 24, 60)
    text = "\n".join(lines)
    assert "Activity detail" in text and "Model step #5" in text
    assert "Child 2" not in text
    assert screen.max_detail_scroll > 0 and screen.detail_scroll == 0

    browser.handle_key("TAB")  # from the action list to page scrolling
    for _ in range(100):
        browser.handle_key("j")
    assert screen.detail_scroll == screen.max_detail_scroll
    scrolled = "\n".join(_render_lines(node, nav, 24, 60))
    assert scrolled != text and "Unit of accounting" in scrolled
    browser.handle_key("k")
    assert screen.detail_scroll == screen.max_detail_scroll - 1
    assert screen.selected_activity_id == saved[1]  # detail scroll never moved list selection

    browser.handle_key("ESC")
    assert (screen.focus, screen.selected_activity_id, screen.activity_offset,
            screen.selected_child_id, screen.child_scroll) == saved
    assert len(nav.stack) == 1 and screen.detail_activity_id is None
    assert _render_lines(node, nav, 24, 60) == before


def test_activity_identity_is_stable_across_updates_and_removal():
    from t3_telemetry.activity import build_activity_entries

    node = _activity_node(n_steps=5, n_children=0)
    nav = _open_overview(node)
    screen = nav.current_screen
    _render_lines(node, nav, 30, 100)
    ids = [e.entry_id for e in build_activity_entries(node, 50)]
    assert len(set(ids)) == 5
    screen.selected_activity_id = ids[2]

    # New entries and changed values keep the selected ID and do not redirect it
    node.steps.append(_activity_step(6))
    node.steps[2].input_tokens = 999
    _render_lines(node, nav, 30, 100)
    assert screen.selected_activity_id == ids[2]

    # Window sliding (--max-steps) keeps identity by ID, not position
    assert build_activity_entries(node, 3)[0].entry_id == ids[3]

    # Removing the selected entry reconciles to the nearest remaining entry
    del node.steps[2]
    lines = _render_lines(node, nav, 30, 100)
    assert screen.selected_activity_id == ids[3]
    assert any("▶" in line and "tool_4" in line for line in lines)

    # An open detail whose entry disappears closes with a notice instead of crashing
    nav.enter()
    assert screen.detail_activity_id == ids[3]
    del node.steps[2]
    _render_lines(node, nav, 30, 100)
    assert screen.detail_activity_id is None
    assert nav.notice and "no longer available" in nav.notice
    assert nav.detail_id == "th" and len(nav.stack) == 1  # node identity untouched

    node.steps.clear()
    lines = _render_lines(node, nav, 30, 100)
    assert screen.selected_activity_id is None
    assert any("No recorded activity" in line for line in lines)


def test_activity_identity_uses_provider_turn_id_and_agent_not_position():
    from t3_telemetry.activity import build_activity_entries
    from t3_telemetry.models import ModelStep

    a = _activity_node(steps=[ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-1")], n_children=0)
    b = _activity_node(steps=[ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-1")], n_children=0)
    b.thread_id = "other"
    b.active_provider_thread_id = "prov-b"
    ea, eb = build_activity_entries(a)[0], build_activity_entries(b)[0]
    assert ea.entry_id != eb.entry_id and "pt-1" in ea.entry_id
    assert ea.accounting_scope == "provider turn" and ea.detail is a.steps[0]
    assert ea.child_thread_id is None


def test_running_turn_without_usage_is_not_shown_as_zero_tokens():
    from t3_telemetry.activity import build_activity_entries
    from t3_telemetry.models import ModelStep

    running = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="p", status="running")
    done = ModelStep(step_index=2, activity_kind="provider_turn", provider_turn_id="q", status="completed")
    node = _activity_node(steps=[running, done], n_children=0)
    first, second = build_activity_entries(node)
    assert first.total_tokens is None
    assert second.total_tokens == 0


def test_live_loop_activity_keys_do_not_read_data(monkeypatch):
    from types import SimpleNamespace
    from t3_telemetry import cli
    monkeypatch.setattr(cli, "UsageCache", lambda: _EmptyUsageCache())

    root = SimpleNamespace(active_provider_thread_id=None, thread_id="th", title="Parent", model="m", parent_thread_id=None,
                           deleted_at=None, updated_at="2026-11-03T10:00:00Z", created_at=None)
    snapshot = SimpleNamespace(provider_turns_by_thread={}, subagents_by_parent={}, provider_threads={}, threads={"th": root}, get_provider_thread=lambda _: None)
    reads = 0
    builds = 0

    def read(_):
        nonlocal reads
        reads += 1
        return snapshot

    def build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return _activity_node()

    monkeypatch.setattr(cli, "read_t3_state", read)
    monkeypatch.setattr(cli, "build_thread_telemetry", build)
    tick = 0.0

    def clock():
        nonlocal tick
        tick += 0.001
        return tick

    keys = ["ENTER", "k", "k", "ENTER", "j", "j", "ESC", "TAB", "j", "j", "j", "TAB", "ENTER", "ESC",
            (( 20, 50), "j"), "UP"]
    capture = CaptureLive()
    args = build_parser().parse_args(["--interval", "1000"])
    assert run_live_loop(args, _adapter=ScriptedAdapter(keys, initial_size=(30, 100)), _clock=clock,
                         _max_iterations=5, _live_cls=lambda **kw: capture) == 0
    assert reads == 1 and builds == 1
    assert len(capture.updates) > len(keys) // 2


def test_focus_falls_back_to_activity_when_subagents_disappear():
    from t3_telemetry.activity import build_activity_entries

    node = _activity_node()
    nav = _open_overview(node)
    nav.sync_activities([e.entry_id for e in build_activity_entries(node, 50)])
    nav.tab()
    assert nav.current_screen.focus == "subagents"

    node.subagents.clear()
    nav.sync_children([c.thread_id for c in node.subagents])
    screen = nav.current_screen
    assert screen.focus == "activity"
    last = screen.selected_activity_id
    nav.move(-1)
    assert screen.selected_activity_id != last
    nav.enter()
    assert screen.detail_activity_id == screen.selected_activity_id


# --- Recorded context in activity titles and detail screens ---


def _detail_text(step, node=None, **kwargs):
    from t3_telemetry.activity import build_activity_entries
    from t3_telemetry.models import ThreadTelemetryNode
    from t3_telemetry.ui import build_activity_detail_lines

    node = node or ThreadTelemetryNode(thread_id="th", title="T", driver="codex",
                                       provider_instance_id="codex", model="m", steps=[step])
    entry = build_activity_entries(node)[0]
    return entry, "\n".join(line.plain for line in build_activity_detail_lines(entry, node, **kwargs))


def test_recorded_command_gives_title_and_inspectable_details():
    from t3_telemetry.models import ModelStep, ToolCallDetail

    step = ModelStep(step_index=3, tool_calls=["run_command", "view_file"], tool_details=[
        ToolCallDetail(name="run_command", fields={"command": "uv run pytest", "cwd": "/repo"}),
        ToolCallDetail(name="view_file", fields={"path": "src/app.py"}),
    ])
    entry, text = _detail_text(step)
    assert entry.title == "Run command · uv run pytest (+1 more) · #3"
    assert "Run command · uv run pytest" in text
    assert "Read file · src/app.py" in text
    assert "Unit of accounting" in text


def test_name_only_and_undescribed_turn_fallbacks():
    from t3_telemetry.models import ModelStep

    entry, text = _detail_text(ModelStep(step_index=2, tool_calls=["run_command"]))
    assert entry.title == "run_command · details unavailable · #2"
    turn = ModelStep(step_index=4, activity_kind="provider_turn", provider_turn_id="pt", input_tokens=5, thinking_tokens=10)
    entry, text = _detail_text(turn)
    assert entry.title == "Provider turn #4 · action details unavailable"
    assert "No actions recorded for this turn." in text
    assert "not per-tool usage" in text


def test_control_characters_neutralized_in_title_and_detail():
    from t3_telemetry.models import ModelStep, RecordedText, ToolCallDetail

    step = ModelStep(step_index=1, tool_calls=["run_command"], tool_details=[
        ToolCallDetail(name="run_command", fields={"command": "echo \x1b[31mred\nnext\x07"}),
    ], context_texts=[RecordedText(role="user", text="ok\x1b]0;title\x07\nline2\x9b")])
    entry, text = _detail_text(step)
    assert "\x1b" not in entry.title and "\n" not in entry.title
    assert "\\x1b[31mred next\\x07" in entry.title
    assert "\x1b" not in text and "\x07" not in text and "\x9b" not in text
    assert "\\x1b]0;title" in text and "line2" in text


def test_truncated_text_is_marked():
    from t3_telemetry.models import ModelStep, RecordedText, ToolCallDetail

    step = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt",
                     tool_details=[ToolCallDetail(name="view_file", fields={"path": "a" * 10}, truncated=["path"])],
                     tool_calls=["view_file"],
                     context_texts=[RecordedText(role="user", text="long prompt", truncated=True)])
    entry, text = _detail_text(step)
    assert entry.title.endswith("aaaaaaaaaa… · #1")
    assert "(truncated)" in text and "─ Prompt" in text


def test_prompt_section_collapsed_and_expanded_ten_lines():
    from t3_telemetry.activity import build_activity_entries
    from t3_telemetry.models import ModelStep, RecordedText, ThreadTelemetryNode
    from t3_telemetry.ui import build_activity_detail_lines

    prompt_lines = [f"line {i}" for i in range(1, 11)]
    step = ModelStep(
        step_index=1,
        activity_kind="provider_turn",
        provider_turn_id="pt-1",
        context_texts=[RecordedText(role="user", text="\n".join(prompt_lines))],
    )
    node = ThreadTelemetryNode(thread_id="th", title="T", driver="codex",
                               provider_instance_id="codex", model="m", steps=[step])
    entry = build_activity_entries(node)[0]

    # Collapsed rendering (default prompt_expanded=False)
    lines_collapsed = build_activity_detail_lines(entry, node, prompt_expanded=False)
    text_collapsed = "\n".join(line.plain for line in lines_collapsed)

    # Section order: header -> Tokens -> What it did -> Prompt -> Details
    tokens_idx = text_collapsed.index("─ Tokens")
    what_idx = text_collapsed.index("─ What it did")
    prompt_idx = text_collapsed.index("─ Prompt")
    details_idx = text_collapsed.index("─ Details")
    assert 0 < tokens_idx < what_idx < prompt_idx < details_idx

    assert "  line 1" in text_collapsed
    assert "  line 2" in text_collapsed
    assert "  line 3" in text_collapsed
    assert "  … 7 more lines · P to expand" in text_collapsed
    assert "  line 4" not in text_collapsed
    assert "  line 10" not in text_collapsed

    hint_line = next(l for l in lines_collapsed if "more lines" in l.plain)
    assert hint_line.style == "dim"

    # Expanded rendering (prompt_expanded=True)
    lines_expanded = build_activity_detail_lines(entry, node, prompt_expanded=True)
    text_expanded = "\n".join(line.plain for line in lines_expanded)

    assert 0 < text_expanded.index("─ Tokens") < text_expanded.index("─ What it did") < text_expanded.index("─ Prompt") < text_expanded.index("─ Details")
    for i in range(1, 11):
        assert f"  line {i}" in text_expanded
    assert "more lines · P to expand" not in text_expanded


def test_prompt_section_no_section_without_prompt():
    from t3_telemetry.models import ModelStep, RecordedText

    # 1. No context_texts
    step1 = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-1")
    _, text1 = _detail_text(step1)
    assert "─ Prompt" not in text1
    assert "─ What it did" in text1 and "─ Details" in text1

    # 2. Empty context_texts
    step2 = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-2", context_texts=[])
    _, text2 = _detail_text(step2)
    assert "─ Prompt" not in text2

    # 3. Only assistant context_texts (no user role)
    step3 = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-3",
                      context_texts=[RecordedText(role="assistant", text="assistant reply")])
    _, text3 = _detail_text(step3)
    assert "─ Prompt" not in text3

    # 4. User role with empty text
    step4 = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-4",
                      context_texts=[RecordedText(role="user", text="")])
    _, text4 = _detail_text(step4)
    assert "─ Prompt" not in text4


def test_prompt_section_truncation_collapsed_and_expanded():
    from t3_telemetry.activity import build_activity_entries
    from t3_telemetry.models import ModelStep, RecordedText, ThreadTelemetryNode
    from t3_telemetry.ui import build_activity_detail_lines

    # Collapsed with truncated=True and <= 3 lines
    short_step = ModelStep(
        step_index=1,
        activity_kind="provider_turn",
        provider_turn_id="pt-1",
        context_texts=[RecordedText(role="user", text="short prompt 1\nshort prompt 2", truncated=True)],
    )
    node_short = ThreadTelemetryNode(thread_id="th", title="T", driver="codex", provider_instance_id="codex", model="m", steps=[short_step])
    entry_short = build_activity_entries(node_short)[0]
    lines_short_col = build_activity_detail_lines(entry_short, node_short, prompt_expanded=False)
    text_short_col = "\n".join(l.plain for l in lines_short_col)
    assert "  short prompt 1" in text_short_col
    assert "  short prompt 2" in text_short_col
    assert "  … (truncated)" in text_short_col
    assert "more lines · P to expand" not in text_short_col
    trunc_line = next(l for l in lines_short_col if "(truncated)" in l.plain)
    assert trunc_line.style == "dim"

    # Expanded with truncated=True and 10 lines
    long_lines = [f"line {i}" for i in range(1, 11)]
    long_step = ModelStep(
        step_index=1,
        activity_kind="provider_turn",
        provider_turn_id="pt-2",
        context_texts=[RecordedText(role="user", text="\n".join(long_lines), truncated=True)],
    )
    node_long = ThreadTelemetryNode(thread_id="th", title="T", driver="codex", provider_instance_id="codex", model="m", steps=[long_step])
    entry_long = build_activity_entries(node_long)[0]
    lines_long_exp = build_activity_detail_lines(entry_long, node_long, prompt_expanded=True)
    text_long_exp = "\n".join(l.plain for l in lines_long_exp)
    for i in range(1, 11):
        assert f"  line {i}" in text_long_exp
    assert "  … (truncated)" in text_long_exp
    assert "more lines · P to expand" not in text_long_exp
    trunc_line_exp = next(l for l in lines_long_exp if "(truncated)" in l.plain)
    assert trunc_line_exp.style == "dim"


def test_prompt_section_picks_first_user_role():
    from t3_telemetry.models import ModelStep, RecordedText

    step = ModelStep(
        step_index=1,
        activity_kind="provider_turn",
        provider_turn_id="pt-1",
        context_texts=[
            RecordedText(role="system", text="system prompt"),
            RecordedText(role="user", text="first user prompt"),
            RecordedText(role="assistant", text="assistant reply"),
            RecordedText(role="user", text="second user prompt"),
        ],
    )
    _, text = _detail_text(step)
    assert "  first user prompt" in text
    assert "system prompt" not in text
    assert "second user prompt" not in text


def test_provider_turn_prompt_title_and_usage_labels():
    from t3_telemetry.models import ModelStep, RecordedText

    turn = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt", status="running",
                     usage_status="unavailable", usage_provenance="unavailable",
                     context_texts=[RecordedText(role="user", text="Fix the\nbug")])
    entry, text = _detail_text(turn)
    assert entry.title == "Prompt · Fix the bug · #1"
    assert entry.total_tokens is None and "usage unavailable" in text
    zero = ModelStep(step_index=2, activity_kind="provider_turn", provider_turn_id="pz")
    entry, text = _detail_text(zero)
    assert entry.total_tokens == 0
    assert next(line for line in text.splitlines() if line.strip().startswith("Total")).split() == ["Total", "0"]
    fb = ModelStep(step_index=3, activity_kind="provider_turn", provider_turn_id="pf", input_tokens=9,
                   usage_provenance="cumulative_fallback")
    entry, text = _detail_text(fb)
    assert entry.usage_provenance == "cumulative_fallback" and "cumulative thread usage fallback" in text


def test_activity_tokens_cell_marks_approximate_totals():
    from t3_telemetry.models import ModelStep
    from t3_telemetry.ui import activity_tokens_cell

    def cell(**kw):
        step = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt", status="completed",
                         input_tokens=1000, **kw)
        return activity_tokens_cell(_detail_text(step)[0])

    assert not cell(usage_status="complete", usage_provenance="exact").endswith("~")
    assert cell(usage_status="complete", usage_provenance="cumulative_fallback").endswith("~")
    assert cell(usage_status="partial", usage_provenance="exact").endswith("~")
    assert cell(usage_status="unavailable", usage_provenance="unavailable") == "—"


def _launch_node():
    from t3_telemetry.models import SubagentLaunch

    node = _activity_node(n_steps=3, n_children=2)
    for i, step in enumerate(node.steps):
        step.started_at = f"2026-11-03T10:0{i}:00Z"
    node.launches = [
        SubagentLaunch(launch_id="subagent:s1", title="review the diff", origin="app_owned", status="running",
                       child_thread_id="c2", child_present=True, launched_at="2026-11-03T10:00:30Z",
                       launch_time_source="subagent record start", sources=["A", "B"]),
        SubagentLaunch(launch_id="subagent:s9", title="lost child", origin="app_owned", status="failed",
                       child_thread_id="gone", sources=["A"]),
    ]
    return node


def _select(nav, browser, entry_id):
    screen = nav.current_screen
    while screen.selected_activity_id != entry_id:
        before = screen.selected_activity_id
        browser.handle_key("UP")
        assert screen.selected_activity_id != before, entry_id


def test_launch_detail_opens_child_and_escape_restores_detail_then_parent():
    node = _launch_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    lines = _render_lines(node, nav, 30, 100)
    text = "\n".join(lines)
    assert "Subagent launched · review the diff" in text and "Undated · Subagent launched · lost child" in text
    parent = nav.current_screen
    launch_id = "th|launch:subagent:s1"
    assert parent.activity_ids.index(launch_id) == 2  # after step 1 (10:00), before step 2 (10:01)
    _select(nav, browser, launch_id)
    saved = (parent.selected_activity_id, parent.activity_offset, parent.focus)

    browser.handle_key("ENTER")
    detail = "\n".join(_render_lines(node, nav, 30, 100))
    for expected in ("Subagent launch", "review the diff", "Child thread: c2", "2026-11-03T10:00:30Z",
                     "running", "not applicable", "Open subagent", "Source A, Source B"):
        assert expected in detail, expected

    browser.handle_key("ENTER")  # Open subagent
    assert [s.node_id for s in nav.stack] == ["th", "c2"]
    child = node.subagents[1]
    child_text = "\n".join(_render_lines(child, nav, 30, 100))
    assert "Child 2" in child_text and "tool_1" not in child_text  # parent steps never leak into the child

    browser.handle_key("ESC")
    assert [s.node_id for s in nav.stack] == ["th"] and parent.detail_activity_id == launch_id
    assert "review the diff" in "\n".join(_render_lines(node, nav, 30, 100))
    browser.handle_key("ESC")
    assert parent.detail_activity_id is None
    assert (parent.selected_activity_id, parent.activity_offset, parent.focus) == saved


def test_launch_without_child_telemetry_is_inspectable_but_not_openable():
    node = _launch_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, 30, 100)
    _select(nav, browser, "th|launch:subagent:s9")
    browser.handle_key("ENTER")
    text = "\n".join(_render_lines(node, nav, 30, 100))
    assert "launch time not recorded" in text and "Child telemetry not found" in text
    browser.handle_key("ENTER")
    assert len(nav.stack) == 1 and nav.current_screen.detail_activity_id == "th|launch:subagent:s9"


def test_launch_status_change_repaints_and_launch_rows_show_no_tokens():
    from t3_telemetry.activity import build_activity_entries
    from t3_telemetry.ui import activity_tokens_cell

    node = _launch_node()
    markers = [e for e in build_activity_entries(node) if e.is_launch]
    assert [activity_tokens_cell(e) for e in markers] == ["—", "—"]
    before = node.to_dict()
    node.launches[0].status = "completed"
    assert node.to_dict() != before  # the repaint signature serializes the node


def test_shift_tab_decodes_and_reverses_focus_without_leaving_overview():
    master, slave = pty.openpty()
    class Stream:
        def fileno(self):
            return slave
    adapter = PosixTerminalAdapter(Stream())
    node = _activity_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(adapter, navigation=nav)
    try:
        adapter.begin()
        os.write(master, b"\x1b[Z")
        key = adapter.read_key(0.2)
        assert key == "SHIFT_TAB"
        browser.handle_key(key)
        assert nav.current_screen.focus == "subagents"
        browser.handle_key(key)
        assert nav.current_screen.focus == "activity"
        assert nav.detail_id == node.thread_id
    finally:
        adapter.end()
        os.close(master)
        os.close(slave)


def test_list_ranges_and_focus_are_in_section_rules():
    node = _activity_node(n_children=12)
    nav = _open_overview(node)
    for focus in ("activity", "subagents"):
        nav.current_screen.focus = focus
        lines = _render_lines(node, nav, 24, 60)
        for label in ("Subagents", "Activity"):
            rules = [i for i, line in enumerate(lines) if f" {label} · " in line and "of 12" in line]
            assert len(rules) == 1
            rule = lines[rules[0]]
            assert "───" in rule
            assert ("▌" in rule) == (focus == label.lower())
            # Column names sit on the line directly under the rule.
            assert "Tokens" in lines[rules[0] + 1] and "Time" in lines[rules[0] + 1]


def test_every_overview_section_uses_the_standard_rule():
    node = _activity_node(n_children=3)
    nav = _open_overview(node)
    text = "\n".join(_render_lines(node, nav, 40, 100))
    for label in ("─ Usage ─", "─ Context ─", "─ Subagents · ", "▌ Activity · "):
        assert label in text


def test_subagent_table_shows_total_tokens_not_in_out():
    node = _activity_node(n_children=3)
    nav = _open_overview(node)
    nav.current_screen.focus = "subagents"
    lines = _render_lines(node, nav, 30, 120)
    header = next(line for line in lines if "Model" in line and "Steps" in line)
    assert "Tokens" in header
    assert " In " not in header and "Out" not in header


def test_focus_switch_preserves_table_capacity_and_section_rules():
    node = _activity_node(n_steps=40, n_children=40)
    nav = _open_overview(node)
    for height, width in ((24, 60), (30, 80), (40, 120), (55, 140)):
        ranges = []
        for focus in ("activity", "subagents", "activity"):
            nav.current_screen.focus = focus
            lines = _render_lines(node, nav, height, width)
            headers = [i for i, line in enumerate(lines) if "of 40" in line]
            assert len(headers) == 2
            ranges.append([" ".join(lines[i].replace("▌", "─").split()) for i in headers])
            for i in headers:
                assert "───" in lines[i] and "Time" in lines[i + 1]
        # Ignore the focus marker, which replaces the rule's first dash.
        assert ranges[0] == ranges[1] == ranges[2]


def test_w_s_move_selection_and_scroll_activity_details():
    node = _activity_node()
    for step in node.steps:
        step.tool_calls = [f"tool_{step.step_index}_{k}" for k in range(3)]
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, 15, 60)
    original = nav.current_screen.selected_activity_id
    browser.handle_key("w")
    assert nav.current_screen.selected_activity_id != original
    browser.handle_key("s")
    assert nav.current_screen.selected_activity_id == original
    browser.handle_key("ENTER")
    _render_lines(node, nav, 15, 60)
    screen = nav.current_screen
    assert screen.max_detail_scroll > 0
    # W/S first move the highlighted action in "What it did".
    assert screen.detail_focus == "actions" and screen.detail_item_count > 1
    page_scroll = screen.detail_scroll
    browser.handle_key("s")
    assert screen.detail_item == 1 and screen.detail_scroll == page_scroll
    browser.handle_key("w")
    assert screen.detail_item == 0
    # Tab hands W/S back to page scrolling.
    browser.handle_key("TAB")
    browser.handle_key("w")
    _render_lines(node, nav, 15, 60)
    browser.handle_key("w")
    _render_lines(node, nav, 15, 60)
    assert screen.detail_scroll == 0
    browser.handle_key("s")
    assert screen.detail_scroll == 1 and screen.detail_item == 0


def test_e_opens_selection_and_ordinary_detail_ignores_e():
    node = _activity_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, 24, 60)
    selected = nav.current_screen.selected_activity_id
    browser.handle_key("e")
    assert nav.current_screen.detail_activity_id == selected
    browser.handle_key("E")
    assert len(nav.stack) == 1
    assert nav.current_screen.detail_activity_id == selected


def test_p_toggles_prompt_on_and_off_in_activity_detail():
    node = _activity_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, 24, 60)
    browser.handle_key("ENTER")
    screen = nav.current_screen
    assert screen.detail_activity_id is not None
    assert screen.detail_prompt_expanded is False

    browser.handle_key("p")
    assert screen.detail_prompt_expanded is True
    browser.handle_key("p")
    assert screen.detail_prompt_expanded is False
    browser.handle_key("P")
    assert screen.detail_prompt_expanded is True


def test_p_noop_on_activity_list_and_launch_details():
    # 1. Activity list: p does nothing
    node = _activity_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, 24, 60)
    screen = nav.current_screen
    assert screen.detail_activity_id is None
    assert screen.detail_prompt_expanded is False
    browser.handle_key("p")
    assert screen.detail_prompt_expanded is False
    browser.handle_key("P")
    assert screen.detail_prompt_expanded is False

    # 2. Launch details: p does nothing
    node_launch = _launch_node()
    nav_launch = _open_overview(node_launch)
    browser_launch = TelemetryBrowser(TerminalAdapter(), navigation=nav_launch)
    _render_lines(node_launch, nav_launch, 30, 100)
    launch_id = "th|launch:subagent:s1"
    _select(nav_launch, browser_launch, launch_id)
    browser_launch.handle_key("ENTER")
    launch_screen = nav_launch.current_screen
    assert launch_screen.detail_activity_id == launch_id
    assert launch_id in launch_screen.activity_targets
    assert launch_screen.detail_prompt_expanded is False
    browser_launch.handle_key("p")
    assert launch_screen.detail_prompt_expanded is False
    browser_launch.handle_key("P")
    assert launch_screen.detail_prompt_expanded is False


def test_p_resets_after_escape_and_reentering():
    node = _activity_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, 24, 60)
    browser.handle_key("ENTER")
    screen = nav.current_screen
    assert screen.detail_prompt_expanded is False

    browser.handle_key("p")
    assert screen.detail_prompt_expanded is True

    browser.handle_key("ESC")
    assert screen.detail_activity_id is None
    assert screen.detail_prompt_expanded is False

    browser.handle_key("ENTER")
    assert screen.detail_activity_id is not None
    assert screen.detail_prompt_expanded is False


def test_footer_line_shows_p_prompt_in_activity_detail_only():
    node = _activity_node()
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    lines = _render_lines(node, nav, 24, 80)
    assert "P prompt" not in lines[-1]

    browser.handle_key("ENTER")
    lines = _render_lines(node, nav, 24, 80)
    assert "W/S select  E/Enter full text  Tab page  P prompt" in lines[-1]
    browser.handle_key("TAB")
    lines = _render_lines(node, nav, 24, 80)
    assert "W/S scroll  Tab actions  P prompt" in lines[-1]


def test_detail_token_split_context_and_reply_filter():
    from t3_telemetry.models import ModelStep, RecordedText

    step = ModelStep(step_index=1, activity_kind="provider_turn", input_tokens=6,
                     cache_read_tokens=241250, cache_creation_tokens=943,
                     output_tokens=907, thinking_tokens=249,
                     context_texts=[RecordedText(role="assistant", text="secret reply")])
    _, text = _detail_text(step)
    for value in ("re-read context", "241,250", "99.6%", "new this turn", "949", "Total", "243,106"):
        assert value in text
    assert "Context at end" not in text
    assert "secret reply" not in text and "Recorded reply" not in text
    step.context_used_tokens = 100
    step.context_max_tokens = 1000
    _, text = _detail_text(step)
    assert "Context at end" in text


def test_detail_view_claude_turn_with_mixed_timeline():
    from t3_telemetry.models import ModelStep, TurnItem

    items = [
        TurnItem(kind="run", text="uv run pytest", started_at="2026-11-04T10:00:00Z", duration_seconds=1.5),
        TurnItem(kind="edit", text="src/ui.py", started_at="2026-11-04T10:00:02Z", duration_seconds=0.8),
        TurnItem(kind="search", text="find pattern", started_at="2026-11-04T10:00:03Z", duration_seconds=0.2),
        TurnItem(kind="tool", text="inspect", started_at="2026-11-04T10:00:04Z", duration_seconds=0.5),
        TurnItem(kind="subagent", text="reviewer", started_at="2026-11-04T10:00:05Z", duration_seconds=3.0),
        TurnItem(kind="think", started_at="2026-11-04T10:00:08Z", duration_seconds=2.0),
        TurnItem(kind="error", text="timeout", started_at="2026-11-04T10:00:10Z", duration_seconds=0.1),
        TurnItem(kind="compact", text="context", started_at="2026-11-04T10:00:11Z", duration_seconds=0.0),
        TurnItem(kind="run", text="git status", started_at="2026-11-04T10:00:12Z", status="running", duration_seconds=0.0),
    ]
    step = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-1", timeline=items)
    entry, text = _detail_text(step)

    assert "─ What it did" in text
    assert "2 commands · 1 edit · 1 tool · 1 search · 1 subagent · 1 thinking · 1 error · 1 compaction" in text
    assert "10:00:00  run       uv run pytest" in text
    assert "1.5s" in text
    assert "10:00:02  edit      src/ui.py" in text
    assert "0.8s" in text
    assert "10:00:03  search    find pattern" in text
    assert "0.2s" in text
    assert "10:00:04  tool      inspect" in text
    assert "0.5s" in text
    assert "10:00:05  subagent  reviewer" in text
    assert "3.0s" in text
    assert "10:00:08  think" in text
    assert "2.0s" in text
    assert "10:00:10  error     timeout" in text
    assert "0.1s" in text
    assert "10:00:11  compact   context" in text
    assert "10:00:12  run       git status (running)" in text


def test_detail_view_lists_every_item_without_a_view():
    from t3_telemetry.models import ModelStep, TurnItem
    from t3_telemetry.ui import _actions_lines

    items = [
        TurnItem(kind="run", text=f"command_{i}", started_at=f"2026-11-04T10:{i:02d}:00Z", duration_seconds=1.0)
        for i in range(35)
    ]
    step = ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-1", timeline=items)
    lines = _actions_lines(step, width=80)
    assert len(lines) == 36
    assert "35 commands" in lines[0].plain
    assert "command_34" in lines[-1].plain


def _long_timeline_step():
    from t3_telemetry.models import ModelStep, TurnItem

    long_text = "pytest " + " ".join(f"tests/test_module_{i}.py" for i in range(30))
    items = [
        TurnItem(kind="run", text=(long_text if i == 20 else f"command_{i}"),
                 started_at=f"2026-11-04T10:{i:02d}:00Z", duration_seconds=1.0)
        for i in range(35)
    ]
    return ModelStep(step_index=1, activity_kind="provider_turn", provider_turn_id="pt-1", timeline=items), long_text


def test_action_window_scrolls_and_truncates_selected_rows_like_the_rest():
    from t3_telemetry.ui import ActionsView, SELECTED_ROW_STYLE, _actions_lines

    step, _ = _long_timeline_step()
    view = ActionsView(selected=0, rows=8)
    lines = _actions_lines(step, width=60, view=view)
    assert view.count == 35 and view.scroll == 0
    assert view.range_label == "1–8 of 35"
    assert len(lines) == 9  # summary + 8 rows
    assert all(line.cell_len <= 60 for line in lines)

    view = ActionsView(selected=25, scroll=0, rows=8)
    lines = _actions_lines(step, width=60, view=view)
    assert view.scroll == 18 and view.range_label == "19–26 of 35"
    row20 = next(line for line in lines if "10:20:00" in line.plain)
    assert row20.plain.rstrip().endswith("1.0s") and "…" in row20.plain
    # A clear gutter separates the cut-off text from the duration column.
    assert row20.plain.index("…") <= 60 - len("1.0s") - 4 - 1

    # Selecting the long row keeps it on one identically truncated line, highlighted.
    view = ActionsView(selected=20, scroll=18, rows=8)
    selected = _actions_lines(step, width=60, view=view)
    assert len(selected) == 9 and view.selected_height == 1
    sel = selected[view.selected_line]
    assert sel.plain.startswith("▶ ") and sel.plain[2:] == row20.plain[2:]
    assert any(span.style == SELECTED_ROW_STYLE for span in sel.spans)

    view = ActionsView(selected=20, scroll=18, rows=8, focused=False)
    lines = _actions_lines(step, width=60, view=view)
    assert not any(span.style == SELECTED_ROW_STYLE for line in lines for span in line.spans)


def test_action_detail_page_shows_full_command_and_tool_fields():
    from t3_telemetry.models import ModelStep, ToolCallDetail
    from t3_telemetry.ui import build_action_detail_lines

    step, long_text = _long_timeline_step()
    lines = build_action_detail_lines(step, 20, width=60)
    plain = "\n".join(line.plain for line in lines)
    assert lines[0].plain.startswith("run · completed · 1.0s")
    assert "─ Command ─" in plain and long_text in plain
    assert build_action_detail_lines(step, 35, width=60) is None

    tool_step = ModelStep(step_index=1, activity_kind="model_step", tool_calls=["run_command"], tool_details=[
        ToolCallDetail(name="run_command", fields={"command": "cargo test --all", "cwd": "/repo"}, truncated=["command"]),
    ])
    plain = "\n".join(line.plain for line in build_action_detail_lines(tool_step, 0, width=60))
    assert "─ Command ─" in plain and "cargo test --all" in plain
    assert "─ Working dir ─" in plain and "/repo" in plain
    assert "cut at the retention limit" in plain


def test_enter_opens_action_page_and_escape_returns_to_the_same_row():
    from t3_telemetry.models import ThreadTelemetryNode

    step, long_text = _long_timeline_step()
    node = ThreadTelemetryNode(thread_id="th", title="Parent", driver="claude", provider_instance_id="x",
                               model="m", status="idle", steps=[step])
    nav = _open_overview(node)
    browser = TelemetryBrowser(TerminalAdapter(), navigation=nav)
    _render_lines(node, nav, 24, 60)
    browser.handle_key("ENTER")  # open the activity detail
    _render_lines(node, nav, 24, 60)
    screen = nav.current_screen
    for _ in range(20):
        browser.handle_key("s")
    assert screen.detail_item == 20

    browser.handle_key("ENTER")
    assert screen.action_open
    page = "\n".join(_render_lines(node, nav, 24, 60))
    assert "Action detail · 21 of 35" in page and "tests/test_module_1.py" in page
    assert "W/S scroll  Esc back" in page
    for _ in range(50):
        browser.handle_key("s")
    assert screen.action_scroll == screen.max_action_scroll
    assert screen.detail_item == 20  # scrolling the page never moves the selection

    browser.handle_key("ESC")
    assert not screen.action_open and screen.detail_activity_id is not None
    assert screen.detail_item == 20
    browser.handle_key("ESC")
    assert screen.detail_activity_id is None


def test_detail_view_antigravity_step_with_thinking_and_two_tools():
    from t3_telemetry.models import ModelStep, ToolCallDetail
    from t3_telemetry.ui import _actions_lines

    step = ModelStep(
        step_index=1,
        activity_kind="model_step",
        thinking_tokens=1500,
        tool_calls=["run_command", "view_file"],
        tool_details=[
            ToolCallDetail(name="run_command", fields={"command": "cargo test"}),
            ToolCallDetail(name="view_file", fields={"path": "src/main.rs"}),
        ],
    )
    entry, text = _detail_text(step)
    lines = _actions_lines(step, width=80)

    assert len(lines) == 4
    assert lines[0].plain == "1 Run command · 1 Read file"
    assert lines[1].plain == "think  1,500 thinking tokens"
    assert lines[2].plain == "Run command · cargo test"
    assert lines[3].plain == "Read file · src/main.rs"

    assert "1 Run command · 1 Read file" in text
    assert "think  1,500 thinking tokens" in text
    assert "Run command · cargo test" in text
    assert "Read file · src/main.rs" in text


def test_detail_view_timeline_row_formatting_and_gotchas():
    from t3_telemetry.models import ModelStep, TurnItem
    from t3_telemetry.ui import _actions_lines, _timeline_item_line

    # 1. Neutralize control chars and newlines in item text
    item_escape = TurnItem(
        kind="run",
        text="echo \x1b[31mred\nnewline\x07",
        started_at="2026-11-04T10:00:00Z",
        duration_seconds=1.2,
    )
    line_esc = _timeline_item_line(item_escape, width=80)
    assert "\x1b" not in line_esc.plain and "\n" not in line_esc.plain
    assert "\\x1b[31mred newline\\x07" in line_esc.plain

    # 2. Truncated item text gets marked with …
    item_trunc = TurnItem(
        kind="edit",
        text="src/very/long/path/to/some/nested/file.py",
        truncated=True,
        started_at="2026-11-04T10:00:00Z",
        duration_seconds=0.5,
    )
    line_trunc = _timeline_item_line(item_trunc, width=80)
    assert "file.py…" in line_trunc.plain

    # 3. Text elided to fit narrow width before right-aligned duration
    item_long = TurnItem(
        kind="run",
        text="a" * 100,
        started_at="2026-11-04T10:00:00Z",
        duration_seconds=2.5,
    )
    line_narrow = _timeline_item_line(item_long, width=50)
    assert len(line_narrow.plain) == 50
    assert line_narrow.plain.endswith("2.5s")
    assert "…" in line_narrow.plain
    item_long.text = "界" * 100
    line_wide_chars = _timeline_item_line(item_long, width=50)
    assert line_wide_chars.cell_len == 50
    assert line_wide_chars.plain.endswith("2.5s")
    assert "…" in line_wide_chars.plain

    # 4. Running tag in yellow for both text items and think items
    item_running_run = TurnItem(kind="run", text="cargo build", status="running", duration_seconds=0.0)
    line_rr = _timeline_item_line(item_running_run, width=60)
    assert "(running)" in line_rr.plain
    assert any(span.style == "yellow" for span in line_rr.spans)

    item_running_think = TurnItem(kind="think", status="running", duration_seconds=0.0)
    line_rt = _timeline_item_line(item_running_think, width=60)
    assert "(running)" in line_rt.plain
    assert any(span.style == "yellow" for span in line_rt.spans)

    # 5. Nothing recorded for empty step
    empty_step = ModelStep(step_index=1, activity_kind="model_step")
    empty_lines = _actions_lines(empty_step, width=80)
    assert len(empty_lines) == 1
    assert empty_lines[0].plain == "No actions recorded for this turn."
    assert empty_lines[0].style == "dim"


def test_usage_navigation_round_trip():
    nav = NavigationState()
    nav.tab()
    nav.move(1)
    nav.enter()
    assert nav.period == "week"
    nav.sync_top(["first", "second"])
    nav.move(1)
    nav.enter()
    assert nav.detail_id == "second"
    nav.escape()
    assert nav.period == "week"
    assert nav.selected_top_id == "second"
    nav.escape()
    assert nav.period is None
    assert nav.home_focus == "usage"
    assert nav.selected_period == "week"


def test_usage_navigation_clamps_and_tabs():
    nav = NavigationState()
    nav.tab()
    nav.move(-10)
    assert nav.selected_period == "today"
    nav.move(10)
    assert nav.selected_period == "month"
    nav.tab()
    assert nav.home_focus == "threads"
    nav.tab()
    nav.enter()
    nav.tab()
    assert nav.home_focus == "usage"
    nav.enter()
    assert not nav.stack
    nav.sync_top(["a", "b"])
    nav.move(-10)
    assert nav.selected_top_id == "a"
    nav.move(10)
    assert nav.selected_top_id == "b"


def test_sync_top_preserves_selection_and_nearest_index():
    nav = NavigationState()
    nav.sync_top(["a", "b", "c"])
    nav.selected_top_id = "b"
    nav.sync_top(["c", "a", "b"])
    assert nav.selected_top_id == "b"
    nav.sync_top(["c", "a"])
    assert nav.selected_top_id == "a"
    nav.sync_top([])
    assert nav.selected_top_id is None
    nav.sync_top(["x"])
    assert nav.selected_top_id == "x"


def test_reconcile_missing_period_thread_preserves_period():
    nav = NavigationState(period="week")
    nav.sync_top(["a"])
    nav.enter()
    nav.reconcile([])
    assert not nav.stack
    assert nav.period == "week"
    assert nav.selected_top_id == "a"


class _EmptyUsageCache:
    def get(self, *args, **kwargs):
        return None


def test_period_live_flow_and_frame_size(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from t3_telemetry import cli
    from t3_telemetry.usage import build_usage_report, ThreadUsage, TokenBreakdown

    now = datetime(2026, 11, 4, tzinfo=timezone.utc)
    report = build_usage_report([], now)
    for period in report.periods.values():
        period.threads.append(ThreadUsage("th", "Parent", "idle", ["Codex"], TokenBreakdown(input_tokens=100)))
    class FakeCache:
        def get(self, *args, **kwargs):
            return report
    root = SimpleNamespace(active_provider_thread_id=None, thread_id="th", title="Parent", model="m", parent_thread_id=None, deleted_at=None, updated_at="", created_at=None)
    snapshot = SimpleNamespace(provider_turns_by_thread={}, subagents_by_parent={}, provider_threads={}, threads={"th": root}, get_provider_thread=lambda _: None)
    monkeypatch.setattr(cli, "read_t3_state", lambda _: snapshot)
    monkeypatch.setattr(cli, "build_thread_telemetry", lambda *a, **kw: _activity_node())
    capture = CaptureLive()
    tick = iter(range(1000))
    args = build_parser().parse_args(["--interval", "1000"])
    assert run_live_loop(args, _adapter=ScriptedAdapter(["TAB", "DOWN", "ENTER", "ENTER", "ESC", "ESC"]), _clock=lambda: next(tick) / 1000, _max_iterations=5, _live_cls=lambda **kw: capture, _usage_cache=FakeCache()) == 0
    def render(frame, width=80, height=24):
        output = io.StringIO()
        Console(file=output, width=width, height=height, force_terminal=True, color_system=None).print(frame)
        return output.getvalue()
    frames = [render(frame) for frame in capture.updates]
    assert any("This week" in f and "Providers" in f and "Top threads" in f for f in frames)
    assert any("This week ›" in f for f in frames)
    assert "Top threads" in frames[-2] and "App Threads" in frames[-1]
    for width, height in [(60, 24), (120, 40)]:
        nav = NavigationState(period="month")
        frame = cli._browser_frame([], None, None, nav, height, width, usage=report, now=now)
        assert len(render(frame, width, height).splitlines()) == height

@pytest.mark.parametrize("width,height", [(60, 24), (120, 40)])
def test_home_sections_and_running_clock(width, height):
    from datetime import datetime, timezone, timedelta
    from types import SimpleNamespace
    from t3_telemetry.models import ThreadTelemetryNode, ModelStep
    from t3_telemetry.usage import build_usage_report
    from t3_telemetry.ui import format_duration

    now = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    summary = ThreadTelemetryNode(thread_id="root", title="Live chat", driver="codex", provider_instance_id="codex", model="m")
    turn = ModelStep(step_index=1, status="running", started_at=(now - timedelta(seconds=134)).isoformat())
    snapshot = SimpleNamespace(provider_turns_by_thread={"root": [turn]}, get_provider_thread=lambda _: None)
    nav = NavigationState()
    frame = _browser_frame([], None, summary, nav, height, width, snapshot=snapshot,
                           usage=build_usage_report([], now), now=now, running_ids=["root", "a", "b"])
    buf = io.StringIO()
    Console(file=buf, width=width, height=height, force_terminal=True, color_system=None).print(frame)
    output = buf.getvalue()
    assert len(output.splitlines()) == height
    for text in ("Live", "Usage", "Today", "This week", "This month", "Chats", "Turn running " + format_duration(134), "+2 other threads running"):
        assert text in output


def test_idle_home_does_not_repaint_on_unchanged_polls(monkeypatch):
    from types import SimpleNamespace
    from t3_telemetry import cli
    snapshot = SimpleNamespace(threads={}, provider_turns_by_thread={}, subagents_by_parent={}, provider_threads={}, get_provider_thread=lambda _: None)
    monkeypatch.setattr(cli, "read_t3_state", lambda _: snapshot)
    capture = CaptureLive()
    tick = iter(range(1000))
    args = build_parser().parse_args(["--interval", "0.01"])
    run_live_loop(args, _adapter=ScriptedAdapter([None, None, None]), _clock=lambda: next(tick) * 0.05,
                  _max_iterations=3, _live_cls=lambda **kw: capture, _usage_cache=_EmptyUsageCache())
    assert len(capture.updates) == 1
