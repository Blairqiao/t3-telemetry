"""CLI entrypoint for T3 Telemetry (t3-top)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Callable

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from t3_telemetry.activity import build_activity_entries, launch_targets
from t3_telemetry.aggregator import (
    running_root_ids,
    resolve_thread_id_in_snapshot,
    snapshot_thread_status,
)
from datetime import datetime, timezone, timedelta
from t3_telemetry.usage import UsageCache, UsageReport, PERIOD_LABELS, node_cost, read_store_usage_report, stats_for_node
from t3_telemetry.collector import Collector, default_roots
from t3_telemetry.store import TelemetryStore, session_rows
from t3_telemetry.db_t3 import T3StateSnapshot, read_t3_state
from t3_telemetry.thread_view import build_thread_node, resolve_thread_node, list_root_nodes

from t3_telemetry.context_history import ContextHistoryStore, make_agent_key
from t3_telemetry.models import AppThreadRecord, ContextSample, ThreadTelemetryNode
from t3_telemetry.terminal import PosixTerminalAdapter
from t3_telemetry.tui import (
    NavigationState,
    ScreenState,
    TelemetryBrowser,
    TerminalAdapter,
    Viewport,
    table_geometry,
    table_header_lines,
    window_start,
)
from t3_telemetry.ui import (
    SELECTED_ROW_STYLE,
    format_measure,
    usage_graph_lines,
    breakdown_lines,
    provider_lines,
    ActionsView,
    build_action_detail_lines,
    section_rule,
    activity_range_label,
    build_activity_detail_lines,
    build_activity_table,
    build_dashboard_renderable,
    context_graph_lines,
    context_graph_min_height,
    render_context_graph,
    build_scopes_summary,
    build_scopes_table,
    build_threads_list_table,
    build_sessions_table,
    format_duration,
    render_context_bar,
    format_tokens,
    format_usd,
    get_status_style,
)


def validate_positive_interval(value: str) -> float:
    try:
        val = float(value)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid interval: '{value}'")
    if not math.isfinite(val) or val <= 0:
        raise argparse.ArgumentTypeError(
            f"--interval must be a finite positive number, got '{value}'"
        )
    return val


def build_parser() -> argparse.ArgumentParser:
    """Build argparse argument parser for t3-top CLI."""
    parser = argparse.ArgumentParser(
        prog="t3-top",
        description="Live telemetry dashboard for T3 Code and subagent runs.",
    )
    parser.add_argument(
        "thread_query",
        nargs="?",
        default=None,
        help="Optional thread ID prefix, native conversation ID prefix, or title substring.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print a single telemetry snapshot to stdout and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON telemetry and exit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent app threads with summary metrics and exit.",
    )
    parser.add_argument(
        "--interval",
        type=validate_positive_interval,
        default=0.5,
        help="Refresh polling interval in seconds for live TUI mode.",
    )
    parser.add_argument(
        "--t3-dir",
        type=Path,
        default=Path("~/.t3/userdata").expanduser(),
        help="Override base T3 userdata directory.",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Pin the initially resolved thread in live TUI mode instead of following latest active.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max recent steps/turns to display in the stream table.",
    )
    parser.add_argument("--store", type=Path, default=Path("~/.t3-telemetry/telemetry.sqlite").expanduser(), help="Path to the local telemetry store.")
    parser.add_argument("--no-store", action="store_true", help="Do not sync native session files; read the store as it is.")
    parser.add_argument("--sync", action="store_true", help="Sync the telemetry store once and exit.")
    parser.add_argument("--sessions", action="store_true", help="List recent native sessions from the telemetry store and exit.")
    return parser


def _synced_store(args: argparse.Namespace, t3_dir: Path) -> tuple[TelemetryStore, T3StateSnapshot]:
    store = TelemetryStore(args.store.expanduser())
    try:
        try:
            snapshot = read_t3_state(t3_dir)
        except (sqlite3.Error, OSError):
            snapshot = T3StateSnapshot()
        if not args.no_store:
            stats = Collector(store, default_roots(t3_dir)).sync(snapshot)
            if args.sync:
                print(f"files_seen={stats.files_seen} files_read={stats.files_read} steps_written={stats.steps_written} errors={stats.errors}")
        return store, snapshot
    except BaseException:
        store.close()
        raise


def run_live_loop(
    args: argparse.Namespace,
    console: Console | None = None,
    _max_iterations: int | None = None,
    _adapter: TerminalAdapter | None = None,
    _clock: Callable[[], float] = time.monotonic,
    _live_cls: type[Live] | None = None,
    _history_store: ContextHistoryStore | None = None,
    _usage_cache: UsageCache | None = None,
    _collector: Collector | None = None,
) -> int:
    """Run live watch TUI loop with graceful refresh and error handling."""
    console = console or Console()

    if not math.isfinite(args.interval) or args.interval <= 0:
        raise argparse.ArgumentTypeError("--interval must be a finite positive number")

    # Noninteractive fallback
    if _adapter is None and _max_iterations is None:
        if not console.is_terminal or not sys.stdin.isatty():
            store, snapshot = _synced_store(args, Path(args.t3_dir).expanduser())
            try:
                node = resolve_thread_node(snapshot, store, args.thread_query)
                if node is not None:
                    console.print(build_dashboard_renderable(node, max_steps=args.max_steps))
                return 0
            finally:
                store.close()

    t3_dir = Path(args.t3_dir).expanduser()
    adapter = _adapter or PosixTerminalAdapter()
    history_store = _history_store if _history_store is not None else ContextHistoryStore()
    browser = TelemetryBrowser(adapter, history_store=history_store)
    live_factory = _live_cls or Live

    store = TelemetryStore(Path(args.store).expanduser())
    collector = None if args.no_store else (_collector or Collector(store, default_roots(t3_dir)))
    usage_cache = _usage_cache if _usage_cache is not None else UsageCache(
        background=True, reader=lambda d, s: read_store_usage_report(store, s)
    )
    sync_worker = None
    last_sync = None

    def start_sync(snapshot, now):
        nonlocal sync_worker, last_sync
        if collector is None or (sync_worker is not None and sync_worker.is_alive()):
            return
        if last_sync is not None and now - last_sync < 5:
            return
        last_sync = now
        def run():
            try:
                collector.sync(snapshot)
            except Exception:
                pass  # Keep worker failures from corrupting the terminal display.
        sync_worker = threading.Thread(target=run, name="telemetry-sync", daemon=True)
        sync_worker.start()
    usage = None
    last_period = None
    cached_snapshot = None
    cached_roots: list[AppThreadRecord] = []
    tree_node_cache: dict[str, ThreadTelemetryNode] = {}
    cached_node: ThreadTelemetryNode | None = None
    cached_summary: ThreadTelemetryNode | None = None
    cached_frame = None
    running_ids = []

    query_resolved = False
    polls = 0
    deadline = _clock()
    last_size = adapter.size()
    last_signature = None

    def index_telemetry_tree(n: ThreadTelemetryNode) -> None:
        tree_node_cache[n.thread_id] = n
        if n.native_id:
            tree_node_cache[n.native_id] = n
        for c in n.subagents:
            index_telemetry_tree(c)

    def compute_signature(roots, summary, node, nav, size):
        roots_sig = tuple(
            (
                r.thread_id,
                r.title,
                (
                    cached_snapshot.get_provider_thread(r.thread_id).status
                    if cached_snapshot and cached_snapshot.get_provider_thread(r.thread_id)
                    else None
                ),
                r.model,
                r.updated_at,
            )
            for r in roots
        )
        # Serialized values include mutable usage, tool labels, child status,
        # and context provenance. Counts alone miss in-flight updates.
        summary_sig = summary.to_dict() if summary else None
        node_sig = node.to_dict() if node else None
        cur_screen = nav.current_screen
        stack_sig = tuple(s.node_id for s in nav.stack)
        nav_sig = (
            nav.selected_id,
            stack_sig,
            cur_screen.focus if cur_screen else None,
            cur_screen.selected_child_id if cur_screen else None,
            cur_screen.selected_activity_id if cur_screen else None,
            cur_screen.activity_offset if cur_screen else None,
            cur_screen.detail_activity_id if cur_screen else None,
            cur_screen.detail_scroll if cur_screen else None,
            (cur_screen.detail_focus, cur_screen.detail_item, cur_screen.action_open, cur_screen.action_scroll) if cur_screen else None,
            cur_screen.child_scroll if cur_screen else None,
            nav.list_scroll,
            nav.notice,
            nav.home_focus, nav.selected_period, nav.period, nav.selected_top_id, nav.top_scroll,
            nav.measure, nav.graph_split,
        )
        return (
            roots_sig,
            usage.generated_at if usage else None,
            nav_sig,
            summary_sig,
            node_sig,
            size,
            tuple(running_ids),
            int(_clock()) if summary and cached_snapshot and any(t.status == "running" and t.started_at for t in cached_snapshot.provider_turns_by_thread.get(summary.thread_id, [])) else None,
        )

    try:
        adapter.begin()
        with live_factory(auto_refresh=False, console=console, screen=True, transient=True) as live:
            while True:
                if _max_iterations is not None and polls >= _max_iterations:
                    break

                now = _clock()
                if now >= deadline:
                    polls += 1
                    try:
                        snapshot = read_t3_state(t3_dir)
                        if polls == 1 and collector is not None and store.get_meta("last_sync_at") is None:
                            collector.sync(snapshot)
                            last_sync = now
                        start_sync(snapshot, now)
                        if _usage_cache is None and not args.no_store and usage_cache.report is None:
                            # UsageCache normally reads its first report synchronously.
                            if (usage_cache.built_at is None or
                                    usage_cache.clock() - usage_cache.built_at >= usage_cache.max_age):
                                usage_cache._start_worker(t3_dir, snapshot)
                        else:
                            usage = usage_cache.get(t3_dir, snapshot, force=polls == 1 or browser.navigation.period != last_period)
                        last_period = browser.navigation.period
                        if usage and browser.navigation.period:
                            browser.navigation.sync_top([t.thread_id for t in usage.periods[browser.navigation.period].top(browser.navigation.measure)])
                        cached_snapshot = snapshot
                        running_ids = running_root_ids(snapshot)
                        tree_node_cache.clear()

                        roots = [
                            t for t in snapshot.threads.values()
                            if t.parent_thread_id is None and t.deleted_at is None
                        ]
                        roots.sort(key=lambda t: t.updated_at or t.created_at or "", reverse=True)
                        cached_roots = roots

                        browser.navigation.reconcile([t.thread_id for t in roots])

                        if args.thread_query and not query_resolved:
                            chosen_id = resolve_thread_id_in_snapshot(snapshot, args.thread_query)
                            if chosen_id:
                                browser.navigation.selected_id = chosen_id
                                browser.navigation.detail_id = chosen_id
                            query_resolved = True

                        if args.no_follow and browser.navigation.pinned_summary_id is None:
                            browser.navigation.pinned_summary_id = (
                                browser.navigation.detail_id or browser.navigation.selected_id
                            )

                        summary_id = browser.navigation.pinned_summary_id or (
                            roots[0].thread_id if roots else None
                        )

                        def get_or_build_node(tid: str | None) -> ThreadTelemetryNode | None:
                            if not tid:
                                return None
                            if tid in tree_node_cache:
                                return tree_node_cache[tid]
                            if tid in snapshot.threads:
                                n = build_thread_node(tid, snapshot, store)
                                index_telemetry_tree(n)
                                return n
                            return None

                        summary = get_or_build_node(summary_id)
                        cached_summary = summary

                        # Build root tree for active detail view or selection
                        if browser.navigation.stack:
                            active_root_id = browser.navigation.stack[0].node_id
                            get_or_build_node(active_root_id)
                        else:
                            if browser.navigation.selected_id:
                                get_or_build_node(browser.navigation.selected_id)

                        # Disappearance reconciliation for tree
                        browser.navigation.reconcile_tree(
                            set(tree_node_cache.keys()),
                            {k: v.title for k, v in tree_node_cache.items()}
                        )

                        active_id = browser.navigation.detail_id or browser.navigation.selected_id
                        node = tree_node_cache.get(active_id) if active_id else None
                        if browser.navigation.stack and node:
                            cur_screen = browser.navigation.current_screen
                            if cur_screen:
                                cur_screen.title = node.title
                                browser.navigation.sync_children([c.thread_id for c in node.subagents])
                                feed = build_activity_entries(node, args.max_steps)
                                browser.navigation.sync_activities([e.entry_id for e in feed], launch_targets(feed))
                        cached_node = node

                        def sample_agent_context(target_node: ThreadTelemetryNode | None) -> None:
                            if target_node is None:
                                return
                            key = make_agent_key(target_node)
                            obs = None
                            if target_node.context_provenance != "unknown":
                                now_iso = datetime.now(timezone.utc).isoformat()
                                obs = ContextSample(
                                    source_id=f"obs:{target_node.active_provider_thread_id or target_node.thread_id}:{now_iso}",
                                    used_tokens=target_node.context_used_tokens,
                                    timestamp=None,
                                    limit=target_node.context_limit,
                                    provenance=target_node.context_provenance,
                                    provider_thread_id=target_node.active_provider_thread_id,
                                    observed_at=now_iso,
                                    timestamp_kind="observation",
                                )
                            target_node.context_samples = history_store.merge(
                                key,
                                target_node.context_samples,
                                obs,
                            )

                        sample_agent_context(cached_summary)
                        sample_agent_context(cached_node)

                        height, width = last_size
                        sig = compute_signature(roots, summary, node, browser.navigation, last_size)

                        if sig != last_signature:
                            last_signature = sig
                            frame = _browser_frame(
                                roots,
                                node,
                                summary,
                                browser.navigation,
                                height,
                                width,
                                args.max_steps,
                                snapshot,
                                usage=usage, now=datetime.now().astimezone(), running_ids=running_ids, syncing=store.get_meta("last_sync_at") is None,
                            )
                            cached_frame = frame
                            live.update(frame, refresh=True)
                            # The frame may settle list offsets; record the settled state.
                            last_signature = compute_signature(roots, summary, node, browser.navigation, last_size)
                    except (sqlite3.Error, OSError) as e:
                        frame = Panel(f"[yellow]Database unavailable: {e}[/]", title="T3 Telemetry")
                        if str(e) != last_signature:
                            last_signature = str(e)
                            cached_frame = frame
                            live.update(frame, refresh=True)

                    deadline = _clock() + args.interval

                if _max_iterations is not None and polls >= _max_iterations:
                    break

                try:
                    time_to_deadline = max(0.0, deadline - _clock())
                    wait_time = min(0.05, time_to_deadline)
                    key = adapter.read_key(wait_time)
                except KeyboardInterrupt:
                    break

                now_size = adapter.size()
                size_changed = (now_size != last_size)
                if size_changed:
                    last_size = now_size

                if key is not None:
                    if not browser.handle_key(key):
                        break

                    if usage and browser.navigation.period:
                        browser.navigation.sync_top([t.thread_id for t in usage.periods[browser.navigation.period].top(browser.navigation.measure)])
                    active_id = browser.navigation.detail_id or browser.navigation.selected_id
                    if active_id and active_id not in tree_node_cache and cached_snapshot and active_id in cached_snapshot.threads:
                        n = build_thread_node(active_id, cached_snapshot, store)
                        index_telemetry_tree(n)
                        cached_node = n
                    else:
                        cached_node = tree_node_cache.get(active_id) if active_id else None

                    if browser.navigation.stack and cached_node:
                        cur_screen = browser.navigation.current_screen
                        if cur_screen:
                            cur_screen.title = cached_node.title
                            browser.navigation.sync_children([c.thread_id for c in cached_node.subagents])
                            feed = build_activity_entries(cached_node, args.max_steps)
                            browser.navigation.sync_activities([e.entry_id for e in feed], launch_targets(feed))

                    if cached_node is not None:
                        node_key = make_agent_key(cached_node)
                        stored_samples = history_store.get(node_key)
                        if stored_samples:
                            cached_node.context_samples = stored_samples
                        else:
                            obs = None
                            if cached_node.context_provenance != "unknown":
                                now_iso = datetime.now(timezone.utc).isoformat()
                                obs = ContextSample(
                                    source_id=f"obs:{cached_node.active_provider_thread_id or cached_node.thread_id}:{now_iso}",
                                    used_tokens=cached_node.context_used_tokens,
                                    timestamp=None,
                                    limit=cached_node.context_limit,
                                    provenance=cached_node.context_provenance,
                                    provider_thread_id=cached_node.active_provider_thread_id,
                                    observed_at=now_iso,
                                    timestamp_kind="observation",
                                )
                            cached_node.context_samples = history_store.merge(
                                node_key,
                                cached_node.context_samples,
                                obs,
                            )

                    height, width = last_size
                    frame = _browser_frame(
                        cached_roots,
                        cached_node,
                        cached_summary,
                        browser.navigation,
                        height,
                        width,
                        args.max_steps,
                        cached_snapshot,
                        usage=usage, now=datetime.now().astimezone(), running_ids=running_ids, syncing=store.get_meta("last_sync_at") is None,
                    )
                    cached_frame = frame
                    live.update(frame, refresh=True)
                    last_signature = compute_signature(
                        cached_roots,
                        cached_summary,
                        cached_node,
                        browser.navigation,
                        last_size,
                    )
                elif size_changed:
                    height, width = last_size
                    frame = _browser_frame(
                        cached_roots,
                        cached_node,
                        cached_summary,
                        browser.navigation,
                        height,
                        width,
                        args.max_steps,
                        cached_snapshot,
                        usage=usage, now=datetime.now().astimezone(), running_ids=running_ids, syncing=store.get_meta("last_sync_at") is None,
                    )
                    cached_frame = frame
                    live.update(frame, refresh=True)
                    last_signature = compute_signature(
                        cached_roots,
                        cached_summary,
                        cached_node,
                        browser.navigation,
                        last_size,
                    )
    finally:
        try:
            adapter.end()
        finally:
            # Bounded joins keep quit fast during a long first sync. Writes are single locked
            # transactions, so close() lands between them and a late write fails as sqlite3.Error.
            if sync_worker is not None:
                sync_worker.join(timeout=1.0)
            if store is not None:
                if getattr(usage_cache, "_worker", None) is not None:
                    usage_cache._worker.join(timeout=1.0)
                store.close()
    return 0


def _context_text(node: ThreadTelemetryNode) -> str:
    if node.context_limit is not None and node.context_limit > 0:
        return f"Context: {format_tokens(node.context_used_tokens)}/{format_tokens(node.context_limit)} ({node.context_pct:.0f}%)"
    if getattr(node, "context_limit_provenance", "") == "measured":
        return f"Context: {format_tokens(node.context_used_tokens)}/{format_tokens(node.context_max_tokens)} ({node.context_pct:.0f}%)"
    return f"Context: {format_tokens(node.context_used_tokens)} / ? (limit unknown)"


def _usage_label(row, measure: str) -> str:
    """A thread's total in the active measure; spend with no list price reads 'unpriced', not $0."""
    if measure == "usd" and row.cost.total <= 0 and row.cost.unpriced_tokens:
        return "unpriced"
    return format_measure(row.value(measure), measure)


def _fit(text: Text, width: int) -> Text:
    text.no_wrap = True
    text.truncate(max(1, width), overflow="ellipsis")
    return text


def _root_strip(summary: ThreadTelemetryNode | None, width: int) -> Text:
    """One-line strip for the followed root: status, context, and title first; extras only if they fit."""
    strip = Text()
    if not summary:
        strip.append("T3 Telemetry  ·  No active app threads", style="bold")
        return _fit(strip, width)
    status = summary.status or "idle"
    style = get_status_style(status)
    ctx = _context_text(summary)
    strip.append("Live ", style="bold cyan")
    strip.append(status.upper(), style=style or "bold")
    strip.append("  ")
    fixed = strip.cell_len + 5 + len(ctx)
    title = Text(summary.title or summary.thread_id[:8], style=style or "bold")
    _fit(title, max(8, width - fixed))
    strip.append_text(title)
    strip.append("  ·  ")
    strip.append(ctx, style="bold")
    for extra, extra_style in (
        (summary.model or "unknown", "bold yellow"),
        (f"Act: {summary.steps[-1].action_summary}" if summary.steps else "", "magenta"),
    ):
        if extra and strip.cell_len + 5 + len(extra) <= width:
            strip.append("  ·  ")
            strip.append(extra, style=extra_style)
    return _fit(strip, width)


def _footer_line(nav: NavigationState, node: ThreadTelemetryNode | None, width: int) -> Text:
    """The only place key help appears; the keys match the current focus."""
    screen = nav.current_screen
    money = "$ tokens" if nav.measure == "usd" else "$ cost"
    if nav.detail_id and node and screen is not None:
        if screen.detail_activity_id in screen.activity_targets:
            keys = "E/Enter open subagent  W/S scroll  Esc back  q quit"
        elif screen.detail_activity_id and screen.action_open:
            keys = "W/S scroll  Esc back  q quit"
        elif screen.detail_activity_id and screen.detail_item_count and screen.detail_focus == "actions":
            keys = f"W/S select  E/Enter full text  Tab page  P prompt  {money}  Esc back  q quit"
        elif screen.detail_activity_id and screen.detail_item_count:
            keys = f"W/S scroll  Tab actions  P prompt  {money}  Esc back  q quit"
        elif screen.detail_activity_id:
            keys = f"W/S scroll  P prompt  {money}  Esc back  q quit"
        elif screen.focus == "overview":
            if node.subagents:
                keys = f"Tab activity  W/S move  E/Enter open subagent  {money}  Esc back  q quit"
            else:
                keys = f"Tab activity  {money}  Esc back  q quit"
        else:
            keys = f"Tab overview  W/S move  E/Enter details  {money}  Esc back  q quit"
    elif nav.period and not nav.stack:
        keys = f"W/S move  E/Enter open thread  {money}  G split  Esc home  q quit"
    else:
        keys = f"Tab section  W/S move  E/Enter open  {money}  q quit"
    foot = Text()
    if nav.notice:
        foot.append(nav.notice, style="bold yellow")
        foot.append("  ·  ")
    foot.append(keys, style="bold")
    return _fit(foot, width)


def _breadcrumb(nav: NavigationState, width: int) -> str:
    """Path above the selected overview, trimmed from the left so the current agent stays visible."""
    names = [s.title or s.node_id[:8] for s in nav.stack]
    if len(names) <= 1 and not nav.period:
        return "Overview"
    crumb = (PERIOD_LABELS[nav.period] + " › " if nav.period and names else "") + " › ".join(names)
    if len(crumb) > width:
        crumb = "… " + crumb[-(width - 2):]
    return crumb


def _agent_total(node: ThreadTelemetryNode, measure: str) -> str:
    """Spend of this agent plus every subagent below it, in the active measure."""
    if measure == "usd":
        cost = node_cost(node)
        return "unpriced" if cost.total <= 0 and cost.unpriced_tokens else format_measure(cost.total, measure)
    return format_measure(node.rollup_metrics.total_tokens, measure)


def _identity_line(node: ThreadTelemetryNode, width: int, measure: str = "tokens") -> Text:
    """Selected agent: title in status color, then textual status, total spend, driver and model."""
    style = get_status_style(node.status)
    tail = Text()
    tail.append("  ")
    tail.append(node.status.upper(), style=style or "bold")
    title = Text(f"{node.title} ({node.thread_id[:8]})", style=style or "bold")
    total = _agent_total(node, measure)
    unit = (" est." if total.startswith("$") else "") if measure == "usd" else " tokens"
    total_text = f"  ·  Total {total}{unit}"
    meta = [node.driver, node.model or "unknown"] if width >= 76 + len(total_text) else [node.model or "unknown"]
    meta_text = "".join(f"  ·  {m}" for m in meta)
    # The total is shown only when the title still fits whole (or keeps 24 cells).
    if width - tail.cell_len - len(meta_text) - len(total_text) >= min(title.cell_len, 24):
        tail.append(total_text, style="bold")
    tail.append(meta_text, style="bold")
    _fit(title, max(12, width - tail.cell_len))
    title.append_text(tail)
    return _fit(title, width)


def _child_table(node: ThreadTelemetryNode, screen: ScreenState, window: list, show_header: bool, focused: bool, width: int, range_label: str, measure: str = "tokens") -> Table:
    # The section rule above carries the name, range and focus; the header names columns only.
    table = Table(box=None, expand=True, padding=(0, 1), show_header=show_header, pad_edge=False, show_edge=False, header_style="dim")
    table.add_column("", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("Model", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("Steps", justify="right", no_wrap=True)
    table.add_column("Cost" if measure == "usd" else "Tokens", justify="right", no_wrap=True)
    table.add_column("Time", justify="right", no_wrap=True)
    for child in window:
        is_sel = child.thread_id == screen.selected_child_id
        highlighted = is_sel and focused
        # On the highlight background the row style supplies bright text.
        title = Text()
        title.append("▶ " if is_sel else "  ", style="bold" if highlighted else "bold cyan")
        title.append(child.title, style="" if highlighted else (get_status_style(child.status) or "bold"))
        m = child.rollup_metrics
        table.add_row(
            title,
            child.model or "unknown",
            str(m.step_count),
            _agent_total(child, measure),
            format_duration(m.duration_seconds),
            style=SELECTED_ROW_STYLE if highlighted else "",
        )
    return table


def _tab_strip(screen: ScreenState, activity_count: int, width: int) -> Text:
    """Two-tab strip for an agent screen: Overview and Activity (N)."""
    strip = Text()
    if screen.focus == "activity":
        strip.append(" Overview", style="dim")
        strip.append("  ")
        strip.append(f"▌ Activity ({activity_count}) ", style="bold cyan")
    else:
        strip.append("▌ Overview", style="bold cyan")
        strip.append("   ")
        strip.append(f"Activity ({activity_count}) ", style="dim")
    return _fit(strip, width)


def _activity_tab_body(
    node: ThreadTelemetryNode,
    nav: NavigationState,
    screen: ScreenState,
    entries: list,
    inner_height: int,
    inner_width: int,
) -> list:
    """Full-height activity tab for an agent screen."""
    identity = _identity_line(node, inner_width, nav.measure)
    tab_strip = _tab_strip(screen, len(entries), inner_width)
    lines = max(0, inner_height - 3)

    entry_ids = [e.entry_id for e in entries]
    act_total = len(entries)
    act_sel = entry_ids.index(screen.selected_activity_id) if screen.selected_activity_id in entry_ids else max(0, act_total - 1)

    if not entries:
        note = "0 of 0"
        table = build_activity_table([], interactive=True, focused=True, range_label=note, measure=nav.measure)
        return [
            identity,
            tab_strip,
            section_rule("Activity", inner_width, note=note, focused=True),
            Viewport(table, height=lines, width=inner_width),
        ]

    header_lines, rows = table_geometry(lines)
    start = window_start(act_total, act_sel, screen.activity_offset, rows)
    screen.activity_offset = start
    window = entries[start : start + rows]
    note = activity_range_label(start, len(window), act_total)
    table = build_activity_table(
        window,
        selected_id=entry_ids[act_sel],
        focused=True,
        interactive=True,
        show_header=header_lines > 0,
        range_label=note,
        measure=nav.measure,
        model=node.model,
        syncing=node.store_state == "syncing",
    )
    return [
        identity,
        tab_strip,
        section_rule("Activity", inner_width, note=note, focused=True),
        Viewport(table, height=lines, width=inner_width),
    ]


def _overview_body(
    node: ThreadTelemetryNode,
    nav: NavigationState,
    screen: ScreenState,
    entries: list,
    inner_height: int,
    inner_width: int,
    width: int,
) -> list:
    """Overview for any agent (root or child): identity, scopes, graph, Subagents, Activity.

    Height budget: identity and scope lines first; then each nonempty list gets
    a heading and one row, and the graph gets its three labeled plot rows.
    Extra lines go to the lists (up to three rows each, in a fixed order),
    then the graph (up to its natural size, at most half of what is left),
    then the lists again until every row is shown.
    Every list row is one line, so the window, selection, and clip agree.
    """
    identity = _identity_line(node, inner_width, nav.measure)
    scope_lines = [_fit(line, inner_width) for line in build_scopes_summary(node, width=inner_width, measure=nav.measure)]

    entry_ids = [e.entry_id for e in entries]
    act_total = len(entries)
    act_sel = entry_ids.index(screen.selected_activity_id) if screen.selected_activity_id in entry_ids else max(0, act_total - 1)
    children = list(node.subagents)
    child_ids = [c.thread_id for c in children]
    child_sel = child_ids.index(screen.selected_child_id) if screen.selected_child_id in child_ids else 0
    sub_focus = bool(children) and screen.focus == "overview"

    # Rows per list (an empty list still shows one message line) and how many it could use.
    counts = ({"subagents": len(children)} if children else {}) | {"activity": act_total}
    rows = {key: 1 for key in counts}
    order = list(counts)

    def _lines(key: str) -> int:
        return rows[key] + table_header_lines(rows[key])

    # Between the minimum (optional lines merged or dropped) and the full
    # three-row graph, the graph would trade plot rows for optional lines, so
    # it only ever grows to `graph_full` or beyond.
    plot_rows = 3 if inner_height <= 20 else (5 if inner_height <= 30 else 7)
    graph_min = context_graph_min_height(node, inner_width, 3)
    graph_full = max(graph_min, len(context_graph_lines(node, inner_width, 3)))
    graph_natural = max(graph_full, len(context_graph_lines(node, inner_width, plot_rows)))

    # One section rule each for Usage, Context, and every list.
    section_rules = 2 + len(counts)
    stats = stats_for_node(node)
    variants = [(True, dict(headline_on=True, legend="compact")), (False, dict(headline_on=True, legend="compact")), (False, dict(headline_on=False, legend="compact")), (False, dict(headline_on=False, legend="none")), (False, None)]
    if node.store_state == "syncing":
        variants = [(False, None)]
    graph_h = graph_min
    for show_provider, variant in variants:
        block = (provider_lines(stats, nav.measure, inner_width) if show_provider else [])
        if variant is not None:
            block += breakdown_lines(stats, nav.measure, inner_width, scope="this chat's", **variant)
        fixed = 2 + len(scope_lines) + section_rules + len(block)
        spare = inner_height - fixed - graph_h - sum(_lines(k) for k in counts)
        if spare >= 0:
            break
    if spare < 0:
        # Below the supported minimum: the graph shrinks before list rows go.
        graph_h = max(1, graph_h + spare)
        spare = inner_height - fixed - graph_h - sum(_lines(k) for k in counts)

    def _grow(cap: int) -> None:
        nonlocal spare
        moved = True
        while moved:
            moved = False
            for key in order:
                if rows[key] >= min(cap, counts[key]):
                    continue
                cost = 1 + table_header_lines(rows[key] + 1) - table_header_lines(rows[key])
                if cost <= spare:
                    rows[key] += 1
                    spare -= cost
                    moved = True

    _grow(3)
    graph_extra = min(graph_natural - graph_h, spare // 2)
    if graph_h + graph_extra >= graph_full:
        graph_h += graph_extra
        spare -= graph_extra
    _grow(10**6)
    if node.context_samples and graph_h + spare >= graph_full:
        # Lists are fully shown; leftover lines become taller bars.
        graph_h += spare
    alloc = {key: _lines(key) for key in counts}

    graph = render_context_graph(node, width=inner_width, as_panel=False, height=graph_h, min_plot_rows=3)

    body: list = [
        identity,
        _tab_strip(screen, act_total, inner_width),
        section_rule("Usage", inner_width),
        *scope_lines,
        *block,
        section_rule("Context", inner_width),
        graph,
    ]

    if children:
        lines = alloc["subagents"]
        header_lines, rows = table_geometry(lines)
        start = window_start(len(children), child_sel, screen.child_scroll, rows)
        screen.child_scroll = start
        window = children[start : start + rows]
        table = _child_table(node, screen, window, header_lines > 0, sub_focus, width, activity_range_label(start, len(window), len(children)), nav.measure)
        body.append(section_rule("Subagents", inner_width, note=activity_range_label(start, len(window), len(children)), focused=sub_focus))
        body.append(Viewport(table, height=lines, width=inner_width))

    lines = alloc["activity"]
    if not entries:
        body.append(section_rule("Activity", inner_width, note="0 of 0", focused=not sub_focus))
        body.append(Viewport(build_activity_table([], interactive=True, focused=not sub_focus, range_label="0 of 0", measure=nav.measure), height=lines, width=inner_width))
    else:
        header_lines, rows = table_geometry(lines)
        start = window_start(act_total, act_sel, screen.activity_offset, rows)
        screen.activity_offset = start
        window = entries[start : start + rows]
        body.append(section_rule("Activity", inner_width, note=activity_range_label(start, len(window), act_total), focused=not sub_focus))
        table = build_activity_table(
            window, selected_id=entry_ids[act_sel], focused=not sub_focus, interactive=True, show_header=header_lines > 0,
            range_label=activity_range_label(start, len(window), act_total),
            measure=nav.measure, model=node.model, syncing=node.store_state == "syncing",
        )
        body.append(Viewport(table, height=lines, width=inner_width))
    return body


def _action_detail_body(node: ThreadTelemetryNode, screen: ScreenState, lines: list, inner_height: int, inner_width: int, unit: str = "tokens") -> list:
    """Scrollable full text of one 'What it did' item."""
    measuring = Console(width=inner_width, force_terminal=True, color_system=None)
    opts = measuring.options.update(width=inner_width, height=None)
    group = Group(*lines)
    total_lines = len(list(Segment.split_lines(measuring.render(group, opts))))
    view_height = max(1, inner_height - 2)
    screen.max_action_scroll = max(0, total_lines - view_height)
    screen.action_scroll = max(0, min(screen.action_scroll, screen.max_action_scroll))
    top = screen.action_scroll + 1
    bottom = min(total_lines, screen.action_scroll + view_height)
    note = f"{screen.detail_item + 1} of {screen.detail_item_count} · lines {top}–{bottom} of {total_lines}"
    return [
        _identity_line(node, inner_width, unit),
        section_rule("Action detail", inner_width, note=note, focused=True),
        Viewport(group, height=view_height, scroll=screen.action_scroll, width=inner_width),
    ]


def _activity_detail_body(node: ThreadTelemetryNode, screen: ScreenState, entry, inner_height: int, inner_width: int, unit: str = "tokens") -> list:
    """Scrollable full detail for one activity; shows everything the one-line row elides."""
    measuring = Console(width=inner_width, force_terminal=True, color_system=None)
    opts = measuring.options.update(width=inner_width, height=None)

    def measure(renderables: list) -> int:
        return len(list(Segment.split_lines(measuring.render(Group(*renderables), opts)))) if renderables else 0

    view_height = max(1, inner_height - 2)
    actions = ActionsView(
        selected=screen.detail_item,
        scroll=screen.detail_item_scroll,
        rows=max(3, view_height // 2),
        focused=screen.detail_focus == "actions",
    )
    detail_lines = build_activity_detail_lines(
        entry, node, width=inner_width, prompt_expanded=screen.detail_prompt_expanded, actions_view=actions
    )
    screen.detail_item_count = actions.count
    screen.detail_item = actions.selected
    screen.detail_item_scroll = actions.scroll
    if not actions.count:
        screen.detail_focus = "actions"
    detail_group = Group(*detail_lines)
    total_lines = measure(detail_lines)
    screen.max_detail_scroll = max(0, total_lines - view_height)
    if actions.focused and actions.selected_line is not None:
        # Keep the highlighted item on screen; the page follows the selection.
        top = measure(detail_lines[: actions.selected_line])
        bottom = top + actions.selected_height
        if bottom > screen.detail_scroll + view_height:
            screen.detail_scroll = bottom - view_height
        if top < screen.detail_scroll:
            screen.detail_scroll = top
    screen.detail_scroll = max(0, min(screen.detail_scroll, screen.max_detail_scroll))
    top = screen.detail_scroll + 1
    bottom = min(total_lines, screen.detail_scroll + view_height)
    head = section_rule(
        "Activity detail", inner_width, note=f"lines {top}–{bottom} of {total_lines}",
        focused=not actions.focused or not actions.count,
    )
    return [
        _identity_line(node, inner_width, unit),
        head,
        Viewport(detail_group, height=view_height, scroll=screen.detail_scroll, width=inner_width),
    ]


def _root_list_body(roots: list[AppThreadRecord], nav: NavigationState, inner_height: int, inner_width: int, snapshot, focused: bool, usage=None, syncing: bool = False) -> Text:
    body = Text()
    visible_rows = max(1, inner_height - 1)
    ids = [t.thread_id for t in roots]
    sel_idx = ids.index(nav.selected_id) if (nav.selected_id and nav.selected_id in ids) else 0
    start = window_start(len(roots), sel_idx, nav.list_scroll, visible_rows)
    nav.list_scroll = start
    shown = len(roots[start : start + visible_rows])
    note = activity_range_label(start, shown, len(roots)) + (" · lifetime est. $" if nav.measure == "usd" else " · lifetime tokens")
    body.append_text(section_rule("Chats", inner_width, note=note, focused=focused))
    body.append("\n")
    if not roots:
        body.append("No app threads found. Waiting for activity.\n", style="italic")
    for root in roots[start : start + visible_rows]:
        # Only the focused section shows its selection; the cursor is kept for when focus returns.
        is_selected = focused and root.thread_id == nav.selected_id
        pt = snapshot.get_provider_thread(root.thread_id) if snapshot else None
        status = pt.status if pt else getattr(root, "status", "idle")
        line = Text()
        line.append("▶ " if is_selected else "  ", style="bold cyan")
        suffix = Text(f"  [{root.model or 'unknown'}]", style="dim")  # optional metadata
        row = usage.all_time.get(root.thread_id) if usage is not None else None
        total = Text("  " + (_usage_label(row, nav.measure) if row is not None else ("syncing…" if syncing or usage is None else "—")))
        title = Text(root.title or root.thread_id, style="" if is_selected else (get_status_style(status) or "bold"))
        room = inner_width - 2 - total.cell_len
        if room - suffix.cell_len < 12:
            suffix = Text()
        _fit(title, max(5, room - suffix.cell_len))
        line.append_text(title)
        line.append_text(suffix)
        line.append(" " * max(0, inner_width - line.cell_len - total.cell_len))
        line.append_text(total)
        if is_selected:
            line.stylize(SELECTED_ROW_STYLE)
        body.append_text(_fit(line, inner_width))
        body.append("\n")
    return body


def _parse_utc(value: str | None) -> datetime | None:
    """T3 timestamps are UTC; treat a missing offset as UTC and bad text as absent."""
    try:
        parsed = datetime.fromisoformat(value) if value else None
    except ValueError:
        return None
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _home_body(roots, summary, usage, nav, running_ids, snapshot, inner_height, inner_width, now, syncing=False):
    card = []
    if summary:
        turns = snapshot.provider_turns_by_thread.get(summary.thread_id, []) if snapshot else []
        running = next((t for t in reversed(turns) if t.status == "running" and t.started_at), None)
        started = _parse_utc(running.started_at) if running else None

        status = summary.status or "idle"
        status_text = status.upper()
        if started:
            status_text += " " + format_duration(max(0.0, (now - started).total_seconds()))
        line = Text(f"{status_text}  {summary.title or summary.thread_id[:8]}", style=get_status_style(status) or "bold")
        line.append(f"  ·  {summary.model or 'unknown'}")
        if inner_width >= 90:
            line.append("  ·  " + (summary.driver or "unknown"))
        card.append(line)

        context_line = Text("Context  ")
        context_line.append_text(Text.from_markup(render_context_bar(summary.context_used_tokens, summary.context_limit, 10)))
        card.append(context_line)

        if getattr(summary, "store_state", "ready") == "syncing":
            card.append(Text("Spent    syncing…"))
        else:
            c = node_cost(summary)
            cost = "unpriced" if (c.total <= 0 and c.unpriced_tokens) else (format_usd(c.total) + " est.")
            rollup = getattr(summary, "rollup_metrics", None)
            total_tokens = rollup.total_tokens if rollup else 0
            spent = f"Spent    {format_tokens(total_tokens)} tokens  ·  {cost}"
            n = rollup.total_subagents if rollup else 0
            if n > 0:
                spent += f"  ·  incl. {n} subagent{'s' if n != 1 else ''}"
            card.append(Text(spent))

        if summary.steps:
            card.append(Text("Act: " + (summary.steps[-1].action_summary or ""), style="magenta"))
        children = [c for c in summary.subagents if c.status == "running"]
        if children:
            card.append(Text(f"Subagents: {len(children)} running — " + ", ".join(c.title or c.thread_id for c in children)))
        others = len([i for i in (running_ids or []) if i != summary.thread_id])
        if others:
            card.append(Text(f"+{others} other thread{'s' if others != 1 else ''} running", style="dim"))
    else:
        card.append(Text("No active app threads"))
    card = card[:max(1, inner_height - 8)]
    result = [section_rule("Live", inner_width), *(_fit(line, inner_width) for line in card)]
    result.append(section_rule("Usage", inner_width, note="est. $" if nav.measure == "usd" else "tokens", focused=nav.home_focus == "usage"))
    for key in ("today", "week", "month"):
        selected = key == nav.selected_period
        total = format_measure(usage.periods[key].total_value(nav.measure), nav.measure) if usage else "…"
        line = Text(("▶ " if selected else "  ") + PERIOD_LABELS[key].ljust(11))
        if usage:
            window = usage.periods[key].window
            last_day = window.end - timedelta(days=1)
            date_range = f"{window.start:%b %d}" if key == "today" else f"{window.start:%b %d}–{last_day:%b %d}"
            if line.cell_len + len(date_range) + len(total) + 2 <= inner_width:
                line.append(date_range, style="dim")
        line.append(" " * max(1, inner_width - line.cell_len - len(total)) + total)
        if selected and nav.home_focus == "usage":
            line.stylize(SELECTED_ROW_STYLE)
        result.append(_fit(line, inner_width))
    result.append(_root_list_body(roots, nav, max(1, inner_height - len(result)), inner_width, snapshot, nav.home_focus == "threads", usage, syncing))
    return result


SCOPE_PHRASE = {"today": "today's", "week": "this week's", "month": "this month's"}


def _period_body(usage, nav, inner_height, inner_width, now) -> list:
    if usage is None:
        return [Text("Loading usage…", style="italic")]
    p = usage.periods[nav.period]
    start = p.window.start
    end = p.window.end - timedelta(days=1)
    date = lambda d: f"{d:%a} {d.day} {d:%b}"
    label = (date(start) if nav.period == "today" else f"{start:%B %Y}" if nav.period == "month" else f"{date(start)} – {date(end)}")
    header = Text(f"{label}  ·  Total ")
    header.append(format_measure(p.total_value(nav.measure), nav.measure), style="bold")
    if nav.measure == "usd":
        header.append(" est.", style="dim")
    block = []
    variants = (
        dict(headline_on=True, legend="detailed", kpis_on=True),
        dict(headline_on=True, legend="compact", kpis_on=True),
        dict(headline_on=True, legend="compact", kpis_on=False),
        dict(headline_on=False, legend="compact", kpis_on=False),
        dict(headline_on=False, legend="none", kpis_on=False),
    )
    rest_min = 12
    for variant in variants:
        lines = breakdown_lines(p.stats, nav.measure, inner_width, scope=SCOPE_PHRASE[nav.period], **variant)
        if 1 + len(lines) + rest_min <= inner_height:
            block = [section_rule("Breakdown", inner_width, note="tokens" if nav.measure == "tokens" else "est. $"), *lines]
            break
    inner_height_rest = inner_height - len(block)
    plot_rows = 4 if inner_height_rest <= 24 else (6 if inner_height_rest <= 34 else 8)
    providers = list(p.by_provider.items())
    graph = usage_graph_lines(p, inner_width, plot_rows, now, split=nav.graph_split, measure=nav.measure)
    if 1 + 1 + len(graph) + 1 + 1 + max(1, len(providers)) + 2 > inner_height_rest:
        plot_rows = 3
        graph = usage_graph_lines(p, inner_width, plot_rows, now, split=nav.graph_split, measure=nav.measure)
    show_header = 1 + 1 + len(graph) + 1 + 1 + max(1, len(providers)) + 2 <= inner_height_rest
    provider_rows = max(1, min(max(1, len(providers)), inner_height_rest - (1 + 1 + len(graph) + 1 + int(show_header) + 2)))
    table = Table(box=None, expand=True, padding=(0, 1), show_header=show_header, pad_edge=False, show_edge=False, header_style="dim")
    # Short headers below 70 columns so token values are never cut off.
    columns = ("Provider", "Input", "Cache read", "Cache write", "Output", "Total") if inner_width >= 70 else ("Provider", "In", "Cache R", "Cache W", "Out", "Total")
    for col in columns:
        table.add_column(col, justify="left" if col == "Provider" else "right", no_wrap=True, overflow="ellipsis")
    for name, tokens in providers[:provider_rows]:
        if nav.measure == "usd":
            c = p.cost_by_provider.get(name)
            values = (c.input, c.cache_read, c.cache_write, c.output, c.total) if c else (0.0,) * 5
        else:
            values = (tokens.input_tokens, tokens.cache_read_tokens, tokens.cache_creation_tokens, tokens.output_tokens, tokens.total)
        table.add_row(name, *(format_measure(v, nav.measure) for v in values))
    if not providers:
        table.add_row("No usage recorded", "", "", "", "", "")
    threads = p.top(nav.measure)
    rows = max(1, inner_height_rest - (1 + 1 + len(graph) + 1 + int(show_header) + provider_rows + 1))
    ids = [t.thread_id for t in threads]
    selected = ids.index(nav.selected_top_id) if nav.selected_top_id in ids else 0
    start_idx = window_start(len(threads), selected, nav.top_scroll, rows)
    nav.top_scroll = start_idx
    window = threads[start_idx:start_idx + rows]
    result = [header, *block, section_rule("Usage over time", inner_width), *graph, section_rule("Providers", inner_width), table, section_rule("Top threads", inner_width, note=activity_range_label(start_idx, len(window), len(threads)), focused=True)]
    # Fixed provider and token columns so titles all end at the same place.
    provider_labels = ["+".join(t.providers) for t in window]
    token_labels = [_usage_label(t, nav.measure) for t in window]
    token_w = max((len(x) for x in token_labels), default=0)
    provider_w = min(max((len(x) for x in provider_labels), default=0), max(0, (inner_width - 2) // 3))
    title_w = max(1, inner_width - 2 - (2 + provider_w) - (2 + token_w))
    for thread, providers, tokens in zip(window, provider_labels, token_labels):
        line = Text("▶ " if thread.thread_id == nav.selected_top_id else "  ", style="bold cyan")
        title = _fit(Text(thread.title, style=get_status_style(thread.status)), title_w)
        line.append_text(title)
        line.append(" " * (title_w - title.cell_len))
        line.append_text(_fit(Text("  " + providers, style="dim"), 2 + provider_w) if provider_w else Text())
        line.append(" " * max(0, inner_width - token_w - line.cell_len))
        line.append(tokens.rjust(token_w))
        result.append(_fit(line, inner_width))
    if not threads:
        result.append(Text("No usage recorded", style="italic"))
    return result


def _browser_frame(
    roots: list[AppThreadRecord],
    node: ThreadTelemetryNode | None,
    summary: ThreadTelemetryNode | None,
    nav: NavigationState,
    height: int,
    width: int,
    max_steps: int = 20,
    snapshot=None,
    *, usage: UsageReport | None = None, now: datetime | None = None, running_ids: list[str] | None = None, syncing: bool = False,
) -> Layout:
    """Construct Rich Layout bounded strictly to (height, width).

    One strip line for the followed root, a bordered body, one footer line.
    """
    inner_height = max(1, height - 4)
    inner_width = max(10, width - 4)

    if nav.detail_id and node:
        screen = nav.current_screen or ScreenState(node_id=node.thread_id)
        entries = build_activity_entries(node, max_steps)
        if nav.current_screen is screen:
            nav.sync_activities([e.entry_id for e in entries], launch_targets(entries))
        detail_entry = next((e for e in entries if e.entry_id == screen.detail_activity_id), None) if screen.detail_activity_id else None
        action_lines = None
        if detail_entry is not None and screen.action_open:
            action_lines = build_action_detail_lines(detail_entry.detail, screen.detail_item, inner_width)
            if action_lines is None:
                screen.close_action()
                nav.notice = "Action is no longer available"
        if action_lines is not None:
            body = _action_detail_body(node, screen, action_lines, inner_height, inner_width, nav.measure)
        elif detail_entry is not None:
            body = _activity_detail_body(node, screen, detail_entry, inner_height, inner_width, nav.measure)
        elif screen.focus == "activity":
            body = _activity_tab_body(node, nav, screen, entries, inner_height, inner_width)
        else:
            body = _overview_body(node, nav, screen, entries, inner_height, inner_width, width)
        title = Text(_breadcrumb(nav, max(10, width - 8)), style="bold cyan")
        body_panel = Panel(Group(*body), title=title, box=box.ROUNDED, expand=True)
    elif nav.period:
        body_panel = Panel(Group(*_period_body(usage, nav, inner_height, inner_width, now or datetime.now().astimezone())), title=Text(PERIOD_LABELS[nav.period], style="bold cyan"), box=box.ROUNDED, expand=True)
    else:
        body_panel = Panel(
            Group(*_home_body(roots, summary, usage, nav, running_ids, snapshot, inner_height, inner_width, now or datetime.now().astimezone(), syncing)),
            title="[bold]App Threads[/]",
            box=box.ROUNDED,
            expand=True,
        )

    header = _root_strip(summary, width) if nav.stack or nav.period else _fit(Text("t3-top · Home · " + (now or datetime.now().astimezone()).strftime("%a %-d %b"), style="bold"), width)
    return _assemble_frame(header, body_panel, _footer_line(nav, node, width))


def _assemble_frame(header, body_panel: Panel, footer) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(header, size=1),
        Layout(body_panel),
        Layout(footer, size=1),
    )
    return layout


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    t3_dir = Path(args.t3_dir).expanduser()
    console = Console()

    if args.sessions or args.sync or args.list or args.json or args.once:
        store, snapshot = _synced_store(args, t3_dir)
        try:
            if args.sessions:
                rows = session_rows(store)
                for row in rows:
                    row["status"] = (snapshot_thread_status(snapshot, row["thread_id"])
                                     if row["thread_id"] in snapshot.threads else None)
                if args.json:
                    print(json.dumps(rows, indent=2))
                else:
                    console.print(build_sessions_table(rows))
                return 0
            if args.sync:
                return 0
            if args.list:
                nodes = list_root_nodes(snapshot, store, limit=50)
                if args.json:
                    print(json.dumps([n.to_dict() for n in nodes], indent=2))
                else:
                    console.print(build_threads_list_table(nodes))
                return 0
            node = resolve_thread_node(snapshot, store, args.thread_query)
            if node is None:
                if args.json:
                    print(json.dumps({"error": f"No thread found matching '{args.thread_query}'"}, indent=2), file=sys.stderr)
                else:
                    query_desc = f"'{args.thread_query}'" if args.thread_query else "any active thread"
                    print(f"Error: No thread found matching {query_desc} in {t3_dir}", file=sys.stderr)
                return 1
            if args.json:
                print(json.dumps(node.to_dict(), indent=2))
            else:
                console.print(build_dashboard_renderable(node, max_steps=args.max_steps))
            return 0
        finally:
            store.close()

    # Mode 4: Live TUI mode
    try:
        return run_live_loop(args, console=console)
    except KeyboardInterrupt:
        return 0


def _entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    _entrypoint()
