"""CLI entrypoint for T3 Telemetry (t3-top)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3
import sys
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
    build_thread_telemetry,
    list_root_threads,
    read_t3_state,
    resolve_thread,
    resolve_thread_id_in_snapshot,
)
from datetime import datetime, timezone

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
    format_duration,
    format_tokens,
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
    return parser


def run_live_loop(
    args: argparse.Namespace,
    console: Console | None = None,
    _max_iterations: int | None = None,
    _adapter: TerminalAdapter | None = None,
    _clock: Callable[[], float] = time.monotonic,
    _live_cls: type[Live] | None = None,
    _history_store: ContextHistoryStore | None = None,
) -> int:
    """Run live watch TUI loop with graceful refresh and error handling."""
    console = console or Console()

    if not math.isfinite(args.interval) or args.interval <= 0:
        raise argparse.ArgumentTypeError("--interval must be a finite positive number")

    # Noninteractive fallback
    if _adapter is None and _max_iterations is None:
        if not console.is_terminal or not sys.stdin.isatty():
            node = resolve_thread(args.t3_dir, args.thread_query)
            if node is not None:
                console.print(build_dashboard_renderable(node, max_steps=args.max_steps))
            return 0

    t3_dir = Path(args.t3_dir).expanduser()
    adapter = _adapter or PosixTerminalAdapter()
    history_store = _history_store if _history_store is not None else ContextHistoryStore()
    browser = TelemetryBrowser(adapter, history_store=history_store)
    live_factory = _live_cls or Live

    cached_snapshot = None
    cached_roots: list[AppThreadRecord] = []
    tree_node_cache: dict[str, ThreadTelemetryNode] = {}
    cached_node: ThreadTelemetryNode | None = None
    cached_summary: ThreadTelemetryNode | None = None
    cached_frame = None

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
            cur_screen.child_scroll if cur_screen else None,
            nav.list_scroll,
            nav.notice,
        )
        return (
            roots_sig,
            nav_sig,
            summary_sig,
            node_sig,
            size,
        )

    adapter.begin()
    try:
        with live_factory(auto_refresh=False, console=console, screen=True, transient=True) as live:
            while True:
                if _max_iterations is not None and polls >= _max_iterations:
                    break

                now = _clock()
                if now >= deadline:
                    polls += 1
                    try:
                        snapshot = read_t3_state(t3_dir)
                        cached_snapshot = snapshot
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
                                n = build_thread_telemetry(tid, t3_dir, snapshot=snapshot)
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

                    active_id = browser.navigation.detail_id or browser.navigation.selected_id
                    if active_id and active_id not in tree_node_cache and cached_snapshot and active_id in cached_snapshot.threads:
                        n = build_thread_telemetry(active_id, t3_dir, snapshot=cached_snapshot)
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
        adapter.end()
    return 0


def _context_text(node: ThreadTelemetryNode) -> str:
    if node.context_limit is not None and node.context_limit > 0:
        return f"Context: {format_tokens(node.context_used_tokens)}/{format_tokens(node.context_limit)} ({node.context_pct:.0f}%)"
    if getattr(node, "context_limit_provenance", "") == "measured":
        return f"Context: {format_tokens(node.context_used_tokens)}/{format_tokens(node.context_max_tokens)} ({node.context_pct:.0f}%)"
    return f"Context: {format_tokens(node.context_used_tokens)} / ? (limit unknown)"


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
    if nav.detail_id and node and screen is not None:
        if screen.detail_activity_id in screen.activity_targets:
            keys = "E/Enter open subagent  W/S scroll  Esc back  q quit"
        elif screen.detail_activity_id:
            keys = "W/S scroll  P prompt  Esc back  q quit"
        elif node.subagents and screen.focus == "subagents":
            keys = "Tab activity  W/S move  E/Enter open  Esc back  q quit"
        elif node.subagents:
            keys = "Tab subagents  W/S move  E/Enter details  Esc back  q quit"
        else:
            keys = "W/S move  E/Enter details  Esc back  q quit"
    else:
        keys = "W/S move  E/Enter open  Esc back  q quit"
    foot = Text()
    if nav.notice:
        foot.append(nav.notice, style="bold yellow")
        foot.append("  ·  ")
    foot.append(keys, style="bold")
    return _fit(foot, width)


def _breadcrumb(nav: NavigationState, width: int) -> str:
    """Path above the selected overview, trimmed from the left so the current agent stays visible."""
    names = [s.title or s.node_id[:8] for s in nav.stack]
    if len(names) <= 1:
        return "Overview"
    crumb = " › ".join(names)
    if len(crumb) > width:
        crumb = "… " + crumb[-(width - 2):]
    return crumb


def _identity_line(node: ThreadTelemetryNode, width: int) -> Text:
    """Selected agent: title in status color, then textual status, driver and model."""
    style = get_status_style(node.status)
    tail = Text()
    tail.append("  ")
    tail.append(node.status.upper(), style=style or "bold")
    meta = [node.driver, node.model or "unknown"] if width >= 76 else [node.model or "unknown"]
    tail.append("".join(f"  ·  {m}" for m in meta), style="bold")
    title = Text(f"{node.title} ({node.thread_id[:8]})", style=style or "bold")
    _fit(title, max(12, width - tail.cell_len))
    title.append_text(tail)
    return _fit(title, width)


def _child_table(node: ThreadTelemetryNode, screen: ScreenState, window: list, show_header: bool, focused: bool, width: int, range_label: str) -> Table:
    table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), show_header=show_header, pad_edge=False, show_edge=False, header_style=SELECTED_ROW_STYLE if focused else "bold")
    table.add_column(f"{'▌' if focused else ' '}Subagents {range_label}", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("Model", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("Steps", justify="right", no_wrap=True)
    table.add_column("Tokens", justify="right", no_wrap=True)
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
            format_tokens(m.total_tokens),
            format_duration(m.duration_seconds),
            style=SELECTED_ROW_STYLE if highlighted else "",
        )
    return table


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
    identity = _identity_line(node, inner_width)
    scope_lines = [_fit(line, inner_width) for line in build_scopes_summary(node, width=inner_width)]

    entry_ids = [e.entry_id for e in entries]
    act_total = len(entries)
    act_sel = entry_ids.index(screen.selected_activity_id) if screen.selected_activity_id in entry_ids else max(0, act_total - 1)
    children = list(node.subagents)
    child_ids = [c.thread_id for c in children]
    child_sel = child_ids.index(screen.selected_child_id) if screen.selected_child_id in child_ids else 0
    sub_focus = bool(children) and screen.focus == "subagents"

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

    section_gaps = 1 + len(counts)
    fixed = 1 + len(scope_lines) + section_gaps
    graph_h = graph_min
    spare = inner_height - fixed - graph_h - sum(_lines(k) for k in counts)
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

    body: list = [identity, *scope_lines, Text(""), graph]

    if children:
        lines = alloc["subagents"]
        header_lines, rows = table_geometry(lines)
        start = window_start(len(children), child_sel, screen.child_scroll, rows)
        screen.child_scroll = start
        window = children[start : start + rows]
        table = _child_table(node, screen, window, header_lines > 0, sub_focus, width, activity_range_label(start, len(window), len(children)))
        body.append(Text(""))
        body.append(Viewport(table, height=lines, width=inner_width))

    body.append(Text(""))
    lines = alloc["activity"]
    if not entries:
        body.append(Viewport(build_activity_table([], interactive=True, focused=not sub_focus, range_label="0 of 0"), height=lines, width=inner_width))
    else:
        header_lines, rows = table_geometry(lines)
        start = window_start(act_total, act_sel, screen.activity_offset, rows)
        screen.activity_offset = start
        window = entries[start : start + rows]
        table = build_activity_table(
            window, selected_id=entry_ids[act_sel], focused=not sub_focus, interactive=True, show_header=header_lines > 0,
            range_label=activity_range_label(start, len(window), act_total)
        )
        body.append(Viewport(table, height=lines, width=inner_width))
    return body


def _activity_detail_body(node: ThreadTelemetryNode, screen: ScreenState, entry, inner_height: int, inner_width: int) -> list:
    """Scrollable full detail for one activity; shows everything the one-line row elides."""
    measuring = Console(width=inner_width, force_terminal=True, color_system=None)
    opts = measuring.options.update(width=inner_width, height=None)
    detail_group = Group(*build_activity_detail_lines(entry, node, width=inner_width, prompt_expanded=screen.detail_prompt_expanded))
    total_lines = len(list(Segment.split_lines(measuring.render(detail_group, opts))))
    view_height = max(1, inner_height - 2)
    screen.max_detail_scroll = max(0, total_lines - view_height)
    screen.detail_scroll = max(0, min(screen.detail_scroll, screen.max_detail_scroll))
    head = Text()
    head.append("▌Activity detail", style="bold cyan")
    top = screen.detail_scroll + 1
    bottom = min(total_lines, screen.detail_scroll + view_height)
    head.append(f"  lines {top}–{bottom} of {total_lines}", style="bold")
    return [
        _identity_line(node, inner_width),
        _fit(head, inner_width),
        Viewport(detail_group, height=view_height, scroll=screen.detail_scroll, width=inner_width),
    ]


def _root_list_body(roots: list[AppThreadRecord], nav: NavigationState, inner_height: int, inner_width: int, snapshot) -> Text:
    body = Text()
    if not roots:
        body.append("No app threads found. Waiting for activity.\n", style="italic")
        return body
    visible_rows = inner_height
    ids = [t.thread_id for t in roots]
    sel_idx = ids.index(nav.selected_id) if (nav.selected_id and nav.selected_id in ids) else 0
    start = window_start(len(roots), sel_idx, nav.list_scroll, visible_rows)
    nav.list_scroll = start
    for root in roots[start : start + visible_rows]:
        is_selected = root.thread_id == nav.selected_id
        pt = snapshot.get_provider_thread(root.thread_id) if snapshot else None
        status = pt.status if pt else getattr(root, "status", "idle")
        line = Text()
        line.append("▶ " if is_selected else "  ", style="bold cyan")
        suffix = Text(f"  [{root.model or 'unknown'}]", style="dim")  # optional metadata
        title = Text(root.title or root.thread_id, style="" if is_selected else (get_status_style(status) or "bold"))
        _fit(title, max(5, inner_width - 2 - suffix.cell_len))
        line.append_text(title)
        line.append_text(suffix)
        if is_selected:
            line.stylize(SELECTED_ROW_STYLE)
        body.append_text(_fit(line, inner_width))
        body.append("\n")
    return body


def _browser_frame(
    roots: list[AppThreadRecord],
    node: ThreadTelemetryNode | None,
    summary: ThreadTelemetryNode | None,
    nav: NavigationState,
    height: int,
    width: int,
    max_steps: int = 20,
    snapshot=None,
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
        if detail_entry is not None:
            body = _activity_detail_body(node, screen, detail_entry, inner_height, inner_width)
        else:
            body = _overview_body(node, nav, screen, entries, inner_height, inner_width, width)
        title = Text(_breadcrumb(nav, max(10, width - 8)), style="bold cyan")
        body_panel = Panel(Group(*body), title=title, box=box.ROUNDED, expand=True)
    else:
        body_panel = Panel(
            _root_list_body(roots, nav, inner_height, inner_width, snapshot),
            title="[bold]App Threads[/]",
            box=box.ROUNDED,
            expand=True,
        )

    return _assemble_frame(_root_strip(summary, width), body_panel, _footer_line(nav, node, width))


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

    # Mode 1: List threads
    if args.list:
        nodes = list_root_threads(t3_dir, limit=50)
        if args.json:
            print(json.dumps([n.to_dict() for n in nodes], indent=2))
        else:
            table = build_threads_list_table(nodes)
            console.print(table)
        return 0

    # Mode 2: JSON dump
    if args.json:
        node = resolve_thread(t3_dir, args.thread_query)
        if node is None:
            print(
                json.dumps({"error": f"No thread found matching '{args.thread_query}'"}, indent=2),
                file=sys.stderr,
            )
            return 1
        print(json.dumps(node.to_dict(), indent=2))
        return 0

    # Mode 3: Print once
    if args.once:
        node = resolve_thread(t3_dir, args.thread_query)
        if node is None:
            query_desc = f"'{args.thread_query}'" if args.thread_query else "any active thread"
            print(f"Error: No thread found matching {query_desc} in {t3_dir}", file=sys.stderr)
            return 1
        renderable = build_dashboard_renderable(node, max_steps=args.max_steps)
        console.print(renderable)
        return 0

    # Mode 4: Live TUI mode
    try:
        return run_live_loop(args, console=console)
    except KeyboardInterrupt:
        return 0


def _entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    _entrypoint()
