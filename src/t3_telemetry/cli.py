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
from t3_telemetry.tui import NavigationState, ScreenState, TelemetryBrowser, TerminalAdapter, Viewport
from t3_telemetry.ui import (
    build_dashboard_renderable,
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
            cur_screen.activity_scroll if cur_screen else None,
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
                                cur_screen.child_ids = [c.thread_id for c in node.subagents]
                                if not cur_screen.selected_child_id and cur_screen.child_ids:
                                    cur_screen.selected_child_id = cur_screen.child_ids[0]
                                elif cur_screen.selected_child_id not in cur_screen.child_ids and cur_screen.child_ids:
                                    cur_screen.selected_child_id = cur_screen.child_ids[0]
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
                            cur_screen.child_ids = [c.thread_id for c in cached_node.subagents]
                            if not cur_screen.selected_child_id and cur_screen.child_ids:
                                cur_screen.selected_child_id = cur_screen.child_ids[0]
                            elif cur_screen.selected_child_id not in cur_screen.child_ids and cur_screen.child_ids:
                                cur_screen.selected_child_id = cur_screen.child_ids[0]

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
    """Construct Rich Layout bounded strictly to (height, width)."""
    head = Text()
    if summary:
        status = summary.status or "idle"
        title_style = get_status_style(status) or "dim"
        status_text = f"[{title_style}]{status.upper()}[/]"
        latest_act = summary.steps[-1].action_summary if summary.steps else "idle"
        if summary.context_limit is not None and summary.context_limit > 0:
            ctx_str = f"Context: {format_tokens(summary.context_used_tokens)}/{format_tokens(summary.context_limit)} ({summary.context_pct:.0f}%)"
        elif getattr(summary, "context_limit_provenance", "") == "measured":
            ctx_str = f"Context: {format_tokens(summary.context_used_tokens)}/{format_tokens(summary.context_max_tokens)} ({summary.context_pct:.0f}%)"
        else:
            ctx_str = f"Context: {format_tokens(summary.context_used_tokens)} / ? (limit unknown)"

        head.append(summary.title or summary.thread_id[:8], style=title_style)
        head.append("  ·  ")
        head.append(summary.model or "unknown", style="bold yellow")
        head.append("  ·  ")
        head.append(Text.from_markup(status_text))
        head.append("  ·  ")
        head.append(ctx_str, style="cyan")
        head.append("  ·  ")
        head.append(f"Act: {latest_act}", style="magenta")
    else:
        head.append("T3 Telemetry  ·  No active app threads", style="dim")

    header_panel = Panel(
        head,
        title="[bold]T3 Code Telemetry[/]",
        expand=True,
        box=box.ROUNDED,
    )

    legend = "[green]●[/] running  [dim]●[/] idle/completed  [red]●[/] failed  [yellow]●[/] cancelled"
    if nav.detail_id and node:
        screen = nav.current_screen
        if node.subagents:
            if screen and screen.focus == "subagents":
                hint = "Tab activity · ↑/↓ select child · Enter open · Esc back · q quit"
            else:
                hint = "Tab subagents · ↑/↓ scroll · Enter open child · Esc back · q quit"
        else:
            hint = "↑/↓ scroll · Esc back · q quit"
    else:
        hint = "↑/↓ or j/k move · Enter open · Esc back · q quit"

    if nav.notice:
        foot_text = Text.from_markup(f"[yellow]{nav.notice}[/]  ·  {hint}")
    else:
        foot_text = Text.from_markup(f"{hint}  {legend}")

    footer_panel = Panel(foot_text, box=box.ROUNDED)

    inner_height = max(1, height - 8)
    inner_width = max(10, width - 4)

    if nav.detail_id and node:
        screen = nav.current_screen or ScreenState(node_id=node.thread_id)
        node_status_color = get_status_style(node.status)
        node_status_text = f"[{node_status_color or 'dim'}]{node.status.upper()}[/]"
        breadcrumb = " › ".join(s.title or s.node_id[:8] for s in nav.stack)
        panel_title = f"[bold {node_status_color or 'white'}]{node.title}[/] [dim]({node.thread_id[:8]})[/]"
        if len(nav.stack) > 1:
            panel_title += f"  ·  [cyan]{breadcrumb}[/]"

        body_elements = []

        # Info line (1 line)
        info_line = Text()
        info_line.append(f"Driver: {node.driver}  ·  Model: {node.model}  ·  Status: ")
        info_line.append(Text.from_markup(node_status_text))
        if node.context_limit is not None and node.context_limit > 0:
            ctx_str = f"{format_tokens(node.context_used_tokens)}/{format_tokens(node.context_limit)} ({node.context_pct:.0f}%)"
        elif getattr(node, "context_limit_provenance", "") == "measured":
            ctx_str = f"{format_tokens(node.context_used_tokens)}/{format_tokens(node.context_max_tokens)} ({node.context_pct:.0f}%)"
        else:
            ctx_str = f"{format_tokens(node.context_used_tokens)} / ? (limit unknown)"
        info_line.append(f"  ·  Context: {ctx_str}")
        body_elements.append(info_line)

        measuring_console = Console(width=inner_width, force_terminal=True, color_system=None)
        opts = measuring_console.options.update(width=inner_width, height=None)

        def _count_lines(renderable: object) -> int:
            segs = measuring_console.render(renderable, opts)
            return len(list(Segment.split_lines(segs)))

        # Summary scopes (3 scopes: Turn, Whole chat, Chat + subagents)
        scope_lines = build_scopes_summary(node, width=inner_width)

        # Context history chart
        is_compact = (inner_height <= 20)
        chart_renderable = render_context_graph(
            node,
            width=inner_width,
            compact=is_compact,
            as_panel=False,
        )

        top_elements = [info_line, *scope_lines, chart_renderable]
        top_used = sum(_count_lines(el) for el in top_elements)

        min_interactive = 4
        if top_used + min_interactive > inner_height:
            top_budget = max(2, inner_height - min_interactive)
            body_elements = [Viewport(Group(*top_elements), height=top_budget, scroll=0, width=inner_width)]
            top_used = top_budget
        else:
            body_elements = list(top_elements)

        avail_for_interactive = inner_height - top_used

        visible_steps = node.steps[-max_steps:] if (max_steps and len(node.steps) > max_steps) else node.steps
        if not visible_steps:
            act_renderable = Text("No recorded activity for this agent.", style="dim")
        else:
            act_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), show_header=True, pad_edge=False, show_edge=False)
            act_table.add_column("Activity", style="dim", no_wrap=True)
            act_table.add_column("Action / Tool", ratio=2, no_wrap=True, overflow="ellipsis")
            if width >= 80:
                act_table.add_column("Source", style="cyan", width=8)
                act_table.add_column("In", justify="right", no_wrap=True)
                act_table.add_column("Cache", justify="right", no_wrap=True)
                act_table.add_column("Think", justify="right", no_wrap=True)
                act_table.add_column("Out", justify="right", no_wrap=True)
            else:
                act_table.add_column("Tokens", justify="right", no_wrap=True)
            act_table.add_column("Latency", justify="right", no_wrap=True)

            for step in visible_steps:
                if width >= 80:
                    act_table.add_row(
                        step.row_label,
                        step.action_summary,
                        step.source,
                        format_tokens(step.input_tokens),
                        format_tokens(step.cache_read_tokens),
                        format_tokens(step.thinking_tokens),
                        format_tokens(step.output_tokens),
                        format_duration(step.duration_seconds),
                    )
                else:
                    act_table.add_row(
                        step.row_label,
                        step.action_summary,
                        format_tokens(step.total_tokens),
                        format_duration(step.duration_seconds),
                    )
            act_renderable = act_table

        if node.subagents:
            is_sub_focus = (screen.focus == "subagents")
            visible_sub_rows = max(1, min(len(node.subagents), 1 if inner_height <= 18 else 5))
            child_ids = [c.thread_id for c in node.subagents]
            sel_idx = child_ids.index(screen.selected_child_id) if (screen.selected_child_id and screen.selected_child_id in child_ids) else 0

            if sel_idx < screen.child_scroll:
                screen.child_scroll = sel_idx
            elif sel_idx >= screen.child_scroll + visible_sub_rows:
                screen.child_scroll = sel_idx - visible_sub_rows + 1
            max_child_start = max(0, len(node.subagents) - visible_sub_rows)
            screen.child_scroll = max(0, min(screen.child_scroll, max_child_start))

            sub_table = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1), show_header=True, pad_edge=False, show_edge=False)
            sub_table.add_column(" ", width=2, no_wrap=True)
            sub_table.add_column("Subagent / Role", ratio=2, no_wrap=True, overflow="ellipsis")
            sub_table.add_column("Model", ratio=1, no_wrap=True, overflow="ellipsis")
            sub_table.add_column("Steps", justify="right", no_wrap=True)
            if width >= 80:
                sub_table.add_column("In", justify="right", no_wrap=True)
                sub_table.add_column("Out", justify="right", no_wrap=True)
            else:
                sub_table.add_column("Tokens", justify="right", no_wrap=True)
            sub_table.add_column("Time", justify="right", no_wrap=True)

            for child in node.subagents[screen.child_scroll : screen.child_scroll + visible_sub_rows]:
                is_sel = (child.thread_id == screen.selected_child_id)
                mark = "▶ " if is_sel else "  "
                child_color = get_status_style(child.status)
                title_style = f"bold {child_color}" if (is_sel and child_color) else (child_color or "")
                title_txt = Text(child.title, style=title_style)

                if width >= 80:
                    sub_table.add_row(
                        mark,
                        title_txt,
                        child.model or "unknown",
                        str(child.rollup_metrics.step_count),
                        format_tokens(child.rollup_metrics.input_tokens),
                        format_tokens(child.rollup_metrics.output_tokens),
                        format_duration(child.rollup_metrics.duration_seconds),
                    )
                else:
                    sub_table.add_row(
                        mark,
                        title_txt,
                        child.model or "unknown",
                        str(child.rollup_metrics.step_count),
                        format_tokens(child.rollup_metrics.total_tokens),
                        format_duration(child.rollup_metrics.duration_seconds),
                    )

            if avail_for_interactive >= 8:
                sub_head = Text()
                if is_sub_focus:
                    sub_head.append("▶ Immediate Subagents", style="bold cyan")
                    sub_head.append(f" ({len(node.subagents)})", style="bold")
                    sub_head.append("  [dim]• Tab for Activity[/]")
                else:
                    sub_head.append(f"  Immediate Subagents ({len(node.subagents)})  [dim]• Tab to focus[/]")
                body_elements.append(sub_head)
                sub_budget = min(_count_lines(sub_table), 5)
                body_elements.append(Viewport(sub_table, height=sub_budget, scroll=0, width=inner_width))

                act_head = Text()
                if not is_sub_focus:
                    act_head.append("▶ Recent Activity", style="bold cyan")
                    act_head.append(f" ({len(node.steps)} steps)", style="bold")
                    act_head.append("  [dim]• Tab for Subagents[/]")
                else:
                    act_head.append(f"  Recent Activity ({len(node.steps)} steps)  [dim]• Tab to focus[/]")
                body_elements.append(act_head)
                rem = max(1, inner_height - top_used - 1 - sub_budget - 1)
                body_elements.append(
                    Viewport(
                        act_renderable,
                        height=rem,
                        scroll=screen.activity_scroll,
                        width=inner_width,
                        on_max_scroll=lambda m: setattr(screen, "max_activity_scroll", m),
                    )
                )
            elif is_sub_focus:
                sub_head = Text()
                sub_head.append("▶ Immediate Subagents", style="bold cyan")
                sub_head.append(f" ({len(node.subagents)})", style="bold")
                sub_head.append("  [dim]• Tab for Activity[/]")
                body_elements.append(sub_head)

                sub_budget = max(1, avail_for_interactive - 2)
                body_elements.append(Viewport(sub_table, height=sub_budget, scroll=0, width=inner_width))

                act_head = Text(f"  Recent Activity ({len(node.steps)} steps)  [dim]• Tab to focus[/]")
                body_elements.append(act_head)
            else:
                selected_child = next((c for c in node.subagents if c.thread_id == screen.selected_child_id), node.subagents[0] if node.subagents else None)
                sub_head = Text()
                sub_head.append(f"  Immediate Subagents ({len(node.subagents)})", style="bold")
                if selected_child:
                    sub_head.append(f": ▶ {selected_child.title} ({selected_child.model or 'unknown'})", style="dim")
                sub_head.append("  [dim]• Tab to focus[/]")
                sub_head.truncate(inner_width)
                body_elements.append(sub_head)

                act_head = Text()
                act_head.append("▶ Recent Activity", style="bold cyan")
                act_head.append(f" ({len(node.steps)} steps)", style="bold")
                act_head.append("  [dim]• Tab for Subagents[/]")
                body_elements.append(act_head)

                act_viewport_height = max(1, inner_height - top_used - 2)
                viewport = Viewport(
                    act_renderable,
                    height=act_viewport_height,
                    scroll=screen.activity_scroll,
                    width=inner_width,
                    on_max_scroll=lambda m: setattr(screen, "max_activity_scroll", m),
                )
                body_elements.append(viewport)
        else:
            act_head = Text(f"▶ Recent Activity ({len(node.steps)} steps)")
            body_elements.append(act_head)
            act_viewport_height = max(1, inner_height - top_used - 1)
            viewport = Viewport(
                act_renderable,
                height=act_viewport_height,
                scroll=screen.activity_scroll,
                width=inner_width,
                on_max_scroll=lambda m: setattr(screen, "max_activity_scroll", m),
            )
            body_elements.append(viewport)

        body_panel = Panel(
            Group(*body_elements),
            title=panel_title,
            box=box.ROUNDED,
            expand=True,
        )
    else:
        body = Text()
        if roots:
            visible_rows = inner_height
            ids = [t.thread_id for t in roots]
            sel_idx = ids.index(nav.selected_id) if (nav.selected_id and nav.selected_id in ids) else 0

            if sel_idx < nav.list_scroll:
                nav.list_scroll = sel_idx
            elif sel_idx >= nav.list_scroll + visible_rows:
                nav.list_scroll = sel_idx - visible_rows + 1
            max_start = max(0, len(roots) - visible_rows)
            start = max(0, min(nav.list_scroll, max_start))
            nav.list_scroll = start

            for root in roots[start : start + visible_rows]:
                is_selected = (root.thread_id == nav.selected_id)
                mark = "▶ " if is_selected else "  "

                pt = snapshot.get_provider_thread(root.thread_id) if snapshot else None
                status = pt.status if pt else getattr(root, "status", "idle")

                color = get_status_style(status)

                suffix = f"  [{root.model or 'unknown'}]"
                avail_title = max(5, inner_width - len(mark) - len(suffix) - 1)
                raw_title = root.title or root.thread_id
                if len(raw_title) > avail_title:
                    disp_title = raw_title[: avail_title - 1] + "…"
                else:
                    disp_title = raw_title

                line_style = f"bold {color}" if (is_selected and color) else (color or "")
                body.append(mark, style="bold cyan" if is_selected else "")
                body.append(disp_title, style=line_style)
                body.append(suffix, style="dim")
                body.append("\n")
        else:
            body.append("No app threads found. Waiting for activity.\n", style="dim")

        body_panel = Panel(body, title="App Threads", box=box.ROUNDED, expand=True)

    layout = Layout()
    layout.split_column(
        Layout(header_panel, size=3),
        Layout(body_panel),
        Layout(footer_panel, size=3),
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
