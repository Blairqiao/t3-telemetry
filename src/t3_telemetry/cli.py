"""CLI Entrypoint for T3 Telemetry (t3-top)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from t3_telemetry.aggregator import list_root_threads, resolve_thread
from t3_telemetry.ui import build_dashboard_renderable, build_threads_list_table


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="t3-top",
        description="Terminal telemetry dashboard and CLI for T3 Code and Antigravity.",
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
        help="Emit structured JSON telemetry for the selected thread and its subagent tree, then exit.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent app threads with provider, model, status, subagent count, and rollup token totals, then exit.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Refresh polling interval in seconds for live TUI mode (default: 0.5s).",
    )
    parser.add_argument(
        "--t3-dir",
        type=Path,
        default=Path("~/.t3/userdata").expanduser(),
        help="Override base T3 userdata directory (default: ~/.t3/userdata).",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Pin the initially resolved thread in live TUI mode instead of switching when another root thread becomes newly active.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max recent steps/turns to display in the stream table (default: 20).",
    )
    return parser


def run_live_loop(
    args: argparse.Namespace,
    console: Console | None = None,
    _max_iterations: int | None = None,
) -> int:
    """Run live watch TUI loop with graceful refresh and error handling."""
    console = console or Console()
    t3_dir = Path(args.t3_dir).expanduser()
    pinned_query: str | None = args.thread_query
    last_renderable = None
    iterations = 0

    refresh_rate = max(1, int(1.0 / max(0.1, args.interval)))

    with Live(auto_refresh=False, console=console) as live:
        while True:
            if _max_iterations is not None and iterations >= _max_iterations:
                break
            iterations += 1

            query = pinned_query or args.thread_query
            try:
                node = resolve_thread(t3_dir, query)
                if node is not None:
                    if args.no_follow and pinned_query is None:
                        pinned_query = node.thread_id
                    last_renderable = build_dashboard_renderable(node, max_steps=args.max_steps)
                    live.update(last_renderable, refresh=True)
                elif last_renderable is not None:
                    live.update(last_renderable, refresh=True)
                else:
                    live.update(
                        Panel(
                            f"[yellow]No thread found matching '{query or 'active'}' in {t3_dir}[/]\n"
                            f"[dim]Waiting for active threads...[/]",
                            title="T3 Telemetry",
                            expand=True,
                        ),
                        refresh=True,
                    )
            except (sqlite3.Error, OSError) as e:
                if last_renderable is not None:
                    live.update(last_renderable, refresh=True)
                else:
                    live.update(
                        Panel(f"[yellow]Waiting for database: {e}[/]", title="T3 Telemetry"),
                        refresh=True,
                    )

            if _max_iterations is not None and iterations >= _max_iterations:
                break

            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    t3_dir = Path(args.t3_dir).expanduser()

    if args.list:
        nodes = list_root_threads(t3_dir)
        if args.json:
            print(json.dumps([n.to_dict() for n in nodes], indent=2))
        else:
            table = build_threads_list_table(nodes)
            console = Console()
            console.print(table)
        return 0

    if args.json:
        node = resolve_thread(t3_dir, args.thread_query)
        if node is None:
            sys.stderr.write(f"No thread found matching query '{args.thread_query}' in {t3_dir}\n")
            return 1
        print(json.dumps(node.to_dict(), indent=2))
        return 0

    if args.once:
        node = resolve_thread(t3_dir, args.thread_query)
        if node is None:
            sys.stderr.write(f"No thread found matching query '{args.thread_query}' in {t3_dir}\n")
            return 1
        console = Console()
        renderable = build_dashboard_renderable(node, max_steps=args.max_steps)
        console.print(renderable)
        return 0

    try:
        return run_live_loop(args)
    except KeyboardInterrupt:
        return 0


def _entrypoint() -> None:
    try:
        sys.exit(main())
    except BrokenPipeError:
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    _entrypoint()
