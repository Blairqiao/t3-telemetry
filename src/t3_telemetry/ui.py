"""Rich Terminal UI Renderer for T3 Code Telemetry."""

from __future__ import annotations

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from t3_telemetry.models import ThreadTelemetryNode


def format_tokens(value: int | float | None) -> str:
    """Format token count into compact string representation."""
    if value is None or value <= 0:
        return "0"
    if value < 1000:
        return f"{int(value)}"
    if value < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.2f}M"


def format_duration(seconds: float | None) -> str:
    """Format duration in seconds to compact human-readable string."""
    if seconds is None or seconds <= 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s:02d}s"


def render_context_bar(used: int, max_tokens: int, width: int = 20) -> str:
    """Render visual context window gauge with Rich color tags."""
    if max_tokens <= 0:
        pct = 0.0
        filled = 0
    else:
        pct = (used / max_tokens) * 100.0
        clamped_pct = max(0.0, min(100.0, pct))
        filled = int(round(width * (clamped_pct / 100.0)))
        filled = max(0, min(width, filled))

    empty = width - filled
    bar = "█" * filled + "░" * empty

    if pct < 60.0:
        color = "green"
    elif pct <= 85.0:
        color = "yellow"
    else:
        color = "red"

    return f"[{color}][{bar}][/] {format_tokens(used)} / {format_tokens(max_tokens)} ({pct:.1f}%)"


def build_dashboard_renderable(node: ThreadTelemetryNode, max_steps: int = 20) -> Group:
    """Build Rich renderable Group with Header, Metrics, Subagents, and Steps."""
    status_str = "[bold green]RUNNING[/]" if node.status == "running" else f"[dim]{node.status.upper()}[/]"
    model_str = f"[bold yellow]{node.model}[/]"
    if node.thinking_level:
        model_str += f" [cyan]({node.thinking_level})[/]"
    driver_str = f"[bold magenta]{node.driver}[/]"

    context_bar = render_context_bar(node.context_used_tokens, node.context_max_tokens, width=20)

    header_content = (
        f"[bold]{node.title}[/]  [dim]({node.thread_id[:8]})[/]\n"
        f"Driver: {driver_str}  Model: {model_str}  Status: {status_str}\n"
        f"Context: {context_bar}"
    )
    header_panel = Panel(header_content, title="[bold]T3 Code Telemetry[/]", expand=True, box=box.ROUNDED)

    # 2. Summary Metrics Strip
    summary_table = Table(title="Summary Metrics", expand=True, box=box.ROUNDED)
    summary_table.add_column("Scope", style="bold cyan")
    summary_table.add_column("Uncached In", justify="right")
    summary_table.add_column("Cache Read", justify="right")
    summary_table.add_column("Cache Hit %", justify="right")
    summary_table.add_column("Thinking", justify="right")
    summary_table.add_column("Response", justify="right")
    summary_table.add_column("Total Out", justify="right")
    summary_table.add_column("Steps/Turns", justify="right")
    summary_table.add_column("Duration", justify="right")

    # Row 1: Self
    summary_table.add_row(
        "Self",
        format_tokens(node.self_metrics.input_tokens),
        format_tokens(node.self_metrics.cache_read_tokens),
        f"{node.self_metrics.cache_hit_pct:.1f}%",
        format_tokens(node.self_metrics.thinking_tokens),
        format_tokens(node.self_metrics.response_tokens),
        format_tokens(node.self_metrics.output_tokens),
        str(node.self_metrics.step_count),
        format_duration(node.self_metrics.duration_seconds),
    )

    # Row 2: Rollup (+N subagents)
    n_sub = node.rollup_metrics.total_subagents
    rollup_scope = f"Rollup (+{n_sub} subagents)" if n_sub > 0 else "Rollup"
    summary_table.add_row(
        rollup_scope,
        format_tokens(node.rollup_metrics.input_tokens),
        format_tokens(node.rollup_metrics.cache_read_tokens),
        f"{node.rollup_metrics.cache_hit_pct:.1f}%",
        format_tokens(node.rollup_metrics.thinking_tokens),
        format_tokens(node.rollup_metrics.response_tokens),
        format_tokens(node.rollup_metrics.output_tokens),
        str(node.rollup_metrics.step_count),
        format_duration(node.rollup_metrics.duration_seconds),
    )

    renderables: list[object] = [header_panel, summary_table]

    # 3. Subagents & Delegation Rollup Table (if present)
    if node.subagents:
        subagent_table = Table(title="Subagents & Delegated Runs", expand=True, box=box.ROUNDED, padding=(0, 1))
        subagent_table.add_column("Subagent / Role", style="bold", ratio=2, no_wrap=True, overflow="ellipsis")
        subagent_table.add_column("Model", ratio=1, no_wrap=True, overflow="ellipsis")
        subagent_table.add_column("Status", no_wrap=True)
        subagent_table.add_column("Steps", justify="right", no_wrap=True)
        subagent_table.add_column("In", justify="right", no_wrap=True)
        subagent_table.add_column("Cache", justify="right", no_wrap=True)
        subagent_table.add_column("Think", justify="right", no_wrap=True)
        subagent_table.add_column("Out", justify="right", no_wrap=True)
        subagent_table.add_column("Time", justify="right", no_wrap=True)

        def _add_subagent_rows(subagents: list[ThreadTelemetryNode], prefix: str = "", counter: int = 1) -> int:
            total = len(subagents)
            for idx, child in enumerate(subagents):
                is_last = (idx == total - 1)
                connector = "└─ " if is_last else "├─ "
                child_prefix = prefix + connector
                next_prefix = prefix + ("   " if is_last else "│  ")

                tag = f"sub-{counter}"
                counter += 1
                status_style = "[bold green]running[/]" if child.status == "running" else f"[dim]{child.status}[/]"

                label = f"{child_prefix}[cyan]{tag}[/] {child.title}"
                subagent_table.add_row(
                    label,
                    child.model or "unknown",
                    status_style,
                    str(child.rollup_metrics.step_count),
                    format_tokens(child.rollup_metrics.input_tokens),
                    format_tokens(child.rollup_metrics.cache_read_tokens),
                    format_tokens(child.rollup_metrics.thinking_tokens),
                    format_tokens(child.rollup_metrics.output_tokens),
                    format_duration(child.rollup_metrics.duration_seconds),
                )
                if child.subagents:
                    counter = _add_subagent_rows(child.subagents, next_prefix, counter)
            return counter

        _add_subagent_rows(node.subagents)
        renderables.append(subagent_table)

    # 4. Step / Turn Stream Table
    stream_table = Table(title="Recent Steps / Turns", expand=True, box=box.ROUNDED)
    stream_table.add_column("#", justify="right", style="dim")
    stream_table.add_column("Source", style="cyan")
    stream_table.add_column("Action / Tool")
    stream_table.add_column("Uncached In", justify="right")
    stream_table.add_column("Cache Read", justify="right")
    stream_table.add_column("Thinking", justify="right")
    stream_table.add_column("Response", justify="right")
    stream_table.add_column("Latency", justify="right")

    steps_to_show = node.combined_steps[-max_steps:] if node.combined_steps else node.steps[-max_steps:]
    for step in steps_to_show:
        stream_table.add_row(
            str(step.step_index),
            step.source,
            step.action_summary,
            format_tokens(step.input_tokens),
            format_tokens(step.cache_read_tokens),
            format_tokens(step.thinking_tokens),
            format_tokens(step.response_tokens),
            format_duration(step.duration_seconds),
        )

    renderables.append(stream_table)
    return Group(*renderables)


def build_threads_list_table(nodes: list[ThreadTelemetryNode]) -> Table:
    """Build Rich table summarizing recent threads for --list mode."""
    table = Table(title="Recent T3 Code App Threads", expand=True, box=box.ROUNDED, padding=(0, 1))
    table.add_column("ID", style="bold cyan", min_width=8, no_wrap=True)
    table.add_column("Title", ratio=2, no_wrap=True, overflow="ellipsis")
    table.add_column("Driver / Model", ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("Status")
    table.add_column("Context", justify="right", no_wrap=True)
    table.add_column("Sub", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("In", justify="right")
    table.add_column("Cache", justify="right")
    table.add_column("Hit%", justify="right")
    table.add_column("Think", justify="right")
    table.add_column("Out", justify="right")

    for node in nodes:
        status_style = "[bold green]running[/]" if node.status == "running" else f"[dim]{node.status}[/]"
        context_str = f"{format_tokens(node.context_used_tokens)} ({node.context_pct:.0f}%)"
        table.add_row(
            node.thread_id[:8],
            node.title,
            f"{node.driver} / {node.model}",
            status_style,
            context_str,
            str(node.rollup_metrics.total_subagents),
            str(node.rollup_metrics.step_count),
            format_tokens(node.rollup_metrics.input_tokens),
            format_tokens(node.rollup_metrics.cache_read_tokens),
            f"{node.rollup_metrics.cache_hit_pct:.1f}%",
            format_tokens(node.rollup_metrics.thinking_tokens),
            format_tokens(node.rollup_metrics.output_tokens),
        )
    return table
