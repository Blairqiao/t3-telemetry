"""Rich Terminal UI Renderer for T3 Code Telemetry."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from t3_telemetry.models import ThreadTelemetryNode


def get_status_style(status: str | None) -> str:
    s = (status or "").lower()
    if s == "running":
        return "bold green"
    elif s == "failed":
        return "bold red"
    elif s == "cancelled":
        return "bold yellow"
    elif s in ("completed", "idle"):
        return "dim"
    return ""


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


def render_context_bar(used: int, max_tokens: int | None, width: int = 20) -> str:
    """Render visual context window gauge with Rich color tags."""
    if max_tokens is None:
        empty = width
        bar = "░" * empty
        return f"[dim][{bar}][/] {format_tokens(used)} / ? (limit unknown)"

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


def build_scopes_table(
    node: ThreadTelemetryNode,
    width: int = 120,
    title: str | None = "Summary Metrics",
    box_style: box.Box = box.ROUNDED,
) -> Table:
    """Build summary table rendering the 3 scopes: Turn, Whole chat, and Chat + subagents."""
    table = Table(title=title, expand=True, box=box_style, padding=(0, 1))

    if width >= 80:
        table.add_column("Scope", style="bold cyan", min_width=25, no_wrap=True)
        table.add_column("Uncached In", justify="right")
        table.add_column("Cache Read", justify="right")
        table.add_column("Cache Hit %", justify="right")
        table.add_column("Thinking", justify="right")
        table.add_column("Response", justify="right")
        table.add_column("Total Out", justify="right")
        table.add_column("Steps/Turns", justify="right")
        table.add_column("Duration", justify="right")
    else:
        table.add_column("Scope", style="bold cyan", min_width=25, no_wrap=True)
        table.add_column("In", justify="right")
        table.add_column("Cache", justify="right")
        table.add_column("Think", justify="right")
        table.add_column("Out", justify="right")
        table.add_column("Steps/Turns", justify="right")
        table.add_column("Duration", justify="right")

    # Scope 1: Current turn (running) or Latest completed turn (idle/completed)
    if node.status == "running":
        turn_label = "Current turn (running)"
    elif node.status == "idle":
        turn_label = "Latest completed turn (idle)"
    else:
        turn_label = "Latest completed turn"
    turn = node.current_turn_metrics
    avail = node.turn_metrics_availability if node.turn_metrics_availability else (turn.availability if turn else "unavailable")

    if avail == "available" and turn is not None:
        step_disp = f"turn #{turn.ordinal}" if turn.ordinal else "1 turn"
        if width >= 80:
            table.add_row(
                turn_label,
                format_tokens(turn.input_tokens),
                format_tokens(turn.cache_read_tokens),
                f"{turn.cache_hit_pct:.1f}%",
                format_tokens(turn.thinking_tokens),
                format_tokens(turn.response_tokens),
                format_tokens(turn.output_tokens),
                step_disp,
                format_duration(turn.duration_seconds),
            )
        else:
            table.add_row(
                turn_label,
                format_tokens(turn.input_tokens),
                format_tokens(turn.cache_read_tokens),
                format_tokens(turn.thinking_tokens),
                format_tokens(turn.output_tokens),
                step_disp,
                format_duration(turn.duration_seconds),
            )
    elif avail == "partial" and turn is not None:
        step_disp = f"turn #{turn.ordinal} (partial)" if turn.ordinal else "[yellow]partial[/]"
        in_str = format_tokens(turn.input_tokens) if turn.input_tokens else "[yellow]partial[/]"
        cr_str = format_tokens(turn.cache_read_tokens) if turn.cache_read_tokens else "[yellow]partial[/]"
        hit_str = f"{turn.cache_hit_pct:.1f}%" if (turn.input_tokens or turn.cache_read_tokens) else "[yellow]partial[/]"
        th_str = format_tokens(turn.thinking_tokens) if turn.thinking_tokens else "[yellow]partial[/]"
        resp_str = format_tokens(turn.response_tokens) if turn.response_tokens else "[yellow]partial[/]"
        out_str = format_tokens(turn.output_tokens) if turn.output_tokens else "[yellow]partial[/]"
        if width >= 80:
            table.add_row(
                turn_label,
                in_str,
                cr_str,
                hit_str,
                th_str,
                resp_str,
                out_str,
                step_disp,
                format_duration(turn.duration_seconds),
            )
        else:
            table.add_row(
                turn_label,
                in_str,
                cr_str,
                th_str,
                out_str,
                step_disp,
                format_duration(turn.duration_seconds),
            )
    else:
        # Unavailable
        step_disp = f"[dim]unavailable ({node.latest_step_label})[/]" if node.latest_step_label else "[dim]unavailable[/]"
        if width >= 80:
            table.add_row(
                turn_label,
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                step_disp,
                "[dim]unavailable[/]",
            )
        else:
            table.add_row(
                turn_label,
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                "[dim]unavailable[/]",
                step_disp,
                "[dim]unavailable[/]",
            )

    # Scope 2: Whole chat · this agent
    if width >= 80:
        table.add_row(
            "Whole chat · this agent",
            format_tokens(node.self_metrics.input_tokens),
            format_tokens(node.self_metrics.cache_read_tokens),
            f"{node.self_metrics.cache_hit_pct:.1f}%",
            format_tokens(node.self_metrics.thinking_tokens),
            format_tokens(node.self_metrics.response_tokens),
            format_tokens(node.self_metrics.output_tokens),
            node.self_metrics.steps_display,
            format_duration(node.self_metrics.duration_seconds),
        )
    else:
        table.add_row(
            "Whole chat · this agent",
            format_tokens(node.self_metrics.input_tokens),
            format_tokens(node.self_metrics.cache_read_tokens),
            format_tokens(node.self_metrics.thinking_tokens),
            format_tokens(node.self_metrics.output_tokens),
            node.self_metrics.steps_display,
            format_duration(node.self_metrics.duration_seconds),
        )

    # Scope 3: Chat + subagents
    if width >= 80:
        table.add_row(
            "Chat + subagents",
            format_tokens(node.rollup_metrics.input_tokens),
            format_tokens(node.rollup_metrics.cache_read_tokens),
            f"{node.rollup_metrics.cache_hit_pct:.1f}%",
            format_tokens(node.rollup_metrics.thinking_tokens),
            format_tokens(node.rollup_metrics.response_tokens),
            format_tokens(node.rollup_metrics.output_tokens),
            node.rollup_metrics.steps_display,
            format_duration(node.rollup_metrics.duration_seconds),
        )
    else:
        table.add_row(
            "Chat + subagents",
            format_tokens(node.rollup_metrics.input_tokens),
            format_tokens(node.rollup_metrics.cache_read_tokens),
            format_tokens(node.rollup_metrics.thinking_tokens),
            format_tokens(node.rollup_metrics.output_tokens),
            node.rollup_metrics.steps_display,
            format_duration(node.rollup_metrics.duration_seconds),
        )

    return table


def format_clock(ts: str | None) -> str:
    """Extract compact HH:MM:SS from ISO timestamp."""
    if not ts:
        return ""
    if "T" in ts:
        part = ts.split("T")[1]
        part = part.rstrip("Z").split("+")[0]
        return part[:8]
    return ts[:8]


BLOCKS = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def render_context_graph(
    node: ThreadTelemetryNode,
    width: int = 60,
    compact: bool = False,
    as_panel: bool = True,
) -> Panel | Group:
    """Render a terminal graph of context occupancy and history."""
    samples = node.context_samples
    current_val = node.context_used_tokens
    peak_val = node.observed_peak_tokens
    limit_val = node.context_limit

    has_drop = False
    for i in range(1, len(samples)):
        if samples[i].used_tokens < samples[i - 1].used_tokens:
            has_drop = True
            break
    if not has_drop and samples and current_val < peak_val:
        has_drop = True

    elements: list[Text] = []
    if node.context_provenance == "measured":
        prov_style = "yellow" if node.context_uncertainty else "bold cyan"
        prov_label = "Measured (uncertain recency)" if node.context_uncertainty else "Measured"
    elif node.context_provenance == "estimated":
        prov_style = "yellow"
        prov_label = "Estimated"
    else:
        prov_style = "dim"
        prov_label = "Unknown"

    if width < 75:
        meta1 = Text()
        meta1.append(prov_label, style=prov_style)
        meta1.append("  ·  Current: ", style="dim")
        meta1.append(format_tokens(current_val), style="bold")
        meta1.append("  ·  Peak: ", style="dim")
        meta1.append(format_tokens(peak_val), style="bold")
        elements.append(meta1)

        meta2 = Text()
        meta2.append("Limit: ", style="dim")
        if limit_val is not None and limit_val > 0:
            meta2.append(format_tokens(limit_val), style="bold")
        else:
            meta2.append("unknown", style="dim")
        if has_drop:
            meta2.append("  ·  ", style="dim")
            meta2.append("Context drop", style="yellow")
        elements.append(meta2)
    else:
        meta = Text()
        meta.append(prov_label, style=prov_style)
        meta.append("  ·  Current: ", style="dim")
        meta.append(format_tokens(current_val), style="bold")
        meta.append("  ·  Peak: ", style="dim")
        meta.append(format_tokens(peak_val), style="bold")
        meta.append("  ·  Limit: ", style="dim")
        if limit_val is not None and limit_val > 0:
            meta.append(format_tokens(limit_val), style="bold")
        else:
            meta.append("unknown", style="dim")
        if has_drop:
            meta.append("  ·  ", style="dim")
            meta.append("Context drop", style="yellow")
        elements.append(meta)

    if not samples:
        elements.append(Text("No context history recorded.", style="dim"))
    elif len(samples) == 1:
        s = samples[0]
        line = Text()
        s_style = "bold cyan" if s.provenance == "measured" else "yellow"
        line.append("● ", style=s_style)
        line.append(f"{format_tokens(s.used_tokens)} tokens ({s.provenance})")
        t_str = format_clock(s.timestamp or getattr(s, "observed_at", None))
        if t_str:
            line.append(f"  [dim]at {t_str}[/]")
        line.append("  [dim]• single reading[/]")
        elements.append(line)
    else:
        chart_rows = 2 if compact else 3
        max_scale = max(peak_val, limit_val or 0, 1)

        cols: list[dict[str, Any]] = []
        for i, s in enumerate(samples):
            if i > 0:
                prev = samples[i - 1]
                if s.provenance != prev.provenance:
                    cols.append({"type": "gap", "char": "┊"})
            is_drop = (i > 0 and s.used_tokens < samples[i - 1].used_tokens)
            cols.append({
                "type": "sample",
                "used": s.used_tokens,
                "prov": s.provenance,
                "is_drop": is_drop,
                "time": s.timestamp or getattr(s, "observed_at", None),
            })

        for r in range(chart_rows - 1, -1, -1):
            row_text = Text()
            y_low = (r / chart_rows) * max_scale
            y_high = ((r + 1) / chart_rows) * max_scale
            for c in cols:
                if c["type"] == "gap":
                    row_text.append(" ┊ ", style="dim")
                    continue
                val = c["used"]
                c_style = "cyan" if c["prov"] == "measured" else "yellow"
                if c["is_drop"]:
                    c_style = "yellow"

                if val <= y_low:
                    row_text.append("   ")
                elif val >= y_high:
                    row_text.append(" █ ", style=c_style)
                else:
                    frac = (val - y_low) / (y_high - y_low)
                    idx = min(8, max(1, int(frac * 8)))
                    ch = BLOCKS[idx]
                    row_text.append(f" {ch} ", style=c_style)
            elements.append(row_text)

        t_first = format_clock(samples[0].timestamp or getattr(samples[0], "observed_at", None))
        t_last = format_clock(samples[-1].timestamp or getattr(samples[-1], "observed_at", None))
        if t_first or t_last:
            time_line = Text()
            time_line.append(t_first or "", style="dim")
            total_len = len(t_first) + len(t_last)
            avail = max(4, width - total_len - 6)
            dash_count = max(2, min(avail, 20))
            time_line.append(" " + "─" * dash_count + " ", style="dim")
            time_line.append(t_last or "", style="dim")
            elements.append(time_line)

    grp = Group(*elements)
    if as_panel:
        return Panel(grp, title="[bold]Context History[/]", box=box.ROUNDED, expand=True)
    return grp


def build_scopes_summary(node: ThreadTelemetryNode, width: int = 80) -> list[Text]:
    """Build compact scope summary lines for interactive live browser frame."""
    if node.status == "running":
        turn_label = "Current turn (running)"
    elif node.status == "idle":
        turn_label = "Latest completed turn (idle)"
    else:
        turn_label = "Latest completed turn"

    turn = node.current_turn_metrics
    avail = node.turn_metrics_availability if node.turn_metrics_availability else (turn.availability if turn else "unavailable")

    if avail == "available" and turn is not None:
        turn_val = f"{turn.steps_display}, {format_tokens(turn.total_tokens)} tokens, {format_duration(turn.duration_seconds)}"
    elif avail == "partial" and turn is not None:
        turn_val = f"partial ({format_tokens(turn.total_tokens)} tokens, {format_duration(turn.duration_seconds)})"
    else:
        turn_val = f"unavailable ({node.latest_step_label})" if node.latest_step_label else "unavailable"

    self_val = f"{node.self_metrics.steps_display}, {format_tokens(node.self_metrics.total_tokens)} tokens, {format_duration(node.self_metrics.duration_seconds)}"
    n_sub = node.rollup_metrics.total_subagents
    rollup_label = f"Chat + subagents (+{n_sub} subs)" if n_sub > 0 else "Chat + subagents"
    rollup_val = f"{node.rollup_metrics.steps_display}, {format_tokens(node.rollup_metrics.total_tokens)} tokens, {format_duration(node.rollup_metrics.duration_seconds)}"

    lines: list[Text] = []
    if width >= 90:
        line1 = Text()
        line1.append(f"{turn_label}: ", style="bold cyan")
        line1.append(turn_val)
        line1.append("  ·  ")
        line1.append("Whole chat · this agent: ", style="bold cyan")
        line1.append(self_val)
        lines.append(line1)

        line2 = Text()
        line2.append(f"{rollup_label}: ", style="bold cyan")
        line2.append(rollup_val)
        lines.append(line2)
    else:
        line1 = Text()
        line1.append(f"{turn_label}: ", style="bold cyan")
        line1.append(turn_val)
        lines.append(line1)

        line2 = Text()
        line2.append("Whole chat · this agent: ", style="bold cyan")
        line2.append(self_val)
        lines.append(line2)

        line3 = Text()
        line3.append(f"{rollup_label}: ", style="bold cyan")
        line3.append(rollup_val)
        lines.append(line3)

    return lines


def build_dashboard_renderable(node: ThreadTelemetryNode, max_steps: int = 20) -> Group:
    """Build Rich renderable Group with Header, Metrics, Subagents, and Steps."""
    status_str = "[bold green]RUNNING[/]" if node.status == "running" else f"[dim]{node.status.upper()}[/]"
    model_str = f"[bold yellow]{node.model}[/]"
    if node.thinking_level:
        model_str += f" [cyan]({node.thinking_level})[/]"
    driver_str = f"[bold magenta]{node.driver}[/]"

    eff_limit = node.context_limit if (node.context_limit is not None and node.context_limit > 0) else (node.context_max_tokens if getattr(node, "context_limit_provenance", "") == "measured" else None)
    context_bar = render_context_bar(node.context_used_tokens, eff_limit, width=20)

    header_lines = [
        f"[bold]{node.title}[/]  [dim]({node.thread_id[:8]})[/]",
        f"Driver: {driver_str}  Model: {model_str}  Status: {status_str}",
        f"Context: {context_bar}",
    ]
    header_panel = Panel(chr(10).join(header_lines), title="[bold]T3 Code Telemetry[/]", expand=True, box=box.ROUNDED)

    # 2. Summary Metrics Strip (3 scopes)
    summary_table = build_scopes_table(node, width=120, title="Summary Metrics", box_style=box.ROUNDED)

    # 3. Context History & Occupancy Chart
    context_chart = render_context_graph(node, width=120, as_panel=True)

    renderables: list[object] = [header_panel, summary_table, context_chart]

    # 3. Subagents & Delegation Rollup Table (if present) - Immediate children only
    if node.subagents:
        subagent_table = Table(title="Subagents & Delegated Runs", expand=True, box=box.ROUNDED, padding=(0, 1))
        subagent_table.add_column("Subagent / Role", ratio=2, no_wrap=True, overflow="ellipsis")
        subagent_table.add_column("Model", ratio=1, no_wrap=True, overflow="ellipsis")
        subagent_table.add_column("Steps", justify="right", no_wrap=True)
        subagent_table.add_column("In", justify="right", no_wrap=True)
        subagent_table.add_column("Cache", justify="right", no_wrap=True)
        subagent_table.add_column("Think", justify="right", no_wrap=True)
        subagent_table.add_column("Out", justify="right", no_wrap=True)
        subagent_table.add_column("Time", justify="right", no_wrap=True)

        for idx, child in enumerate(node.subagents):
            tag = f"sub-{idx + 1}"
            child_style = get_status_style(child.status)
            label = Text()
            label.append(f"{tag} ", style="cyan")
            label.append(child.title, style=child_style)
            subagent_table.add_row(
                label,
                child.model or "unknown",
                str(child.rollup_metrics.step_count),
                format_tokens(child.rollup_metrics.input_tokens),
                format_tokens(child.rollup_metrics.cache_read_tokens),
                format_tokens(child.rollup_metrics.thinking_tokens),
                format_tokens(child.rollup_metrics.output_tokens),
                format_duration(child.rollup_metrics.duration_seconds),
            )

        renderables.append(subagent_table)

    # 4. Step / Turn Stream Table - Node steps only (never combined_steps)
    stream_table = Table(title="Recent Steps / Turns", expand=True, box=box.ROUNDED)
    stream_table.add_column("Activity", style="dim", no_wrap=True)
    stream_table.add_column("Source", style="cyan")
    stream_table.add_column("Action / Tool")
    stream_table.add_column("Uncached In", justify="right")
    stream_table.add_column("Cache Read", justify="right")
    stream_table.add_column("Thinking", justify="right")
    stream_table.add_column("Response", justify="right")
    stream_table.add_column("Latency", justify="right")

    steps_to_show = node.steps[-max_steps:] if node.steps else []
    for step in steps_to_show:
        stream_table.add_row(
            step.row_label,
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
    table.add_column("Context", justify="right", no_wrap=True)
    table.add_column("Sub", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("In", justify="right")
    table.add_column("Cache", justify="right")
    table.add_column("Hit%", justify="right")
    table.add_column("Think", justify="right")
    table.add_column("Out", justify="right")

    for node in nodes:
        if node.context_limit is not None and node.context_limit > 0:
            context_str = f"{format_tokens(node.context_used_tokens)} ({node.context_pct:.0f}%)"
        elif getattr(node, "context_limit_provenance", "") == "measured":
            context_str = f"{format_tokens(node.context_used_tokens)} ({node.context_pct:.0f}%)"
        else:
            context_str = f"{format_tokens(node.context_used_tokens)} (?)"
        status_style = get_status_style(node.status)
        title_text = Text(node.title, style=status_style) if status_style else Text(node.title)
        table.add_row(
            node.thread_id[:8],
            title_text,
            f"{node.driver} / {node.model}",
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
