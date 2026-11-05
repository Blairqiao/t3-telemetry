"""Rich Terminal UI Renderer for T3 Code Telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import textwrap
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from t3_telemetry.activity import DETAIL_FIELD_LABELS, TOOL_VERBS, ActivityEntry, build_activity_entries, describe_tool, neutralize
from t3_telemetry.models import ModelStep, ThreadTelemetryNode, ToolCallDetail, TurnItem
from t3_telemetry.tui import window_start
from t3_telemetry.usage import PROVIDER_ORDER, PeriodUsage


# Focused selection: bright text on a solid background; the ▶ marker carries it without color.
SELECTED_ROW_STYLE = "bold bright_yellow"


def get_status_style(status: str | None) -> str:
    s = (status or "").lower()
    if s == "running":
        return "bold bright_green"
    elif s == "failed":
        return "bold bright_red"
    elif s == "cancelled":
        return "bold bright_yellow"
    elif s == "completed":
        return "bold bright_blue"
    elif s == "idle":
        return "bold bright_white"
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


BLOCKS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
RANGE_CHAR = "░"


def nice_ceiling(value: float) -> int:
    """Smallest 1, 2, or 5 × 10^n at or above value; 1k for zero or negative data."""
    if value <= 0:
        return 1000
    magnitude = 1
    while True:
        for step in (1, 2, 5):
            if step * magnitude >= value:
                return step * magnitude
        magnitude *= 10


def format_axis_tokens(value: float) -> str:
    """Short axis label such as 0, 500, 2.5k, 100k, 1M."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1000:
        return f"{value / 1000:g}k"
    return f"{value:g}"


PROVIDER_STYLES = {"Claude": "dark_orange", "Codex": "green3", "Antigravity": "medium_purple1", "Other": "grey62"}
PROVIDER_GLYPHS = {"Claude": "█", "Codex": "▓", "Antigravity": "▒", "Other": "░"}


def merge_buckets(buckets: list[dict[str, int]], starts: list[datetime], max_columns: int) -> list[tuple[datetime, dict[str, int]]]:
    """Combine adjacent buckets, retaining the first start and provider sums."""
    if not buckets or max_columns <= 0:
        return []
    size = (len(buckets) + max_columns - 1) // max_columns
    result = []
    for i in range(0, len(buckets), size):
        totals: dict[str, int] = {}
        for bucket in buckets[i:i + size]:
            for provider, value in bucket.items():
                totals[provider] = totals.get(provider, 0) + value
        result.append((starts[i], totals))
    return result


def stack_column(values: dict[str, int], ceiling: int, rows: int) -> list[str | None]:
    """Assign each row, bottom first, to the provider covering its midpoint."""
    result: list[str | None] = [None] * rows
    if ceiling <= 0 or rows <= 0:
        return result
    low = 0.0
    for provider in PROVIDER_ORDER:
        high = low + values.get(provider, 0) / ceiling * rows
        for row in range(rows):
            if low <= row + 0.5 < high:
                result[row] = provider
        low = high
    if values and sum(values.values()) > 0 and not any(result):
        result[0] = max(PROVIDER_ORDER, key=lambda provider: values.get(provider, 0))
    return result


def usage_graph_lines(period: PeriodUsage, width: int, plot_rows: int, now: datetime) -> list[Text]:
    """Render a legend, stacked usage plot, and time labels."""
    legend = Text()
    for provider in PROVIDER_ORDER:
        if provider in period.by_provider:
            if legend:
                legend.append("   ")
            legend.append("■", style=PROVIDER_STYLES[provider])
            legend.append(f" {provider} ({PROVIDER_GLYPHS[provider]}) {format_tokens(period.by_provider[provider].total)}")
    if not legend:
        legend.append("No usage recorded")
    legend.truncate(width, overflow="ellipsis")
    merged = merge_buckets(period.buckets, period.window.bucket_starts, max(1, (width - 6) // 2))
    ceiling = nice_ceiling(max((sum(values.values()) for _, values in merged), default=0))
    n = len(merged)
    bar_w = min(3, max(1, (width - 6) // n - 1)) if n else 1
    columns = [stack_column(values, ceiling, plot_rows) for _, values in merged]
    lines = [legend]
    for row in range(plot_rows - 1, -1, -1):
        labels = {0: "0", plot_rows - 1: format_axis_tokens(ceiling)}
        if plot_rows >= 5:
            labels[plot_rows // 2] = format_axis_tokens(ceiling / 2)
        label = labels.get(row)
        line = Text((label or "")[-5:].rjust(5), style="bold")
        line.append("┤" if label is not None else "│", style="bold")
        for (start, _), column in zip(merged, columns):
            provider = column[row]
            if start >= now:
                line.append("·" * bar_w if row == 0 else " " * bar_w, style="dim")
            elif provider:
                line.append(PROVIDER_GLYPHS[provider] * bar_w, style=PROVIDER_STYLES[provider])
            else:
                line.append(" " * bar_w)
            line.append(" ")
        line.truncate(width)
        lines.append(line)
    labels_line = Text(" " * 6)
    for i, (start, _) in enumerate(merged):
        offset = 6 + i * (bar_w + 1)
        if period.window.bucket == "hour":
            label = str(start.hour) if start.hour in (0, 6, 12, 18) else ""
        elif period.window.key == "week":
            label = start.strftime("%a")
        else:
            label = str(start.day)
        if label and offset >= labels_line.cell_len and offset + len(label) <= width:
            labels_line.append(" " * (offset - labels_line.cell_len) + label)
    lines.append(labels_line)
    return lines


@dataclass(frozen=True, slots=True)
class ContextBucket:
    """A chronological run of samples drawn as one bar column."""

    first_index: int
    count: int
    low: int
    high: int
    latest: int
    provenances: frozenset[str]
    latest_provenance: str
    has_drop: bool
    provenance_boundary: bool


def bucket_context_samples(samples: Sequence[Any], capacity: int) -> list[ContextBucket]:
    """Split samples into at most `capacity` contiguous chronological buckets.

    Bucket i covers samples [i*n//B, (i+1)*n//B). Each keeps its min/max range,
    its most recent value, whether it contains a drop (including the step from
    the previous bucket), and whether provenance changes inside it or at its start.
    """
    n = len(samples)
    if n == 0 or capacity <= 0:
        return []
    count = min(n, capacity)
    buckets: list[ContextBucket] = []
    for b in range(count):
        start = b * n // count
        end = (b + 1) * n // count
        chunk = samples[start:end]
        values = [s.used_tokens for s in chunk]
        provs = [s.provenance for s in chunk]
        drop = any(samples[i].used_tokens < samples[i - 1].used_tokens for i in range(max(1, start), end))
        boundary = any(samples[i].provenance != samples[i - 1].provenance for i in range(max(1, start), end))
        buckets.append(
            ContextBucket(
                first_index=start,
                count=end - start,
                low=min(values),
                high=max(values),
                latest=values[-1],
                provenances=frozenset(provs),
                latest_provenance=provs[-1],
                has_drop=drop,
                provenance_boundary=boundary,
            )
        )
    return buckets


def _bar_cell(bucket: ContextBucket, row: int, rows: int, ceiling: int) -> tuple[str, str]:
    """Glyph and style for one plot cell (row 0 is the bottom).

    The latest value is drawn with full and eighth blocks. The range above it,
    up to the bucket max, is drawn with RANGE_CHAR only in cells the max covers
    at least halfway, so a peak is never overstated by more than half a row.
    """
    cell = ceiling / rows
    y_low = row * cell
    style = "cyan" if bucket.latest_provenance == "measured" else "yellow"
    latest = bucket.latest
    if latest > y_low:
        if latest >= y_low + cell:
            return "█", style
        # Partial cell; any nonzero value shows at least one eighth.
        return BLOCKS[min(8, max(1, int((latest - y_low) / cell * 8)))], style
    if bucket.high > latest and bucket.high >= y_low + cell / 2:
        return RANGE_CHAR, style
    return " ", ""


def _sample_time(s: Any) -> str:
    return format_clock(s.timestamp or getattr(s, "observed_at", None))


def _graph_parts(node: ThreadTelemetryNode, width: int, plot_rows: int) -> dict[str, Any]:
    """Build the graph's line groups for one width and row count.

    Keys: header (lines), plot (lines), marker (line or None), marker_required
    (drop/switch present), time (line or None), scale (str or None), extras (legend items).
    """
    plot_rows = max(1, plot_rows)
    samples = list(node.context_samples)
    current_val = node.context_used_tokens or 0
    peak_val = node.observed_peak_tokens or 0
    limit_val = node.context_limit

    has_drop = any(samples[i].used_tokens < samples[i - 1].used_tokens for i in range(1, len(samples)))
    if not has_drop and samples and current_val < peak_val:
        has_drop = True

    if node.context_provenance == "measured":
        prov_style = "yellow" if node.context_uncertainty else "bold cyan"
        prov_label = "Measured (uncertain recency)" if node.context_uncertainty else "Measured"
    elif node.context_provenance == "estimated":
        prov_style = "yellow"
        prov_label = "Estimated"
    else:
        prov_style = "bold"
        prov_label = "Unknown"

    def _limit_parts(with_drop: bool = True) -> list[tuple[str, str]]:
        parts = [("Limit: ", "")]
        if limit_val is not None and limit_val > 0:
            parts.append((format_tokens(limit_val), "bold"))
        else:
            parts.append(("unknown", "bold yellow"))
        if has_drop and with_drop:
            parts += [("  ·  ", "dim"), ("Context drop", "yellow")]
        return parts

    head = [
        (prov_label, prov_style),
        ("  ·  Current: ", ""),
        (format_tokens(current_val), "bold"),
    ]
    peak = [("Peak: ", ""), (format_tokens(peak_val), "bold")]
    head += [("  ·  ", "")] + peak
    def _header(with_drop: bool) -> list[Text]:
        limit = _limit_parts(with_drop)
        if width >= 75:
            return [Text.assemble(*head, ("  ·  ", ""), *limit)]
        first = Text.assemble(*head)
        if first.cell_len <= width:
            return [first, Text.assemble(*limit)]
        return [Text.assemble(*head[:3]), Text.assemble(*peak, ("  ·  ", ""), *limit)]

    header = _header(with_drop=True)

    parts: dict[str, Any] = {
        "header": header,
        # Same lines without the optional "Context drop" tail, so a tight
        # budget can trade that tail for the scale and keep the limit.
        "header_lean": _header(with_drop=False),
        "plot": [],
        "marker": None,
        "marker_required": False,
        "time": None,
        "scale": None,
        "extras": [],
    }
    if not samples:
        parts["plot"] = [Text("No context history recorded.", style="bold")]
        return parts

    ceiling = nice_ceiling(max([current_val] + [s.used_tokens for s in samples]))
    # Top row always carries the ceiling; with one row there is no room for 0.
    labels = {0: "0"}
    labels[plot_rows - 1] = format_axis_tokens(ceiling)
    if plot_rows >= 3 and plot_rows % 2 == 1:
        labels[plot_rows // 2] = format_axis_tokens(ceiling / 2)
    axis_w = max(len(v) for v in labels.values()) + 1
    if width - axis_w < 1:
        # Too narrow for labels: keep only the plot.
        labels, axis_w = {}, 0
    plot_w = max(1, width - axis_w)

    buckets = bucket_context_samples(samples, plot_w)
    col_w = max(1, min(3, plot_w // len(buckets)))
    bucketed = any(b.count > 1 for b in buckets)

    def _pad(glyph: str) -> str:
        if col_w == 1:
            return glyph
        if col_w == 2:
            return glyph + " "
        return " " + glyph + " "

    for row in range(plot_rows - 1, -1, -1):
        line = Text()
        if axis_w:
            line.append(labels.get(row, "").rjust(axis_w - 1), style="bold")
            line.append("┤" if row in labels else "│", style="bold")
        for b in buckets:
            glyph, style = _bar_cell(b, row, plot_rows, ceiling)
            line.append(_pad(glyph), style=style)
        parts["plot"].append(line)

    # Marker row: drops, provenance switches, estimated columns.
    # With 3-cell columns a source switch sits in the gap before the column,
    # so it stays visible alongside a drop; narrower columns show one marker.
    markers: list[tuple[str, str, str]] = []
    for b in buckets:
        gap = "┊" if (col_w == 3 and b.provenance_boundary) else " "
        if b.has_drop:
            markers.append((gap, "↓", "bold yellow"))
        elif b.provenance_boundary:
            markers.append((" ", "┊", "bold"))
        elif "estimated" in b.provenances:
            markers.append((gap, "~", "yellow"))
        else:
            markers.append((gap, " ", ""))
    glyphs = {m[0] for m in markers} | {m[1] for m in markers}
    if glyphs - {" "}:
        line = Text(" " * axis_w)
        for gap, glyph, style in markers:
            cell = (gap + glyph + " ") if col_w == 3 else _pad(glyph)
            line.append(cell, style=style)
        parts["marker"] = line
        parts["marker_required"] = bool(glyphs & {"↓", "┊"})

    # Time / order axis; the latest label ends under the last bar.
    used_w = len(buckets) * col_w
    end_col = axis_w + used_w
    first_t, last_t = _sample_time(samples[0]), _sample_time(samples[-1])
    right = ""
    if len(samples) == 1:
        left = f"at {first_t}" if first_t else "sample 1"
    elif not (first_t or last_t):
        left = f"#1 … #{len(samples)} (no times)"
    else:
        left = first_t or "sample 1"
        right = last_t or f"sample {len(samples)}"
    time_line = Text(" " * axis_w)
    if not right:
        time_line.append(left, style="bold")
    elif len(left) + 1 + len(right) <= used_w:
        time_line.append(left, style="bold")
        time_line.append(" " * (used_w - len(left) - len(right)))
        time_line.append(right, style="bold")
    else:
        # Few columns: end the latest label under the last bar, shifting the
        # oldest label left into the axis gutter; overflow only if both cannot fit.
        start = max(len(left) + 1, end_col - len(right))
        left_start = max(0, min(axis_w, start - len(left) - 1))
        time_line = Text(" " * left_start)
        time_line.append(left, style="bold")
        time_line.append(" " * (start - left_start - len(left)))
        time_line.append(right, style="bold")
    parts["time"] = time_line

    parts["scale"] = f"auto axis 0–{format_axis_tokens(ceiling)}, not limit"
    extras: list[str] = []
    if len(samples) == 1:
        extras.append("single reading")
    if bucketed:
        per = -(-len(samples) // len(buckets))
        extras.append(f"█ latest ░ col peak, ≤{per}/col")
    extras += [label for glyph, label in (("↓", "↓ drop"), ("┊", "┊ source switch"), ("~", "~ estimated")) if glyph in glyphs]
    parts["extras"] = extras
    return parts


def _pack_legend(items: list[str], width: int) -> list[Text]:
    """Pack legend items into as few width-bounded lines as possible."""
    if not items:
        return []
    lines: list[Text] = []
    legend = Text(style="bold")
    for item in items:
        if legend.plain and len(legend.plain) + 3 + len(item) > width:
            lines.append(legend)
            legend = Text(style="bold")
        legend.append(("  · " if legend.plain else "") + item)
    lines.append(legend)
    return lines


def _assemble(parts: dict[str, Any], width: int, level: int) -> list[Text]:
    """Join parts, dropping optional lines up to `level`.

    Level 1 drops an optional marker row (no drop/switch), 2 the time line,
    3 extra legend items, 4 merges the scale onto the last header line.
    """
    header = [h.copy() for h in parts["header"]]
    merge = level >= 4 and bool(parts["scale"])
    if merge:
        # Merge the scale onto a header line with room. Failing that, drop the
        # optional "Context drop" tail to make room, keeping the limit; as a
        # last resort the scale replaces the last header line.
        def _try_merge(candidates: list[Text]) -> bool:
            for i in range(len(candidates) - 1, -1, -1):
                merged = candidates[i].copy()
                merged.append("  · " + parts["scale"], style="bold")
                if merged.cell_len <= width:
                    header[i] = merged
                    return True
            return False

        if not _try_merge(header) and not _try_merge(parts["header_lean"]):
            header[-1] = Text(parts["scale"], style="bold")
    lines = header + list(parts["plot"])
    if parts["marker"] is not None and not (level >= 1 and not parts["marker_required"]):
        lines.append(parts["marker"])
    if parts["time"] is not None and level < 2:
        lines.append(parts["time"])
    if not merge:
        items = ([parts["scale"]] if parts["scale"] else []) + (parts["extras"] if level < 3 else [])
        lines += _pack_legend(items, width)
    for line in lines:
        line.no_wrap = True
        line.truncate(width, overflow="ellipsis")
    return lines


def context_graph_min_height(node: ThreadTelemetryNode, width: int, plot_rows: int = 3) -> int:
    """Fewest lines that keep the graph header and `plot_rows` labeled rows (legend merged)."""
    parts = _graph_parts(node, max(1, width), plot_rows)
    return len(_assemble(parts, max(1, width), 4))


def context_graph_lines(
    node: ThreadTelemetryNode,
    width: int,
    plot_rows: int = 3,
    height: int | None = None,
    min_plot_rows: int = 1,
) -> list[Text]:
    """Lines of the context graph, each at most `width` cells.

    With `height`, returns exactly that many lines: plot rows absorb the space,
    and optional lines are dropped (see `_assemble`) before rows go below
    `min_plot_rows` (then below 1).
    """
    width = max(1, width)
    if height is None:
        return _assemble(_graph_parts(node, width, plot_rows), width, 0)
    height = max(1, height)
    best: list[Text] | None = None
    for level in range(5):
        # Axis width (and so buckets and markers) can depend on the row count,
        # so build each candidate for real rather than estimating overhead.
        for rows in range(height, 0, -1):
            if rows < min_plot_rows and level < 4:
                break
            lines = _assemble(_graph_parts(node, width, rows), width, level)
            if len(lines) == height:
                return lines
            if len(lines) < height and best is None:
                best = lines
            if len(lines) < height:
                break
        if best is not None:
            break
    if best is None:
        best = _assemble(_graph_parts(node, width, 1), width, 4)
    return best[:height] + [Text("") for _ in range(height - len(best))]


class _ContextGraphBody:
    """Renders context graph lines at the narrower of `width` and the console width."""

    def __init__(self, node: ThreadTelemetryNode, width: int, plot_rows: int, height: int | None, min_plot_rows: int = 1) -> None:
        self.min_plot_rows = min_plot_rows
        self.node = node
        self.width = width
        self.plot_rows = plot_rows
        self.height = height

    def lines(self, width: int) -> list[Text]:
        return context_graph_lines(self.node, width, self.plot_rows, self.height, self.min_plot_rows)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        yield from self.lines(min(self.width, options.max_width))


def render_context_graph(
    node: ThreadTelemetryNode,
    width: int = 60,
    compact: bool = False,
    as_panel: bool = True,
    height: int | None = None,
    plot_rows: int | None = None,
    min_plot_rows: int = 1,
) -> Panel | Group:
    """Render context occupancy as bars on an automatic token axis.

    `width` caps the line width; `height`, when given, is the exact number of
    lines to produce (plot rows absorb the remainder). Otherwise `plot_rows`
    applies, defaulting to 3 when compact and 5 otherwise.
    """
    rows = plot_rows if plot_rows is not None else (3 if compact else 5)
    body = _ContextGraphBody(node, width - 4 if as_panel else width, rows, height, min_plot_rows)
    if as_panel:
        return Panel(Group(body), title="[bold]Context History[/]", box=box.ROUNDED, expand=True)
    return Group(body)


def activity_range_label(start: int, shown: int, total: int) -> str:
    """Visible range such as '4–6 of 12' (1-based, inclusive)."""
    if total <= 0 or shown <= 0:
        return f"0 of {total}"
    return f"{start + 1}–{start + shown} of {total}"


def activity_tokens_cell(entry: ActivityEntry) -> str:
    if entry.total_tokens is None:
        return "—"
    approx = entry.usage_provenance == "cumulative_fallback" or entry.usage_status == "partial"
    return format_tokens(entry.total_tokens) + ("~" if approx else "")


def build_activity_table(
    entries: list[ActivityEntry],
    *,
    selected_id: str | None = None,
    focused: bool = False,
    interactive: bool = False,
    show_header: bool = True,
    title: str | None = None,
    box_style: box.Box = box.SIMPLE_HEAD,
    range_label: str | None = None,
) -> Table:
    """Compact activity feed: descriptive title, total tokens, duration.

    Shared by the live browser (``interactive`` adds the selection marker and
    highlight) and the one-shot snapshot.
    """
    if interactive:
        # The section rule above carries the name, range and focus; the header names columns only.
        table = Table(box=None, expand=True, padding=(0, 1), show_header=show_header, pad_edge=False, show_edge=False, header_style="dim")
    else:
        table = Table(title=title, box=box_style, expand=True, show_header=show_header)
    header = "" if interactive else "Activity"
    table.add_column(header, ratio=1, no_wrap=True, overflow="ellipsis")
    table.add_column("Tokens", justify="right", no_wrap=True)
    table.add_column("Time", justify="right", no_wrap=True)
    for entry in entries:
        duration = "—" if entry.is_launch else format_duration(entry.duration_seconds)
        cells: list[Any] = [Text(entry.title), activity_tokens_cell(entry), duration]
        if interactive:
            is_sel = entry.entry_id == selected_id
            highlighted = is_sel and focused
            title = Text()
            title.append("▶ " if is_sel else "  ", style="bold" if highlighted else "bold cyan")
            title.append_text(cells[0])
            table.add_row(title, *cells[1:], style=SELECTED_ROW_STYLE if highlighted else "")
        else:
            table.add_row(*cells)
    if interactive and not entries:
        table.add_row("No recorded activity for this agent.", "", "")
    return table


def section_rule(title: str, width: int, *, note: str = "", focused: bool = False) -> Text:
    """The one section separator used across the live TUI: ``─ Title · note ─────``.

    A focused section (the one W/S acts on) starts with ``▌`` and is drawn in
    bold cyan; every other rule is dim. The rule always fills ``width``.
    """
    label = f" {title}" + (f" · {note}" if note else "") + " "
    style = "bold cyan" if focused else "dim"
    rule = Text("▌" if focused else "─", style=style)
    rule.append(label, style=style)
    rule.append("─" * max(3, width - 1 - len(label)), style=style)
    rule.no_wrap = True
    rule.truncate(max(1, width), overflow="ellipsis")
    return rule


def _section_rule(title: str, width: int) -> Text:
    return section_rule(title, width)


ORDERED_TIMELINE_KINDS = [
    ("run", "command", "commands"),
    ("edit", "edit", "edits"),
    ("tool", "tool", "tools"),
    ("search", "search", "searches"),
    ("subagent", "subagent", "subagents"),
    ("think", "thinking", "thinking"),
    ("error", "error", "errors"),
    ("compact", "compaction", "compactions"),
]


TIMELINE_DURATION_GAP = 4
# Durations right-align in a fixed column ("59m59s"), so every row's text stops at the same place.
TIMELINE_DURATION_COL = 6


def _timeline_item_line(item: TurnItem, width: int) -> Text:
    clock = neutralize(format_clock(item.started_at), single_line=True)
    clock_col = f"{clock:<8}"
    kind_col = f"{neutralize(item.kind, single_line=True):<8}"

    if item.kind == "think":
        text = ""
    else:
        text = neutralize(item.text, single_line=True) if item.text else ""
        if item.truncated and not text.endswith("…"):
            text += "…"

    dur_str = ""
    if item.duration_seconds is not None and item.duration_seconds > 0:
        dur_str = format_duration(item.duration_seconds)
    dur_len = len(dur_str)

    prefix = f"{clock_col}  {kind_col}  "
    prefix_len = len(prefix)

    is_running = item.status == "running"
    running_tag = " (running)" if text else ("(running)" if is_running else "")
    running_len = len(running_tag) if is_running else 0

    if not text and not is_running and dur_len == 0:
        return Text(f"{clock_col}  {kind_col}")

    # Keep a clear gutter between the text and the duration column.
    min_gap = TIMELINE_DURATION_GAP if dur_len > 0 else 0
    avail_width = width - max(dur_len, TIMELINE_DURATION_COL) - min_gap if dur_len > 0 else width
    max_text_len = avail_width - prefix_len - running_len

    fitted_text = Text(text)
    fitted_text.truncate(max(0, max_text_len), overflow="ellipsis")
    text = fitted_text.plain

    current_left_len = prefix_len + fitted_text.cell_len + running_len
    line = Text(prefix)
    if text:
        line.append(text)
    if is_running:
        line.append(running_tag, style="yellow")
    if dur_len > 0:
        pad_len = max(min_gap, width - current_left_len - dur_len)
        line.append(" " * pad_len)
        line.append(dur_str)
    return line


def _wrap_plain(text: str, width: int) -> list[str]:
    """Hard-wrap recorded text (newlines kept) into lines of at most ``width`` cells."""
    width = max(1, width)
    out: list[str] = []
    for para in text.splitlines() or [""]:
        out.extend(textwrap.wrap(para, width, break_long_words=True, break_on_hyphens=False) or [""])
    return out


TIMELINE_TEXT_LABELS = {
    "run": "Command",
    "edit": "Edit",
    "tool": "Tool call",
    "search": "Search",
    "subagent": "Task",
    "think": "Thinking",
    "error": "Error",
    "compact": "Compaction",
}


def _full_text_lines(text: str, truncated: bool) -> list[Text]:
    """Recorded text in full, newlines kept; Rich wraps long lines to the page width."""
    lines = [Text(line) for line in neutralize(text).rstrip().splitlines()] or [Text("")]
    if truncated:
        lines.append(Text("… recorded text was cut at the retention limit", style="dim"))
    return lines


def _timeline_item_detail(item: TurnItem, width: int) -> list[Text]:
    clean = lambda value: neutralize(str(value), single_line=True)
    header = f"{clean(item.kind)} · {clean(item.status)}"
    if item.duration_seconds:
        header += f" · {format_duration(item.duration_seconds)}"
    if item.started_at:
        span = clean(format_clock(item.started_at))
        if item.completed_at:
            span += f"–{clean(format_clock(item.completed_at))}"
        header += f" · {span}"
    lines = [Text(header, style="bold")]
    lines.append(section_rule(TIMELINE_TEXT_LABELS.get(item.kind, "Recorded text"), width))
    if item.text and item.kind != "think":
        lines.extend(_full_text_lines(item.text, item.truncated))
    else:
        lines.append(Text("No text recorded for this action.", style="dim"))
    return lines


def _tool_detail_page(detail: ToolCallDetail | None, text: str, width: int) -> list[Text]:
    if detail is None:  # the thinking summary line of a model step
        return [Text("Thinking", style="bold"), section_rule("Recorded", width), Text(text)]
    lines = [Text(neutralize(TOOL_VERBS.get(detail.name, detail.name), single_line=True), style="bold")]
    recorded = [(key, label) for key, label in DETAIL_FIELD_LABELS if detail.fields.get(key)]
    if not recorded:
        lines.append(section_rule("Recorded", width))
        lines.append(Text("details unavailable (no fields recorded)", style="dim"))
    for key, label in recorded:
        lines.append(section_rule(label, width))
        lines.extend(_full_text_lines(detail.fields[key], key in detail.truncated))
    lines.append(section_rule("Details", width))
    lines.append(Text(f"Tool: {neutralize(detail.name, single_line=True)}", style="dim"))
    return lines


@dataclass
class ActionsView:
    """Selection state for the scrollable 'What it did' list.

    Inputs: ``selected`` item index, ``scroll`` (first visible item), ``rows``
    (line budget for items), ``focused``. After rendering, ``selected`` and
    ``scroll`` are clamped, ``count`` is the item total, and ``selected_line``
    / ``selected_height`` locate the selected item within the returned lines.
    """

    selected: int = 0
    scroll: int = 0
    rows: int = 10
    focused: bool = True
    count: int = 0
    selected_line: int | None = None
    selected_height: int = 0
    range_label: str = ""


def _action_items(step: ModelStep, width: int) -> tuple[str | None, list[tuple[Text, Any]]]:
    """Counts summary plus (one-line row, full-detail page builder) per action item."""
    if step.timeline:
        kind_counts: dict[str, int] = {}
        for item in step.timeline:
            kind_counts[item.kind] = kind_counts.get(item.kind, 0) + 1
        counts_parts: list[str] = []
        for kind, singular, plural in ORDERED_TIMELINE_KINDS:
            c = kind_counts.get(kind, 0)
            if c > 0:
                counts_parts.append(f"{c} {singular if c == 1 else plural}")
        known = {k for k, _, _ in ORDERED_TIMELINE_KINDS}
        counts_parts.extend(f"{c} {kind}" for kind, c in kind_counts.items() if kind not in known and c > 0)
        items = [
            (_timeline_item_line(item, width), (lambda w, it=item: _timeline_item_detail(it, w)))
            for item in step.timeline
        ]
        return (" · ".join(counts_parts) or None), items

    if step.activity_kind == "model_step" and (step.tool_details or step.tool_calls or step.thinking_tokens > 0):
        tool_names = [d.name for d in step.tool_details] if step.tool_details else list(step.tool_calls)
        counts: dict[str, int] = {}
        for name in tool_names:
            counts[name] = counts.get(name, 0) + 1
        entries: list[tuple[ToolCallDetail | None, str]] = []
        if step.thinking_tokens > 0:
            entries.append((None, f"think  {step.thinking_tokens:,} thinking tokens"))
        details = step.tool_details or [ToolCallDetail(name=name) for name in step.tool_calls]
        entries.extend((d, describe_tool(d)) for d in details)

        def collapsed(t: str) -> Text:
            line = Text(t)
            line.truncate(width, overflow="ellipsis")
            return line

        items = [(collapsed(t), (lambda w, d=d, t=t: _tool_detail_page(d, t, w))) for d, t in entries]
        summary = " · ".join(f"{count} {TOOL_VERBS.get(name, name)}" for name, count in counts.items())
        return (summary or None), items

    return None, []


def build_action_detail_lines(step: ModelStep | None, index: int, width: int = 80) -> list[Text] | None:
    """Full page for one 'What it did' item, or None when that item no longer exists."""
    if step is None:
        return None
    _, items = _action_items(step, width)
    if not 0 <= index < len(items):
        return None
    return items[index][1](width)


def _actions_lines(step: ModelStep, width: int = 80, view: ActionsView | None = None) -> list[Text]:
    """Render 'What it did' for an activity step: one truncated line per item.

    With a view, only a window of ``view.rows`` items is shown, each row gets a
    selection marker column, and the selected row is highlighted while the list
    is focused. Selected and unselected rows truncate identically; the full
    text lives on the action detail page.
    """
    if view is not None:
        width = max(10, width - 2)  # room for the "▶ " marker column
    summary, items = _action_items(step, width)
    if not items:
        if view is not None:
            view.count = 0
            view.selected_line = None
        return [Text("No actions recorded for this turn.", style="dim")]

    summary = neutralize(summary, single_line=True) if summary else ""
    if view is None:
        return ([Text(summary)] if summary else []) + [collapsed for collapsed, _ in items]

    count = len(items)
    view.count = count
    sel = max(0, min(count - 1, view.selected))
    view.selected = sel
    rows = max(1, view.rows)
    start = window_start(count, sel, view.scroll, rows)
    view.scroll = start
    shown = items[start : start + rows]
    # Summary: at most two lines, so it never pushes the list off screen.
    head = _wrap_plain(summary, width + 2) if summary else []
    if len(head) > 2:
        head = head[:2]
        head[1] = head[1][: max(0, width + 1)] + "…"
    lines = [Text(h) for h in head]
    if count <= len(shown):
        view.range_label = ""
    elif len(shown) == 1:
        view.range_label = f"{start + 1} of {count}"
    else:
        view.range_label = f"{start + 1}–{start + len(shown)} of {count}"
    for offset, (collapsed, _) in enumerate(shown):
        is_sel = start + offset == sel
        highlighted = is_sel and view.focused
        row = Text("▶ " if is_sel else "  ", style="bold" if highlighted else "bold cyan")
        row.append_text(collapsed)
        if highlighted:
            row.stylize(SELECTED_ROW_STYLE)
        if is_sel:
            view.selected_line = len(lines)
            view.selected_height = 1
        lines.append(row)
    return lines


def build_activity_detail_lines(
    entry: ActivityEntry, node: ThreadTelemetryNode | None = None, *,
    width: int = 80, prompt_expanded: bool = False, actions_view: ActionsView | None = None,
) -> list[Text]:
    """Full detail for one activity entry; the unit of accounting is stated explicitly."""
    step = entry.detail
    lines: list[Text] = []

    def field(label: str, value: str) -> None:
        line = Text()
        line.append(f"{label}: ", style="bold cyan")
        line.append(value)
        lines.append(line)

    if entry.launch is not None:
        return _launch_detail_lines(entry, node, field, lines)

    clean = lambda value: neutralize(str(value), single_line=True)
    header = f"{clean(step.row_label if step is not None else entry.title)} · {clean(entry.status)} · {format_duration(entry.duration_seconds)}"
    if step is not None and step.started_at and step.completed_at:
        header += f" · {clean(format_clock(step.started_at))}–{clean(format_clock(step.completed_at))}"
    if node is not None:
        header += f" · {clean(node.model or 'unknown')}"
    lines.append(Text(header, style="bold"))
    lines.append(_section_rule("Tokens", width))
    if step is not None and entry.usage_status == "unavailable":
        lines.append(Text("usage unavailable (not recorded; not zero)"))
    elif entry.total_tokens is None or step is None:
        lines.append(Text("usage not recorded yet"))
    else:
        if entry.usage_provenance == "cumulative_fallback":
            field("Usage source", "cumulative thread usage fallback; may include earlier turns")
        elif entry.usage_status == "partial":
            field("Usage source", "partial usage recorded")
        total_input = step.input_tokens + step.cache_read_tokens + step.cache_creation_tokens
        new_input = step.input_tokens + step.cache_creation_tokens
        bar_width = max(10, min(40, width - 36))
        read_width = round(bar_width * step.cache_read_tokens / total_input) if total_input else 0
        bar = Text(f"{' Input':<20} {total_input:>10,}  ")
        bar.append("█" * read_width, style="cyan")
        bar.append("█" * (bar_width - read_width), style="yellow")
        lines.append(bar)
        read_pct = f"{step.cache_read_tokens / total_input * 100:.1f}%" if total_input else "—"
        new_pct = f"{new_input / total_input * 100:.1f}%" if total_input else "—"
        lines.append(Text(f"{'   ├ re-read context':<20} {step.cache_read_tokens:>10,}  {read_pct:>6}   cache read"))
        breakdown = f"uncached {step.input_tokens:,}" if step.activity_kind == "model_step" else f"cache write {step.cache_creation_tokens:,} + uncached {step.input_tokens:,}"
        lines.append(Text(f"{'   └ new this turn':<20} {new_input:>10,}  {new_pct:>6}   {breakdown}"))
        lines.append(Text(f"{' Output':<20} {step.output_tokens:>10,}  thinking {step.thinking_tokens:,} · response {step.response_tokens:,}"))
        lines.append(Text(f"{' Total':<20} {step.total_tokens:>10,}"))
        if step.context_used_tokens is not None:
            context = Text(" Context at end  ")
            context.append_text(Text.from_markup(render_context_bar(step.context_used_tokens, step.context_max_tokens, width=20)))
            lines.append(context)
    if step is not None:
        rule_index = len(lines)
        lines.append(_section_rule("What it did", width))
        actions_start = len(lines)
        lines.extend(_actions_lines(step, width, actions_view))
        if actions_view is not None:
            lines[rule_index] = section_rule(
                "What it did", width, note=actions_view.range_label,
                focused=actions_view.focused and actions_view.count > 0,
            )
            if actions_view.selected_line is not None:
                actions_view.selected_line += actions_start
        user_rec = next((r for r in step.context_texts if r.role == "user"), None) if step.context_texts else None
        if user_rec is not None and user_rec.text:
            prompt_lines = neutralize(user_rec.text).splitlines()
            if prompt_lines:
                lines.append(_section_rule("Prompt", width))
                if not prompt_expanded and len(prompt_lines) > 3:
                    for text_line in prompt_lines[:3]:
                        lines.append(Text(f"  {text_line}"))
                    remaining = len(prompt_lines) - 3
                    lines.append(Text(f"  … {remaining} more lines · P to expand", style="dim"))
                else:
                    for text_line in prompt_lines:
                        lines.append(Text(f"  {text_line}"))
                    if user_rec.truncated:
                        lines.append(Text("  … (truncated)", style="dim"))
    lines.append(_section_rule("Details", width))
    if node is not None:
        lines.append(Text(f"Agent: {clean(node.title)} ({clean(node.thread_id[:8])})", style="dim"))
        lines.append(Text(f"Driver / model: {clean(node.driver)} · {clean(node.model or 'unknown')}", style="dim"))
    if step is not None:
        if step.provider_turn_id:
            lines.append(Text(f"Provider turn id: {clean(step.provider_turn_id)} (ordinal {step.step_index})", style="dim"))
        else:
            lines.append(Text(f"Step index: {step.step_index}", style="dim"))
        lines.append(Text(f"Source: {clean(step.source)}", style="dim"))
    lines.append(Text(f"Unit of accounting: {clean(entry.accounting_scope)}", style="dim"))
    lines.append(Text(neutralize(entry.scope_note), style="dim"))
    return lines


ORIGIN_LABELS = {"app_owned": "app-owned", "native": "native", "lineage": "thread lineage"}


def _launch_detail_lines(entry: ActivityEntry, node: ThreadTelemetryNode | None, field: Any, lines: list[Text]) -> list[Text]:
    launch = entry.launch
    assert launch is not None
    lines.append(Text("Subagent launch", style="bold"))
    lines.append(Text(""))
    if node is not None:
        field("Parent agent", f"{neutralize(node.title, single_line=True)} ({node.thread_id[:8]})")
    field("Task", neutralize(launch.title) if launch.title else "not recorded")
    field("Child thread", neutralize(launch.child_thread_id, single_line=True) if launch.child_thread_id else "not recorded")
    if launch.child_native_id and launch.child_native_id != launch.child_thread_id:
        field("Child conversation", neutralize(launch.child_native_id, single_line=True))
    field("Origin", ORIGIN_LABELS.get(launch.origin, neutralize(launch.origin, single_line=True)))
    field("Status", neutralize(launch.status, single_line=True))
    if launch.launched_at and launch.launch_time_source == "child thread created":
        # Lineage only knows when the child thread was created, not when it was launched.
        field("Child created", f"{neutralize(launch.launched_at, single_line=True)} (launch time not recorded)")
    elif launch.launched_at:
        field("Launched", f"{neutralize(launch.launched_at, single_line=True)} ({launch.launch_time_source})")
    else:
        field("Launched", "launch time not recorded")
    if launch.parent_step_index is not None:
        tool = neutralize(launch.tool_name or "tool", single_line=True)
        field("Recorded in", f"Model step #{launch.parent_step_index} ({tool})")
    field("Discovered via", ", ".join(f"Source {src}" for src in launch.sources) or "unknown")
    lines.append(Text(""))
    field("Unit of accounting", entry.accounting_scope)
    lines.append(Text(entry.scope_note, style="dim"))
    field("Tokens", "not applicable (launch marker)")
    lines.append(Text(""))
    if entry.child_thread_id:
        lines.append(Text("Enter: Open subagent (full overview)", style="bold green"))
    else:
        lines.append(Text("Child telemetry not found; this launch can be inspected but not opened.", style="dim"))
    return lines


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

    scopes = [
        (turn_label, turn_val),
        ("Whole chat · this agent", self_val),
        (rollup_label, rollup_val),
    ]
    label_width = max(len(label) for label, _ in scopes)
    lines: list[Text] = []
    for label, value in scopes:
        line = Text()
        line.append(f"{label}:", style="bold cyan")
        line.append(" " * (label_width - len(label) + 1))
        line.append("│ ", style="cyan")
        line.append(value.replace(", ", " · "))
        line.no_wrap = True
        lines.append(line)

    return lines


def build_dashboard_renderable(node: ThreadTelemetryNode, max_steps: int = 20) -> Group:
    """Build Rich renderable Group with Header, Metrics, Subagents, and Steps."""
    status_str = f"[{get_status_style(node.status) or 'bold'}]{node.status.upper()}[/]"
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

    # 4. Activity feed - Node's own entries only (never combined_steps)
    stream_table = build_activity_table(
        build_activity_entries(node, max_steps),
        title="Recent Steps / Turns",
        box_style=box.ROUNDED,
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
