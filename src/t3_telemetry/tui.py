"""Navigation state and injectable browser driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from t3_telemetry.context_history import ContextHistoryStore
from t3_telemetry.usage import PERIODS


@dataclass
class ScreenState:
    node_id: str
    title: str = ""
    focus: str = "overview"  # session tab
    child_ids: list[str] = field(default_factory=list)
    selected_child_id: str | None = None
    child_scroll: int = 0
    # Activity identity is separate from node identity: an activity is never a stack entry.
    activity_ids: list[str] = field(default_factory=list)
    selected_activity_id: str | None = None
    activity_offset: int = 0
    detail_activity_id: str | None = None
    detail_scroll: int = 0
    max_detail_scroll: int = 0
    detail_prompt_expanded: bool = False
    # "What it did" list inside the detail: W/S moves the highlighted item while
    # focus is "actions"; Tab switches to "page", where W/S scrolls the page.
    detail_focus: str = "actions"
    detail_item: int = 0
    detail_item_scroll: int = 0
    detail_item_count: int = 0
    # Full-text page for the highlighted action (E/Enter opens, Esc closes).
    action_open: bool = False
    action_scroll: int = 0
    max_action_scroll: int = 0
    # Launch entry ID -> child node ID, only for launches whose child is present.
    activity_targets: dict[str, str] = field(default_factory=dict)

    def reset_detail(self) -> None:
        self.detail_scroll = 0
        self.detail_prompt_expanded = False
        self.detail_focus = "actions"
        self.detail_item = 0
        self.detail_item_scroll = 0
        self.detail_item_count = 0
        self.close_action()

    def close_action(self) -> None:
        self.action_open = False
        self.action_scroll = 0
        self.max_action_scroll = 0


@dataclass
class NavigationState:
    ids: list[str] = field(default_factory=list)
    selected_id: str | None = None
    stack: list[ScreenState] = field(default_factory=list)
    pinned_summary_id: str | None = None
    list_scroll: int = 0
    notice: str | None = None
    home_focus: str = "threads"
    selected_period: str = "today"
    period: str | None = None
    top_ids: list[str] = field(default_factory=list)
    selected_top_id: str | None = None
    top_scroll: int = 0
    measure: str = "tokens"
    graph_split: str = "provider"

    @property
    def detail_id(self) -> str | None:
        if self.stack:
            return self.stack[-1].node_id
        return None

    @detail_id.setter
    def detail_id(self, val: str | None) -> None:
        if val is None:
            self.stack.clear()
        else:
            if self.stack and self.stack[-1].node_id == val:
                return
            self.stack = [ScreenState(node_id=val)]

    @property
    def current_screen(self) -> ScreenState | None:
        return self.stack[-1] if self.stack else None

    def reconcile(self, ids: list[str]) -> None:
        prior = self.selected_id
        old_index = self.ids.index(prior) if (prior and prior in self.ids) else 0
        self.ids = list(ids)
        if prior in ids:
            self.selected_id = prior
        elif ids:
            self.selected_id = ids[min(old_index, len(ids) - 1)]
        else:
            self.selected_id = None

        if self.stack and self.stack[0].node_id not in ids:
            disappeared = self.stack[0].title or self.stack[0].node_id[:8]
            self.notice = f"Thread '{disappeared}' is no longer available"
            self.stack.clear()

    def reconcile_tree(self, available_ids: set[str], titles: dict[str, str] | None = None) -> None:
        if not self.stack:
            return
        titles = titles or {}
        new_stack = [self.stack[0]]
        for screen in self.stack[1:]:
            if screen.node_id in available_ids:
                if screen.node_id in titles and titles[screen.node_id]:
                    screen.title = titles[screen.node_id]
                new_stack.append(screen)
            else:
                disappeared = screen.title or screen.node_id[:8]
                ancestor = new_stack[-1].title or new_stack[-1].node_id[:8]
                self.notice = f"'{disappeared}' disappeared; returned to '{ancestor}'"
                break
        self.stack = new_stack

    def sync_activities(self, ids: list[str], targets: dict[str, str] | None = None) -> None:
        """Reconcile the current screen's activity selection with the latest entry IDs.

        ``targets`` maps launch entries to the child they open.
        """
        screen = self.current_screen
        if screen is None:
            return
        screen.activity_targets = dict(targets or {})
        prior_ids = screen.activity_ids
        screen.activity_ids = list(ids)
        selected = screen.selected_activity_id
        if not ids:
            screen.selected_activity_id = None
        elif selected in ids:
            pass
        elif selected is None:
            screen.selected_activity_id = ids[-1]  # default to the most recent entry
        else:
            old_index = prior_ids.index(selected) if selected in prior_ids else len(ids) - 1
            screen.selected_activity_id = ids[min(old_index, len(ids) - 1)]
        if screen.detail_activity_id and screen.detail_activity_id not in ids:
            screen.detail_activity_id = None
            screen.reset_detail()
            self.notice = "Activity is no longer available"

    def sync_children(self, ids: list[str]) -> None:
        """Reconcile the current screen's subagent list."""
        screen = self.current_screen
        if screen is None:
            return
        screen.child_ids = list(ids)
        if screen.child_ids and screen.selected_child_id not in screen.child_ids:
            screen.selected_child_id = screen.child_ids[0]

    def sync_top(self, ids: list[str]) -> None:
        prior = self.selected_top_id
        old_index = self.top_ids.index(prior) if prior in self.top_ids else 0
        self.top_ids = list(ids)
        if prior in ids:
            self.selected_top_id = prior
        elif ids:
            self.selected_top_id = ids[min(old_index, len(ids) - 1)]
        else:
            self.selected_top_id = None

    def move(self, delta: int) -> None:
        if self.stack:
            screen = self.stack[-1]
            if screen.detail_activity_id and screen.action_open:
                screen.action_scroll = max(0, min(screen.max_action_scroll, screen.action_scroll + delta))
            elif screen.detail_activity_id:
                if screen.detail_focus == "actions" and screen.detail_item_count > 0:
                    screen.detail_item = max(0, min(screen.detail_item_count - 1, screen.detail_item + delta))
                else:
                    screen.detail_scroll = max(0, min(screen.max_detail_scroll, screen.detail_scroll + delta))
            elif screen.focus == "activity":
                if screen.activity_ids:
                    ids = screen.activity_ids
                    curr_idx = ids.index(screen.selected_activity_id) if screen.selected_activity_id in ids else len(ids) - 1
                    screen.selected_activity_id = ids[max(0, min(len(ids) - 1, curr_idx + delta))]
            elif screen.focus == "overview":
                if screen.child_ids:
                    curr_idx = (
                        screen.child_ids.index(screen.selected_child_id)
                        if (screen.selected_child_id and screen.selected_child_id in screen.child_ids)
                        else 0
                    )
                    new_idx = max(0, min(len(screen.child_ids) - 1, curr_idx + delta))
                    screen.selected_child_id = screen.child_ids[new_idx]
        elif self.period:
            if self.top_ids:
                i = self.top_ids.index(self.selected_top_id) if self.selected_top_id in self.top_ids else 0
                self.selected_top_id = self.top_ids[max(0, min(len(self.top_ids) - 1, i + delta))]
        elif self.home_focus == "usage":
            i = PERIODS.index(self.selected_period)
            self.selected_period = PERIODS[max(0, min(len(PERIODS) - 1, i + delta))]
        elif self.ids:
            i = self.ids.index(self.selected_id) if (self.selected_id and self.selected_id in self.ids) else 0
            self.selected_id = self.ids[max(0, min(len(self.ids) - 1, i + delta))]

    def tab(self) -> None:
        if self.stack:
            screen = self.stack[-1]
            if screen.detail_activity_id:
                if screen.detail_item_count > 0 and not screen.action_open:
                    screen.detail_focus = "page" if screen.detail_focus == "actions" else "actions"
                return
            if screen.focus == "activity":
                screen.focus = "overview"
                if screen.child_ids and screen.selected_child_id not in screen.child_ids:
                    screen.selected_child_id = screen.child_ids[0]
            else:
                screen.focus = "activity"

        elif not self.period:
            self.home_focus = "threads" if self.home_focus == "usage" else "usage"

    def enter(self) -> None:
        if not self.stack:
            if self.period:
                if self.selected_top_id and self.selected_top_id.startswith("session:"):
                    self.notice = "Not a T3 thread: no detail view"
                elif self.selected_top_id:
                    self.stack = [ScreenState(node_id=self.selected_top_id)]
            elif self.home_focus == "usage":
                self.period = self.selected_period
                self.selected_top_id = None
                self.top_scroll = 0
            elif self.selected_id:
                self.stack = [ScreenState(node_id=self.selected_id)]
        else:
            screen = self.stack[-1]
            if screen.detail_activity_id:
                # Open subagent from a launch detail; the parent keeps its detail for Escape.
                target = screen.activity_targets.get(screen.detail_activity_id)
                if target and target in screen.child_ids:
                    self.stack.append(ScreenState(node_id=target))
                elif not screen.action_open and screen.detail_focus == "actions" and screen.detail_item_count > 0:
                    screen.action_open = True
                    screen.action_scroll = 0
                return
            if screen.focus == "overview":
                target = screen.selected_child_id
                if target and target in screen.child_ids:
                    self.stack.append(ScreenState(node_id=target))
            elif screen.focus == "activity":
                target = screen.selected_activity_id
                if target and target in screen.activity_ids:
                    screen.detail_activity_id = target
                    screen.reset_detail()

    def escape(self) -> None:
        if self.stack:
            screen = self.stack[-1]
            if screen.detail_activity_id and screen.action_open:
                screen.close_action()  # back to the list, highlight unchanged
            elif screen.detail_activity_id:
                screen.detail_activity_id = None
                screen.reset_detail()
            else:
                self.stack.pop()
        elif self.period:
            self.period = None

    def toggle_prompt(self) -> None:
        screen = self.current_screen
        if screen is not None and screen.detail_activity_id and not screen.action_open and screen.detail_activity_id not in screen.activity_targets:
            screen.detail_prompt_expanded = not screen.detail_prompt_expanded

    def toggle_measure(self) -> None:
        self.measure = "usd" if self.measure == "tokens" else "tokens"

    def toggle_graph_split(self) -> None:
        if self.period and not self.stack:
            self.graph_split = "scope" if self.graph_split == "provider" else "provider"


def table_header_lines(rows: int) -> int:
    """Interactive lists reserve one column-header line; the section rule above is counted separately."""
    return 1


def table_geometry(lines: int) -> tuple[int, int]:
    """Split the budget into a column header and visible data rows."""
    if lines <= 0:
        return 0, 0
    return min(1, lines), max(0, lines - 1)


def window_start(total: int, selected_index: int, offset: int, rows: int) -> int:
    """First visible index for a list window of ``rows`` that keeps the selection visible."""
    rows = max(1, rows)
    start = min(max(0, offset), selected_index)
    if selected_index >= start + rows:
        start = selected_index - rows + 1
    return max(0, min(start, max(0, total - rows)))


class TerminalAdapter:
    """Input and terminal lifecycle boundary; Rich Live owns all drawing."""

    def begin(self) -> None:
        pass

    def end(self) -> None:
        pass

    def read_key(self, timeout: float) -> str | None:
        return None

    def size(self) -> tuple[int, int]:
        return (24, 80)


class TelemetryBrowser:
    def __init__(
        self,
        adapter: TerminalAdapter,
        navigation: NavigationState | None = None,
        history_store: ContextHistoryStore | None = None,
    ) -> None:
        self.adapter = adapter
        self.navigation = navigation or NavigationState()
        self.history_store = history_store or ContextHistoryStore()
        self.last_signature: Any = None

    def handle_key(self, key: str) -> bool:
        if self.navigation.notice:
            self.navigation.notice = None
        if key in ("UP", "k", "w", "W"):
            self.navigation.move(-1)
        elif key in ("DOWN", "j", "s", "S"):
            self.navigation.move(1)
        elif key in ("TAB", "SHIFT_TAB", "	"):
            self.navigation.tab()
        elif key in ("ENTER", "e", "E", chr(13), chr(10)):
            self.navigation.enter()
        elif key in ("p", "P"):
            self.navigation.toggle_prompt()
        elif key in ("$", "c", "C"):
            self.navigation.toggle_measure()
        elif key in ("g", "G"):
            self.navigation.toggle_graph_split()
        elif key in ("ESC", chr(27)):
            self.navigation.escape()
        elif key in ("q", chr(3)):
            return False
        return True


class Viewport:
    """Bounded viewport renderable that clips content to height and supports vertical scrolling."""

    def __init__(
        self,
        renderable: object,
        height: int,
        scroll: int = 0,
        width: int | None = None,
        on_max_scroll: Callable[[int], None] | None = None,
    ) -> None:
        self.renderable = renderable
        self.height = max(1, height)
        self.scroll = max(0, scroll)
        self.width = width
        self.on_max_scroll = on_max_scroll

    def __rich_console__(self, console: Any, options: Any):
        from rich.segment import Segment

        target_width = self.width or options.max_width
        opts = options.update(width=target_width, height=None)
        segments = console.render(self.renderable, opts)
        lines = list(Segment.split_lines(segments))
        max_scroll = max(0, len(lines) - self.height)
        if self.on_max_scroll is not None:
            self.on_max_scroll(max_scroll)
        scroll = min(self.scroll, max_scroll)
        visible_lines = lines[scroll : scroll + self.height]
        for line in visible_lines:
            yield from line
            yield Segment.line()
