"""Navigation state and injectable browser driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from t3_telemetry.context_history import ContextHistoryStore


@dataclass
class ScreenState:
    node_id: str
    title: str = ""
    focus: str = "activity"  # "activity" or "subagents"
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
    # Launch entry ID -> child node ID, only for launches whose child is present.
    activity_targets: dict[str, str] = field(default_factory=dict)


@dataclass
class NavigationState:
    ids: list[str] = field(default_factory=list)
    selected_id: str | None = None
    stack: list[ScreenState] = field(default_factory=list)
    pinned_summary_id: str | None = None
    list_scroll: int = 0
    notice: str | None = None

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
            screen.detail_scroll = 0
            self.notice = "Activity is no longer available"

    def sync_children(self, ids: list[str]) -> None:
        """Reconcile the current screen's subagent list; focus falls back to activity when it empties."""
        screen = self.current_screen
        if screen is None:
            return
        screen.child_ids = list(ids)
        if not screen.child_ids:
            screen.focus = "activity"
        elif screen.selected_child_id not in screen.child_ids:
            screen.selected_child_id = screen.child_ids[0]

    def move(self, delta: int) -> None:
        if self.stack:
            screen = self.stack[-1]
            if screen.detail_activity_id:
                screen.detail_scroll = max(0, min(screen.max_detail_scroll, screen.detail_scroll + delta))
            elif screen.focus == "activity":
                if screen.activity_ids:
                    ids = screen.activity_ids
                    curr_idx = ids.index(screen.selected_activity_id) if screen.selected_activity_id in ids else len(ids) - 1
                    screen.selected_activity_id = ids[max(0, min(len(ids) - 1, curr_idx + delta))]
            elif screen.focus == "subagents":
                if screen.child_ids:
                    curr_idx = (
                        screen.child_ids.index(screen.selected_child_id)
                        if (screen.selected_child_id and screen.selected_child_id in screen.child_ids)
                        else 0
                    )
                    new_idx = max(0, min(len(screen.child_ids) - 1, curr_idx + delta))
                    screen.selected_child_id = screen.child_ids[new_idx]
        elif self.ids:
            i = self.ids.index(self.selected_id) if (self.selected_id and self.selected_id in self.ids) else 0
            self.selected_id = self.ids[max(0, min(len(self.ids) - 1, i + delta))]

    def tab(self) -> None:
        if self.stack:
            screen = self.stack[-1]
            if screen.detail_activity_id:
                return
            if screen.child_ids:
                if screen.focus == "activity":
                    screen.focus = "subagents"
                    if not screen.selected_child_id or screen.selected_child_id not in screen.child_ids:
                        screen.selected_child_id = screen.child_ids[0]
                else:
                    screen.focus = "activity"

    def enter(self) -> None:
        if not self.stack:
            if self.selected_id:
                self.stack = [ScreenState(node_id=self.selected_id)]
        else:
            screen = self.stack[-1]
            if screen.detail_activity_id:
                # Open subagent from a launch detail; the parent keeps its detail for Escape.
                target = screen.activity_targets.get(screen.detail_activity_id)
                if target and target in screen.child_ids:
                    self.stack.append(ScreenState(node_id=target))
                return
            if screen.focus == "subagents":
                target = screen.selected_child_id
                if target and target in screen.child_ids:
                    self.stack.append(ScreenState(node_id=target))
            elif screen.focus == "activity":
                target = screen.selected_activity_id
                if target and target in screen.activity_ids:
                    screen.detail_activity_id = target
                    screen.detail_scroll = 0

    def escape(self) -> None:
        if self.stack:
            screen = self.stack[-1]
            if screen.detail_activity_id:
                screen.detail_activity_id = None
                screen.detail_scroll = 0
            else:
                self.stack.pop()


def table_header_lines(rows: int) -> int:
    """Interactive lists reserve a column header and its divider."""
    return 2


def table_geometry(lines: int) -> tuple[int, int]:
    """Split the budget into a header, divider, and visible data rows."""
    if lines <= 0:
        return 0, 0
    return min(2, lines), max(0, lines - 2)


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
