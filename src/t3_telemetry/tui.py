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
    activity_scroll: int = 0
    child_ids: list[str] = field(default_factory=list)
    selected_child_id: str | None = None
    child_scroll: int = 0
    max_activity_scroll: int = 0


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
    def detail_scroll(self) -> int:
        if self.stack:
            return self.stack[-1].activity_scroll
        return 0

    @detail_scroll.setter
    def detail_scroll(self, val: int) -> None:
        if self.stack:
            self.stack[-1].activity_scroll = val

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

    def move(self, delta: int) -> None:
        if self.stack:
            screen = self.stack[-1]
            if screen.focus == "activity":
                new_scroll = screen.activity_scroll + delta
                screen.activity_scroll = max(0, min(screen.max_activity_scroll, new_scroll))
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
            target = screen.selected_child_id
            if target and screen.child_ids and (screen.focus == "subagents" or screen.selected_child_id):
                self.stack.append(ScreenState(node_id=target))

    def escape(self) -> None:
        if self.stack:
            self.stack.pop()


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
        if key in ("UP", "k"):
            self.navigation.move(-1)
        elif key in ("DOWN", "j"):
            self.navigation.move(1)
        elif key in ("TAB", "	"):
            self.navigation.tab()
        elif key in ("ENTER", chr(13), chr(10)):
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
