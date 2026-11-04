"""POSIX terminal input adapter with buffered escape sequence parsing."""

from __future__ import annotations

import io

import os
import selectors
import shutil
import sys
import termios
import time
import tty
from typing import TextIO

from .tui import TerminalAdapter


class PosixTerminalAdapter(TerminalAdapter):
    ESC_TIMEOUT: float = 0.05
    SEQUENCES: dict[bytes, str] = {
        b"\x1b[Z": "SHIFT_TAB",
        b"[A": "UP",
        b"OA": "UP",
        b"[B": "DOWN",
        b"OB": "DOWN",
        b"[C": "RIGHT",
        b"OC": "RIGHT",
        b"[D": "LEFT",
        b"OD": "LEFT",
    }

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdin
        self.buffer = bytearray()
        self.saved: list[int | list[bytes | int]] | None = None
        self.fd: int | None = None
        self._esc_time: float | None = None
        self._eof: bool = False

    def begin(self) -> None:
        if self.saved is not None:
            return
        try:
            if not hasattr(self.stream, "fileno"):
                self.fd = None
                return
            self.fd = self.stream.fileno()
            is_tty = self.stream.isatty() if hasattr(self.stream, "isatty") else os.isatty(self.fd)
            if not is_tty:
                self.fd = None
                return
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except (io.UnsupportedOperation, AttributeError, ValueError, OSError, termios.error):
            self.fd = None
            self.saved = None

    def end(self) -> None:
        if self.saved is not None and self.fd is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            except OSError:
                pass
            finally:
                self.saved = None

    def size(self) -> tuple[int, int]:
        s = shutil.get_terminal_size((80, 24))
        return s.lines, s.columns

    def read_key(self, timeout: float = 0.0) -> str | None:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            # 1. Complete sequence matches
            for seq, key in self.SEQUENCES.items():
                if self.buffer.startswith(seq):
                    del self.buffer[: len(seq)]
                    self._esc_time = None
                    return key

            # 2. Normal non-ESC byte at head of buffer
            if self.buffer and self.buffer[0] != 0x1B:
                byte = self.buffer.pop(0)
                if byte in (0x0D, 0x0A):
                    return "ENTER"
                if byte == 0x09:
                    return "TAB"
                if byte == 0x03:
                    return ""
                return bytes([byte]).decode(errors="replace")

            # 3. ESC prefix in buffer
            if self.buffer and self.buffer[0] == 0x1B:
                prefixes = [s for s in self.SEQUENCES if s.startswith(self.buffer)]
                now = time.monotonic()
                if not prefixes or len(self.buffer) >= 4:
                    self.buffer.pop(0)
                    self._esc_time = None
                    return "ESC"
                if self._esc_time is None:
                    self._esc_time = now
                elif (now - self._esc_time) >= self.ESC_TIMEOUT:
                    self.buffer.pop(0)
                    self._esc_time = None
                    return "ESC"

            # 4. If EOF and nothing left or ESC timed out
            if self._eof:
                if self.buffer and self.buffer[0] == 0x1B:
                    self.buffer.pop(0)
                    self._esc_time = None
                    return "ESC"
                return None

            # 5. Wait for fd read
            now = time.monotonic()
            if self.buffer and self.buffer[0] == 0x1B and self._esc_time is not None:
                remaining_esc = max(0.0, (self._esc_time + self.ESC_TIMEOUT) - now)
                wait = min(remaining_esc, max(0.0, deadline - now))
            else:
                wait = max(0.0, deadline - now)

            if wait <= 0.0:
                if self.buffer and self.buffer[0] == 0x1B and self._esc_time is not None:
                    if (now - self._esc_time) >= self.ESC_TIMEOUT:
                        self.buffer.pop(0)
                        self._esc_time = None
                        return "ESC"
                return None

            if self.fd is None:
                return None

            with selectors.DefaultSelector() as sel:
                sel.register(self.fd, selectors.EVENT_READ)
                events = sel.select(wait)
                if events:
                    try:
                        chunk = os.read(self.fd, 4096)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        self._eof = True
                    else:
                        self.buffer.extend(chunk)
                else:
                    now = time.monotonic()
                    if self.buffer and self.buffer[0] == 0x1B and self._esc_time is not None:
                        if (now - self._esc_time) >= self.ESC_TIMEOUT:
                            self.buffer.pop(0)
                            self._esc_time = None
                            return "ESC"
                    return None
