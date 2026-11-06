"""Read complete JSONL records with their byte offsets."""

import json
from pathlib import Path
from typing import Iterator


def read_jsonl(path: Path, offset: int = 0) -> Iterator[tuple[int, dict | None]]:
    with path.open("rb") as stream:
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeError, ValueError):
                value = None
            yield stream.tell(), value if isinstance(value, dict) else None
