"""Pure-Python protobuf wire format decoder and encoder helpers.

Handles decoding Antigravity SQLite steps.metadata blobs without third-party
protobuf runtime dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def decode_varint(data: bytes, pos: int = 0) -> tuple[int, int]:
    """Reads a base-128 varint from data starting at pos.

    Returns:
        (value, new_pos)

    Raises:
        ValueError: If data is truncated.
    """
    res = 0
    shift = 0
    data_len = len(data)
    while True:
        if pos >= data_len:
            raise ValueError("Truncated varint")
        b = data[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return res, pos


def encode_varint(value: int) -> bytes:
    """Encodes a non-negative integer as varint bytes."""
    if value < 0:
        raise ValueError("Negative varints not supported in unsigned encoder")
    if value == 0:
        return b"\x00"
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def encode_field(field_no: int, wire_type: int, payload: int | bytes) -> bytes:
    """Constructs a protobuf field tag and payload bytes (helper for tests)."""
    tag = (field_no << 3) | (wire_type & 0x07)
    tag_bytes = encode_varint(tag)
    if wire_type == 0:
        if not isinstance(payload, int):
            raise TypeError("Wire type 0 expects int payload")
        return tag_bytes + encode_varint(payload)
    if wire_type == 2:
        if not isinstance(payload, bytes):
            raise TypeError("Wire type 2 expects bytes payload")
        return tag_bytes + encode_varint(len(payload)) + payload
    if wire_type == 1:
        if not isinstance(payload, bytes) or len(payload) != 8:
            raise ValueError("Wire type 1 expects 8 bytes payload")
        return tag_bytes + payload
    if wire_type == 5:
        if not isinstance(payload, bytes) or len(payload) != 4:
            raise ValueError("Wire type 5 expects 4 bytes payload")
        return tag_bytes + payload
    raise ValueError(f"Unsupported wire type: {wire_type}")


def decode_message(data: bytes) -> dict[int, list[tuple[int, int | bytes]]]:
    """Decodes protobuf wire format message into {field_no: [(wire_type, val), ...]}.

    Gracefully terminates on malformed or truncated data, returning partially
    decoded fields.
    """
    fields: dict[int, list[tuple[int, int | bytes]]] = {}
    pos = 0
    data_len = len(data)

    try:
        while pos < data_len:
            tag, pos = decode_varint(data, pos)
            field_no = tag >> 3
            wire_type = tag & 0x07

            if wire_type == 0:  # varint
                val, pos = decode_varint(data, pos)
                fields.setdefault(field_no, []).append((wire_type, val))
            elif wire_type == 1:  # 64-bit
                if pos + 8 > data_len:
                    break
                val = data[pos : pos + 8]
                pos += 8
                fields.setdefault(field_no, []).append((wire_type, val))
            elif wire_type == 2:  # length-delimited
                length, pos = decode_varint(data, pos)
                if pos + length > data_len:
                    break
                val = data[pos : pos + length]
                pos += length
                fields.setdefault(field_no, []).append((wire_type, val))
            elif wire_type == 5:  # 32-bit
                if pos + 4 > data_len:
                    break
                val = data[pos : pos + 4]
                pos += 4
                fields.setdefault(field_no, []).append((wire_type, val))
            else:
                # Unsupported wire type (e.g. groups), abort parsing gracefully
                break
    except (IndexError, ValueError):
        # Truncated or malformed message
        pass

    return fields


def decode_timestamp(blob: bytes | None) -> tuple[float | None, str | None]:
    """Decodes a protobuf google.protobuf.Timestamp submessage.

    Field 1: seconds (int64 varint)
    Field 2: nanos (int32 varint)

    Returns:
        (epoch_float, iso8601_utc_str) or (None, None)
    """
    if not blob:
        return None, None

    fields = decode_message(blob)
    sec_entries = fields.get(1, [])
    nano_entries = fields.get(2, [])

    seconds = sec_entries[0][1] if sec_entries and isinstance(sec_entries[0][1], int) else 0
    nanos = nano_entries[0][1] if nano_entries and isinstance(nano_entries[0][1], int) else 0

    if seconds == 0 and nanos == 0:
        return None, None

    epoch = float(seconds) + nanos / 1_000_000_000.0
    try:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        millis = (nanos // 1_000_000) % 1000
        iso_str = dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{millis:03d}Z"
        return epoch, iso_str
    except (OverflowError, OSError, ValueError):
        return None, None


def decode_step_metadata(blob: bytes | None) -> dict[str, Any]:
    """Decodes Antigravity steps.metadata protobuf blob.

    Extracts:
        - input_tokens, output_tokens, cache_read_tokens, thinking_tokens, response_tokens
        - started_at, completed_at, duration_seconds
        - model_name
    """
    result: dict[str, Any] = {
        "started_at": None,
        "completed_at": None,
        "duration_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "thinking_tokens": 0,
        "response_tokens": 0,
        "model_name": None,
    }

    if not blob:
        return result

    fields = decode_message(blob)

    # 1. Started / Created timestamp (Field 1)
    created_ts: float | None = None
    if 1 in fields:
        created_val = fields[1][0][1]
        if isinstance(created_val, bytes):
            created_ts, started_at_str = decode_timestamp(created_val)
            result["started_at"] = started_at_str

    # 2. Completed / Updated timestamps (Fields 6, 7, 8)
    completed_candidates: list[tuple[float, str]] = []
    for f_no in (6, 7, 8):
        if f_no in fields:
            for _, val in fields[f_no]:
                if isinstance(val, bytes):
                    epoch, iso_str = decode_timestamp(val)
                    if epoch is not None and iso_str is not None:
                        completed_candidates.append((epoch, iso_str))

    if completed_candidates:
        max_completed_ts, max_completed_str = max(completed_candidates, key=lambda c: c[0])
        result["completed_at"] = max_completed_str
        if created_ts is not None:
            result["duration_seconds"] = max(0.0, max_completed_ts - created_ts)
    elif result["started_at"] is not None:
        result["completed_at"] = result["started_at"]

    # 3. Token usage (Field 9)
    if 9 in fields:
        f9_bytes = fields[9][0][1]
        if isinstance(f9_bytes, bytes):
            sub9 = decode_message(f9_bytes)

            def _get_varint(field_no: int) -> int:
                entries = sub9.get(field_no, [])
                if entries and isinstance(entries[0][1], int):
                    return max(0, entries[0][1])
                return 0

            inp = _get_varint(2)
            out = _get_varint(3)
            cache_read = _get_varint(5)
            thinking = _get_varint(9)
            resp = _get_varint(10)
            f11 = _get_varint(11)

            # Variant handling: if thinking (f9) absent, f11 present, and resp == out
            if thinking == 0 and f11 > 0 and resp == out and out > 0:
                thinking = f11
                resp = max(0, out - f11)

            # Adjustment: if output > 0 and both thinking and response are 0
            if out > 0 and resp == 0 and thinking == 0:
                resp = out
            elif out == 0 and (thinking + resp) > 0:
                out = thinking + resp

            result["input_tokens"] = inp
            result["output_tokens"] = out
            result["cache_read_tokens"] = cache_read
            result["thinking_tokens"] = thinking
            result["response_tokens"] = resp

    # 4. Model name (Field 13 or Field 24)
    model_name: str | None = None
    if 13 in fields:
        f13_val = fields[13][0][1]
        if isinstance(f13_val, bytes):
            try:
                s = f13_val.decode("utf-8").strip()
                if s and all(c.isprintable() or c in "\r\n\t" for c in s):
                    model_name = s
            except UnicodeDecodeError:
                pass

    if not model_name and 24 in fields:
        f24_val = fields[24][0][1]
        if isinstance(f24_val, bytes):
            sub24 = decode_message(f24_val)
            for sub_f in (8, 20):
                if sub_f in sub24:
                    val = sub24[sub_f][0][1]
                    if isinstance(val, bytes):
                        try:
                            s = val.decode("utf-8").strip()
                            if s and all(c.isprintable() for c in s):
                                model_name = s
                                break
                        except UnicodeDecodeError:
                            pass

    result["model_name"] = model_name
    return result
