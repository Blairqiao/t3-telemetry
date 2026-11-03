"""Unit and integration tests for protobuf decoder and Antigravity conversation telemetry reader."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from t3_telemetry.db_agy import (
    find_agy_db_path,
    find_agy_transcript_path,
    read_agy_conversation,
    read_transcript_data,
)
from t3_telemetry.models import ConversationTelemetry, ModelStep
from t3_telemetry.protobuf import (
    decode_message,
    decode_step_metadata,
    decode_timestamp,
    decode_varint,
    encode_field,
    encode_varint,
)


# ============================================================================
# Protobuf wire decoder tests
# ============================================================================


def test_varint_encode_decode_roundtrip():
    values = [0, 1, 127, 128, 300, 19452, 1048576, 2**31 - 1, 2**63 - 1]
    for val in values:
        encoded = encode_varint(val)
        decoded, pos = decode_varint(encoded, 0)
        assert decoded == val
        assert pos == len(encoded)


def test_varint_truncated_raises():
    truncated = b"\x80\x80"
    with pytest.raises(ValueError, match="Truncated varint"):
        decode_varint(truncated, 0)


def test_encode_field_and_decode_message():
    # Build a message with wire_type 0, 1, 2, 5
    m1 = encode_field(1, 0, 42)
    m2 = encode_field(2, 2, b"hello world")
    m3 = encode_field(3, 5, b"\x01\x02\x03\x04")
    m4 = encode_field(4, 1, b"\x01\x02\x03\x04\x05\x06\x07\x08")
    payload = m1 + m2 + m3 + m4

    decoded = decode_message(payload)
    assert decoded[1] == [(0, 42)]
    assert decoded[2] == [(2, b"hello world")]
    assert decoded[3] == [(5, b"\x01\x02\x03\x04")]
    assert decoded[4] == [(1, b"\x01\x02\x03\x04\x05\x06\x07\x08")]


def test_decode_message_truncated_graceful():
    # Tag indicates length-delimited 50 bytes, but only 5 provided
    data = encode_varint((2 << 3) | 2) + encode_varint(50) + b"short"
    decoded = decode_message(data)
    # Should not crash, returns empty or partial fields
    assert isinstance(decoded, dict)


def test_decode_timestamp():
    # Subfield 1: seconds (1793722761), Subfield 2: nanos (62106000 = 62.1ms)
    ts_blob = encode_field(1, 0, 1793722761) + encode_field(2, 0, 62106000)
    epoch, iso_str = decode_timestamp(ts_blob)
    assert epoch is not None
    assert abs(epoch - (1793722761 + 62106000 / 1e9)) < 1e-6
    assert iso_str is not None
    assert iso_str.endswith("Z")
    assert "2026-" in iso_str

    # Empty blob returns (None, None)
    assert decode_timestamp(b"") == (None, None)
    assert decode_timestamp(None) == (None, None)


def test_decode_step_metadata_live_sample_hex():
    sample_hex = (
        "0A0C08A5A5A8D70610F081D38D021802"
        "320C08AFA5A8D70610D0E2C69102"
        "3A0C08B5A5A8D70610B0D6A7B801"
        "420C08B5A5A8D70610B0D6A7B801"
        "4A0D10D81818DC0728DE900150DC07"
        "58C602"
    )
    blob = bytes.fromhex(sample_hex)
    meta = decode_step_metadata(blob)

    assert meta["input_tokens"] == 3160
    assert meta["output_tokens"] == 988
    assert meta["cache_read_tokens"] == 18526
    assert meta["response_tokens"] == 988
    assert meta["thinking_tokens"] == 0
    assert meta["started_at"] is not None
    assert meta["completed_at"] is not None
    assert meta["duration_seconds"] >= 0.0


def test_decode_step_metadata_ac1_synthetic():
    # Build token usage submessage (field 9)
    # f2: input_tokens = 1000
    # f3: output_tokens = 500
    # f5: cache_read_tokens = 4000
    # f9: thinking_tokens = 350
    # f10: response_tokens = 150
    f9_sub = (
        encode_field(2, 0, 1000)
        + encode_field(3, 0, 500)
        + encode_field(5, 0, 4000)
        + encode_field(9, 0, 350)
        + encode_field(10, 0, 150)
    )

    # Created ts (field 1)
    ts1 = encode_field(1, 0, 1700000000) + encode_field(2, 0, 100000000)
    # Completed ts (field 6)
    ts6 = encode_field(1, 0, 1700000005) + encode_field(2, 0, 600000000)

    # Model name in field 13
    f13_model = b"gemini-4-argon-medium"

    blob = (
        encode_field(1, 2, ts1)
        + encode_field(6, 2, ts6)
        + encode_field(9, 2, f9_sub)
        + encode_field(13, 2, f13_model)
    )

    meta = decode_step_metadata(blob)
    assert meta["input_tokens"] == 1000
    assert meta["output_tokens"] == 500
    assert meta["cache_read_tokens"] == 4000
    assert meta["thinking_tokens"] == 350
    assert meta["response_tokens"] == 150
    assert round(meta["duration_seconds"], 2) == 5.50
    assert meta["model_name"] == "gemini-4-argon-medium"


def test_decode_step_metadata_output_and_thinking_adjustments():
    # Case 1: output_tokens > 0, response_tokens == 0, thinking_tokens == 0 -> response_tokens = output_tokens
    f9_sub1 = encode_field(2, 0, 100) + encode_field(3, 0, 50)
    meta1 = decode_step_metadata(encode_field(9, 2, f9_sub1))
    assert meta1["output_tokens"] == 50
    assert meta1["response_tokens"] == 50
    assert meta1["thinking_tokens"] == 0

    # Case 2: output_tokens == 0, thinking + response > 0 -> output_tokens = thinking + response
    f9_sub2 = encode_field(9, 0, 40) + encode_field(10, 0, 60)
    meta2 = decode_step_metadata(encode_field(9, 2, f9_sub2))
    assert meta2["output_tokens"] == 100
    assert meta2["thinking_tokens"] == 40
    assert meta2["response_tokens"] == 60

    # Case 3: f9 absent, f11 present, f10 == f3
    # Subfield 10 = 100, Subfield 3 = 100, Subfield 11 = 65 -> thinking = 65, response = 35
    f9_sub3 = (
        encode_field(3, 0, 100)
        + encode_field(10, 0, 100)
        + encode_field(11, 0, 65)
    )
    meta3 = decode_step_metadata(encode_field(9, 2, f9_sub3))
    assert meta3["output_tokens"] == 100
    assert meta3["thinking_tokens"] == 65
    assert meta3["response_tokens"] == 35


def test_decode_step_metadata_resilience():
    assert decode_step_metadata(None)["input_tokens"] == 0
    assert decode_step_metadata(b"")["output_tokens"] == 0
    assert decode_step_metadata(b"\xff\xff\xff")["duration_seconds"] == 0.0


# ============================================================================
# Antigravity conversation telemetry reader tests
# ============================================================================


@pytest.fixture
def sample_agy_fixture(tmp_path: Path):
    """Creates a synthetic Antigravity .db and transcript.jsonl structure."""
    base_dir = tmp_path / "userdata"
    conv_id = "11111111-2222-3333-4444-555555555555"
    child_id = "99999999-8888-7777-6666-555555555555"
    parent_id = "00000000-aaaa-bbbb-cccc-dddddddddddd"

    # Setup directories
    conv_dir = base_dir / "providers" / "antigravity" / "inst-1" / "antigravity-acp" / "conversations"
    conv_dir.mkdir(parents=True)
    brain_dir = base_dir / "providers" / "antigravity" / "inst-1" / "antigravity-acp" / "brain" / conv_id / ".system_generated" / "logs"
    brain_dir.mkdir(parents=True)

    db_path = conv_dir / f"{conv_id}.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE steps (
            idx INTEGER PRIMARY KEY,
            step_type INTEGER NOT NULL,
            status INTEGER NOT NULL,
            metadata BLOB
        )
    """)
    c.execute("""
        CREATE TABLE parent_references (
            idx INTEGER PRIMARY KEY,
            data BLOB
        )
    """)

    # Insert parent reference
    c.execute("INSERT INTO parent_references (idx, data) VALUES (1, ?)", (f"ref to {parent_id}".encode(),))

    # Insert 2 inference steps (step_type = 15)
    f9_1 = encode_field(2, 0, 500) + encode_field(3, 0, 100) + encode_field(5, 0, 2000)
    meta_1 = encode_field(9, 2, f9_1) + encode_field(13, 2, b"gemini-3.8-flash")
    c.execute("INSERT INTO steps (idx, step_type, status, metadata) VALUES (1, 15, 3, ?)", (meta_1,))

    f9_2 = encode_field(2, 0, 800) + encode_field(3, 0, 250) + encode_field(5, 0, 2500)
    meta_2 = encode_field(9, 2, f9_2)
    c.execute("INSERT INTO steps (idx, step_type, status, metadata) VALUES (5, 15, 3, ?)", (meta_2,))

    conn.commit()
    conn.close()

    # Create transcript.jsonl
    transcript_path = brain_dir / "transcript.jsonl"
    lines = [
        json.dumps({
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "tool_calls": [{"name": "view_file"}, {"name": "run_command"}],
            "input_tokens": 500,
            "output_tokens": 100,
            "cache_read_tokens": 2000,
        }),
        json.dumps({
            "step_index": 2,
            "type": "AGENCY_TOOL_CALL",
            "content": f"Launched agent with conversationId: {child_id}",
        }),
        json.dumps({
            "step_index": 5,
            "type": "PLANNER_RESPONSE",
            "tool_calls": [{"name": "replace_file_content"}],
            "input_tokens": 800,
            "output_tokens": 250,
            "cache_read_tokens": 2500,
        }),
    ]
    transcript_path.write_text("\n".join(lines) + "\n")

    return base_dir, conv_id, child_id, parent_id


def test_read_agy_conversation_with_db_and_transcript(sample_agy_fixture):
    base_dir, conv_id, child_id, parent_id = sample_agy_fixture

    telemetry = read_agy_conversation(conv_id, base_dir, source_label="main")

    assert telemetry.native_id == conv_id
    assert telemetry.source == "db"
    assert telemetry.model_name == "gemini-3.8-flash"
    assert telemetry.parent_native_ids == [parent_id]
    assert telemetry.child_native_ids == [child_id]
    assert len(telemetry.steps) == 2

    step1 = telemetry.steps[0]
    assert step1.step_index == 1
    assert step1.input_tokens == 500
    assert step1.output_tokens == 100
    assert step1.cache_read_tokens == 2000
    assert step1.tool_calls == ["view_file", "run_command"]

    step2 = telemetry.steps[1]
    assert step2.step_index == 5
    assert step2.input_tokens == 800
    assert step2.output_tokens == 250
    assert step2.cache_read_tokens == 2500
    assert step2.tool_calls == ["replace_file_content"]


def test_read_agy_conversation_fallback_to_transcript(tmp_path: Path):
    base_dir = tmp_path / "userdata"
    conv_id = "22222222-3333-4444-5555-666666666666"

    brain_dir = base_dir / "brain" / conv_id / ".system_generated" / "logs"
    brain_dir.mkdir(parents=True)
    transcript_path = brain_dir / "transcript.jsonl"

    lines = [
        json.dumps({
            "step_index": 1,
            "type": "PLANNER_RESPONSE",
            "created_at": "2026-11-03T10:00:00.000Z",
            "input_tokens": 1200,
            "output_tokens": 300,
            "cache_read_tokens": 5000,
            "tool_calls": [{"name": "run_command"}],
        }),
    ]
    transcript_path.write_text("\n".join(lines) + "\n")

    # DB is absent!
    telemetry = read_agy_conversation(conv_id, base_dir)
    assert telemetry.native_id == conv_id
    assert telemetry.source == "transcript"
    assert len(telemetry.steps) == 1
    step = telemetry.steps[0]
    assert step.step_index == 1
    assert step.input_tokens == 1200
    assert step.output_tokens == 300
    assert step.cache_read_tokens == 5000
    assert step.tool_calls == ["run_command"]
    assert step.total_tokens == 6500


def test_read_agy_conversation_missing(tmp_path: Path):
    telemetry = read_agy_conversation("nonexistent-id", tmp_path)
    assert telemetry.native_id == "nonexistent-id"
    assert telemetry.source == "missing"
    assert telemetry.steps == []


def test_sqlite_ro_mode(sample_agy_fixture):
    base_dir, conv_id, _, _ = sample_agy_fixture
    db_path = find_agy_db_path(conv_id, base_dir)
    assert db_path is not None

    # Connect in read-only mode and verify write fails
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    c = conn.cursor()
    with pytest.raises(sqlite3.OperationalError):
        c.execute("INSERT INTO steps (idx, step_type, status) VALUES (999, 15, 1)")
    conn.close()
