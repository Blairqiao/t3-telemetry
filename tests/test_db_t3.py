from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pytest

from t3_telemetry.db_t3 import (
    connect_readonly_sqlite,
    parse_iso_duration,
    read_t3_state,
    T3StateSnapshot,
)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE orchestration_v2_projection_threads (
        thread_id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL,
        title TEXT NOT NULL,
        default_provider TEXT NOT NULL,
        runtime_mode TEXT NOT NULL,
        interaction_mode TEXT NOT NULL,
        active_provider_thread_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        archived_at TEXT,
        deleted_at TEXT,
        payload_json TEXT NOT NULL,
        provider_instance_id TEXT
    );

    CREATE TABLE orchestration_v2_projection_provider_threads (
        provider_thread_id TEXT PRIMARY KEY,
        thread_id TEXT,
        owner_node_id TEXT,
        provider TEXT NOT NULL,
        provider_session_id TEXT,
        status TEXT NOT NULL,
        first_run_ordinal INTEGER,
        last_run_ordinal INTEGER,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        driver TEXT,
        provider_instance_id TEXT
    );

    CREATE TABLE orchestration_v2_projection_provider_turns (
        provider_turn_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        provider_thread_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        run_attempt_id TEXT,
        ordinal INTEGER NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        payload_json TEXT NOT NULL
    );

    CREATE TABLE orchestration_v2_projection_subagents (
        subagent_id TEXT PRIMARY KEY,
        thread_id TEXT NOT NULL,
        run_id TEXT,
        parent_node_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        provider_thread_id TEXT,
        child_thread_id TEXT,
        origin TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        driver TEXT,
        provider_instance_id TEXT,
        completion_delivery_state TEXT
    );
    """)


def test_connect_readonly_sqlite_enforcement(tmp_path: Path):
    db_file = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE foo (id INT)")
    conn.execute("INSERT INTO foo VALUES (1)")
    conn.commit()
    conn.close()

    ro_conn = connect_readonly_sqlite(db_file)
    cur = ro_conn.cursor()
    cur.execute("SELECT id FROM foo")
    assert cur.fetchone()[0] == 1

    with pytest.raises(sqlite3.OperationalError, match="attempt to write a readonly database"):
        ro_conn.execute("INSERT INTO foo VALUES (2)")

    ro_conn.close()


def test_parse_iso_duration():
    assert parse_iso_duration(None, None) == 0.0
    assert parse_iso_duration("invalid", "2026-11-03T16:53:01.000Z") == 0.0
    assert parse_iso_duration("2026-11-03T16:53:01.000Z", None) == 0.0

    t1 = "2026-11-03T16:53:01.200Z"
    t2 = "2026-11-03T16:53:05.700Z"
    assert pytest.approx(parse_iso_duration(t1, t2), rel=1e-3) == 4.5

    # completed before started returns 0.0
    assert parse_iso_duration(t2, t1) == 0.0


def test_read_t3_state_nonexistent_dir(tmp_path: Path):
    snapshot = read_t3_state(tmp_path / "nonexistent")
    assert isinstance(snapshot, T3StateSnapshot)
    assert len(snapshot.threads) == 0
    assert len(snapshot.provider_threads) == 0
    assert len(snapshot.provider_turns_by_thread) == 0
    assert len(snapshot.subagents_by_parent) == 0
    assert snapshot.get_provider_thread("any") is None


def test_read_t3_state_synthetic_db(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = sqlite3.connect(db_file)
    create_schema(conn)

    # Insert root Antigravity thread
    thread1_payload = {
        "modelSelection": {
            "model": "gemini-3.8-flash",
            "options": [{"id": "thinking_level", "value": "high"}]
        },
        "lineage": {
            "parentThreadId": None,
            "rootThreadId": "thread-1",
            "relationshipToParent": None
        }
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "thread-1", "proj-1", "Main Task", "antigravity", "full-access", "default",
        "pt-1", "2026-11-03T10:00:00.000Z", "2026-11-03T10:05:00.000Z", None,
        json.dumps(thread1_payload), "antigravity_gemini"
    ))

    # Insert child Claude thread
    thread2_payload = {
        "modelSelection": {
            "model": "claude-3-7-sonnet",
            "options": [{"id": "effort", "value": "medium"}]
        },
        "lineage": {
            "parentThreadId": "thread-1",
            "rootThreadId": "thread-1",
            "relationshipToParent": "delegate"
        }
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "thread-2", "proj-1", "Delegated Subtask", "claudeAgent", "full-access", "default",
        "pt-2", "2026-11-03T10:01:00.000Z", "2026-11-03T10:04:00.000Z", None,
        json.dumps(thread2_payload), "claude_instance"
    ))

    # Insert deleted thread (should be ignored)
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "thread-deleted", "proj-1", "Deleted Thread", "claudeAgent", "full-access", "default",
        "pt-del", "2026-11-03T09:00:00.000Z", "2026-11-03T09:01:00.000Z", "2026-11-03T09:02:00.000Z",
        "{}", "claude_instance"
    ))

    # Provider threads
    pt1_payload = {
        "nativeThreadRef": {"driver": "antigravity", "nativeId": "conv-agy-uuid-1"},
        "contextUsage": {"usedTokens": 4500, "maxTokens": 1048576}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "pt-1", "thread-1", "node-1", "antigravity", "idle", "2026-11-03T10:05:00.000Z",
        json.dumps(pt1_payload), "antigravity", "antigravity_gemini"
    ))

    pt2_payload = {
        "nativeThreadRef": {"driver": "claudeAgent", "nativeId": "claude-session-2"},
        "contextUsage": {"usedTokens": 32000, "maxTokens": 200000}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "pt-2", "thread-2", "node-2", "claudeAgent", "completed", "2026-11-03T10:04:00.000Z",
        json.dumps(pt2_payload), "claudeAgent", "claude_instance"
    ))

    # Provider turns: Antigravity turn (unavailable) and Claude turn (complete with breakdown)
    turn1_payload = {
        "nativeTurnRef": {"driver": "antigravity", "nativeId": "conv-agy-uuid-1:turn:1"},
        "turnTokenUsage": {"usageStatus": "unavailable"}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "turn-1", "thread-1", "pt-1", "node-1", 1, "completed",
        "2026-11-03T10:00:10.000Z", "2026-11-03T10:00:15.000Z", json.dumps(turn1_payload)
    ))

    turn2_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-claude-abc"},
        "tokenUsage": {"usedTokens": 32000, "maxTokens": 200000},
        "turnTokenUsage": {
            "usageStatus": "complete",
            "inputTokens": 25000,
            "cachedInputTokens": 18000,
            "cacheCreationTokens": 4000,
            "outputTokens": 2500,
            "reasoningTokens": 1000
        }
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "turn-2", "thread-2", "pt-2", "node-2", 1, "completed",
        "2026-11-03T10:01:10.000Z", "2026-11-03T10:01:25.000Z", json.dumps(turn2_payload)
    ))

    # Subagent link
    sub_payload = {
        "title": "Child Subagent Task",
        "model": "claude-3-7-sonnet"
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_subagents
    (subagent_id, thread_id, parent_node_id, provider, provider_thread_id, child_thread_id,
     origin, status, started_at, completed_at, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "sub-1", "thread-1", "node-1", "claudeAgent", "pt-2", "thread-2",
        "app_owned", "completed", "2026-11-03T10:01:00.000Z", "2026-11-03T10:01:30.000Z",
        "2026-11-03T10:01:30.000Z", json.dumps(sub_payload), "claudeAgent", "claude_instance"
    ))

    conn.commit()
    conn.close()

    snapshot = read_t3_state(tmp_path)
    assert len(snapshot.threads) == 2
    assert "thread-1" in snapshot.threads
    assert "thread-2" in snapshot.threads
    assert "thread-deleted" not in snapshot.threads

    t1 = snapshot.threads["thread-1"]
    assert t1.model == "gemini-3.8-flash"
    assert t1.thinking_level == "high"
    assert t1.parent_thread_id is None
    assert t1.active_provider_thread_id == "pt-1"

    t2 = snapshot.threads["thread-2"]
    assert t2.model == "claude-3-7-sonnet"
    assert t2.thinking_level == "medium"
    assert t2.parent_thread_id == "thread-1"

    pt1 = snapshot.provider_threads["pt-1"]
    assert pt1.driver == "antigravity"
    assert pt1.native_id == "conv-agy-uuid-1"
    assert pt1.used_tokens == 4500
    assert pt1.max_tokens == 1048576
    assert snapshot.get_provider_thread("thread-1") == pt1

    turns2 = snapshot.provider_turns_by_thread["thread-2"]
    assert len(turns2) == 1
    tr2 = turns2[0]
    assert tr2.input_tokens == 25000
    assert tr2.cache_read_tokens == 18000
    assert tr2.cache_creation_tokens == 4000
    assert tr2.output_tokens == 2500
    assert tr2.thinking_tokens == 1000
    assert tr2.response_tokens == 1500  # output_tokens - thinking_tokens
    assert tr2.usage_status == "complete"
    assert tr2.duration_seconds == 15.0

    subs1 = snapshot.subagents_by_parent["thread-1"]
    assert len(subs1) == 1
    sub1 = subs1[0]
    assert sub1.child_thread_id == "thread-2"
    assert sub1.title == "Child Subagent Task"
    assert sub1.model == "claude-3-7-sonnet"
    assert sub1.duration_seconds == 30.0


def test_read_t3_state_fallback_token_usage_when_turn_token_usage_absent(tmp_path: Path):
    """Verify fallback to tokenUsage for Claude/Codex turns without turnTokenUsage."""
    db_file = tmp_path / "statev2.sqlite"
    conn = sqlite3.connect(db_file)
    create_schema(conn)

    turn_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-fallback"},
        "tokenUsage": {
            "usedTokens": 5000,
            "maxTokens": 200000,
            "inputTokens": 4000,
            "cachedInputTokens": 1000,
            "outputTokens": 500,
            "reasoningOutputTokens": 100
        }
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-fb", "t-fb", "pt-fb", "n-1", 1, "completed",
            "2026-11-03T10:00:00Z", "2026-11-03T10:00:05Z", json.dumps(turn_payload)))
    conn.commit()
    conn.close()

    snapshot = read_t3_state(tmp_path)
    turn = snapshot.provider_turns_by_thread["t-fb"][0]
    assert turn.input_tokens == 4000
    assert turn.cache_read_tokens == 1000
    assert turn.output_tokens == 500
    assert turn.thinking_tokens == 100
    assert turn.response_tokens == 400
    assert turn.usage_status == "complete"
