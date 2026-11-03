from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pytest

from t3_telemetry.models import ConversationTelemetry, ModelStep, ThreadTelemetryNode
from t3_telemetry.db_t3 import read_t3_state
from t3_telemetry.aggregator import (
    build_thread_telemetry,
    list_root_threads,
    resolve_thread,
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


def setup_test_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    return conn


def test_build_thread_telemetry_claude_turns(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    thread_payload = {
        "modelSelection": {"model": "claude-3-7-sonnet", "options": [{"id": "effort", "value": "low"}]},
        "lineage": {"parentThreadId": None, "rootThreadId": "t-claude", "relationshipToParent": None}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-claude", "proj", "Claude Solo", "claudeAgent", "full-access", "default",
            "pt-claude", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(thread_payload), "claude_inst"))

    pt_payload = {
        "nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-native-1"},
        "contextUsage": {"usedTokens": 30000, "maxTokens": 200000}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-claude", "t-claude", "node-1", "claudeAgent", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    # Turn 1
    turn1_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-1"},
        "tokenUsage": {"usedTokens": 35000, "maxTokens": 200000},
        "turnTokenUsage": {
            "usageStatus": "complete",
            "inputTokens": 1000,
            "cachedInputTokens": 4000,
            "cacheCreationTokens": 500,
            "outputTokens": 800,
            "reasoningTokens": 300
        }
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-1", "t-claude", "pt-claude", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(turn1_payload)))

    # Turn 2
    turn2_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-2"},
        "tokenUsage": {"usedTokens": 45000, "maxTokens": 200000},
        "turnTokenUsage": {
            "usageStatus": "complete",
            "inputTokens": 2000,
            "cachedInputTokens": 6000,
            "cacheCreationTokens": 0,
            "outputTokens": 1200,
            "reasoningTokens": 200
        }
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-2", "t-claude", "pt-claude", "node-1", 2, "completed",
            "2026-11-03T10:02:00Z", "2026-11-03T10:02:15Z", json.dumps(turn2_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-claude", tmp_path)
    assert node.thread_id == "t-claude"
    assert node.title == "Claude Solo"
    assert node.driver == "claudeAgent"
    assert node.model == "claude-3-7-sonnet"
    assert len(node.steps) == 2

    # Context usage should reflect turn 2 (45000 / 200000 = 22.5%)
    assert node.context_used_tokens == 45000
    assert node.context_max_tokens == 200000
    assert pytest.approx(node.context_pct, rel=1e-3) == 22.5

    # Step checks
    assert node.steps[0].step_index == 1
    assert node.steps[0].input_tokens == 1000
    assert node.steps[0].cache_read_tokens == 4000
    assert node.steps[0].output_tokens == 800
    assert node.steps[0].thinking_tokens == 300
    assert node.steps[0].response_tokens == 500
    assert node.steps[0].total_tokens == 5800

    # Self metrics:
    # input = 3000, cache_read = 10000, output = 2000, thinking = 500, response = 1500
    sm = node.self_metrics
    assert sm.input_tokens == 3000
    assert sm.cache_read_tokens == 10000
    assert sm.output_tokens == 2000
    assert sm.thinking_tokens == 500
    assert sm.response_tokens == 1500
    assert sm.total_tokens == 15000
    assert pytest.approx(sm.cache_hit_pct, rel=1e-3) == (10000 / 13000) * 100
    assert pytest.approx(sm.thinking_pct, rel=1e-3) == (500 / 2000) * 100

    # Rollup matches self metrics when no subagents
    rm = node.rollup_metrics
    assert rm.total_tokens == sm.total_tokens
    assert rm.total_subagents == 0
    assert rm.active_subagents == 0

    # to_dict verification
    data = node.to_dict()
    assert data["thread_id"] == "t-claude"
    assert data["context_used_tokens"] == 45000
    assert data["context_pct"] == 22.5
    assert data["self_metrics"]["total_tokens"] == 15000


def test_build_thread_telemetry_delegated_subagent_rollup(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    # Parent thread (Antigravity)
    parent_payload = {
        "modelSelection": {"model": "gemini-3.8-flash", "options": [{"id": "thinking_level", "value": "high"}]},
        "lineage": {"parentThreadId": None, "rootThreadId": "t-parent", "relationshipToParent": None}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-parent", "proj", "Parent Agy Thread", "antigravity", "full-access", "default",
            "pt-parent", "2026-11-03T10:00:00Z", "2026-11-03T10:10:00Z", None,
            json.dumps(parent_payload), "antigravity_gemini"))

    pt_parent_payload = {
        "nativeThreadRef": {"driver": "antigravity", "nativeId": "agy-parent-uuid"},
        "contextUsage": {"usedTokens": 10000, "maxTokens": 1048576}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-parent", "t-parent", "node-1", "antigravity", "idle", "2026-11-03T10:10:00Z",
            json.dumps(pt_parent_payload), "antigravity", "antigravity_gemini"))

    # Child thread (Claude delegated)
    child_payload = {
        "modelSelection": {"model": "claude-3-7-sonnet"},
        "lineage": {"parentThreadId": "t-parent", "rootThreadId": "t-parent", "relationshipToParent": "delegate"}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-child", "proj", "Child Claude Thread", "claudeAgent", "full-access", "default",
            "pt-child", "2026-11-03T10:02:00Z", "2026-11-03T10:08:00Z", None,
            json.dumps(child_payload), "claude_inst"))

    pt_child_payload = {
        "nativeThreadRef": {"driver": "claudeAgent", "nativeId": "claude-child-native"},
        "contextUsage": {"usedTokens": 8000, "maxTokens": 200000}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-child", "t-child", "node-2", "claudeAgent", "running", "2026-11-03T10:08:00Z",
            json.dumps(pt_child_payload), "claudeAgent", "claude_inst"))

    # Turn on child
    child_turn_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-c1"},
        "turnTokenUsage": {
            "usageStatus": "complete",
            "inputTokens": 2000,
            "cachedInputTokens": 1000,
            "cacheCreationTokens": 100,
            "outputTokens": 400,
            "reasoningTokens": 100
        }
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-c1", "t-child", "pt-child", "node-2", 1, "completed",
            "2026-11-03T10:03:00Z", "2026-11-03T10:03:10Z", json.dumps(child_turn_payload)))

    # Subagents table link with metadata
    sub_payload = {"title": "Delegated Review", "model": "claude-3-7-sonnet"}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_subagents
    (subagent_id, thread_id, parent_node_id, provider, provider_thread_id, child_thread_id,
     origin, status, started_at, completed_at, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("sub-link-1", "t-parent", "node-1", "claudeAgent", "pt-child", "t-child",
            "app_owned", "running", "2026-11-03T10:02:00Z", None, "2026-11-03T10:08:00Z",
            json.dumps(sub_payload), "claudeAgent", "claude_inst"))

    conn.commit()
    conn.close()

    def mock_agy_reader(native_id: str, t3_dir: Path, source_label: str = "main") -> ConversationTelemetry:
        assert native_id == "agy-parent-uuid"
        step = ModelStep(
            step_index=1,
            source=source_label,
            status="completed",
            input_tokens=1000,
            cache_read_tokens=500,
            output_tokens=200,
            thinking_tokens=50,
            response_tokens=150,
            started_at="2026-11-03T10:01:00Z",
            completed_at="2026-11-03T10:01:05Z",
            duration_seconds=5.0,
        )
        return ConversationTelemetry(native_id=native_id, steps=[step])

    node = build_thread_telemetry("t-parent", tmp_path, agy_reader=mock_agy_reader)
    assert len(node.subagents) == 1
    child = node.subagents[0]
    assert child.thread_id == "t-child"
    assert child.driver == "claudeAgent"
    assert child.status == "running"

    # Self metrics: only parent step
    assert node.self_metrics.input_tokens == 1000
    assert node.self_metrics.cache_read_tokens == 500
    assert node.self_metrics.output_tokens == 200
    assert node.self_metrics.total_tokens == 1700

    # Rollup metrics: parent (1000+500+200=1700) + child (2000+1000+400=3400) = 5100
    rm = node.rollup_metrics
    assert rm.input_tokens == 3000
    assert rm.cache_read_tokens == 1500
    assert rm.output_tokens == 600
    assert rm.thinking_tokens == 150
    assert rm.response_tokens == 450
    assert rm.total_tokens == 5100
    assert pytest.approx(rm.cache_hit_pct, rel=1e-3) == (1500 / 4500) * 100
    assert pytest.approx(rm.thinking_pct, rel=1e-3) == (150 / 600) * 100
    assert rm.total_subagents == 1
    assert rm.active_subagents == 1

    # Combined steps sorted chronologically
    assert len(node.combined_steps) == 2
    assert node.combined_steps[0].source == "main"
    assert node.combined_steps[1].source == "sub-1"


def test_build_thread_telemetry_native_antigravity_subagent(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    parent_payload = {
        "modelSelection": {"model": "gemini-3.8-flash"},
        "lineage": {"parentThreadId": None, "rootThreadId": "t-parent-native", "relationshipToParent": None}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-parent-native", "proj", "Parent Native", "antigravity", "full-access", "default",
            "pt-pnat", "2026-11-03T10:00:00Z", "2026-11-03T10:10:00Z", None,
            json.dumps(parent_payload), "antigravity_gemini"))

    pt_payload = {
        "nativeThreadRef": {"driver": "antigravity", "nativeId": "agy-parent-id"},
        "contextUsage": {"usedTokens": 5000, "maxTokens": 1048576}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-pnat", "t-parent-native", "node-1", "antigravity", "idle", "2026-11-03T10:10:00Z",
            json.dumps(pt_payload), "antigravity", "antigravity_gemini"))

    conn.commit()
    conn.close()

    def mock_agy_reader(native_id: str, t3_dir: Path, source_label: str = "main") -> ConversationTelemetry:
        if native_id == "agy-parent-id":
            return ConversationTelemetry(
                native_id=native_id,
                child_native_ids=["native-child-id-1"],
                steps=[ModelStep(step_index=1, source=source_label, status="completed",
                                 input_tokens=1000, cache_read_tokens=200, output_tokens=300,
                                 started_at="2026-11-03T10:01:00Z")]
            )
        elif native_id == "native-child-id-1":
            return ConversationTelemetry(
                native_id=native_id,
                steps=[ModelStep(step_index=1, source=source_label, status="completed",
                                 input_tokens=500, cache_read_tokens=100, output_tokens=150,
                                 started_at="2026-11-03T10:02:00Z")]
            )
        raise ValueError(f"Unknown native_id: {native_id}")

    node = build_thread_telemetry("t-parent-native", tmp_path, agy_reader=mock_agy_reader)
    assert len(node.subagents) == 1
    child = node.subagents[0]
    assert child.native_id == "native-child-id-1"
    assert child.driver == "antigravity"
    assert node.rollup_metrics.input_tokens == 1500
    assert node.rollup_metrics.cache_read_tokens == 300
    assert node.rollup_metrics.output_tokens == 450
    assert node.rollup_metrics.total_subagents == 1


def test_circular_lineage_protection(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    # Thread A claims Thread B as child, and Thread B claims Thread A as child
    payload_a = {"modelSelection": {"model": "m-a"}, "lineage": {"parentThreadId": "t-b"}}
    payload_b = {"modelSelection": {"model": "m-b"}, "lineage": {"parentThreadId": "t-a"}}

    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-a", "proj", "Thread A", "codex", "full-access", "default",
            "pt-a", "2026-11-03T10:00:00Z", "2026-11-03T10:01:00Z", None,
            json.dumps(payload_a), "codex"))

    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-b", "proj", "Thread B", "codex", "full-access", "default",
            "pt-b", "2026-11-03T10:00:00Z", "2026-11-03T10:01:00Z", None,
            json.dumps(payload_b), "codex"))

    # Link subagents table also pointing A -> B and B -> A
    conn.execute("""
    INSERT INTO orchestration_v2_projection_subagents
    (subagent_id, thread_id, parent_node_id, provider, provider_thread_id, child_thread_id,
     origin, status, started_at, completed_at, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("sub-ab", "t-a", "node-a", "codex", "pt-b", "t-b",
            "app_owned", "completed", None, None, "2026-11-03T10:01:00Z", "{}", "codex", "codex"))

    conn.execute("""
    INSERT INTO orchestration_v2_projection_subagents
    (subagent_id, thread_id, parent_node_id, provider, provider_thread_id, child_thread_id,
     origin, status, started_at, completed_at, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("sub-ba", "t-b", "node-b", "codex", "pt-a", "t-a",
            "app_owned", "completed", None, None, "2026-11-03T10:01:00Z", "{}", "codex", "codex"))

    conn.commit()
    conn.close()

    # Must complete without RecursionError or infinite loop
    node_a = build_thread_telemetry("t-a", tmp_path)
    assert node_a.thread_id == "t-a"
    assert len(node_a.subagents) == 1
    assert node_a.subagents[0].thread_id == "t-b"
    # Thread B does not re-include Thread A as a child
    assert len(node_a.subagents[0].subagents) == 0


def test_list_root_threads_and_resolve(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    t1 = {"modelSelection": {"model": "m1"}, "lineage": {"parentThreadId": None}}
    t2 = {"modelSelection": {"model": "m2"}, "lineage": {"parentThreadId": None}}
    t_child = {"modelSelection": {"model": "m3"}, "lineage": {"parentThreadId": "t1"}}
    t_del = {"modelSelection": {"model": "m4"}, "lineage": {"parentThreadId": None}}

    rows = [
        ("t1", "proj", "First Task", "codex", "2026-11-03T08:00:00Z", "2026-11-03T08:30:00Z", None, json.dumps(t1)),
        ("t2", "proj", "Second Feature Work", "codex", "2026-11-03T09:00:00Z", "2026-11-03T09:30:00Z", None, json.dumps(t2)),
        ("t-child", "proj", "Sub task", "codex", "2026-11-03T09:10:00Z", "2026-11-03T09:20:00Z", None, json.dumps(t_child)),
        ("t-del", "proj", "Deleted", "codex", "2026-11-03T07:00:00Z", "2026-11-03T07:10:00Z", "2026-11-03T07:15:00Z", json.dumps(t_del)),
    ]

    for tid, pid, title, prov, cat, uat, dat, pj in rows:
        conn.execute("""
        INSERT INTO orchestration_v2_projection_threads
        (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
         active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tid, pid, title, prov, "full-access", "default", f"pt-{tid}", cat, uat, dat, pj, prov))

    # Add a provider thread for t1 with native_id for testing prefix resolve
    pt1_payload = {
        "nativeThreadRef": {"driver": "codex", "nativeId": "native-codex-xyz"}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-t1", "t1", "node-1", "codex", "idle", "2026-11-03T08:30:00Z", json.dumps(pt1_payload), "codex", "codex"))

    conn.commit()
    conn.close()

    roots = list_root_threads(tmp_path, limit=10)
    # Roots should only be t2 and t1 in descending order of updated_at (t2 was 09:30, t1 was 08:30)
    assert len(roots) == 2
    assert roots[0].thread_id == "t2"
    assert roots[1].thread_id == "t1"

    # Test resolve_thread default (no query -> latest root)
    res_default = resolve_thread(tmp_path, None)
    assert res_default is not None
    assert res_default.thread_id == "t2"

    # Test resolve by exact id
    res_id = resolve_thread(tmp_path, "t1")
    assert res_id is not None
    assert res_id.thread_id == "t1"

    # Test resolve by native_id prefix
    res_native = resolve_thread(tmp_path, "native-codex")
    assert res_native is not None
    assert res_native.thread_id == "t1"

    # Test resolve by title substring (case-insensitive)
    res_title = resolve_thread(tmp_path, "feature")
    assert res_title is not None
    assert res_title.thread_id == "t2"

    # Test resolve not found
    assert resolve_thread(tmp_path, "nonexistent-xyz") is None
