from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pytest

from t3_telemetry.models import ConversationTelemetry, ModelStep, ProviderTurnRecord, ThreadTelemetryNode
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
            "inputTokens": 5500,
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
            "inputTokens": 8000,
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
    assert node.steps[0].total_tokens == 6300
    assert node.steps[0].context_used_tokens == 35000
    assert node.steps[0].context_max_tokens == 200000
    assert node.steps[0].to_dict()["context_used_tokens"] == 35000
    assert node.steps[0].to_dict()["context_max_tokens"] == 200000
    assert node.steps[1].context_used_tokens == 45000
    assert node.steps[1].context_max_tokens == 200000
    assert node.steps[1].to_dict()["context_used_tokens"] == 45000
    assert node.steps[1].to_dict()["context_max_tokens"] == 200000

    # Self metrics:
    # input = 3000, cache_read = 10000, output = 2000, thinking = 500, response = 1500
    sm = node.self_metrics
    assert sm.input_tokens == 3000
    assert sm.cache_read_tokens == 10000
    assert sm.output_tokens == 2000
    assert sm.thinking_tokens == 500
    assert sm.response_tokens == 1500
    assert sm.total_tokens == 15500
    assert pytest.approx(sm.cache_hit_pct, rel=1e-3) == (10000 / 13500) * 100
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
    assert data["self_metrics"]["total_tokens"] == 15500


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
            "inputTokens": 3100,
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
    assert rm.total_tokens == 5200
    assert pytest.approx(rm.cache_hit_pct, rel=1e-3) == (1500 / 4600) * 100
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
    # Verify native Antigravity child context telemetry reconstruction (Task 4 review issue 2)
    assert child.context_used_tokens == 750
    assert child.context_provenance == "estimated"
    assert child.context_limit is None
    assert child.context_max_tokens == 1_048_576
    assert child.context_limit_provenance == "default"
    assert len(child.context_samples) == 1
    assert child.context_samples[0].used_tokens == 750
    assert child.context_samples[0].source_id == "native-child-id-1:step:1"
    assert child.context_samples[0].provider_thread_id == "native-child-id-1"
    assert child.active_provider_thread_id == "native-child-id-1"
    assert child.latest_step_label == "Model step #1"


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


def test_current_running_turn_selection(tmp_path: Path):
    """When a thread has running status, current turn metrics selects the running turn."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-run", "proj", "Running Thread", "claudeAgent", "full-access", "default",
            "pt-run", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "claude_inst"))

    pt_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-run-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-run", "t-run", "node-1", "claudeAgent", "running", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    # Turn 1 completed
    t1_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-1"},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-1", "t-run", "pt-run", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t1_payload)))

    # Turn 2 running
    t2_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-2"},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 500, "outputTokens": 50}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-2", "t-run", "pt-run", "node-1", 2, "running",
            "2026-11-03T10:02:00Z", None, json.dumps(t2_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-run", tmp_path)
    assert node.status == "running"
    assert node.current_turn_metrics is not None
    assert node.current_turn_metrics.ordinal == 2
    assert node.current_turn_metrics.input_tokens == 500
    assert node.current_turn_metrics.output_tokens == 50
    assert node.turn_metrics_availability == "available"
    # Whole chat totals unchanged: sum of both turns
    assert node.self_metrics.input_tokens == 1500
    assert node.self_metrics.output_tokens == 250
    assert node.self_metrics.step_count == 2
    assert node.self_metrics.provider_turn_count == 2


def test_latest_completed_turn_selection(tmp_path: Path):
    """When a thread is idle, current turn metrics selects the latest completed turn."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "codex"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-idle", "proj", "Idle Thread", "codex", "full-access", "default",
            "pt-idle", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "codex"))

    pt_payload = {"nativeThreadRef": {"driver": "codex", "nativeId": "c-idle-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-idle", "t-idle", "node-1", "codex", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "codex", "codex"))

    # Turn 1 completed
    t1_payload = {
        "nativeTurnRef": {"driver": "codex", "nativeId": "turn-1"},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-1", "t-idle", "pt-idle", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t1_payload)))

    # Turn 2 completed
    t2_payload = {
        "nativeTurnRef": {"driver": "codex", "nativeId": "turn-2"},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 3000, "outputTokens": 600}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-2", "t-idle", "pt-idle", "node-1", 2, "completed",
            "2026-11-03T10:02:00Z", "2026-11-03T10:02:15Z", json.dumps(t2_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-idle", tmp_path)
    assert node.status in ("idle", "completed")
    assert node.current_turn_metrics is not None
    assert node.current_turn_metrics.ordinal == 2
    assert node.current_turn_metrics.input_tokens == 3000
    assert node.current_turn_metrics.output_tokens == 600
    assert node.turn_metrics_availability == "available"


def test_cumulative_fallback_not_claimed_as_exact_turn(tmp_path: Path):
    """Cumulative tokenUsage fallback is retained in whole chat but not claimed as exact turn metric."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "codex"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-fb", "proj", "Fallback Thread", "codex", "full-access", "default",
            "pt-fb", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "codex"))

    pt_payload = {"nativeThreadRef": {"driver": "codex", "nativeId": "c-fb-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-fb", "t-fb", "node-1", "codex", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "codex", "codex"))

    # Turn with only tokenUsage (cumulative fallback), no turnTokenUsage
    turn_payload = {
        "nativeTurnRef": {"driver": "codex", "nativeId": "turn-1"},
        "tokenUsage": {"inputTokens": 4500, "outputTokens": 800}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-1", "t-fb", "pt-fb", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(turn_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-fb", tmp_path)
    # Turn metrics must be marked unavailable (not claimed as exact turn usage)
    assert node.turn_metrics_availability == "unavailable"
    assert node.current_turn_metrics.availability == "unavailable"
    assert node.current_turn_metrics.provenance == "cumulative_fallback"
    # Whole chat totals remain unchanged
    assert node.self_metrics.input_tokens == 4500
    assert node.self_metrics.output_tokens == 800
    assert node.self_metrics.step_count == 1


def test_zero_exact_usage_distinguished_from_unavailable(tmp_path: Path):
    """Zero reported exact usage produces available turn metrics with 0 tokens, not unavailable."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claudeAgent"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-zero", "proj", "Zero Exact Thread", "claudeAgent", "full-access", "default",
            "pt-zero", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "claude_inst"))

    pt_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-zero-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-zero", "t-zero", "node-1", "claudeAgent", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    turn_payload = {
        "nativeTurnRef": {"driver": "claudeAgent", "nativeId": "turn-1"},
        "tokenUsage": {"inputTokens": 5000, "outputTokens": 1000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 0, "outputTokens": 0}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-1", "t-zero", "pt-zero", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(turn_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-zero", tmp_path)
    assert node.turn_metrics_availability == "available"
    assert node.current_turn_metrics.availability == "available"
    assert node.current_turn_metrics.input_tokens == 0
    assert node.current_turn_metrics.output_tokens == 0


def test_antigravity_current_turn_unavailable_and_step_labeled_separately(tmp_path: Path):
    """Antigravity has unavailable current-turn scope and exposes latest model step separately."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "gemini-flash"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-agy", "proj", "Antigravity Chat", "antigravity", "full-access", "default",
            "pt-agy", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "antigravity_gemini"))

    pt_payload = {"nativeThreadRef": {"driver": "antigravity", "nativeId": "agy-conv-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-agy", "t-agy", "node-1", "antigravity", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "antigravity", "antigravity_gemini"))

    conn.commit()
    conn.close()

    def mock_reader(native_id: str, t3_dir: Path, source_label: str = "main"):
        return ConversationTelemetry(
            native_id=native_id,
            steps=[
                ModelStep(step_index=1, source=source_label, status="completed", input_tokens=1000, output_tokens=200),
                ModelStep(step_index=2, source=source_label, status="completed", input_tokens=1500, output_tokens=300),
            ]
        )

    node = build_thread_telemetry("t-agy", tmp_path, agy_reader=mock_reader)
    assert node.driver == "antigravity"
    assert node.turn_metrics_availability == "unavailable"
    assert node.current_turn_metrics.availability == "unavailable"
    assert node.latest_step_label == "Model step #2"
    assert node.steps[0].row_label == "Model step #1"
    assert node.steps[0].activity_kind == "model_step"
    assert node.self_metrics.model_step_count == 2
    assert node.self_metrics.provider_turn_count == 0


def test_mixed_provider_rollup_counts_and_unchanged_totals(tmp_path: Path):
    """Mixed provider rollup counts turns and inference steps separately while preserving legacy totals."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    # Parent thread: Codex with 2 provider turns
    p_payload = {"modelSelection": {"model": "gpt-5-codex"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-parent", "proj", "Parent Orchestrator", "codex", "full-access", "default",
            "pt-parent", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(p_payload), "codex"))

    pt_p_payload = {"nativeThreadRef": {"driver": "codex", "nativeId": "c-parent-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-parent", "t-parent", "node-1", "codex", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_p_payload), "codex", "codex"))

    # Parent 2 turns
    for ord_val, in_tok, out_tok in [(1, 2000, 400), (2, 3000, 600)]:
        t_pay = {
            "nativeTurnRef": {"driver": "codex", "nativeId": f"turn-{ord_val}"},
            "turnTokenUsage": {"usageStatus": "complete", "inputTokens": in_tok, "outputTokens": out_tok}
        }
        conn.execute("""
        INSERT INTO orchestration_v2_projection_provider_turns
        (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (f"pturn-{ord_val}", "t-parent", "pt-parent", "node-1", ord_val, "completed",
                "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t_pay)))

    # Child thread: Antigravity delegated subagent
    c_payload = {"modelSelection": {"model": "gemini-flash"}, "lineage": {"parentThreadId": "t-parent"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-child", "proj", "Child Worker", "antigravity", "full-access", "default",
            "pt-child", "2026-11-03T10:02:00Z", "2026-11-03T10:04:00Z", None,
            json.dumps(c_payload), "antigravity_gemini"))

    pt_c_payload = {"nativeThreadRef": {"driver": "antigravity", "nativeId": "agy-child-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-child", "t-child", "node-2", "antigravity", "idle", "2026-11-03T10:04:00Z",
            json.dumps(pt_c_payload), "antigravity", "antigravity_gemini"))

    sub_pay = {"title": "Child Worker", "model": "gemini-flash"}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_subagents
    (subagent_id, thread_id, parent_node_id, provider, provider_thread_id, child_thread_id,
     origin, status, started_at, completed_at, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("sub-c1", "t-parent", "node-1", "antigravity", "pt-child", "t-child",
            "app_owned", "completed", "2026-11-03T10:02:00Z", "2026-11-03T10:04:00Z",
            "2026-11-03T10:04:00Z", json.dumps(sub_pay), "antigravity", "antigravity_gemini"))

    conn.commit()
    conn.close()

    def mock_reader(native_id: str, t3_dir: Path, source_label: str = "main"):
        return ConversationTelemetry(
            native_id=native_id,
            steps=[
                ModelStep(step_index=1, source=source_label, status="completed", input_tokens=500, output_tokens=100),
                ModelStep(step_index=2, source=source_label, status="completed", input_tokens=600, output_tokens=150),
                ModelStep(step_index=3, source=source_label, status="completed", input_tokens=700, output_tokens=200),
            ]
        )

    node = build_thread_telemetry("t-parent", tmp_path, agy_reader=mock_reader)
    assert len(node.subagents) == 1
    child_node = node.subagents[0]

    # Verify self metrics separation
    assert node.self_metrics.step_count == 2
    assert node.self_metrics.provider_turn_count == 2
    assert node.self_metrics.model_step_count == 0
    assert node.self_metrics.steps_display == "2 turns"

    assert child_node.self_metrics.step_count == 3
    assert child_node.self_metrics.provider_turn_count == 0
    assert child_node.self_metrics.model_step_count == 3
    assert child_node.self_metrics.steps_display == "3 steps"

    # Verify rollup metrics separation
    assert node.rollup_metrics.step_count == 5  # 2 + 3 (legacy step_count preserved)
    assert node.rollup_metrics.provider_turn_count == 2
    assert node.rollup_metrics.model_step_count == 3
    assert node.rollup_metrics.steps_display == "2 turns, 3 steps"

    # Whole chat totals unchanged
    assert node.rollup_metrics.input_tokens == 5000 + 1800  # (2000 + 3000) + (500 + 600 + 700)
    assert node.rollup_metrics.output_tokens == 1000 + 450  # (400 + 600) + (100 + 150 + 200)

    # Provider steps have no synthetic turn in tool_calls, row_label is Provider turn #N, action is unknown
    assert node.steps[0].activity_kind == "provider_turn"
    assert node.steps[0].row_label == "Provider turn #1"
    assert node.steps[0].tool_calls == []
    assert node.steps[0].action_summary == "unknown"


def test_context_high_to_low_sequence_retains_peak(tmp_path: Path):
    """A new reading below an old peak lowers the current reading while retaining the peak."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-drop", "proj", "Drop Test", "claudeAgent", "full-access", "default",
            "pt-drop", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "claude_inst"))

    pt_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-drop-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-drop", "t-drop", "node-1", "claudeAgent", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    # Turn 1: 80,000 tokens
    t1_payload = {
        "tokenUsage": {"usedTokens": 80000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-turn-1", "t-drop", "pt-drop", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t1_payload)))

    # Turn 2: 25,000 tokens (Context drop!)
    t2_payload = {
        "tokenUsage": {"usedTokens": 25000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 500, "outputTokens": 100}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-turn-2", "t-drop", "pt-drop", "node-1", 2, "completed",
            "2026-11-03T10:02:00Z", "2026-11-03T10:02:10Z", json.dumps(t2_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-drop", tmp_path)
    assert node.context_used_tokens == 25000
    assert node.observed_peak_tokens == 80000
    assert node.context_limit == 200000
    assert pytest.approx(node.context_pct, rel=1e-3) == 12.5
    assert node.context_provenance == "measured"
    assert len(node.context_samples) == 2
    assert node.context_samples[0].used_tokens == 80000
    assert node.context_samples[1].used_tokens == 25000


def test_context_timestamp_recency_over_token_magnitude(tmp_path: Path):
    """Timestamp recency determines freshest reading rather than token magnitude."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-recency", "proj", "Recency Test", "claudeAgent", "full-access", "default",
            "pt-recency", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "claude_inst"))

    pt_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-rec-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-recency", "t-recency", "node-1", "claudeAgent", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    # Turn A: larger tokens but earlier timestamp
    tA_payload = {
        "tokenUsage": {"usedTokens": 95000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-A", "t-recency", "pt-recency", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(tA_payload)))

    # Turn B: smaller tokens but later timestamp
    tB_payload = {
        "tokenUsage": {"usedTokens": 32000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 500, "outputTokens": 100}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-B", "t-recency", "pt-recency", "node-1", 2, "completed",
            "2026-11-03T10:05:00Z", "2026-11-03T10:05:15Z", json.dumps(tB_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-recency", tmp_path)
    assert node.context_used_tokens == 32000
    assert node.observed_peak_tokens == 95000


def test_context_explicit_zero_valid(tmp_path: Path):
    """Explicit zero context usage is valid and does not fallback to older peak."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-zero", "proj", "Zero Test", "claudeAgent", "full-access", "default",
            "pt-zero", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "claude_inst"))

    pt_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-zero-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-zero", "t-zero", "node-1", "claudeAgent", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    t1_payload = {
        "tokenUsage": {"usedTokens": 40000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-z1", "t-zero", "pt-zero", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t1_payload)))

    # Turn 2: explicit zero
    t2_payload = {
        "tokenUsage": {"usedTokens": 0, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 0, "outputTokens": 0}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-z2", "t-zero", "pt-zero", "node-1", 2, "completed",
            "2026-11-03T10:02:00Z", "2026-11-03T10:02:10Z", json.dumps(t2_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-zero", tmp_path)
    assert node.context_used_tokens == 0
    assert node.context_pct == 0.0
    assert node.observed_peak_tokens == 40000


def test_context_missing_limits_visibly_unknown(tmp_path: Path):
    """Unknown limits remain visibly unknown, preserving legacy 1M JSON default with explicit provenance."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-nolim", "proj", "No Limit Test", "claudeAgent", "full-access", "default",
            "pt-nolim", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "claude_inst"))

    pt_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-nolim-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-nolim", "t-nolim", "node-1", "claudeAgent", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    # Turn with usedTokens but no maxTokens
    t_payload = {
        "tokenUsage": {"usedTokens": 18000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-nl", "t-nolim", "pt-nolim", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-nolim", tmp_path)
    assert node.context_used_tokens == 18000
    assert node.context_limit is None
    assert node.context_limit_provenance == "default"
    assert node.context_max_tokens == 1048576  # Legacy JSON numeric default preserved

    data = node.to_dict()
    assert data["context_limit"] is None
    assert data["context_limit_provenance"] == "default"
    assert data["context_max_tokens"] == 1048576


def test_context_missing_timestamps_labels_uncertainty(tmp_path: Path):
    """Missing timestamps do not fabricate event times, and ordering uncertainty is labeled."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "claude-3-7-sonnet"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-nots", "proj", "No TS Test", "claudeAgent", "full-access", "default",
            "pt-nots", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "claude_inst"))

    pt_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-nots-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-nots", "t-nots", "node-1", "claudeAgent", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt_payload), "claudeAgent", "claude_inst"))

    t1_payload = {
        "tokenUsage": {"usedTokens": 22000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-ts1", "t-nots", "pt-nots", "node-1", 1, "completed",
            None, None, json.dumps(t1_payload)))

    t2_payload = {
        "tokenUsage": {"usedTokens": 38000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-ts2", "t-nots", "pt-nots", "node-1", 2, "completed",
            None, None, json.dumps(t2_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-nots", tmp_path)
    assert node.context_used_tokens == 38000
    assert node.context_uncertainty is True
    # Verify timestamps are honestly None and not fabricated
    assert node.context_samples[0].timestamp is None
    assert node.context_samples[1].timestamp is None


def test_context_provider_switch_constrains_history_and_limits(tmp_path: Path):
    """Switching providers constrains context history to active provider thread and does not join incompatible histories."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    # App thread points to active provider thread pt-2 (codex)
    th_payload = {"modelSelection": {"model": "gpt-5-codex"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-switch", "proj", "Switch Test", "codex", "full-access", "default",
            "pt-2", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None,
            json.dumps(th_payload), "codex_inst"))

    # Old provider thread pt-1 (claude)
    pt1_payload = {"nativeThreadRef": {"driver": "claudeAgent", "nativeId": "c-old"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-1", "t-switch", "node-1", "claudeAgent", "idle", "2026-11-03T10:01:00Z",
            json.dumps(pt1_payload), "claudeAgent", "claude_inst"))

    # Active provider thread pt-2 (codex)
    pt2_payload = {"nativeThreadRef": {"driver": "codex", "nativeId": "c-new"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-2", "t-switch", "node-2", "codex", "idle", "2026-11-03T10:05:00Z",
            json.dumps(pt2_payload), "codex", "codex_inst"))

    # Turn under pt-1 (Claude: 85,000 / 200,000)
    t1_payload = {
        "tokenUsage": {"usedTokens": 85000, "maxTokens": 200000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 2000, "outputTokens": 500}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-pt1", "t-switch", "pt-1", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t1_payload)))

    # Turn under pt-2 (Codex: 12,000 / 128,000)
    t2_payload = {
        "tokenUsage": {"usedTokens": 12000, "maxTokens": 128000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 300}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-pt2", "t-switch", "pt-2", "node-2", 2, "completed",
            "2026-11-03T10:04:00Z", "2026-11-03T10:04:15Z", json.dumps(t2_payload)))

    conn.commit()
    conn.close()

    node = build_thread_telemetry("t-switch", tmp_path)
    # Context must reflect active pt-2 only
    assert node.context_used_tokens == 12000
    assert node.context_limit == 128000
    assert node.observed_peak_tokens == 12000
    assert node.active_provider_thread_id == "pt-2"
    # History must contain only pt-2 samples
    assert len(node.context_samples) == 1
    assert node.context_samples[0].provider_thread_id == "pt-2"
    assert node.context_samples[0].used_tokens == 12000


def test_context_measured_wins_over_antigravity_unless_provably_stale(tmp_path: Path):
    """Measured provider context wins unless provably stale relative to timestamped Antigravity inference."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    th_payload = {"modelSelection": {"model": "gemini-2.5-pro"}, "lineage": {"parentThreadId": None}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-agy-stale", "proj", "Agy Stale", "antigravity", "full-access", "default",
            "pt-agy-s", "2026-11-03T10:00:00Z", "2026-11-03T10:06:00Z", None,
            json.dumps(th_payload), "antigravity_gemini"))

    pt_payload = {"nativeThreadRef": {"driver": "antigravity", "nativeId": "agy-stale-1"}}
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-agy-s", "t-agy-stale", "node-1", "antigravity", "idle", "2026-11-03T10:06:00Z",
            json.dumps(pt_payload), "antigravity", "antigravity_gemini"))

    # Provider turn completed at 10:01:00Z with measured 40,000 tokens
    t_payload = {
        "tokenUsage": {"usedTokens": 40000, "maxTokens": 1000000},
        "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 1000, "outputTokens": 200}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-meas", "t-agy-stale", "pt-agy-s", "node-1", 1, "completed",
            "2026-11-03T10:01:00Z", "2026-11-03T10:01:10Z", json.dumps(t_payload)))

    conn.commit()
    conn.close()

    # Case A: Antigravity step completed at 10:05:00Z (provably newer than 10:01:10Z)
    def mock_reader_newer(native_id: str, t3_dir: Path, source_label: str = "main"):
        return ConversationTelemetry(
            native_id=native_id,
            steps=[
                ModelStep(
                    step_index=1,
                    source=source_label,
                    status="completed",
                    input_tokens=10000,
                    cache_read_tokens=4000,
                    output_tokens=1000,
                    started_at="2026-11-03T10:04:50Z",
                    completed_at="2026-11-03T10:05:00Z",
                )
            ]
        )

    node_newer = build_thread_telemetry("t-agy-stale", tmp_path, agy_reader=mock_reader_newer)
    # Provably stale provider measurement falls back to freshest Antigravity estimate (10000+4000+1000=15000)
    assert node_newer.context_used_tokens == 15000
    assert node_newer.context_provenance == "estimated"

    # Case B: Antigravity step completed at 10:00:30Z (older than provider 10:01:10Z)
    def mock_reader_older(native_id: str, t3_dir: Path, source_label: str = "main"):
        return ConversationTelemetry(
            native_id=native_id,
            steps=[
                ModelStep(
                    step_index=1,
                    source=source_label,
                    status="completed",
                    input_tokens=5000,
                    cache_read_tokens=1000,
                    output_tokens=500,
                    started_at="2026-11-03T10:00:00Z",
                    completed_at="2026-11-03T10:00:30Z",
                )
            ]
        )

    # Fresh provider measurement wins over older Antigravity estimate
    node_older = build_thread_telemetry("t-agy-stale", tmp_path, agy_reader=mock_reader_older)
    assert node_older.context_used_tokens == 40000
    assert node_older.context_provenance == "measured"



def test_native_subagent_context_estimation_with_zero_tokens(tmp_path: Path):
    """Explicit zero token estimate is preserved with estimated provenance and single sample."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    parent_payload = {
        "modelSelection": {"model": "gemini-3.8-flash"},
        "lineage": {"parentThreadId": None, "rootThreadId": "t-parent-zero"}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-parent-zero", "proj", "Parent Zero", "antigravity", "full-access", "default",
            "pt-pzero", "2026-11-03T10:00:00Z", "2026-11-03T10:10:00Z", None,
            json.dumps(parent_payload), "antigravity_gemini"))

    pt_payload = {
        "nativeThreadRef": {"driver": "antigravity", "nativeId": "agy-parent-zero"},
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-pzero", "t-parent-zero", "node-1", "antigravity", "idle", "2026-11-03T10:10:00Z",
            json.dumps(pt_payload), "antigravity", "antigravity_gemini"))
    conn.commit()
    conn.close()

    def mock_agy_reader(native_id: str, t3_dir: Path, source_label: str = "main") -> ConversationTelemetry:
        if native_id == "agy-parent-zero":
            return ConversationTelemetry(
                native_id=native_id,
                child_native_ids=["native-child-zero"],
                steps=[ModelStep(step_index=1, source=source_label, status="completed",
                                 input_tokens=100, output_tokens=50, started_at="2026-11-03T10:01:00Z")]
            )
        elif native_id == "native-child-zero":
            return ConversationTelemetry(
                native_id=native_id,
                steps=[ModelStep(step_index=1, source=source_label, status="completed",
                                 input_tokens=0, cache_read_tokens=0, output_tokens=0,
                                 started_at="2026-11-03T10:02:00Z")]
            )
        raise ValueError(f"Unknown native_id: {native_id}")

    node = build_thread_telemetry("t-parent-zero", tmp_path, agy_reader=mock_agy_reader)
    assert len(node.subagents) == 1
    child = node.subagents[0]
    assert child.native_id == "native-child-zero"
    assert child.context_used_tokens == 0
    assert child.context_provenance == "estimated"
    assert child.context_limit is None
    assert len(child.context_samples) == 1
    assert child.context_samples[0].used_tokens == 0
    assert child.context_samples[0].source_id == "native-child-zero:step:1"
    assert child.context_samples[0].provider_thread_id == "native-child-zero"
    assert child.active_provider_thread_id == "native-child-zero"
    assert child.latest_step_label == "Model step #1"


def test_native_subagent_context_empty_no_data(tmp_path: Path):
    """Empty native subagent without steps remains unknown with no samples and no fake ceiling."""
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    parent_payload = {
        "modelSelection": {"model": "gemini-3.8-flash"},
        "lineage": {"parentThreadId": None, "rootThreadId": "t-parent-empty"}
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-parent-empty", "proj", "Parent Empty", "antigravity", "full-access", "default",
            "pt-pempty", "2026-11-03T10:00:00Z", "2026-11-03T10:10:00Z", None,
            json.dumps(parent_payload), "antigravity_gemini"))

    pt_payload = {
        "nativeThreadRef": {"driver": "antigravity", "nativeId": "agy-parent-empty"},
    }
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-pempty", "t-parent-empty", "node-1", "antigravity", "idle", "2026-11-03T10:10:00Z",
            json.dumps(pt_payload), "antigravity", "antigravity_gemini"))
    conn.commit()
    conn.close()

    def mock_agy_reader(native_id: str, t3_dir: Path, source_label: str = "main") -> ConversationTelemetry:
        if native_id == "agy-parent-empty":
            return ConversationTelemetry(
                native_id=native_id,
                child_native_ids=["native-child-empty"],
                steps=[ModelStep(step_index=1, source=source_label, status="completed",
                                 input_tokens=100, output_tokens=50, started_at="2026-11-03T10:01:00Z")]
            )
        elif native_id == "native-child-empty":
            return ConversationTelemetry(native_id=native_id, steps=[])
        raise ValueError(f"Unknown native_id: {native_id}")

    node = build_thread_telemetry("t-parent-empty", tmp_path, agy_reader=mock_agy_reader)
    assert len(node.subagents) == 1
    child = node.subagents[0]
    assert child.native_id == "native-child-empty"
    assert child.context_used_tokens == 0
    assert child.context_provenance == "unknown"
    assert child.context_limit is None
    assert child.context_max_tokens == 1_048_576
    assert child.context_limit_provenance == "default"
    assert len(child.context_samples) == 0
    assert child.latest_step_label is None


# --- Recorded context carried into steps ---


def _claude_thread_db(db_file: Path, with_context: bool) -> None:
    conn = setup_test_db(db_file)
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-c", "proj", "Claude", "claudeAgent", "full-access", "default", "pt-c",
          "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None, "{}", "claude_inst"))
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-c", "t-c", "n", "claudeAgent", "idle", "2026-11-03T10:05:00Z", "{}", "claudeAgent", "claude_inst"))
    usage = {"nativeTurnRef": {"driver": "claudeAgent"},
             "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 100, "outputTokens": 40}}
    zero = {"nativeTurnRef": {"driver": "claudeAgent"},
            "turnTokenUsage": {"usageStatus": "complete", "inputTokens": 0, "outputTokens": 0}}
    for tid, node, ordn, payload in (("turn-1", "root-1", 1, usage), ("turn-2", "root-2", 2, zero)):
        conn.execute("""
        INSERT INTO orchestration_v2_projection_provider_turns
        (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tid, "t-c", "pt-c", node, ordn, "completed", "2026-11-03T10:00:00Z", "2026-11-03T10:01:00Z",
              json.dumps(payload)))
    if with_context:
        conn.executescript("""
        CREATE TABLE orchestration_v2_projection_nodes (node_id TEXT PRIMARY KEY, thread_id TEXT, kind TEXT,
            provider_turn_id TEXT, payload_json TEXT);
        CREATE TABLE orchestration_v2_projection_messages (message_id TEXT PRIMARY KEY, thread_id TEXT, node_id TEXT,
            role TEXT, created_at TEXT, payload_json TEXT);
        INSERT INTO orchestration_v2_projection_nodes VALUES ('am-1', 't-c', 'assistant_message', 'turn-1', '{}');
        INSERT INTO orchestration_v2_projection_messages VALUES ('u1', 't-c', 'root-1', 'user', '2026-11-03T10:00:00Z', '{"text": "Run tests"}');
        INSERT INTO orchestration_v2_projection_messages VALUES ('a1', 't-c', 'am-1', 'assistant', '2026-11-03T10:00:05Z', '{"text": "Done"}');
        """)
    conn.commit()
    conn.close()


def test_provider_turn_context_carried_and_totals_unchanged(tmp_path: Path):
    from t3_telemetry.activity import build_activity_entries

    plain_dir, ctx_dir = tmp_path / "plain", tmp_path / "ctx"
    plain_dir.mkdir()
    ctx_dir.mkdir()
    _claude_thread_db(plain_dir / "statev2.sqlite", with_context=False)
    _claude_thread_db(ctx_dir / "statev2.sqlite", with_context=True)
    plain = build_thread_telemetry("t-c", plain_dir)
    rich = build_thread_telemetry("t-c", ctx_dir)

    assert plain.self_metrics.to_dict() == rich.self_metrics.to_dict()
    assert plain.rollup_metrics.to_dict() == rich.rollup_metrics.to_dict()
    assert [(r.role, r.text) for r in rich.steps[0].context_texts] == [("user", "Run tests"), ("assistant", "Done")]
    assert rich.steps[0].tool_calls == [] and rich.steps[0].tool_details == []
    assert rich.steps[1].context_texts == []

    entries = build_activity_entries(rich)
    assert entries[0].title == "Prompt · Run tests · #1"
    assert entries[1].title == "Provider turn #2 · action details unavailable"
    assert entries[1].total_tokens == 0 and entries[1].usage_status == "complete"
    assert entries[0].usage_provenance == "exact"
    # Existing JSON keys remain; new ones are additive.
    step_json = rich.to_dict()["steps"][0]
    assert {"tool_calls", "total_tokens", "action_summary", "row_label"} <= step_json.keys()
    assert step_json["context_texts"][0]["text"] == "Run tests"


def test_cumulative_fallback_turn_entry_labeled(tmp_path: Path):
    from t3_telemetry.activity import build_activity_entries

    conn = setup_test_db(tmp_path / "statev2.sqlite")
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_turns
    (provider_turn_id, thread_id, provider_thread_id, node_id, ordinal, status, started_at, completed_at, payload_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("turn-f", "t-f", "pt-f", "n", 1, "completed", None, None,
          json.dumps({"nativeTurnRef": {"driver": "codex"}, "tokenUsage": {"inputTokens": 50, "outputTokens": 5}})))
    conn.commit()
    conn.close()
    node = build_thread_telemetry("t-f", tmp_path)
    entry = build_activity_entries(node)[0]
    assert entry.usage_provenance == "cumulative_fallback" and entry.total_tokens == 55


def test_antigravity_root_and_native_child_retain_tool_details(tmp_path: Path):
    from t3_telemetry.activity import build_activity_entries

    root_id = "aaaaaaaa-1111-2222-3333-444444444444"
    child_id = "bbbbbbbb-1111-2222-3333-444444444444"
    conn = setup_test_db(tmp_path / "statev2.sqlite")
    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t-a", "proj", "Agy", "antigravity", "full-access", "default", "pt-a",
          "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None, "{}", "antigravity"))
    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt-a", "t-a", "n", "antigravity", "idle", "2026-11-03T10:05:00Z",
          json.dumps({"nativeThreadRef": {"nativeId": root_id}}), "antigravity", "antigravity"))
    conn.commit()
    conn.close()

    def transcript(native_id, records):
        path = tmp_path / "brain" / native_id / ".system_generated" / "logs" / "transcript.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    transcript(root_id, [
        {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 10, "tool_calls": [
            {"name": "invoke_subagent", "args": {"TypeName": "reviewer"}}]},
        {"step_index": 2, "type": "AGENCY_TOOL_CALL", "content": f"conversationId: {child_id}"},
    ])
    transcript(child_id, [
        {"step_index": 1, "type": "PLANNER_RESPONSE", "input_tokens": 7, "tool_calls": [
            {"name": "run_command", "args": json.dumps({"CommandLine": "uv run pytest"})}]},
    ])

    node = build_thread_telemetry("t-a", tmp_path)
    assert build_activity_entries(node)[0].title == "Launch subagent · reviewer · #1"
    child = node.subagents[0]
    assert child.native_id == child_id
    assert build_activity_entries(child)[0].title == "Run command · uv run pytest · #1"
    assert node.rollup_metrics.input_tokens == 17


# --- Subagent launches in the parent feed ---------------------------------

from t3_telemetry.db_t3 import T3StateSnapshot
from t3_telemetry.models import (
    AppThreadRecord,
    NativeLaunchRecord,
    ProviderThreadRecord,
    ProviderTurnRecord,
    SubagentLinkRecord,
)


def _app_thread(tid, title, provider="claudeAgent", parent=None, created=None, pt=None):
    return AppThreadRecord(
        thread_id=tid, project_id="p", title=title, default_provider=provider, provider_instance_id=provider,
        model="m", thinking_level=None, active_provider_thread_id=pt, parent_thread_id=parent,
        root_thread_id=None, relationship_to_parent=None, created_at=created, updated_at=created,
    )


def _turn(tid, ordinal, started, tokens=100):
    return ProviderTurnRecord(
        provider_turn_id=f"{tid}-turn-{ordinal}", thread_id=tid, provider_thread_id=f"pt-{tid}", driver="claudeAgent",
        native_turn_id=None, ordinal=ordinal, status="completed", started_at=started, completed_at=started,
        input_tokens=tokens, output_tokens=10, usage_status="complete", usage_provenance="exact",
    )


def _link(sid, parent, child, started=None, title="review", status="completed"):
    return SubagentLinkRecord(
        subagent_id=sid, parent_thread_id=parent, child_thread_id=child, driver="claudeAgent",
        provider_instance_id=None, model=None, title=title, status=status, started_at=started,
    )


def _launch_snapshot():
    snap = T3StateSnapshot()
    snap.threads["par"] = _app_thread("par", "Parent", created="2026-11-03T10:00:00Z")
    snap.threads["kid"] = _app_thread("kid", "Kid", parent="par", created="2026-11-03T10:01:30Z")
    snap.provider_turns_by_thread["par"] = [
        _turn("par", 1, "2026-11-03T10:01:00Z"),
        _turn("par", 2, "2026-11-03T10:03:00Z"),
    ]
    snap.provider_turns_by_thread["kid"] = [_turn("kid", 1, "2026-11-03T10:02:00Z", tokens=40)]
    return snap


def test_app_owned_launch_dedups_lineage_and_orders_by_time(tmp_path: Path):
    from t3_telemetry.activity import KIND_LAUNCH, build_activity_entries

    snap = _launch_snapshot()
    snap.subagents_by_parent["par"] = [_link("s1", "par", "kid", started="2026-11-03T10:02:00Z")]
    node = build_thread_telemetry("par", tmp_path, snapshot=snap)

    assert len(node.launches) == 1  # Source A and lineage (B) found the same invocation
    launch = node.launches[0]
    assert launch.launch_id == "subagent:s1" and launch.sources == ["A", "B"] and launch.child_present
    entries = build_activity_entries(node)
    assert [e.kind for e in entries] == ["provider_turn", KIND_LAUNCH, "provider_turn"]
    marker = entries[1]
    assert marker.total_tokens is None and marker.child_thread_id == "kid"
    assert "review" in marker.title
    # Metrics are unchanged by launches: the child's turn counts once, only in rollup.
    assert node.self_metrics.provider_turn_count == 2 and node.self_metrics.input_tokens == 200
    assert node.rollup_metrics.input_tokens == 240
    assert all(e.detail is None or e.detail.provider_turn_id.startswith("par-") for e in entries)
    assert node.to_dict()["launches"][0]["launch_id"] == "subagent:s1"


def test_repeated_invocations_stay_distinct_and_undated_is_labeled(tmp_path: Path):
    from t3_telemetry.activity import UNDATED_LABEL, build_activity_entries

    snap = _launch_snapshot()
    snap.subagents_by_parent["par"] = [
        _link("s1", "par", "kid", started="2026-11-03T10:02:00Z"),
        _link("s2", "par", "kid", started=None, title="second pass"),
        _link("s3", "par", None, title="never materialized"),
    ]
    node = build_thread_telemetry("par", tmp_path, snapshot=snap)
    assert [l.launch_id for l in node.launches] == ["subagent:s1", "subagent:s2", "subagent:s3"]
    assert len(node.subagents) == 1 and node.self_metrics.total_subagents == 1
    entries = build_activity_entries(node)
    titles = [e.title for e in entries]
    assert titles[0].startswith(f"{UNDATED_LABEL} · ") and "second pass" in titles[0]
    assert titles[1].startswith(f"{UNDATED_LABEL} · ") and "never materialized" in titles[1]
    assert entries[1].child_thread_id is None  # no child: inspectable, not navigable
    assert entries[1].launch.launched_at is None
    assert len({e.entry_id for e in entries}) == len(entries)


def test_lineage_only_child_uses_parent_child_fallback(tmp_path: Path):
    snap = _launch_snapshot()
    node = build_thread_telemetry("par", tmp_path, snapshot=snap)
    assert [(l.launch_id, l.sources, l.launch_time_source) for l in node.launches] == [
        ("child:kid", ["B"], "child thread created")
    ]


def test_max_steps_keeps_counted_activity_and_merges_launch_markers(tmp_path: Path):
    from t3_telemetry.activity import KIND_LAUNCH, build_activity_entries

    snap = _launch_snapshot()
    snap.provider_turns_by_thread["par"] = [_turn("par", i, f"2026-11-03T10:0{i}:00Z") for i in range(1, 6)]
    snap.subagents_by_parent["par"] = [
        _link(f"s{i}", "par", None, started=f"2026-11-03T10:0{i}:30Z", title=f"job {i}") for i in range(1, 6)
    ]
    node = build_thread_telemetry("par", tmp_path, snapshot=snap)
    entries = build_activity_entries(node, max_steps=2)
    counted = [e for e in entries if e.kind != KIND_LAUNCH]
    markers = [e for e in entries if e.kind == KIND_LAUNCH]
    assert [e.detail.step_index for e in counted] == [4, 5]  # --max-steps unchanged for counted activity
    assert {m.launch.launch_id for m in markers} == {launch.launch_id for launch in node.launches}
    assert [m.launch.title for m in markers if m.launch.launch_id.startswith("subagent:")] == [
        f"job {i}" for i in range(1, 6)
    ]
    assert entries[0].launch.title == "job 1" and entries[-1].launch.title == "job 5"
    assert len(build_activity_entries(node, max_steps=0)) == 10 + 1  # + lineage child "kid"


def test_native_launch_merges_with_app_link_by_conversation_id(tmp_path: Path):
    snap = T3StateSnapshot()
    snap.threads["root"] = _app_thread("root", "Root", provider="antigravity", pt="pt-root")
    snap.provider_threads["pt-root"] = ProviderThreadRecord(
        provider_thread_id="pt-root", thread_id="root", driver="antigravity", provider_instance_id="antigravity",
        native_id="nat-root", status="idle",
    )
    snap.threads["kid"] = _app_thread("kid", "Kid", provider="antigravity", pt="pt-kid")
    snap.provider_threads["pt-kid"] = ProviderThreadRecord(
        provider_thread_id="pt-kid", thread_id="kid", driver="antigravity", provider_instance_id="antigravity",
        native_id="nat-kid", status="idle",
    )
    snap.subagents_by_parent["root"] = [_link("s1", "root", "kid", title="")]

    def reader(native_id, _dir, label):
        if native_id == "nat-root":
            return ConversationTelemetry(
                native_id=native_id,
                steps=[ModelStep(step_index=4, source=label, tool_calls=["invoke_subagent"], input_tokens=5,
                                 started_at="2026-11-03T10:00:00Z")],
                child_native_ids=["nat-kid", "nat-other"],
                launches=[NativeLaunchRecord("nat-kid", 4, "2026-11-03T10:00:01Z", "reviewer", "invoke_subagent")],
            )
        return ConversationTelemetry(native_id=native_id, steps=[ModelStep(step_index=1, source=label, input_tokens=7)])

    node = build_thread_telemetry("root", tmp_path, snapshot=snap, agy_reader=reader)
    by_id = {l.launch_id: l for l in node.launches}
    assert set(by_id) == {"subagent:s1", "native:nat-other"}
    merged = by_id["subagent:s1"]
    assert merged.sources == ["A", "C"] and merged.parent_step_index == 4 and merged.title == "reviewer"
    assert merged.launched_at == "2026-11-03T10:00:01Z" and merged.launch_time_source == "transcript tool call"
    other = by_id["native:nat-other"]
    assert other.launched_at is None and other.child_present  # native child built from Source C
    assert node.rollup_metrics.input_tokens == 5 + 7 + 7


def test_token_metrics_include_cache_creation():
    from t3_telemetry.models import TokenMetricsSummary

    step = ModelStep(step_index=1, input_tokens=6, cache_read_tokens=241250,
                     cache_creation_tokens=943, output_tokens=907)
    summary = TokenMetricsSummary()
    summary.add_step(step)
    for metrics in (step, summary):
        assert metrics.total_tokens == 243106
        assert metrics.new_input_tokens == 949
        assert metrics.context_read_tokens == 241250
        data = metrics.to_dict()
        assert data["total_tokens"] == 243106
        assert data["new_input_tokens"] == 949
        assert data["context_read_tokens"] == 241250
    assert summary.cache_hit_pct == pytest.approx(241250 / 242199 * 100)
    assert summary.to_dict()["cache_hit_pct"] == round(241250 / 242199 * 100, 2)


def test_provider_turn_step_context_used_and_max_tokens(tmp_path: Path):
    from t3_telemetry.db_t3 import T3StateSnapshot
    from t3_telemetry.models import AppThreadRecord

    snap = T3StateSnapshot()
    snap.threads["t-test"] = AppThreadRecord(
        thread_id="t-test",
        project_id="proj",
        title="Claude Turn Context",
        default_provider="claudeAgent",
        provider_instance_id="claude_inst",
        model="claude-3-7-sonnet",
        thinking_level=None,
        active_provider_thread_id="pt-test",
        parent_thread_id=None,
        root_thread_id="t-test",
        relationship_to_parent=None,
        created_at="2026-11-03T10:00:00Z",
        updated_at="2026-11-03T10:05:00Z",
    )
    snap.provider_turns_by_thread["t-test"] = [
        ProviderTurnRecord(
            provider_turn_id="turn-ctx-1",
            thread_id="t-test",
            provider_thread_id="pt-test",
            driver="claudeAgent",
            native_turn_id="turn-1",
            ordinal=1,
            status="completed",
            started_at="2026-11-03T10:00:00Z",
            completed_at="2026-11-03T10:00:10Z",
            duration_seconds=10.0,
            used_tokens=163201,
            max_tokens=1000000,
            input_tokens=1000,
            output_tokens=200,
            usage_status="complete",
            usage_provenance="exact",
        )
    ]

    node = build_thread_telemetry("t-test", tmp_path, snapshot=snap)
    assert len(node.steps) == 1
    step = node.steps[0]
    assert step.context_used_tokens == 163201
    assert step.context_max_tokens == 1000000
    step_dict = step.to_dict()
    assert step_dict["context_used_tokens"] == 163201
    assert step_dict["context_max_tokens"] == 1000000

    # Antigravity steps leave them None
    agy_step = ModelStep(step_index=1, source="main", status="completed")
    assert agy_step.context_used_tokens is None
    assert agy_step.context_max_tokens is None
    agy_dict = agy_step.to_dict()
    assert agy_dict["context_used_tokens"] is None
    assert agy_dict["context_max_tokens"] is None
