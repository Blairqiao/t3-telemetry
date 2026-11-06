from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import pytest

from t3_telemetry.aggregator import (
    build_subagent_launches,
    resolve_thread_id_in_snapshot,
    running_root_ids,
    snapshot_thread_status,
)
from t3_telemetry.db_t3 import T3StateSnapshot, read_t3_state
from t3_telemetry.models import (
    AppThreadRecord,
    ConversationTelemetry,
    ModelStep,
    NativeLaunchRecord,
    ProviderThreadRecord,
    ProviderTurnRecord,
    SubagentLinkRecord,
    ThreadTelemetryNode,
    TokenMetricsSummary,
)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE orchestration_v2_projection_threads (\n        thread_id TEXT PRIMARY KEY,\n        project_id TEXT NOT NULL,\n        title TEXT NOT NULL,\n        default_provider TEXT NOT NULL,\n        runtime_mode TEXT NOT NULL,\n        interaction_mode TEXT NOT NULL,\n        active_provider_thread_id TEXT,\n        created_at TEXT NOT NULL,\n        updated_at TEXT NOT NULL,\n        archived_at TEXT,\n        deleted_at TEXT,\n        payload_json TEXT NOT NULL,\n        provider_instance_id TEXT\n    );\n
    CREATE TABLE orchestration_v2_projection_provider_threads (\n        provider_thread_id TEXT PRIMARY KEY,\n        thread_id TEXT,\n        owner_node_id TEXT,\n        provider TEXT NOT NULL,\n        provider_session_id TEXT,\n        status TEXT NOT NULL,\n        first_run_ordinal INTEGER,\n        last_run_ordinal INTEGER,\n        updated_at TEXT NOT NULL,\n        payload_json TEXT NOT NULL,\n        driver TEXT,\n        provider_instance_id TEXT\n    );\n
    CREATE TABLE orchestration_v2_projection_provider_turns (\n        provider_turn_id TEXT PRIMARY KEY,\n        thread_id TEXT NOT NULL,\n        provider_thread_id TEXT NOT NULL,\n        node_id TEXT NOT NULL,\n        run_attempt_id TEXT,\n        ordinal INTEGER NOT NULL,\n        status TEXT NOT NULL,\n        started_at TEXT,\n        completed_at TEXT,\n        payload_json TEXT NOT NULL\n    );\n
    CREATE TABLE orchestration_v2_projection_subagents (\n        subagent_id TEXT PRIMARY KEY,\n        thread_id TEXT NOT NULL,\n        run_id TEXT,\n        parent_node_id TEXT NOT NULL,\n        provider TEXT NOT NULL,\n        provider_thread_id TEXT,\n        child_thread_id TEXT,\n        origin TEXT NOT NULL,\n        status TEXT NOT NULL,\n        started_at TEXT,\n        completed_at TEXT,\n        updated_at TEXT NOT NULL,\n        payload_json TEXT NOT NULL,\n        driver TEXT,\n        provider_instance_id TEXT,\n        completion_delivery_state TEXT\n    );\n    """)


def setup_test_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    return conn


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


def test_resolve_thread_id_in_snapshot(tmp_path: Path):
    db_file = tmp_path / "statev2.sqlite"
    conn = setup_test_db(db_file)

    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t1", "p", "Feature implementation", "claudeAgent", "full-access", "default",
          "pt1", "2026-11-03T10:00:00Z", "2026-11-03T10:05:00Z", None, "{}", "claude"))

    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t2", "p", "Bug fix", "codex", "full-access", "default",
          "pt2", "2026-11-03T10:01:00Z", "2026-11-03T10:06:00Z", None, "{}", "codex"))

    conn.execute("""
    INSERT INTO orchestration_v2_projection_threads
    (thread_id, project_id, title, default_provider, runtime_mode, interaction_mode,
     active_provider_thread_id, created_at, updated_at, deleted_at, payload_json, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("t3", "p", "Docs update", "antigravity", "full-access", "default",
          "pt3", "2026-11-03T10:02:00Z", "2026-11-03T10:07:00Z", None, "{}", "antigravity"))

    conn.execute("""
    INSERT INTO orchestration_v2_projection_provider_threads
    (provider_thread_id, thread_id, owner_node_id, provider, status, updated_at, payload_json, driver, provider_instance_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ("pt2", "t2", "node", "codex", "idle", "2026-11-03T10:06:00Z",
          json.dumps({"nativeThreadRef": {"driver": "codex", "nativeId": "native-codex-session-id"}}),
          "codex", "codex"))

    conn.commit()
    conn.close()

    snapshot = read_t3_state(tmp_path)
    assert resolve_thread_id_in_snapshot(snapshot, None) == "t3"
    assert resolve_thread_id_in_snapshot(snapshot, "t1") == "t1"
    assert resolve_thread_id_in_snapshot(snapshot, "native-codex") == "t2"
    assert resolve_thread_id_in_snapshot(snapshot, "feature") == "t1"
    assert resolve_thread_id_in_snapshot(snapshot, "nonexistent-xyz") is None


def _node(tid, title="node", driver="claudeAgent", prov=None, model="m", native_id=None, status="completed", subagents=None):
    return ThreadTelemetryNode(
        thread_id=tid, title=title, driver=driver, provider_instance_id=prov, model=model,
        native_id=native_id, status=status, subagents=list(subagents or ()),
    )


def test_app_owned_launch_dedups_lineage_and_orders_by_time():
    from t3_telemetry.activity import KIND_LAUNCH, build_activity_entries

    snap = _launch_snapshot()
    snap.subagents_by_parent["par"] = [_link("s1", "par", "kid", started="2026-11-03T10:02:00Z")]
    node = _node("par", subagents=[_node("kid", "Kid", status="completed")])
    node.launches = build_subagent_launches(node, snap.subagents_by_parent.get("par", []), snap, None)

    assert len(node.launches) == 1  # Source A and lineage (B) found the same invocation
    launch = node.launches[0]
    assert launch.launch_id == "subagent:s1" and launch.sources == ["A", "B"] and launch.child_present
    node.steps = [
        ModelStep(step_index=1, status="completed", activity_kind="provider_turn", started_at="2026-11-03T10:01:00Z", provider_turn_id="par-turn-1", input_tokens=100),
        ModelStep(step_index=2, status="completed", activity_kind="provider_turn", started_at="2026-11-03T10:03:00Z", provider_turn_id="par-turn-2", input_tokens=100),
    ]
    entries = build_activity_entries(node)
    assert [e.kind for e in entries] == ["provider_turn", KIND_LAUNCH, "provider_turn"]
    marker = entries[1]
    assert marker.total_tokens is None and marker.child_thread_id == "kid"
    assert "review" in marker.title
    assert all(e.detail is None or e.detail.provider_turn_id.startswith("par-") for e in entries)
    assert node.to_dict()["launches"][0]["launch_id"] == "subagent:s1"


def test_repeated_invocations_stay_distinct_and_undated_is_labeled():
    from t3_telemetry.activity import UNDATED_LABEL, build_activity_entries

    snap = _launch_snapshot()
    snap.subagents_by_parent["par"] = [
        _link("s1", "par", "kid", started="2026-11-03T10:02:00Z"),
        _link("s2", "par", "kid", started=None, title="second pass"),
        _link("s3", "par", None, title="never materialized"),
    ]
    node = _node("par", subagents=[_node("kid", "Kid", status="completed")])
    node.launches = build_subagent_launches(node, snap.subagents_by_parent.get("par", []), snap, None)
    assert [l.launch_id for l in node.launches] == ["subagent:s1", "subagent:s2", "subagent:s3"]
    entries = build_activity_entries(node)
    titles = [e.title for e in entries]
    assert titles[0].startswith(f"{UNDATED_LABEL} · ") and "second pass" in titles[0]
    assert titles[1].startswith(f"{UNDATED_LABEL} · ") and "never materialized" in titles[1]
    assert entries[1].child_thread_id is None  # no child: inspectable, not navigable
    assert entries[1].launch.launched_at is None
    assert len({e.entry_id for e in entries}) == len(entries)


def test_lineage_only_child_uses_parent_child_fallback():
    snap = _launch_snapshot()
    node = _node("par", subagents=[_node("kid", "Kid", status="completed")])
    node.launches = build_subagent_launches(node, snap.subagents_by_parent.get("par", []), snap, None)
    assert [(l.launch_id, l.sources, l.launch_time_source) for l in node.launches] == [
        ("child:kid", ["B"], "child thread created")
    ]


def test_max_steps_keeps_counted_activity_and_merges_launch_markers():
    from t3_telemetry.activity import KIND_LAUNCH, build_activity_entries

    snap = _launch_snapshot()
    snap.subagents_by_parent["par"] = [
        _link(f"s{i}", "par", None, started=f"2026-11-03T10:0{i}:30Z", title=f"job {i}") for i in range(1, 6)
    ]
    node = _node("par", subagents=[_node("kid", "Kid", status="completed")])
    node.launches = build_subagent_launches(node, snap.subagents_by_parent.get("par", []), snap, None)
    node.steps = [
        ModelStep(step_index=i, status="completed", started_at=f"2026-11-03T10:0{i}:00Z", provider_turn_id=f"par-turn-{i}", input_tokens=100)
        for i in range(1, 6)
    ]
    entries = build_activity_entries(node, max_steps=2)
    counted = [e for e in entries if e.kind != KIND_LAUNCH]
    markers = [e for e in entries if e.kind == KIND_LAUNCH]
    assert [e.detail.step_index for e in counted] == [4, 5]  # --max-steps unchanged for counted activity
    assert {m.launch.launch_id for m in markers} == {launch.launch_id for launch in node.launches}
    assert [m.launch.title for m in markers if m.launch.launch_id.startswith("subagent:")] == [
        f"job {i}" for i in range(1, 6)
    ]
    assert entries[0].launch.title == "job 1" and entries[-1].launch.title == "job 5"
    assert len(build_activity_entries(node, max_steps=0)) == 5 + 6  # 5 steps + 6 launches


def test_native_launch_merges_with_app_link_by_conversation_id():
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

    node = _node(
        "root", "Root", driver="antigravity", prov="antigravity", native_id="nat-root",
        subagents=[
            _node("kid", "Kid", driver="antigravity", prov="antigravity", native_id="nat-kid", status="idle"),
            _node("nat-other", "Other", driver="antigravity", prov="antigravity", native_id="nat-other", status="completed"),
        ],
    )
    agy_telemetry = reader("nat-root", None, "main")
    node.launches = build_subagent_launches(node, snap.subagents_by_parent.get("root", []), snap, agy_telemetry)
    by_id = {l.launch_id: l for l in node.launches}
    assert set(by_id) == {"subagent:s1", "native:nat-other"}
    merged = by_id["subagent:s1"]
    assert merged.sources == ["A", "C"] and merged.parent_step_index == 4 and merged.title == "reviewer"
    assert merged.launched_at == "2026-11-03T10:00:01Z" and merged.launch_time_source == "transcript tool call"
    other = by_id["native:nat-other"]
    assert other.sources == ["C"] and other.child_thread_id == "nat-other"


def test_token_metrics_include_cache_creation():
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


def test_snapshot_status_triggers_and_running_roots():
    from types import SimpleNamespace

    snapshot = T3StateSnapshot()
    def thread(identifier, parent=None, deleted=None):
        return SimpleNamespace(thread_id=identifier, parent_thread_id=parent, deleted_at=deleted,
                               active_provider_thread_id=None)
    snapshot.threads = {t.thread_id: t for t in [thread('provider'), thread('turn'), thread('link'),
                        thread('deleted', deleted='yes'), thread('child', parent='provider')]}
    snapshot.provider_turns_by_thread['turn'] = [SimpleNamespace(status='running', provider_thread_id='missing')]
    snapshot.subagents_by_parent['link'] = [SimpleNamespace(status='running')]
    snapshot.subagents_by_parent['deleted'] = [SimpleNamespace(status='running')]
    snapshot.subagents_by_parent['child'] = [SimpleNamespace(status='running')]
    snapshot.provider_threads['pt'] = SimpleNamespace(provider_thread_id='pt', status='running', thread_id='provider')
    snapshot.threads['provider'].active_provider_thread_id = 'pt'
    for identifier in ('provider', 'turn', 'link'):
        assert snapshot_thread_status(snapshot, identifier) == 'running'
    assert running_root_ids(snapshot) == ['provider', 'turn', 'link']
