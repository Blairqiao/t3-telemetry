from pathlib import Path

from t3_telemetry.db_t3 import T3StateSnapshot
from t3_telemetry.models import AppThreadRecord, ProviderThreadRecord
from t3_telemetry.store import TelemetryStore
from t3_telemetry.t3_mirror import mirror_t3_graph


def _make_app_thread(
    thread_id: str,
    title: str = "Test Thread",
    default_provider: str = "claudeAgent",
    provider_instance_id: str | None = "claude_inst",
    parent_thread_id: str | None = None,
    root_thread_id: str | None = None,
    created_at: str | None = "2026-11-03T10:00:00Z",
    updated_at: str | None = "2026-11-03T10:05:00Z",
) -> AppThreadRecord:
    return AppThreadRecord(
        thread_id=thread_id,
        project_id="proj-1",
        title=title,
        default_provider=default_provider,
        provider_instance_id=provider_instance_id,
        model="test-model",
        thinking_level=None,
        active_provider_thread_id=f"pt-{thread_id}",
        parent_thread_id=parent_thread_id,
        root_thread_id=root_thread_id or thread_id,
        relationship_to_parent=None,
        created_at=created_at,
        updated_at=updated_at,
    )


def _make_provider_thread(
    provider_thread_id: str,
    thread_id: str | None,
    driver: str,
    provider_instance_id: str,
    native_id: str | None,
    status: str = "idle",
) -> ProviderThreadRecord:
    return ProviderThreadRecord(
        provider_thread_id=provider_thread_id,
        thread_id=thread_id,  # type: ignore[arg-type]
        driver=driver,
        provider_instance_id=provider_instance_id,
        native_id=native_id,
        status=status,
    )


def test_hand_built_snapshot_three_bindings(tmp_path: Path):
    """A hand-built snapshot with one Claude, one Codex, one pending and one Antigravity

    provider thread gives three bindings with the right keys.
    """
    snap = T3StateSnapshot()
    snap.threads["t-claude"] = _make_app_thread("t-claude", title="Claude Thread", default_provider="claudeAgent")
    snap.threads["t-codex"] = _make_app_thread("t-codex", title="Codex Thread", default_provider="codex")
    snap.threads["t-agy"] = _make_app_thread("t-agy", title="AGY Thread", default_provider="antigravity")
    snap.threads["t-pending"] = _make_app_thread("t-pending", title="Pending Thread", default_provider="claudeAgent")

    snap.provider_threads["pt-claude"] = _make_provider_thread(
        "pt-claude", "t-claude", driver="claudeAgent", provider_instance_id="claude_inst", native_id="c-123"
    )
    snap.provider_threads["pt-codex"] = _make_provider_thread(
        "pt-codex", "t-codex", driver="codex", provider_instance_id="codex_inst", native_id="cdx-456"
    )
    snap.provider_threads["pt-agy"] = _make_provider_thread(
        "pt-agy", "t-agy", driver="antigravity", provider_instance_id="antigravity_gemini", native_id="agy-789"
    )
    snap.provider_threads["pt-pending"] = _make_provider_thread(
        "pt-pending", "t-pending", driver="claudeAgent", provider_instance_id="claude_inst", native_id="pending:temp-001"
    )

    store = TelemetryStore(tmp_path / "store.db")
    mirror_t3_graph(store, snap)

    bindings = store.bindings()
    assert len(bindings) == 3
    assert bindings == {
        "claude:c-123": "t-claude",
        "codex:cdx-456": "t-codex",
        "antigravity:agy-789": "t-agy",
    }

    # Verify provider_thread_id stored in bindings table
    rows = store.conn.execute("SELECT session_key, thread_id, provider_thread_id FROM t3_bindings ORDER BY session_key").fetchall()
    row_tuples = [(r["session_key"], r["thread_id"], r["provider_thread_id"]) for r in rows]
    assert row_tuples == [
        ("antigravity:agy-789", "t-agy", "pt-agy"),
        ("claude:c-123", "t-claude", "pt-claude"),
        ("codex:cdx-456", "t-codex", "pt-codex"),
    ]

    store.close()


def test_driver_mapping_variations(tmp_path: Path):
    """Test driver and provider_instance_id combinations for provider mapping."""
    snap = T3StateSnapshot()

    # Antigravity variants
    snap.provider_threads["pt-agy-1"] = _make_provider_thread(
        "pt-agy-1", "th-1", driver="antigravity", provider_instance_id="custom_inst", native_id="a1"
    )
    snap.provider_threads["pt-agy-2"] = _make_provider_thread(
        "pt-agy-2", "th-2", driver="unknown_driver", provider_instance_id="antigravity_gemini", native_id="a2"
    )
    snap.provider_threads["pt-agy-3"] = _make_provider_thread(
        "pt-agy-3", "th-3", driver="antigravity_harness", provider_instance_id="inst", native_id="a3"
    )

    # Claude variants
    snap.provider_threads["pt-claude-1"] = _make_provider_thread(
        "pt-claude-1", "th-4", driver="claudeAgent", provider_instance_id="inst", native_id="c1"
    )
    snap.provider_threads["pt-claude-2"] = _make_provider_thread(
        "pt-claude-2", "th-5", driver="claude-3-7-sonnet", provider_instance_id="inst", native_id="c2"
    )
    snap.provider_threads["pt-claude-3"] = _make_provider_thread(
        "pt-claude-3", "th-6", driver="custom", provider_instance_id="claude_default", native_id="c3"
    )

    # Codex variants
    snap.provider_threads["pt-codex-1"] = _make_provider_thread(
        "pt-codex-1", "th-7", driver="codex", provider_instance_id="inst", native_id="x1"
    )
    snap.provider_threads["pt-codex-2"] = _make_provider_thread(
        "pt-codex-2", "th-8", driver="codexAgent", provider_instance_id="inst", native_id="x2"
    )
    snap.provider_threads["pt-codex-3"] = _make_provider_thread(
        "pt-codex-3", "th-9", driver="unknown", provider_instance_id="codex_inst", native_id="x3"
    )

    # Unrecognized provider variants (should be skipped)
    snap.provider_threads["pt-other-1"] = _make_provider_thread(
        "pt-other-1", "th-10", driver="other", provider_instance_id="other_inst", native_id="o1"
    )
    snap.provider_threads["pt-other-2"] = _make_provider_thread(
        "pt-other-2", "th-11", driver="", provider_instance_id="", native_id="o2"
    )

    store = TelemetryStore(tmp_path / "store.db")
    mirror_t3_graph(store, snap)

    bindings = store.bindings()
    assert bindings == {
        "antigravity:a1": "th-1",
        "antigravity:a2": "th-2",
        "antigravity:a3": "th-3",
        "claude:c1": "th-4",
        "claude:c2": "th-5",
        "claude:c3": "th-6",
        "codex:x1": "th-7",
        "codex:x2": "th-8",
        "codex:x3": "th-9",
    }
    store.close()


def test_skipped_records(tmp_path: Path):
    """Test that records without native_id, with pending: prefix, or missing thread_id are skipped."""
    snap = T3StateSnapshot()

    # native_id is None
    snap.provider_threads["pt-none"] = _make_provider_thread(
        "pt-none", "th-1", driver="claudeAgent", provider_instance_id="claude_inst", native_id=None
    )
    # native_id is empty string
    snap.provider_threads["pt-empty"] = _make_provider_thread(
        "pt-empty", "th-2", driver="claudeAgent", provider_instance_id="claude_inst", native_id=""
    )
    # native_id starts with pending:
    snap.provider_threads["pt-pending"] = _make_provider_thread(
        "pt-pending", "th-3", driver="claudeAgent", provider_instance_id="claude_inst", native_id="pending:abc"
    )
    # thread_id is None
    snap.provider_threads["pt-nothread"] = _make_provider_thread(
        "pt-nothread", None, driver="claudeAgent", provider_instance_id="claude_inst", native_id="c-valid"
    )
    # thread_id is empty string
    snap.provider_threads["pt-emptythread"] = _make_provider_thread(
        "pt-emptythread", "", driver="claudeAgent", provider_instance_id="claude_inst", native_id="c-valid2"
    )

    store = TelemetryStore(tmp_path / "store.db")
    mirror_t3_graph(store, snap)

    assert store.bindings() == {}
    store.close()


def test_threads_columns_mapping(tmp_path: Path):
    """Verify all columns of t3_threads table are mapped correctly."""
    snap = T3StateSnapshot()
    snap.threads["t-1"] = AppThreadRecord(
        thread_id="t-1",
        project_id="proj-abc",
        title="My Feature Thread",
        default_provider="antigravity",
        provider_instance_id="inst-xyz",
        model="gemini-2.5-pro",
        thinking_level="high",
        active_provider_thread_id="pt-1",
        parent_thread_id="t-parent",
        root_thread_id="t-root",
        relationship_to_parent="child",
        created_at="2026-11-03T10:00:00.000Z",
        updated_at="2026-11-03T10:05:00.000Z",
    )

    store = TelemetryStore(tmp_path / "store.db")
    mirror_t3_graph(store, snap)

    threads = store.t3_threads()
    assert "t-1" in threads
    row = threads["t-1"]
    assert row["thread_id"] == "t-1"
    assert row["title"] == "My Feature Thread"
    assert row["parent_thread_id"] == "t-parent"
    assert row["root_thread_id"] == "t-root"
    assert row["driver"] == "antigravity"
    assert row["provider_instance_id"] == "inst-xyz"
    assert row["created_at"] == "2026-11-03T10:00:00.000Z"
    assert row["updated_at"] == "2026-11-03T10:05:00.000Z"
    store.close()


def test_upsert_only_thread_vanishes(tmp_path: Path):
    """A second mirror run after a thread vanishes from the snapshot still has the thread (upsert only)."""
    store = TelemetryStore(tmp_path / "store.db")

    # Run 1: snapshot has thread A and thread B
    snap1 = T3StateSnapshot()
    snap1.threads["t-a"] = _make_app_thread("t-a", title="Thread A")
    snap1.threads["t-b"] = _make_app_thread("t-b", title="Thread B")
    snap1.provider_threads["pt-a"] = _make_provider_thread(
        "pt-a", "t-a", driver="claudeAgent", provider_instance_id="claude_inst", native_id="c-a"
    )
    snap1.provider_threads["pt-b"] = _make_provider_thread(
        "pt-b", "t-b", driver="codex", provider_instance_id="codex_inst", native_id="x-b"
    )
    mirror_t3_graph(store, snap1)

    assert set(store.t3_threads().keys()) == {"t-a", "t-b"}
    assert store.bindings() == {"claude:c-a": "t-a", "codex:x-b": "t-b"}

    # Run 2: Thread A has vanished from snapshot (e.g. deleted or archived and excluded by read_t3_state),
    # and thread B has updated title
    snap2 = T3StateSnapshot()
    snap2.threads["t-b"] = _make_app_thread("t-b", title="Thread B Updated")
    snap2.provider_threads["pt-b"] = _make_provider_thread(
        "pt-b", "t-b", driver="codex", provider_instance_id="codex_inst", native_id="x-b"
    )
    mirror_t3_graph(store, snap2)

    # Thread A is still present in store (upsert-only)
    threads = store.t3_threads()
    assert set(threads.keys()) == {"t-a", "t-b"}
    assert threads["t-a"]["title"] == "Thread A"
    assert threads["t-b"]["title"] == "Thread B Updated"

    # Binding for thread A also still present
    assert store.bindings() == {"claude:c-a": "t-a", "codex:x-b": "t-b"}

    store.close()
