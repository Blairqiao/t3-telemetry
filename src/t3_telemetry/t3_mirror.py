"""Mirror T3 threads and provider session bindings into TelemetryStore."""

from __future__ import annotations

from typing import TYPE_CHECKING

from t3_telemetry.providers import provider_key

if TYPE_CHECKING:
    from t3_telemetry.db_t3 import T3StateSnapshot
    from t3_telemetry.store import TelemetryStore


def mirror_t3_graph(store: TelemetryStore, snapshot: T3StateSnapshot) -> None:
    """Mirror snapshot app threads and provider session bindings into store."""
    raw_threads = snapshot.threads.values() if isinstance(snapshot.threads, dict) else snapshot.threads
    threads = [
        {
            "thread_id": t.thread_id,
            "title": t.title,
            "parent_thread_id": t.parent_thread_id,
            "root_thread_id": t.root_thread_id,
            "driver": t.default_provider,
            "provider_instance_id": t.provider_instance_id,
            "created_at": t.created_at,
            "updated_at": t.updated_at,
        }
        for t in raw_threads
    ]

    raw_provider_threads = (
        snapshot.provider_threads.values()
        if isinstance(snapshot.provider_threads, dict)
        else snapshot.provider_threads
    )
    bindings: list[dict[str, str | None]] = []
    for pt in raw_provider_threads:
        if not pt.native_id or pt.native_id.startswith("pending:") or not pt.thread_id:
            continue
        provider = provider_key(pt.driver, pt.provider_instance_id)
        if provider is None:
            continue
        bindings.append({
            "session_key": f"{provider}:{pt.native_id}",
            "thread_id": pt.thread_id,
            "provider_thread_id": pt.provider_thread_id,
        })

    store.replace_t3_graph(threads, bindings)
