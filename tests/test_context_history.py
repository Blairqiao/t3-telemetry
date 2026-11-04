"""Tests for ContextHistoryStore: merge, deduplication, in-flight correction, chronology, and LRU eviction."""

from datetime import datetime, timezone
import pytest

from t3_telemetry.context_history import ContextHistoryStore, clone_sample, make_agent_key
from t3_telemetry.models import ContextSample, ThreadTelemetryNode


def test_merge_deduplicates_identical_observations_even_with_different_timestamps():
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    # Poll 1 at T1
    obs1 = ContextSample(
        source_id="obs:pt-1:t1",
        used_tokens=10000,
        timestamp=None,
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
        observed_at="2026-11-03T12:00:00Z",
        timestamp_kind="observation",
    )
    res1 = store.merge(key, [], obs1)
    assert len(res1) == 1
    assert res1[0].used_tokens == 10000

    # Poll 2 at T2 (2 seconds later, context values unchanged)
    obs2 = ContextSample(
        source_id="obs:pt-1:t2",
        used_tokens=10000,
        timestamp=None,
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
        observed_at="2026-11-03T12:00:02Z",
        timestamp_kind="observation",
    )
    res2 = store.merge(key, [], obs2)
    assert len(res2) == 1
    assert res2[0].used_tokens == 10000


def test_merge_preserves_observed_drops():
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    # Poll 1: 50,000 tokens
    obs1 = ContextSample(
        source_id="obs:1",
        used_tokens=50000,
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
    )
    store.merge(key, [], obs1)

    # Poll 2: Context compaction / reset drops context to 12,000 tokens
    obs2 = ContextSample(
        source_id="obs:2",
        used_tokens=12000,
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
    )
    res = store.merge(key, [], obs2)
    assert len(res) == 2
    assert res[0].used_tokens == 50000
    assert res[1].used_tokens == 12000
    assert res[1].used_tokens < res[0].used_tokens  # Drop is preserved!


def test_merge_in_flight_mutable_upsert_by_source_id():
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    # Poll 1: in-flight turn-1 running with 5,000 tokens
    turn1_running = ContextSample(
        source_id="turn-1",
        used_tokens=5000,
        timestamp="2026-11-03T12:00:00Z",
        provenance="measured",
        provider_thread_id="pt-1",
    )
    res1 = store.merge(key, [turn1_running], None)
    assert len(res1) == 1
    assert res1[0].used_tokens == 5000

    # Poll 2: turn-1 completes with final 5,800 tokens
    turn1_completed = ContextSample(
        source_id="turn-1",
        used_tokens=5800,
        timestamp="2026-11-03T12:00:05Z",
        provenance="measured",
        provider_thread_id="pt-1",
    )
    res2 = store.merge(key, [turn1_completed], None)
    assert len(res2) == 1  # Updated in-place, NOT duplicated!
    assert res2[0].used_tokens == 5800
    assert res2[0].timestamp == "2026-11-03T12:00:05Z"


def test_provider_thread_snapshot_preserves_graph_shape_and_peak():
    """Provider-thread snapshot reading source_id stays same while values change:

    Historical source update may replace mutable record, but distinct live observed
    values should remain historical observations so graph shape and peak are preserved.
    """
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    # Poll 1: initial reading at 1,000 tokens
    pt_sample1 = ContextSample(
        source_id="pt:pt-1",
        used_tokens=1000,
        timestamp="2026-11-03T12:00:00Z",
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
    )
    res1 = store.merge(key, [pt_sample1], pt_sample1)
    assert len(res1) == 1
    assert res1[0].used_tokens == 1000

    # Poll 2: provider thread context grows to peak 25,000 tokens (same source_id pt:pt-1)
    pt_sample2 = ContextSample(
        source_id="pt:pt-1",
        used_tokens=25000,
        timestamp="2026-11-03T12:01:00Z",
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
    )
    res2 = store.merge(key, [pt_sample2], pt_sample2)
    # Both initial 1,000 and peak 25,000 must be preserved!
    assert len(res2) == 2
    assert res2[0].used_tokens == 1000
    assert res2[1].used_tokens == 25000

    # Poll 3: context drops to 15,000 tokens (same source_id pt:pt-1)
    pt_sample3 = ContextSample(
        source_id="pt:pt-1",
        used_tokens=15000,
        timestamp="2026-11-03T12:02:00Z",
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
    )
    res3 = store.merge(key, [pt_sample3], pt_sample3)
    # Graph shape (1000 -> 25000 -> 15000) and peak (25000) are preserved!
    assert len(res3) == 3
    assert [s.used_tokens for s in res3] == [1000, 25000, 15000]
    peak = max(s.used_tokens for s in res3)
    assert peak == 25000

    # Poll 4: poll repeats unchanged at 15,000 tokens -> deduplicated!
    res4 = store.merge(key, [pt_sample3], pt_sample3)
    assert len(res4) == 3


def test_chronology_robust_to_missing_timestamps():
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    # Sequence of readings where some or all have timestamp=None
    s1 = ContextSample(source_id="turn-1", used_tokens=1000, timestamp="2026-11-03T12:00:00Z")
    s2 = ContextSample(source_id="turn-2", used_tokens=2000, timestamp=None)
    s3 = ContextSample(source_id="turn-3", used_tokens=3000, timestamp=None)
    s4 = ContextSample(source_id="turn-4", used_tokens=4000, timestamp="2026-11-03T12:05:00Z")

    res = store.merge(key, [s1, s2, s3, s4], None)
    # Arrival sequence must NOT scramble s2/s3 to epoch 0 or beginning of list
    tokens = [s.used_tokens for s in res]
    assert tokens == [1000, 2000, 3000, 4000]


def test_provider_identity_separation_and_switches():
    store = ContextHistoryStore()

    # Thread 1 on Provider Thread pt-A
    node_pt_a = ThreadTelemetryNode(
        thread_id="th-1",
        title="Thread 1",
        driver="claude",
        provider_instance_id="p1",
        model="claude-3-7-sonnet",
        active_provider_thread_id="pt-A",
    )
    key_a = make_agent_key(node_pt_a)
    assert key_a == ("th-1", "pt-A")

    store.merge(key_a, [ContextSample(source_id="turn-1", used_tokens=5000)], None)
    assert len(store.get(key_a)) == 1
    assert store.get(key_a)[0].used_tokens == 5000

    # Thread 1 switches provider / provider thread to pt-B
    node_pt_b = ThreadTelemetryNode(
        thread_id="th-1",
        title="Thread 1",
        driver="claude",
        provider_instance_id="p1",
        model="claude-3-7-sonnet",
        active_provider_thread_id="pt-B",
    )
    key_b = make_agent_key(node_pt_b)
    assert key_b == ("th-1", "pt-B")
    assert key_a != key_b

    # History for pt-B starts fresh, no cross-contamination from pt-A
    assert len(store.get(key_b)) == 0

    store.merge(key_b, [ContextSample(source_id="turn-2", used_tokens=8000)], None)
    assert len(store.get(key_b)) == 1
    assert store.get(key_b)[0].used_tokens == 8000
    # Original pt-A history remains unchanged
    assert len(store.get(key_a)) == 1
    assert store.get(key_a)[0].used_tokens == 5000


def test_native_subagent_key_supported():
    child_node = ThreadTelemetryNode(
        thread_id="native-sub-1234",
        title="Native subagent",
        driver="antigravity",
        provider_instance_id=None,
        model="gemini-3.8-flash",
        native_id="native-sub-1234",
        active_provider_thread_id="native-sub-1234",
    )
    key = make_agent_key(child_node)
    assert key == ("native-sub-1234", "native-sub-1234")


def test_bounded_eviction_lru_64_agents():
    store = ContextHistoryStore(max_agents=64)

    for i in range(64):
        key = (f"agent-{i}", f"pt-{i}")
        store.merge(key, [ContextSample(source_id=f"turn-{i}", used_tokens=i * 100)])

    assert len(store) == 64
    assert ("agent-0", "pt-0") in store

    # Touch agent-0 to make it recently used
    store.touch(("agent-0", "pt-0"))

    # Add 65th agent -> least recently used (agent-1) should be evicted, agent-0 retained
    store.merge(("agent-64", "pt-64"), [ContextSample(source_id="turn-64", used_tokens=6400)])
    assert len(store) == 64
    assert ("agent-0", "pt-0") in store
    assert ("agent-1", "pt-1") not in store  # Evicted!
    assert ("agent-64", "pt-64") in store


def test_bounded_samples_per_agent_256():
    store = ContextHistoryStore(max_samples_per_agent=256)
    key = ("th-1", "pt-1")

    samples = [
        ContextSample(source_id=f"step-{i}", used_tokens=i * 10)
        for i in range(300)
    ]
    res = store.merge(key, samples)
    assert len(res) == 256
    # Retains latest 256
    assert res[0].used_tokens == 44 * 10
    assert res[-1].used_tokens == 299 * 10


def test_immutability_guarantee():
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    sample = ContextSample(source_id="turn-1", used_tokens=5000)
    input_list = [sample]
    res = store.merge(key, input_list)

    # Mutate input object and input list
    sample.used_tokens = 99999
    input_list.append(ContextSample(source_id="turn-2", used_tokens=88888))

    # Store must be unaffected
    stored = store.get(key)
    assert len(stored) == 1
    assert stored[0].used_tokens == 5000

    # Mutate returned list
    res[0].used_tokens = 77777
    res.clear()
    stored_again = store.get(key)
    assert len(stored_again) == 1
    assert stored_again[0].used_tokens == 5000


def test_unknown_context_does_not_fabricate_zero_measurement():
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    # Unknown context observation
    obs = ContextSample(
        source_id="obs:1",
        used_tokens=0,
        limit=None,
        provenance="unknown",
    )
    res = store.merge(key, [], obs)
    # Must NOT record 0 tokens when context is unknown!
    assert len(res) == 0


def test_no_duplicate_on_same_node_processed_twice_in_one_poll():
    store = ContextHistoryStore()
    key = ("th-1", "pt-1")

    obs = ContextSample(
        source_id="obs:1",
        used_tokens=15000,
        limit=100000,
        provenance="measured",
        provider_thread_id="pt-1",
        observed_at="2026-11-03T12:00:00Z",
        timestamp_kind="observation",
    )

    # First call (e.g. processing as summary)
    res1 = store.merge(key, [], obs)
    assert len(res1) == 1

    # Second call in same poll (e.g. processing as selected node)
    res2 = store.merge(key, res1, obs)
    assert len(res2) == 1
    assert res2[0].used_tokens == 15000


def test_repeated_source_snapshot_cannot_resurrect_evicted_history():
    store = ContextHistoryStore(max_samples_per_agent=10)
    key = ("thread", "provider")
    historical = [ContextSample(source_id=f"step-{i}", used_tokens=i * 100)
                  for i in range(1, 4)]
    for i in range(20):
        samples = store.merge(key, historical,
                              ContextSample(source_id=f"obs:{i}", used_tokens=350 + i * 10))
    assert len(samples) == 10
    assert all(s.source_id.startswith("obs:") for s in samples)
    assert [s.used_tokens for s in samples] == list(range(450, 550, 10))
    assert store.merge(key, historical, samples[-1]) == samples


def test_repeated_oversized_source_snapshot_remains_stable():
    store = ContextHistoryStore(max_samples_per_agent=10)
    historical = [ContextSample(source_id=f"step-{i}", used_tokens=i) for i in range(30)]
    first = store.merge("agent", historical)
    assert store.merge("agent", historical) == first
