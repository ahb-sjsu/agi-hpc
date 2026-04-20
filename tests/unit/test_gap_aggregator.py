"""Unit tests for agi.metacognition.gap_aggregator (Gap Mapping Phase 3).

Covers the six relevant acceptance criteria from the spec §9:

  AC1 — one event + one node per unsatisfied conversation
  AC2 — reprocessing idempotent
  AC3 — satisfied/neutral produces zero writes
  AC4 — None detector result produces zero writes
  AC5 — topic normalization is deterministic
  AC6 — signal_count / first_signal_at / last_signal_at correct
  AC8 — dissatisfaction gap never eligible for teaching context

Plus Phase-3-specific behavior: threshold gate, empty-topic rejection,
node-id scheme, evidence accumulation, rebuild_aggregates helper.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agi.knowledge.graph import get_node, is_context_eligible, load_latest
from agi.metacognition.dissatisfaction import ConversationSignal
from agi.metacognition.dissatisfaction_events import iter_events
from agi.metacognition.gap_aggregator import (
    MIN_SCORE_FOR_EMIT,
    NODE_ID_PREFIX,
    aggregate_event,
    rebuild_aggregates,
)

# ── fixtures ─────────────────────────────────────────────────────


def _sig(
    verdict: str = "unsatisfied",
    topic: str = "why matrix rank is not factorization",
    score: float = 0.85,
    signal_turns: list[int] | None = None,
) -> ConversationSignal:
    return ConversationSignal(
        verdict=verdict,
        topic=topic,
        signal_turns=signal_turns or [2, 4],
        rationale="user repeated question",
        score=score,
        detector_model="qwen3",
        detector_version="gap-det-0.1.0",
    )


@pytest.fixture
def paths(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Return (graph_path, events_path) pointed at fresh tmp files.

    Patches module-level DEFAULT_PATHs too so that code calling through
    those default seams still hits the tmp files — important because
    aggregate_event's event_for_topic_key path resolves via the events
    module's DEFAULT_PATH when events_path is None.
    """
    graph = tmp_path / "graph.jsonl"
    events = tmp_path / "events.jsonl"

    import agi.knowledge.graph as g
    import agi.metacognition.dissatisfaction_events as ev

    monkeypatch.setattr(g, "DEFAULT_PATH", graph)
    monkeypatch.setattr(ev, "DEFAULT_PATH", events)
    return graph, events


# ── AC3 — zero writes for satisfied / neutral ────────────────────


def test_satisfied_produces_nothing(paths):
    graph, events = paths
    nid = aggregate_event(
        _sig(verdict="satisfied"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid is None
    assert not events.exists() or events.read_text() == ""
    assert not graph.exists() or graph.read_text() == ""


def test_neutral_produces_nothing(paths):
    graph, events = paths
    nid = aggregate_event(
        _sig(verdict="neutral"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid is None


# ── AC4 — zero writes for None detector result ──────────────────


def test_none_signal_produces_nothing(paths):
    graph, events = paths
    nid = aggregate_event(
        None, conversation_id="c1", graph_path=graph, events_path=events
    )
    assert nid is None


# ── threshold gate (spec §1.6) ──────────────────────────────────


def test_below_threshold_rejected(paths, caplog):
    graph, events = paths
    caplog.set_level(logging.INFO, logger="metacognition.gap_aggregator")
    nid = aggregate_event(
        _sig(score=MIN_SCORE_FOR_EMIT - 0.01),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid is None
    assert any("gap_skip_low_score" in m for m in caplog.messages)
    assert not events.exists() or events.read_text() == ""


def test_exactly_at_threshold_accepted(paths):
    graph, events = paths
    nid = aggregate_event(
        _sig(score=MIN_SCORE_FOR_EMIT),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid is not None


# ── empty-topic and unnormalizable-topic rejection ──────────────


def test_empty_topic_rejected(paths, caplog):
    graph, events = paths
    caplog.set_level(logging.WARNING, logger="metacognition.gap_aggregator")
    nid = aggregate_event(
        _sig(topic=""),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid is None
    assert any("gap_skip_empty_topic" in m for m in caplog.messages)


def test_whitespace_topic_rejected(paths, caplog):
    graph, events = paths
    caplog.set_level(logging.WARNING, logger="metacognition.gap_aggregator")
    nid = aggregate_event(
        _sig(topic="    "),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid is None


def test_punctuation_only_topic_rejected(paths, caplog):
    """topic=".,!?" normalizes to empty key — distinct from the whitespace case."""
    graph, events = paths
    caplog.set_level(logging.WARNING, logger="metacognition.gap_aggregator")
    nid = aggregate_event(
        _sig(topic=".,!?"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid is None
    assert any("gap_skip_unnormalizable_topic" in m for m in caplog.messages)


# ── AC1 — one event + one node on happy path ────────────────────


def test_unsatisfied_produces_one_event_and_one_node(paths):
    graph, events = paths
    nid = aggregate_event(
        _sig(topic="matrix rank confusion"),
        conversation_id="abc",
        graph_path=graph,
        events_path=events,
        now=1_700_000_000,
    )

    assert nid == f"{NODE_ID_PREFIX}matrix-rank-confusion"

    # Exactly one event
    evs = list(iter_events(path=events))
    assert len(evs) == 1
    assert evs[0]["conversation_id"] == "abc"
    assert evs[0]["topic_key"] == "matrix-rank-confusion"

    # Exactly one node
    latest = load_latest(path=graph)
    assert set(latest.keys()) == {nid}
    node = latest[nid]
    assert node["type"] == "gap"
    assert node["status"] == "active"
    assert node["source"] == "dissatisfaction"
    assert node["verified"] is False
    assert node["body_ref"] is None
    assert node["topic_key"] == "matrix-rank-confusion"
    assert "dissatisfaction" in node["tags"]
    assert node["title"].startswith("[gap]")
    assert f"event:{evs[0]['event_id']}" in node["evidence"]

    # Aggregates
    assert node["signal_count"] == 1
    assert node["first_signal_at"] == 1_700_000_000
    assert node["last_signal_at"] == 1_700_000_000


# ── AC2 — reprocessing idempotent ───────────────────────────────


def test_reprocessing_same_conversation_idempotent(paths, caplog):
    graph, events = paths
    aggregate_event(_sig(), conversation_id="abc", graph_path=graph, events_path=events)

    caplog.set_level(logging.INFO, logger="metacognition.gap_aggregator")
    nid = aggregate_event(
        _sig(), conversation_id="abc", graph_path=graph, events_path=events
    )
    assert nid is None
    assert any("gap_skip_already_aggregated" in m for m in caplog.messages)

    evs = list(iter_events(path=events))
    assert len(evs) == 1  # no duplicate event
    latest = load_latest(path=graph)
    # Graph log has a single line — idempotent even at the JSONL level
    graph_lines = [line for line in graph.read_text().splitlines() if line.strip()]
    assert len(graph_lines) == 1
    assert len(latest) == 1


# ── AC6 — aggregates correct across N events on same topic ──────


def test_multiple_conversations_same_topic_aggregate_correctly(paths):
    graph, events = paths
    # Three conversations, same topic, ascending timestamps
    aggregate_event(
        _sig(topic="matrix rank"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
        now=100,
    )
    aggregate_event(
        _sig(topic="matrix rank"),
        conversation_id="c2",
        graph_path=graph,
        events_path=events,
        now=200,
    )
    aggregate_event(
        _sig(topic="matrix rank"),
        conversation_id="c3",
        graph_path=graph,
        events_path=events,
        now=300,
    )

    node = get_node(f"{NODE_ID_PREFIX}matrix-rank", path=graph)
    assert node is not None
    assert node["signal_count"] == 3
    assert node["first_signal_at"] == 100
    assert node["last_signal_at"] == 300
    # Evidence accumulates with first-seen stable order
    assert len(node["evidence"]) == 3


def test_aggregate_survives_out_of_order_timestamps(paths):
    graph, events = paths
    aggregate_event(
        _sig(topic="x"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
        now=200,
    )
    aggregate_event(
        _sig(topic="x"),
        conversation_id="c2",
        graph_path=graph,
        events_path=events,
        now=100,  # earlier
    )
    node = get_node(f"{NODE_ID_PREFIX}x", path=graph)
    assert node["first_signal_at"] == 100  # derived from full event stream
    assert node["last_signal_at"] == 200


# ── AC5 — topic normalization deterministic ─────────────────────


def test_two_topic_strings_normalize_to_same_node(paths):
    graph, events = paths
    aggregate_event(
        _sig(topic="Matrix Rank!"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    aggregate_event(
        _sig(topic="matrix rank"),
        conversation_id="c2",
        graph_path=graph,
        events_path=events,
    )
    latest = load_latest(path=graph)
    assert set(latest.keys()) == {f"{NODE_ID_PREFIX}matrix-rank"}
    assert latest[f"{NODE_ID_PREFIX}matrix-rank"]["signal_count"] == 2


def test_different_topics_produce_different_nodes(paths):
    graph, events = paths
    aggregate_event(
        _sig(topic="matrix rank"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    aggregate_event(
        _sig(topic="vector spaces"),
        conversation_id="c2",
        graph_path=graph,
        events_path=events,
    )
    latest = load_latest(path=graph)
    assert set(latest.keys()) == {
        f"{NODE_ID_PREFIX}matrix-rank",
        f"{NODE_ID_PREFIX}vector-spaces",
    }
    assert all(n["signal_count"] == 1 for n in latest.values())


# ── AC8 — dissatisfaction gap is never context-eligible ─────────


def test_aggregated_gap_is_never_context_eligible(paths):
    graph, events = paths
    nid = aggregate_event(
        _sig(), conversation_id="c1", graph_path=graph, events_path=events
    )
    node = get_node(nid, path=graph)
    # No body_ref, type=gap, verified=false — the gate must reject on
    # multiple grounds. This is the load-bearing invariant.
    assert is_context_eligible(node) is False


# ── rebuild_aggregates helper ───────────────────────────────────


def test_rebuild_aggregates_matches_node_state(paths):
    graph, events = paths
    aggregate_event(
        _sig(topic="x"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
        now=100,
    )
    aggregate_event(
        _sig(topic="x"),
        conversation_id="c2",
        graph_path=graph,
        events_path=events,
        now=200,
    )
    agg = rebuild_aggregates("x", events_path=events)
    node = get_node(f"{NODE_ID_PREFIX}x", path=graph)
    assert agg["signal_count"] == node["signal_count"]
    assert agg["first_signal_at"] == node["first_signal_at"]
    assert agg["last_signal_at"] == node["last_signal_at"]


def test_rebuild_aggregates_empty_topic(paths):
    _, events = paths
    agg = rebuild_aggregates("never-seen", events_path=events)
    assert agg == {
        "signal_count": 0,
        "first_signal_at": None,
        "last_signal_at": None,
    }


# ── node_id scheme ──────────────────────────────────────────────


def test_node_id_scheme_uses_prefix(paths):
    graph, events = paths
    nid = aggregate_event(
        _sig(topic="foo bar"),
        conversation_id="c1",
        graph_path=graph,
        events_path=events,
    )
    assert nid == "gap_foo-bar"
    assert nid.startswith(NODE_ID_PREFIX)


# ── upsert_node.extra contract ──────────────────────────────────


def test_extra_cannot_clobber_required_fields(paths, caplog):
    """If a caller sneaks a required-field key into extra, it must be
    dropped with a warning rather than silently overwriting load-bearing
    state. Tested via the aggregator's upstream path would require a
    bug; here we exercise upsert_node directly to pin the contract."""
    from agi.knowledge.graph import upsert_node

    graph, _ = paths
    caplog.set_level(logging.WARNING, logger="knowledge.graph")
    upsert_node(
        id="n1",
        type="gap",
        topic="t",
        title="x",
        source="manual",
        path=graph,
        extra={"type": "filled", "signal_count": 5},  # type should be dropped
    )
    node = get_node("n1", path=graph)
    assert node["type"] == "gap"  # required field not clobbered
    assert node["signal_count"] == 5  # non-reserved extra passed through
    assert any("collides with required field" in m for m in caplog.messages)
