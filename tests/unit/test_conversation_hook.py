"""Unit tests for agi.metacognition.conversation_hook (Gap Mapping Phase 4).

Covers the buffer contract, synchronous finalization semantics, and
the lazy idle-sweep behavior. End-to-end plumbing (chat handler →
buffer → finalize → events + UKG) is tested with an injected
detector so no network traffic happens.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import pytest

from agi.knowledge.graph import get_node, load_latest
from agi.metacognition.conversation_hook import (
    DEFAULT_IDLE_THRESHOLD_S,
    ConversationBuffer,
    finalize_conversation,
    finalize_conversation_sync,
    record_turn,
    sweep_idle,
)
from agi.metacognition.dissatisfaction_events import iter_events

# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def buf() -> ConversationBuffer:
    return ConversationBuffer()


@pytest.fixture
def paths(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """Fresh graph + events paths; also patches default paths so any
    code that bypasses explicit kwargs still lands in tmp."""
    graph = tmp_path / "graph.jsonl"
    events = tmp_path / "events.jsonl"
    import agi.knowledge.graph as g
    import agi.metacognition.dissatisfaction_events as ev

    monkeypatch.setattr(g, "DEFAULT_PATH", graph)
    monkeypatch.setattr(ev, "DEFAULT_PATH", events)
    return graph, events


def _stub_llm(verdict: str, **overrides) -> Callable[..., str]:
    """Build a detector LLM stub that returns a fixed-verdict response."""

    def _call(*, model: str, messages: list[dict]) -> str:
        payload = {
            "verdict": verdict,
            "topic": overrides.get("topic", "matrix rank confusion"),
            "signal_turns": overrides.get("signal_turns", [0, 2]),
            "rationale": overrides.get("rationale", "test"),
            "score": overrides.get("score", 0.85),
        }
        return json.dumps(payload)

    return _call


# ── ConversationBuffer ──────────────────────────────────────────


def test_buffer_record_and_get(buf):
    buf.record_turn("c1", "user", "hello", ts=100.0)
    buf.record_turn("c1", "assistant", "hi", ts=101.0)
    conv = buf.get("c1")
    assert conv is not None
    assert [t["role"] for t in conv.turns] == ["user", "assistant"]
    assert [t["content"] for t in conv.turns] == ["hello", "hi"]
    assert conv.last_ts == 101.0
    assert conv.finalized is False


def test_buffer_different_conversations_isolated(buf):
    buf.record_turn("c1", "user", "a")
    buf.record_turn("c2", "user", "b")
    assert buf.get("c1").turns[0]["content"] == "a"
    assert buf.get("c2").turns[0]["content"] == "b"
    assert len(buf) == 2


def test_buffer_drops_turns_after_finalize(buf, caplog):
    buf.record_turn("c1", "user", "hello")
    buf.mark_finalized("c1")
    caplog.set_level(logging.WARNING, logger="metacognition.conversation_hook")
    buf.record_turn("c1", "user", "late arrival")
    # Length is still 1 — second turn was rejected
    assert len(buf.get("c1").turns) == 1
    assert any("already finalized" in m for m in caplog.messages)


def test_buffer_idle_conversations_filter(buf):
    buf.record_turn("old", "user", "x", ts=100.0)
    buf.record_turn("fresh", "user", "y", ts=200.0)
    # Threshold=50 at now=210: old (210-100=110 >= 50) idle, fresh (210-200=10) not
    idle = buf.idle_conversations(now=210.0, idle_threshold_s=50.0)
    assert [c.conversation_id for c in idle] == ["old"]


def test_buffer_idle_skips_finalized(buf):
    buf.record_turn("c1", "user", "x", ts=100.0)
    buf.mark_finalized("c1")
    assert buf.idle_conversations(now=1000.0, idle_threshold_s=1.0) == []


def test_buffer_idle_skips_empty_conversations(buf):
    # An empty-conversation entry should never fire — nothing to detect.
    buf.record_turn("c1", "user", "x", ts=100.0)
    buf.drop("c1")
    assert buf.idle_conversations(now=1000.0, idle_threshold_s=1.0) == []


def test_buffer_active_count(buf):
    buf.record_turn("a", "user", "x")
    buf.record_turn("b", "user", "y")
    buf.mark_finalized("a")
    assert buf.active_count() == 1
    assert len(buf) == 2


def test_record_turn_requires_conversation_id(buf, caplog):
    caplog.set_level(logging.WARNING, logger="metacognition.conversation_hook")
    buf.record_turn("", "user", "hello")
    assert len(buf) == 0
    assert any("missing conversation_id" in m for m in caplog.messages)


# ── finalize_conversation_sync ──────────────────────────────────


def test_finalize_sync_emits_on_unsatisfied(buf, paths):
    graph, events = paths
    buf.record_turn("c1", "user", "what's the rank of a matrix?")
    buf.record_turn("c1", "assistant", "the number of rows")
    buf.record_turn("c1", "user", "no, that's wrong")

    node_id = finalize_conversation_sync(
        "c1",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("unsatisfied"),
    )
    assert node_id is not None
    assert node_id.startswith("gap_")

    # Exactly one event
    assert len(list(iter_events(path=events))) == 1

    # Buffer is now marked finalized
    assert buf.get("c1").finalized is True


def test_finalize_sync_silent_on_satisfied(buf, paths):
    graph, events = paths
    buf.record_turn("c1", "user", "thanks, got it")
    node_id = finalize_conversation_sync(
        "c1",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("satisfied"),
    )
    assert node_id is None
    assert not events.exists() or events.read_text() == ""
    # Still marked finalized — the conversation IS over, just nothing to emit
    assert buf.get("c1").finalized is True


def test_finalize_sync_missing_conversation(buf, paths):
    graph, events = paths
    node_id = finalize_conversation_sync(
        "never-existed",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("unsatisfied"),
    )
    assert node_id is None


def test_finalize_sync_already_finalized_is_noop(buf, paths):
    graph, events = paths
    buf.record_turn("c1", "user", "x")
    buf.mark_finalized("c1")
    node_id = finalize_conversation_sync(
        "c1",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("unsatisfied"),
    )
    assert node_id is None
    # Nothing was written
    assert not events.exists() or events.read_text() == ""


def test_finalize_sync_empty_turns_is_noop(buf, paths):
    graph, events = paths
    # Force-register a conversation without turns via direct buffer mut
    buf.record_turn("c1", "user", "x")
    # Manually pop all turns to simulate empty
    conv = buf.get("c1")
    conv.turns.clear()
    node_id = finalize_conversation_sync(
        "c1",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("unsatisfied"),
    )
    assert node_id is None
    assert buf.get("c1").finalized is True


def test_finalize_sync_detector_exception_does_not_raise(buf, paths, caplog):
    graph, events = paths
    buf.record_turn("c1", "user", "x")

    def _boom(*, model, messages):
        raise RuntimeError("NRP is down")

    caplog.set_level(logging.WARNING, logger="metacognition.conversation_hook")
    node_id = finalize_conversation_sync(
        "c1", buffer=buf, graph_path=graph, events_path=events, llm_call=_boom
    )
    # The hook module catches the detector transport error path via
    # classify_conversation (which returns None); a raise from INSIDE
    # the hook's orchestration would be caught by the outer try. So
    # node_id is None either way.
    assert node_id is None
    # Buffer is still marked finalized (the whole point — don't retry
    # a failing detector on every sweep)
    assert buf.get("c1").finalized is True


# ── finalize_conversation (threaded wrapper) ────────────────────


def test_finalize_threaded_returns_immediately(buf, paths):
    graph, events = paths
    buf.record_turn("c1", "user", "x")

    result = finalize_conversation(
        "c1",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("unsatisfied"),
        threaded=True,
    )
    # Threaded mode never returns the node_id inline — it's in the log / sidecar
    assert result is None

    # Wait for the event to land — mark_finalized fires early, but the
    # event write happens after detector + aggregator. Poll the events
    # file with a 2s upper bound.
    for _ in range(40):
        if list(iter_events(path=events)):
            break
        time.sleep(0.05)
    assert buf.get("c1").finalized is True
    assert len(list(iter_events(path=events))) == 1


def test_finalize_threaded_false_returns_node_id(buf, paths):
    graph, events = paths
    buf.record_turn("c1", "user", "x")
    result = finalize_conversation(
        "c1",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("unsatisfied"),
        threaded=False,
    )
    assert result is not None
    assert result.startswith("gap_")


# ── sweep_idle ──────────────────────────────────────────────────


def test_sweep_idle_finalizes_idle_not_fresh(buf, paths):
    graph, events = paths
    buf.record_turn("old", "user", "x", ts=100.0)
    buf.record_turn("fresh", "user", "y", ts=200.0)

    triggered = sweep_idle(
        now=210.0,
        idle_threshold_s=50.0,
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("neutral"),
        threaded=False,
    )
    assert triggered == ["old"]
    assert buf.get("old").finalized is True
    assert buf.get("fresh").finalized is False


def test_sweep_idle_empty_buffer(buf, paths):
    graph, events = paths
    triggered = sweep_idle(
        buffer=buf, graph_path=graph, events_path=events, threaded=False
    )
    assert triggered == []


def test_sweep_idle_skips_already_finalized(buf, paths):
    graph, events = paths
    buf.record_turn("c1", "user", "x", ts=100.0)
    buf.mark_finalized("c1")
    triggered = sweep_idle(
        now=1_000_000.0,
        idle_threshold_s=1.0,
        buffer=buf,
        graph_path=graph,
        events_path=events,
        threaded=False,
    )
    assert triggered == []


def test_default_idle_threshold_matches_spec():
    # Spec §1.1 wording is "inactivity timeout"; implementation default is 10 min.
    assert DEFAULT_IDLE_THRESHOLD_S >= 60
    assert DEFAULT_IDLE_THRESHOLD_S <= 3600


# ── record_turn convenience wrapper ─────────────────────────────


def test_record_turn_module_level_delegates_to_buffer(buf):
    record_turn("c1", "user", "hello", ts=100.0, buffer=buf)
    record_turn("c1", "assistant", "hi", ts=101.0, buffer=buf)
    conv = buf.get("c1")
    assert len(conv.turns) == 2


# ── end-to-end (detector → aggregator → graph) ──────────────────


def test_end_to_end_unsatisfied_flow(buf, paths):
    """A complete conversation: record turns, finalize, check the
    event + UKG node exist with correct fields."""
    graph, events = paths
    ts0 = 1_700_000_000
    buf.record_turn("abc", "user", "what's matrix rank?", ts=ts0)
    buf.record_turn("abc", "assistant", "number of rows", ts=ts0 + 1)
    buf.record_turn("abc", "user", "wrong — corrected: independent columns", ts=ts0 + 2)

    node_id = finalize_conversation_sync(
        "abc",
        buffer=buf,
        graph_path=graph,
        events_path=events,
        llm_call=_stub_llm("unsatisfied", topic="matrix rank"),
    )
    assert node_id == "gap_matrix-rank"

    evs = list(iter_events(path=events))
    assert len(evs) == 1
    assert evs[0]["conversation_id"] == "abc"
    assert evs[0]["topic_key"] == "matrix-rank"

    node = get_node(node_id, path=graph)
    assert node is not None
    assert node["source"] == "dissatisfaction"
    assert node["signal_count"] == 1
    assert f"event:{evs[0]['event_id']}" in node["evidence"]
