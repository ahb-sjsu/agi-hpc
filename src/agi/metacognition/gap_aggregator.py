# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Phase 3 — dissatisfaction event aggregator.

What the module *does* is: accept one detector signal + its
``conversation_id``, gate on verdict/score, write the raw event to the
sidecar log, and upsert the topic-keyed UKG aggregate node with
``signal_count`` / ``first_signal_at`` / ``last_signal_at`` carried on
the node itself.

The module is intentionally narrow. All policy (threshold, topic
normalization, event dedup) lives here. Downstream consumers
(dashboard, clustering, dreaming prioritizer) read the UKG node and/or
the events sidecar; they do not reach back into detector logic.

Spec: ``docs/KNOWLEDGE_GAP_MAPPING_v1_spec.md`` — §4 "Event aggregator"
(previously "Gap emitter" — renamed because what this module actually
does is *aggregate* per-conversation events into per-topic nodes).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from agi.knowledge.graph import normalize_topic_key, upsert_node
from agi.metacognition.dissatisfaction import ConversationSignal
from agi.metacognition.dissatisfaction_events import (
    append_event,
    conversation_has_event,
    events_for_topic_key,
    make_event,
)

log = logging.getLogger("metacognition.gap_aggregator")

# ── constants ────────────────────────────────────────────────────

MIN_SCORE_FOR_EMIT: float = 0.7
"""Spec §1.6 — emit only when the detector's confidence meets this bar.
Weak model noise would otherwise accumulate into false-positive topics.
Tunable here only; no env override in v1 (tuning should be deliberate
and explicit, not scattered across deployments)."""

NODE_ID_PREFIX: str = "gap_"
"""UKG nodes for dissatisfaction gaps are keyed as ``gap_<topic_key>``
so the same topic across conversations collapses into one aggregate
per spec §1.3."""


# ── helpers ──────────────────────────────────────────────────────


def _auto_tags_from_topic(topic_key: str) -> list[str]:
    """Search-friendly tags derived from a normalized topic key.

    v1 just includes the key itself so ``query_nodes(tags_any=["foo"])``
    finds it. Future versions could add synonyms or split on hyphens.
    """
    return [topic_key] if topic_key else []


# ── public entrypoint ────────────────────────────────────────────


def aggregate_event(
    signal: ConversationSignal | None,
    *,
    conversation_id: str,
    graph_path: Path | None = None,
    events_path: Path | None = None,
    now: int | None = None,
) -> str | None:
    """Process a detector signal: threshold → sidecar → UKG upsert.

    Returns the UKG node id (``"gap_<topic_key>"``) on a successful
    emit, ``None`` on any gate rejection:

    - ``signal is None`` (detector failure upstream)
    - ``signal.verdict != "unsatisfied"``
    - ``signal.score < MIN_SCORE_FOR_EMIT``
    - ``signal.topic`` is empty or normalizes to an empty key
    - ``conversation_id`` already has an event in the sidecar

    Never raises on normal operation. A node's ``evidence`` list
    accumulates ``event:<id>`` handles for each conversation that
    contributed, and the node-level aggregate counters are recomputed
    from the full event stream for ``topic_key`` so they stay
    consistent even if upstream writes interleave.
    """
    if signal is None:
        return None
    if signal.verdict != "unsatisfied":
        log.debug(
            "gap_skip_verdict: conv=%s verdict=%s", conversation_id, signal.verdict
        )
        return None
    if signal.score < MIN_SCORE_FOR_EMIT:
        log.info(
            "gap_skip_low_score: conv=%s score=%.2f threshold=%.2f",
            conversation_id,
            signal.score,
            MIN_SCORE_FOR_EMIT,
        )
        return None
    if not signal.topic or not signal.topic.strip():
        log.warning(
            "gap_skip_empty_topic: conv=%s verdict=unsatisfied score=%.2f",
            conversation_id,
            signal.score,
        )
        return None

    # Short-circuit on dedup BEFORE doing any graph work — the sidecar
    # would reject anyway, but graph.load_latest is O(n) and we can
    # spare it.
    if conversation_has_event(conversation_id, path=events_path):
        log.info(
            "gap_skip_already_aggregated: conv=%s already produced an event",
            conversation_id,
        )
        return None

    topic_key = normalize_topic_key(signal.topic)
    if not topic_key:
        log.warning(
            "gap_skip_unnormalizable_topic: conv=%s topic=%r",
            conversation_id,
            signal.topic[:80],
        )
        return None

    ts = int(now if now is not None else time.time())
    event = make_event(
        signal=signal,
        conversation_id=conversation_id,
        topic_key=topic_key,
        now=ts,
    )
    written = append_event(event, path=events_path)
    if not written:
        # Dedup race — should be rare given the pre-check above, but
        # possible under concurrent writers. Swallow and return None.
        log.warning(
            "gap_race_rejected: conv=%s — event dedup lost race", conversation_id
        )
        return None

    # Aggregates are derived from the full sidecar, not incrementally,
    # so a crashed/replayed run heals itself on the next successful
    # write. O(events_for_topic_key) — fine at v1 volumes.
    tk_events = events_for_topic_key(topic_key, path=events_path)
    first_ts = min(e["ts"] for e in tk_events)
    last_ts = max(e["ts"] for e in tk_events)
    signal_count = len(tk_events)

    node_id = f"{NODE_ID_PREFIX}{topic_key}"
    tags = ["dissatisfaction", *_auto_tags_from_topic(topic_key)]
    upsert_node(
        id=node_id,
        type="gap",
        status="active",
        topic=signal.topic.strip(),
        topic_key=topic_key,
        tags=tags,
        title=f"[gap] {signal.topic.strip()}",
        body_ref=None,
        verified=False,
        source="dissatisfaction",
        evidence=[f"event:{event['event_id']}"],
        now=last_ts,  # node last_touched_at tracks newest event
        extra={
            "first_signal_at": first_ts,
            "last_signal_at": last_ts,
            "signal_count": signal_count,
        },
        path=graph_path,
    )
    log.info(
        "gap_aggregated: conv=%s node=%s signal_count=%d score=%.2f",
        conversation_id,
        node_id,
        signal_count,
        signal.score,
    )
    return node_id


# ── query helper (used by dashboard / dreaming prioritizer) ─────


def rebuild_aggregates(
    topic_key: str,
    *,
    events_path: Path | None = None,
) -> dict[str, Any]:
    """Recompute (signal_count, first_signal_at, last_signal_at) from the
    events sidecar for one topic_key.

    Provided so the dashboard / dreaming prioritizer can cross-check a
    node's stored aggregates against the raw log without going through
    the full aggregator path. Returns an empty-shape dict if no events
    exist for ``topic_key``.
    """
    events = events_for_topic_key(topic_key, path=events_path)
    if not events:
        return {"signal_count": 0, "first_signal_at": None, "last_signal_at": None}
    return {
        "signal_count": len(events),
        "first_signal_at": min(e["ts"] for e in events),
        "last_signal_at": max(e["ts"] for e in events),
    }
