# Knowledge Gap Mapping v1 — Implementation Specification

**Status:** decisions locked 2026-04-19. Ready for Phase 1 coding.
**Purpose:** define a shippable v1 for detecting, storing, clustering, and prioritizing "things Atlas has been asked about but answered poorly," using the Unified Knowledge Graph as the storage substrate and a parallel sidecar event log as the raw audit trail.

This spec is scoped against the UKG v1 spec (`reference_ukg_v1_spec`). Only decisions not already settled there appear below.

## Design intent

Atlas today accumulates verified knowledge (sensei notes) and open tasks (help queue). It does not record conversations where the user expressed dissatisfaction, asked for clarification, or corrected Atlas. Roadmap item 2.3 ("Knowledge Gap Mapping") calls for surfacing those signals so the dreaming subsystem can prioritize consolidation around weak areas and a future curiosity module can autonomously seek information in those gaps.

**Core principle: reuse the UKG; do not build a second store.**

A dissatisfaction-derived gap is a `type=gap` node with `source="dissatisfaction"` in the existing Unified Knowledge Graph. What this spec adds is a **producer** (the detector), a **sidecar event log** (raw audit trail), and **consumers** (clustering, dashboard, dreaming prioritizer). Storage model:

| Layer | Contents | Shape |
|---|---|---|
| UKG node | lightweight aggregate index, one per `topic_key`, with counters + timestamps | JSONL (existing graph) |
| Sidecar event log | raw audit trail, one record per classification event | JSONL (new) |
| UKG node → events | `evidence[]` carries `event:<event_id>` handles, never free text | — |

This matches the UKG design philosophy: nodes are cheap indexes; everything else lives in a parallel log the node points at.

**Load-bearing invariants (carried from the UKG spec):**

- Only `filled ∧ verified ∧ active` nodes are teaching context. Dissatisfaction gaps are never fed back to a generator as truth.
- Append-only JSONL with full-state snapshots per write.
- The centralized trust gate (`is_context_eligible`) is the single place that decides what is safe for a generator to consume.

## 1. Settled decisions (v1)

1. **Detector runs post-conversation, not inline.** Hook sits on the conversation finalization path (session close or inactivity timeout), reads the final N turns, classifies the outcome, and emits at most one event. No inline token-level cost during chat.
2. **Classification is three-way plus a failure mode.** `verdict in {"satisfied", "neutral", "unsatisfied"}`. A detector that cannot produce a usable result returns `None`. `neutral` means the detector ran successfully and found no strong signal — distinct from a classifier failure.
3. **Topic-keyed aggregation; one event per conversation.** The UKG node is keyed by `gap_<topic_key>` and aggregates every dissatisfaction event that shares that topic. Each conversation contributes at most one event. The graph grows with topics, not with conversations.
4. **Topic extraction is free-text, clustered periodically.** Detector returns `topic` only; `topic_key = normalize_topic_key(topic)` is computed once in the emitter so canonicalization lives in exactly one place. A separate nightly clustering job proposes near-duplicate merges.
5. **Detector is a small LLM call.** Default model is Qwen3 (fast, cheap, already warm on NRP). Env override: `EREBUS_DETECTOR_MODEL`. Every event records `detector_model` + `detector_version` so drift and A/B comparisons are tractable without digging into logs.
6. **Minimum-confidence gate.** Emit only if `verdict == "unsatisfied" AND score >= 0.7`. Weak model noise otherwise accumulates into false-positive topics. Threshold is tunable in a config constant.
7. **Classify every conversation; no sampling.** Volume is low enough that the small compute overhead is cheaper than the complexity of interpreting sampled counts and priorities.
8. **Gaps persist indefinitely; staleness is computed, not destructive.** The UKG node carries `last_signal_at`, `signal_count`, `first_signal_at`. The dashboard fades stale gaps visually but never deletes — deletion loses signal.
9. **Atlas-originated conversations only for v1.** The Erebus chat handler (`_erebus_chat` in `scripts/telemetry_server.py`) is the one in-scope source. Other producers (a future Primer `/ask` endpoint, Claude Code sessions) plug into the stable `classify_conversation` signature without protocol change.
10. **Dashboard: extend the existing UKG panel.** No new card.

## 2. Detector contract

```python
def classify_conversation(
    *,
    conversation_id: str,
    turns: list[dict],       # [{role, content, ts}, ...]
    ego_model: str,          # the ego backend that served this conversation
) -> ConversationSignal | None:
    ...
```

### ConversationSignal

| Field | Type | Notes |
|---|---|---|
| `verdict` | `"satisfied" \| "neutral" \| "unsatisfied"` | classifier's call |
| `topic` | `str` | free-text description; normalization is emitter's job |
| `signal_turns` | `list[int]` | turn indices carrying the signal |
| `rationale` | `str` | short free-text audit; stored in sidecar, NEVER re-fed to a generator |
| `score` | `float` | 0..1 confidence; 1.0 for overt dissatisfaction |
| `detector_model` | `str` | model id used for this classification |
| `detector_version` | `str` | git short-sha or semver of the detector code |

Returning `None` means "classifier failure or insufficient evidence" — use liberally, false-negatives are fine, false-positives pollute the graph.

### Prompt sketch

```
System: You are a conversation auditor for Atlas AI. Classify whether
the user was satisfied with Atlas's answer.

Rules:
- "satisfied" requires evidence: thanks, next-topic transition, no corrections.
- "unsatisfied" requires evidence: user correction, clarification request,
  repeated question, explicit dissatisfaction.
- "neutral" is the default. When in doubt, return neutral.
- Extract a topic if and only if the verdict is "unsatisfied."

User: <last 10 turns of the conversation>

Return: {"verdict": "...", "topic": "...", "signal_turns": [...],
         "rationale": "...", "score": 0..1}
```

## 3. Sidecar event log

Path: `/archive/neurogolf/dissatisfaction_events.jsonl`. Append-only, one JSON object per line, same atomicity discipline as the UKG log.

### Event schema

| Field | Example |
|---|---|
| `event_id` | UUIDv4 or `conv-abc123-sig` — stable, dedup-able |
| `conversation_id` | opaque string from the chat handler |
| `topic` | `"why is matrix rank not factorization"` |
| `topic_key` | `"why-is-matrix-rank-not-factorization"` |
| `signal_turns` | `[4, 7]` |
| `rationale` | `"user repeated question after assistant reply"` |
| `score` | `0.82` |
| `ts` | `1713574800` |
| `detector_model` | `"qwen3"` |
| `detector_version` | `"gap-det-0.1.0"` |

### Module surface

```python
def append_event(event: dict) -> None: ...
def iter_events(*, since: int | None = None) -> Iterator[dict]: ...
def event_exists(event_id: str) -> bool: ...
```

`append_event` enforces a schema check and a conversation-id dedup gate (one event per conversation — a second event for the same `conversation_id` is rejected with a warning, preserving the "one event per conversation" invariant).

## 4. Event aggregator (replaces "gap emitter")

What this module *does* is: accept one dissatisfaction event, upsert a topic-keyed UKG aggregate node. "Emitter" was the wrong mental model — the module is an aggregator.

```python
def aggregate_event(sig: ConversationSignal, *, graph_path: Path | None = None) -> str | None:
    """Append the event to the sidecar and upsert the UKG aggregate node.

    Returns the UKG node id on success, None when the event is
    rejected (verdict != unsatisfied, score below threshold, detector
    returned None upstream, or conversation_id already has an event).
    """
```

Behavior:

1. Reject if `sig.verdict != "unsatisfied"` or `sig.score < 0.7`.
2. Compute `topic_key = normalize_topic_key(sig.topic)`.
3. Compose the event record, write to the sidecar (skip if duplicate conversation_id).
4. Upsert the UKG node `id = f"gap_{topic_key}"` with:
   - `type="gap"`, `status="active"`, `source="dissatisfaction"`
   - `tags = ["dissatisfaction", *auto_tags_from_topic(topic_key)]`
   - `title = f"[gap] {topic}"`
   - `body_ref = None`, `verified = False`
   - `evidence = [f"event:{event_id}"]` (union-appended per the UKG spec)
5. Derive node-level aggregate fields from the event stream at upsert time:
   - `first_signal_at` (preserved from the first event ever)
   - `last_signal_at` (the incoming event's `ts`)
   - `signal_count` (current event count for this topic_key)

These aggregate fields are carried in the UKG node record itself — the existing append-only-snapshot semantics handle them correctly.

## 5. Consumers

### 5.1 Dashboard (extension of Phase 5 UKG panel)

The existing `/api/ukg/status` handler adds two cheap rows:

- **Top dissatisfaction topics** — a view over `source=="dissatisfaction"` gaps, ranked by `signal_count` (primary) and `last_signal_at` (tiebreaker). Implementation: a second call to `graph.summary(...)` with a `source_filter="dissatisfaction"` kwarg, returning a parallel `top_dissatisfaction_topics` list.
- **Recent dissatisfaction events** — last N events from the sidecar, each as conversation-id + topic + age.

Ranking is on the materialized node fields (`signal_count`, `last_signal_at`). The dashboard does not scan `evidence[]` to count; that would be O(n) per render.

### 5.2 Dreaming prioritizer

```python
def dreaming_priority(path: Path | None = None, *, top_n: int = 5) -> list[str]:
    """Return top-N topic_keys to rehearse in the next dream cycle."""
```

Ranks `type=gap ∧ source=="dissatisfaction"` by `signal_count * recency_weight(last_signal_at)` where `recency_weight` decays exponentially over days. Dreaming reads the list, fetches related sensei notes via the UKG topic index, and composes synthesis prompts for the next consolidation round.

### 5.3 Curiosity (out of scope for v1)

Reserved for roadmap Phase 4 and a separate spec.

## 6. Clustering

A nightly job reduces near-duplicate topic keys:

```python
def cluster_topics(path: Path | None = None, *, threshold: float = 0.85) -> list[Cluster]:
    """Return equivalence classes of topic_keys that should be merged."""
```

v1 uses character-level Jaro-Winkler on `topic_key` strings. v2 may upgrade to sentence-embedding similarity from the existing PCA-384 index, gated on evidence that Jaro-Winkler is insufficient in practice.

**Clusters are reported, not auto-merged.** The dashboard surfaces proposed merges; an operator confirms each one. Merge-lineage semantics (how the resulting node preserves provenance of its constituents) are explicitly **deferred to a v2 spec** — v1 clustering reports proposals only and leaves the graph untouched until an operator acts.

## 7. Delivery phases for v1

| Phase | Component | Deliverable |
|---|---|---|
| 1 | Detector module | `src/agi/metacognition/dissatisfaction.py` + unit tests against recorded conversations. |
| 2 | Events log | `src/agi/metacognition/dissatisfaction_events.py` — append/read/dedup + tests. |
| 3 | Event aggregator | `src/agi/metacognition/gap_aggregator.py` — threshold gate, sidecar write, UKG upsert (with `signal_count`, `first_signal_at`, `last_signal_at` maintained on the node). |
| 4 | Conversation hook | Hook the detector into the **conversation finalization path** in `scripts/telemetry_server.py` (triggers on session close or inactivity timeout), emitting at most one event per conversation. |
| 5 | Dashboard rows | Extend `graph.summary(source_filter=...)` + the existing UKG panel in `schematic.html`. |
| 6 | Clustering proposal | `src/agi/knowledge/clustering.py` + nightly cron + dashboard review surface (read-only, no auto-merge). |
| 7 | Dreaming priority | `dreaming_priority()` + integration into the existing dream scheduler. |

Each phase is a separate commit, tested in isolation, green on CI before the next.

## 8. Out of scope for v1

- Non-Atlas conversation sources (Claude Code sessions, Jupyter, shell history).
- Auto-merge of topic clusters (human-in-the-loop only).
- Merge-lineage semantics for cluster operations (deferred to v2 spec).
- Curiosity module (Phase 4 of the roadmap, separate spec).
- Sentence-embedding clustering (Jaro-Winkler for v1; upgrade gated on data).
- Re-classifying historical conversations (v1 is forward-looking only).

## 9. Acceptance criteria

1. A conversation that ends in user dissatisfaction produces **exactly one dissatisfaction event** (sidecar append) and updates **exactly one topic-keyed gap node** (UKG upsert).
2. Reprocessing the same conversation is idempotent: no duplicate event, no duplicate `evidence[]` handle on the node.
3. A satisfied or neutral conversation produces zero events and zero graph mutations.
4. A missing or failed detector result (detector returned `None`) produces zero events and zero graph mutations.
5. Topic normalization is deterministic: semantically identical `topic` strings normalize to the same `topic_key`.
6. The UKG node's `signal_count`, `first_signal_at`, and `last_signal_at` are correct after N events for the same topic.
7. Dashboard ranking is based on the node-level `signal_count` and `last_signal_at` fields — never on a full scan of `evidence[]`.
8. No dissatisfaction-sourced gap is ever returned by `is_context_eligible()`.
9. The clustering job reports proposed merges without mutating the graph. (Merge semantics themselves are out of scope for v1.)
10. The dreaming priority list is stable across two tickless polls and re-prioritizes within one tick when a fresh dissatisfaction signal arrives for a high-weighted topic.

## 10. Recommended first tests

1. Classify a known-satisfied conversation → `verdict=="satisfied"`, zero emits.
2. Classify a known-unsatisfied conversation above threshold → one event, one node, `signal_count == 1`.
3. Two unsatisfied conversations on the same topic → two events, one node, `signal_count == 2`, `first_signal_at` preserved from the earlier, `last_signal_at` matches the later.
4. Two unsatisfied conversations on different topics → two events, two distinct nodes.
5. Unsatisfied verdict with `score < 0.7` → zero emits.
6. Detector returns `None` → zero emits, zero graph mutations.
7. Aggregator called twice with the same `conversation_id` → one event, no duplicate; warning logged.
8. Emitted gap never satisfies `is_context_eligible()` regardless of other fields.
9. `cluster_topics()` on three near-duplicate keys returns one proposed merge; the graph is unchanged.
10. `dreaming_priority()` on an empty graph returns `[]` without error.
