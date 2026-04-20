# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unified Knowledge Graph v1 — append-only JSONL store.

Each line is a full-state snapshot for one node write event. The reader
materializes the latest valid state per ``id`` using ``last_touched_at``
(ties broken by file-position — later line wins).

The canonical spec is
``C:\\Users\\abptl\\Documents\\personal\\unified_knowledge_graph_v1_spec.docx``.
Settled v1 decisions (do not rediscover):

- body_ref existence is validated at read/query time, never at append.
- source is strict-by-type, lenient-by-vocabulary: unknown non-empty
  strings are accepted with a warning.
- evidence dedup preserves first-seen stable order.
- Context reader mode is config-backed with env-var override
  (``EREBUS_CONTEXT_READER=wiki|graph``), default ``wiki``.
- query_nodes is keyword-only and intentionally narrow.
- Teaching-context eligibility lives in one helper
  (``is_context_eligible``) — never re-encode at call sites.

Warning categories (see ``logging.getLogger("knowledge.graph")``):
``invalid_record_skipped``, ``unknown_source``, ``body_ref_missing``,
``filled_excluded_missing_body``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from agi.common.atomic_write import atomic_write_text

log = logging.getLogger("knowledge.graph")

# ── constants ────────────────────────────────────────────────────

SCHEMA_VERSION = 1

ALLOWED_TYPES: frozenset[str] = frozenset({"filled", "gap", "stub"})
ALLOWED_STATUSES: frozenset[str] = frozenset({"active", "archived"})
KNOWN_SOURCES: frozenset[str] = frozenset(
    {"primer", "help_queue", "dissatisfaction", "backfill", "manual"}
)

# Type alias — a node is always a plain JSON-serializable dict. Using
# dict-of-dict instead of a dataclass keeps JSONL round-trips trivial
# and avoids a second validation surface.
Node = dict

REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "id",
    "type",
    "status",
    "topic",
    "topic_key",
    "tags",
    "title",
    "body_ref",
    "verified",
    "verified_at",
    "source",
    "created_at",
    "last_touched_at",
    "evidence",
)

DEFAULT_PATH = Path(
    os.environ.get("KNOWLEDGE_GRAPH_PATH", "/archive/neurogolf/knowledge_graph.jsonl")
)


# ── normalization ────────────────────────────────────────────────

_SLUG_STRIP = re.compile(r"[^a-z0-9\- ]+")
_HYPHEN_COLLAPSE = re.compile(r"-+")


def normalize_topic_key(topic: str) -> str:
    """Lowercase, trim, collapse whitespace, hyphenate, drop punctuation.

    Mirrors the spec's normalization rule for ``topic_key``:
    ``symmetry completion`` → ``symmetry-completion``,
    ``Count  distinct__colors!`` → ``count-distinct-colors``.
    """
    s = (topic or "").strip().lower()
    # unify word separators to a single hyphen
    s = re.sub(r"[\s_]+", "-", s)
    # strip anything that isn't a-z0-9 or hyphen
    s = _SLUG_STRIP.sub("", s)
    # collapse repeated hyphens and trim leading/trailing
    s = _HYPHEN_COLLAPSE.sub("-", s).strip("-")
    return s


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Lowercase, trim, hyphenate, dedupe while preserving first-seen order."""
    seen: dict[str, None] = {}
    for raw in tags or []:
        if not isinstance(raw, str):
            continue
        t = raw.strip().lower()
        t = re.sub(r"[\s_]+", "-", t)
        t = _HYPHEN_COLLAPSE.sub("-", t).strip("-")
        if t and t not in seen:
            seen[t] = None
    return list(seen.keys())


def merge_evidence(old: Iterable[str], new: Iterable[str]) -> list[str]:
    """Union old + new, preserving first-seen stable order.

    ``old = ["conv:a", "help:t1"]``,
    ``new = ["help:t1", "wiki:x", "conv:b"]``
    → ``["conv:a", "help:t1", "wiki:x", "conv:b"]``.
    """
    seen: dict[str, None] = {}
    for src in (old, new):
        for e in src or []:
            if isinstance(e, str) and e and e not in seen:
                seen[e] = None
    return list(seen.keys())


# ── validation ───────────────────────────────────────────────────


class ValidationError(ValueError):
    """Raised when a node record violates the v1 invariants."""


def _validate_source(source: Any) -> None:
    if not isinstance(source, str) or not source:
        raise ValidationError("source must be a non-empty string")
    if source not in KNOWN_SOURCES:
        log.warning("unknown_source: %r (accepted with warning)", source)


def validate_record(record: Any) -> None:
    """Raise ``ValidationError`` iff ``record`` violates v1 invariants.

    Does NOT consult the filesystem (``body_ref`` existence is a
    read-time concern per the settled spec).
    """
    if not isinstance(record, dict):
        raise ValidationError(f"record must be a dict, got {type(record).__name__}")

    # schema_version
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"schema_version must be {SCHEMA_VERSION}, got {record.get('schema_version')!r}"
        )

    # required non-empty strings
    for f in ("id", "topic", "topic_key", "title"):
        v = record.get(f)
        if not isinstance(v, str) or not v:
            raise ValidationError(f"{f!r} must be a non-empty string")

    # type / status enums
    if record.get("type") not in ALLOWED_TYPES:
        raise ValidationError(
            f"type must be one of {sorted(ALLOWED_TYPES)}, got {record.get('type')!r}"
        )
    if record.get("status") not in ALLOWED_STATUSES:
        raise ValidationError(
            f"status must be one of {sorted(ALLOWED_STATUSES)}, got {record.get('status')!r}"
        )

    # tags
    tags = record.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise ValidationError("tags must be a list of strings")
    if len(set(tags)) != len(tags):
        raise ValidationError("tags must be unique (post-normalization)")

    # body_ref — filled must have one; others may or may not
    body_ref = record.get("body_ref")
    if record["type"] == "filled":
        if not isinstance(body_ref, str) or not body_ref:
            raise ValidationError("filled nodes must have a non-empty body_ref")
    else:
        if body_ref is not None and not isinstance(body_ref, str):
            raise ValidationError("body_ref must be str or null")

    # verified / verified_at interlock
    verified = record.get("verified")
    verified_at = record.get("verified_at")
    if not isinstance(verified, bool):
        raise ValidationError("verified must be a boolean")
    if verified:
        if record["type"] != "filled":
            raise ValidationError("verified=true requires type='filled'")
        if not isinstance(verified_at, int) or verified_at <= 0:
            raise ValidationError("verified=true requires a positive int verified_at")
    else:
        if verified_at is not None:
            raise ValidationError("verified=false requires verified_at=null")

    # source (lenient vocab, strict type)
    _validate_source(record.get("source"))

    # timestamps
    for f in ("created_at", "last_touched_at"):
        v = record.get(f)
        if not isinstance(v, int) or v <= 0:
            raise ValidationError(f"{f} must be a positive int unix seconds")
    if record["created_at"] > record["last_touched_at"]:
        raise ValidationError("created_at must be <= last_touched_at")

    # evidence
    ev = record.get("evidence")
    if not isinstance(ev, list) or not all(isinstance(e, str) for e in ev):
        raise ValidationError("evidence must be a list of strings")


def is_valid_record(record: Any) -> bool:
    """Non-raising predicate wrapper around ``validate_record``."""
    try:
        validate_record(record)
        return True
    except ValidationError:
        return False


# ── context eligibility (the teaching trust gate) ────────────────


def is_context_eligible(node: Node, *, wiki_root: Path | None = None) -> bool:
    """Gate used to decide whether a node can be shown to a generator as truth.

    A node is eligible only when:
      - type == "filled"
      - verified is True
      - status == "active"
      - body_ref is non-null
      - the referenced body file exists on disk

    ``wiki_root`` is an optional prefix to resolve relative ``body_ref``
    values against. If None, ``body_ref`` is treated as-is (absolute
    path or CWD-relative). Missing body files emit a
    ``filled_excluded_missing_body`` warning and return False.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") != "filled":
        return False
    if not node.get("verified"):
        return False
    if node.get("status") != "active":
        return False
    body_ref = node.get("body_ref")
    if not isinstance(body_ref, str) or not body_ref:
        return False
    p = Path(body_ref)
    if wiki_root is not None and not p.is_absolute():
        p = Path(wiki_root) / body_ref
    if not p.exists():
        log.warning(
            "filled_excluded_missing_body: node=%s body_ref=%s",
            node.get("id"),
            body_ref,
        )
        return False
    return True


# ── reader / writer surface ──────────────────────────────────────


def _path(path: Path | None) -> Path:
    return Path(path) if path is not None else DEFAULT_PATH


def append_node(node: Node, *, path: Path | None = None) -> None:
    """Validate ``node`` and append it as one JSONL line.

    Does not check body_ref on the filesystem — per spec, existence is a
    read-time concern. Raises ``ValidationError`` on invariant violation
    so a caller can see its bug loudly rather than silently corrupting
    the log.
    """
    validate_record(node)
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(node, separators=(",", ":"), sort_keys=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def iter_records(path: Path | None = None) -> Iterator[Node]:
    """Yield every valid record in the log in file order.

    Invalid lines are skipped with an ``invalid_record_skipped`` warning
    so corruption doesn't hide silently but also doesn't hard-fail the
    reader (partial writes, concurrent readers, future schema bumps).
    """
    target = _path(path)
    if not target.exists():
        return
    with open(target, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception as e:
                log.warning(
                    "invalid_record_skipped: line=%d parse_error=%r",
                    lineno,
                    str(e)[:200],
                )
                continue
            try:
                validate_record(rec)
            except ValidationError as e:
                log.warning(
                    "invalid_record_skipped: line=%d validation=%r id=%r",
                    lineno,
                    str(e)[:200],
                    (rec or {}).get("id"),
                )
                continue
            yield rec


def load_latest(path: Path | None = None) -> dict[str, Node]:
    """Materialize latest state per id.

    Later ``last_touched_at`` wins; on ties, later line in the file wins.
    Invalid records are skipped (with warning). Missing body files do
    NOT exclude a node from the materialized map — that's a
    context-eligibility concern, not a materialization concern.
    """
    latest: dict[str, Node] = {}
    for rec in iter_records(path):
        nid = rec["id"]
        cur = latest.get(nid)
        if cur is None or rec["last_touched_at"] >= cur["last_touched_at"]:
            latest[nid] = rec
    return latest


def get_node(node_id: str, *, path: Path | None = None) -> Node | None:
    """Return the latest materialized state for one id, or None."""
    return load_latest(path).get(node_id)


def query_nodes(
    *,
    type: str | None = None,
    status: str | None = None,
    verified: bool | None = None,
    topic_key: str | None = None,
    tags_any: list[str] | None = None,
    source: str | None = None,
    updated_since: int | None = None,
    limit: int | None = None,
    sort: str = "last_touched_desc",
    path: Path | None = None,
) -> list[Node]:
    """Filtered/sorted view over the materialized graph.

    Keyword-only on purpose — v1 signature is narrow by design. Richer
    retrieval helpers should be layered on top of ``load_latest`` later
    rather than widening this API.
    """
    nodes = list(load_latest(path).values())
    if type is not None:
        nodes = [n for n in nodes if n.get("type") == type]
    if status is not None:
        nodes = [n for n in nodes if n.get("status") == status]
    if verified is not None:
        nodes = [n for n in nodes if bool(n.get("verified")) is verified]
    if topic_key is not None:
        nodes = [n for n in nodes if n.get("topic_key") == topic_key]
    if tags_any:
        wanted = set(tags_any)
        nodes = [n for n in nodes if wanted.intersection(n.get("tags") or [])]
    if source is not None:
        nodes = [n for n in nodes if n.get("source") == source]
    if updated_since is not None:
        nodes = [n for n in nodes if (n.get("last_touched_at") or 0) >= updated_since]
    if sort == "last_touched_desc":
        nodes.sort(key=lambda n: n.get("last_touched_at", 0), reverse=True)
    elif sort == "last_touched_asc":
        nodes.sort(key=lambda n: n.get("last_touched_at", 0))
    elif sort == "created_desc":
        nodes.sort(key=lambda n: n.get("created_at", 0), reverse=True)
    elif sort == "created_asc":
        nodes.sort(key=lambda n: n.get("created_at", 0))
    else:
        raise ValueError(
            "sort must be one of last_touched_{desc,asc} or created_{desc,asc}"
        )
    if limit is not None:
        nodes = nodes[:limit]
    return nodes


# ── upsert / promotion helper ────────────────────────────────────


def upsert_node(
    *,
    id: str,
    type: str,
    status: str = "active",
    topic: str,
    tags: list[str] | None = None,
    title: str,
    body_ref: str | None = None,
    verified: bool = False,
    source: str,
    evidence: list[str] | None = None,
    topic_key: str | None = None,
    now: int | None = None,
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> Node:
    """Load current state for ``id``, merge in the patch, append full snapshot.

    Rules per spec:
      - Full-state snapshot on every write (never mutate prior lines).
      - On first creation: created_at = last_touched_at = now.
      - On update: preserve created_at; set last_touched_at = now.
      - Replace most fields with incoming values; union evidence with
        first-seen stable order.
      - Type transitions enforced: gap→filled, stub→filled, gap→stub
        allowed; filled→gap forbidden.

    ``verified_at`` is derived: set to ``now`` when the incoming
    ``verified`` flips True on a filled node.

    ``extra`` — optional arbitrary fields merged into the record after
    the standard fields. Used by the dissatisfaction aggregator to
    carry node-level aggregates (``signal_count``, ``first_signal_at``,
    ``last_signal_at``) without widening the graph's required-field
    schema. Keys that collide with required fields are dropped (with a
    warning) so a caller can't accidentally clobber load-bearing state.
    """
    ts = int(now if now is not None else time.time())
    existing = get_node(id, path=path)

    # Enforce allowed type transitions.
    if existing is not None:
        prev_type = existing.get("type")
        if prev_type == "filled" and type != "filled":
            raise ValidationError(
                f"type transition not allowed: filled→{type!r} (id={id!r})"
            )

    norm_tags = normalize_tags(tags or (existing.get("tags") if existing else []))
    # If caller omitted topic_key, derive from topic; explicit override wins.
    tkey = topic_key or normalize_topic_key(topic)

    # verified / verified_at interlock
    if verified and type != "filled":
        raise ValidationError("verified=true requires type='filled'")
    if verified:
        # Preserve original verified_at if already verified; else stamp now.
        if existing and existing.get("verified") and existing.get("verified_at"):
            verified_at: int | None = int(existing["verified_at"])
        else:
            verified_at = ts
    else:
        verified_at = None

    created_at = int(existing["created_at"]) if existing else ts
    merged_evidence = merge_evidence(
        existing.get("evidence") if existing else [],
        evidence or [],
    )

    record: Node = {
        "schema_version": SCHEMA_VERSION,
        "id": id,
        "type": type,
        "status": status,
        "topic": topic,
        "topic_key": tkey,
        "tags": norm_tags,
        "title": title,
        "body_ref": body_ref,
        "verified": verified,
        "verified_at": verified_at,
        "source": source,
        "created_at": created_at,
        "last_touched_at": ts,
        "evidence": merged_evidence,
    }
    if extra:
        _reserved = set(REQUIRED_FIELDS)
        for k, v in extra.items():
            if k in _reserved:
                log.warning(
                    "upsert_node: extra[%r] collides with required field; ignored",
                    k,
                )
                continue
            record[k] = v
    append_node(record, path=path)
    return record


# ── compaction ───────────────────────────────────────────────────


def compact(path: Path | None = None) -> int:
    """Rewrite the log keeping only the latest snapshot per id.

    Uses ``atomic_write_text`` so readers always see either the old log
    or the fully-written compacted log, never a partial file. Returns
    the count of nodes in the compacted file.
    """
    target = _path(path)
    latest = load_latest(target)
    # Preserve a stable order for the rewritten file: by last_touched_at
    # ascending so tail-style readers still see newest at the end.
    ordered = sorted(
        latest.values(),
        key=lambda n: (n.get("last_touched_at", 0), n.get("id", "")),
    )
    body = "\n".join(
        json.dumps(n, separators=(",", ":"), sort_keys=True) for n in ordered
    )
    if body:
        body += "\n"
    atomic_write_text(target, body)
    return len(ordered)


# ── dashboard summary ────────────────────────────────────────────


def summary(
    path: Path | None = None,
    *,
    top_topics: int = 8,
    recent_fills: int = 8,
) -> dict[str, Any]:
    """Aggregate the graph into the shape the dashboard card consumes.

    Returns::

        {
          "total": int,
          "by_type":   {"filled": N, "gap": N, "stub": N},
          "by_status": {"active": N, "archived": N},
          "fill_rate": float,              # 0..1, filled / (filled + gap)
          "top_topics_by_gap": [
            {"topic_key": str, "gaps": N, "filled": N,
             "total": N, "fill_rate": float}, ...
          ],
          "recent_fills": [
            {"id": str, "topic_key": str, "title": str,
             "verified_at": int, "last_touched_at": int}, ...
          ],
        }

    Intentionally narrow per the Phase 5 spec: node counts by type,
    top topics by gap density, and recent fills. No "recent gaps" list —
    add that in a later phase if the dashboard needs it.
    """
    latest = load_latest(path)
    nodes = list(latest.values())

    by_type = {t: 0 for t in ALLOWED_TYPES}
    by_status = {s: 0 for s in ALLOWED_STATUSES}
    # topic_key -> {"gaps": int, "filled": int}
    per_topic: dict[str, dict[str, int]] = {}

    for n in nodes:
        t = n.get("type")
        if t in by_type:
            by_type[t] += 1
        s = n.get("status")
        if s in by_status:
            by_status[s] += 1
        tk = n.get("topic_key") or "unknown"
        slot = per_topic.setdefault(tk, {"gaps": 0, "filled": 0})
        if t == "gap":
            slot["gaps"] += 1
        elif t == "filled":
            slot["filled"] += 1

    filled_count = by_type["filled"]
    gap_count = by_type["gap"]
    denom = filled_count + gap_count
    fill_rate = (filled_count / denom) if denom > 0 else 0.0

    # Top topics by gap density: sort by (gaps desc, then total desc) so a
    # topic with 5 gaps ranks above one with 2 even if both have the same
    # fill_rate. Zero-gap topics are dropped — they don't help the user
    # spot where the graph needs work.
    topic_rows = []
    for tk, counts in per_topic.items():
        total = counts["gaps"] + counts["filled"]
        if counts["gaps"] == 0:
            continue
        topic_rows.append(
            {
                "topic_key": tk,
                "gaps": counts["gaps"],
                "filled": counts["filled"],
                "total": total,
                "fill_rate": round(counts["filled"] / total, 3) if total else 0.0,
            }
        )
    topic_rows.sort(key=lambda r: (-r["gaps"], -r["total"], r["topic_key"]))
    topic_rows = topic_rows[:top_topics]

    # Recent fills: verified filled nodes sorted by verified_at desc,
    # fallback to last_touched_at when verified_at is missing. Archived
    # fills are included — archival is a lifecycle signal, not a fill
    # retraction, so "recent fills" honestly means "recently published."
    fills = [n for n in nodes if n.get("type") == "filled"]
    fills.sort(
        key=lambda n: (n.get("verified_at") or n.get("last_touched_at") or 0),
        reverse=True,
    )
    recent = [
        {
            "id": n.get("id"),
            "topic_key": n.get("topic_key"),
            "title": n.get("title"),
            "verified_at": n.get("verified_at"),
            "last_touched_at": n.get("last_touched_at"),
            "status": n.get("status"),
        }
        for n in fills[:recent_fills]
    ]

    return {
        "total": len(nodes),
        "by_type": by_type,
        "by_status": by_status,
        "fill_rate": round(fill_rate, 3),
        "top_topics_by_gap": topic_rows,
        "recent_fills": recent,
    }


# ── config flag (context reader rollout) ─────────────────────────


def context_reader_mode(default: str = "wiki") -> str:
    """Return ``"wiki"`` or ``"graph"`` — env override, else default.

    Resolution order per settled spec:
      1. ``EREBUS_CONTEXT_READER`` env var, if set to a valid value.
      2. Default (``"wiki"`` for initial rollout).

    A config-file path may layer in between later; Phase 1 only needs
    the env + default surface so consumers can switch behavior without
    a code change.
    """
    env = os.environ.get("EREBUS_CONTEXT_READER")
    if env:
        v = env.strip().lower()
        if v in ("wiki", "graph"):
            return v
        log.warning("EREBUS_CONTEXT_READER=%r invalid; using default=%r", env, default)
    return default
