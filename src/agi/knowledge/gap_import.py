# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Phase 4 — help-queue → UKG gap import.

Erebus writes stuck-task questions to ``/archive/neurogolf/erebus_help_queue.json``
(list of up-to-20 dicts, each with ``task``, ``timestamp``, ``attempts``,
``error_types``, ``question``, ...). This module emits one ``gap`` node
per distinct task in the queue, using the newest help-queue entry for
that task to populate the topic/tags/title.

Idempotent: a second call is a no-op unless a help-queue entry for the
same task is strictly newer than the existing gap's ``last_touched_at``.
``filled``/``stub`` nodes are never regressed — the graph enforces that.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graph import get_node, normalize_tags, upsert_node

log = logging.getLogger("knowledge.gap_import")

DEFAULT_HELP_PATH = Path("/archive/neurogolf/erebus_help_queue.json")


@dataclass
class GapImportReport:
    imported: int = 0
    refreshed: int = 0
    skipped_up_to_date: int = 0
    skipped_filled: int = 0
    failed: int = 0
    imported_ids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"imported={self.imported} refreshed={self.refreshed} "
            f"skipped_up_to_date={self.skipped_up_to_date} "
            f"skipped_filled={self.skipped_filled} failed={self.failed}"
        )


# ── parsing ──────────────────────────────────────────────────────


def _parse_ts(raw: Any) -> int | None:
    """Accept ISO8601 strings (with or without tz) or int unix seconds."""
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw)
    if not isinstance(raw, str) or not raw:
        return None
    s = raw.strip()
    try:
        # datetime.fromisoformat tolerates "2026-04-19T10:30:00" on 3.11+
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _latest_per_task(queue: list[dict]) -> dict[int, dict]:
    """Reduce a help-queue list to one entry per task: the newest by timestamp.

    Falls back to list-order recency (later list position wins) when a
    timestamp is missing or unparseable. Matches arc_scientist's
    behavior of appending to the queue, so the last entry is newest.
    """
    out: dict[int, dict] = {}
    for idx, entry in enumerate(queue):
        if not isinstance(entry, dict):
            continue
        try:
            tn = int(entry.get("task"))
        except (TypeError, ValueError):
            continue
        ts = _parse_ts(entry.get("timestamp"))
        if ts is None:
            # Fallback: synthesize a monotonic pseudo-ts from list index so
            # the "newest by list position" rule still works.
            ts = -1_000_000_000 + idx
        prev = out.get(tn)
        prev_ts = prev and prev.get("_ts")
        if prev is None or prev_ts is None or ts > prev_ts:
            entry = dict(entry)  # don't mutate the caller's payload
            entry["_ts"] = ts
            out[tn] = entry
    return out


# ── one-entry → gap-node mapping ─────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def _topic_from_entry(entry: dict) -> str:
    """Pick a human-readable topic from the help entry.

    Prefer the first parseable error_type; fall back to a generic
    "stuck task" label. Error-type strings like
    ``spatial_primitive`` become ``spatial primitive`` for the human
    topic; ``topic_key`` is re-derived by graph.normalize_topic_key.
    """
    ets = entry.get("error_types") or []
    for et in ets:
        if isinstance(et, str) and et.strip():
            return et.strip().replace("_", " ").replace("-", " ")
    return "stuck task"


def _tags_from_entry(entry: dict) -> list[str]:
    base = ["arc", "stuck"]
    for et in entry.get("error_types") or []:
        if isinstance(et, str) and et.strip():
            base.append(et)
    return normalize_tags(base)


def _title_from_entry(entry: dict, task_num: int) -> str:
    """Short human-readable title derived from the question text."""
    q = entry.get("question")
    if isinstance(q, str) and q.strip():
        collapsed = _WHITESPACE_RE.sub(" ", q.strip())
        if len(collapsed) > 120:
            collapsed = collapsed[:117].rstrip() + "…"
        return f"[gap] {collapsed}"
    return f"[gap] Task {task_num:03d}"


# ── public entrypoint ────────────────────────────────────────────


def import_help_queue(
    help_path: Path | str | None = None,
    *,
    graph_path: Path | None = None,
    dry_run: bool = False,
) -> GapImportReport:
    """Sync the UKG with the current help-queue contents.

    For each distinct task in the queue, upsert a ``gap`` node (or
    refresh an existing gap if the queue entry is strictly newer than
    the stored ``last_touched_at``). ``filled`` nodes are left alone.
    Safe to call on every Primer tick — idempotent when nothing new.
    """
    path = Path(help_path) if help_path is not None else DEFAULT_HELP_PATH
    report = GapImportReport()
    if not path.exists():
        return report
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("help_queue_parse_failed: %s", str(e)[:200])
        report.failed += 1
        return report

    # arc_scientist writes a list; defensively support {"queue": [...]}.
    if isinstance(raw, dict):
        raw = raw.get("queue") or []
    if not isinstance(raw, list):
        log.warning("help_queue_unexpected_shape: %s", type(raw).__name__)
        report.failed += 1
        return report

    latest = _latest_per_task(raw)
    for task_num, entry in sorted(latest.items()):
        node_id = f"sensei_task_{task_num:03d}"
        existing = get_node(node_id, path=graph_path)
        if existing is not None and existing.get("type") == "filled":
            report.skipped_filled += 1
            continue

        entry_ts = int(entry.get("_ts") or 0)
        if existing is not None and existing.get("type") == "gap":
            prev_ts = int(existing.get("last_touched_at") or 0)
            if entry_ts <= prev_ts:
                report.skipped_up_to_date += 1
                continue

        topic = _topic_from_entry(entry)
        tags = _tags_from_entry(entry)
        title = _title_from_entry(entry, task_num)
        evidence = [f"help:t{task_num:03d}"]
        ts_str = entry.get("timestamp")
        if isinstance(ts_str, str) and ts_str:
            evidence.append(f"help:ts_{ts_str}")

        if dry_run:
            log.info(
                "gap_would_import: id=%s topic=%s tags=%s ts=%d",
                node_id,
                topic,
                tags,
                entry_ts,
            )
            if existing is None:
                report.imported += 1
            else:
                report.refreshed += 1
            report.imported_ids.append(node_id)
            continue

        try:
            # Use a positive timestamp — graph validation requires >0.
            now_arg = entry_ts if entry_ts > 0 else None
            upsert_node(
                id=node_id,
                type="gap",
                status="active",
                topic=topic,
                tags=tags,
                title=title,
                body_ref=None,
                verified=False,
                source="help_queue",
                evidence=evidence,
                now=now_arg,
                path=graph_path,
            )
            if existing is None:
                report.imported += 1
            else:
                report.refreshed += 1
            report.imported_ids.append(node_id)
        except Exception as e:  # noqa: BLE001
            log.warning("gap_upsert_failed: id=%s error=%s", node_id, str(e)[:200])
            report.failed += 1

    return report
