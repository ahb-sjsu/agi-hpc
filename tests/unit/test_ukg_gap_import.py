"""Unit tests for agi.knowledge.gap_import (Phase 4).

Covers: help-queue parse, newest-per-task reduction, idempotent
refresh, filled-node protection, malformed queue handling, gap tag
derivation from error_types, dry-run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agi.knowledge.gap_import import (
    _latest_per_task,
    _parse_ts,
    _title_from_entry,
    _topic_from_entry,
    import_help_queue,
)
from agi.knowledge.graph import get_node, load_latest, upsert_node


def _write_queue(p: Path, entries: list[dict]) -> None:
    p.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _entry(task: int, **overrides) -> dict:
    base = {
        "task": task,
        "timestamp": "2026-04-19T10:30:00",
        "attempts": 12,
        "best_score": "2/3",
        "error_types": ["spatial_primitive"],
        "recent_failures": [],
        "insights": [],
        "question": f"I have tried task{task:03d} 12 times; is this local or global?",
    }
    base.update(overrides)
    return base


# ── helpers ─────────────────────────────────────────────────────


def test_parse_ts_iso_and_int_and_bad():
    assert _parse_ts("2026-04-19T10:30:00") > 0
    assert _parse_ts("2026-04-19T10:30:00Z") > 0
    assert _parse_ts(1713600000) == 1713600000
    assert _parse_ts("not-a-date") is None
    assert _parse_ts(None) is None
    assert _parse_ts("") is None


def test_latest_per_task_picks_newest_by_timestamp():
    q = [
        _entry(167, timestamp="2026-04-18T08:00:00"),
        _entry(168, timestamp="2026-04-18T09:00:00"),
        _entry(167, timestamp="2026-04-19T10:00:00"),  # newer for 167
    ]
    out = _latest_per_task(q)
    assert set(out.keys()) == {167, 168}
    assert "2026-04-19" in out[167]["timestamp"]


def test_latest_per_task_falls_back_to_list_order_when_ts_missing():
    q = [
        _entry(167, timestamp=None, question="first"),
        _entry(167, timestamp=None, question="second"),
    ]
    out = _latest_per_task(q)
    assert out[167]["question"] == "second"


def test_latest_per_task_skips_bad_entries():
    q = [
        {"task": "not-an-int"},  # skipped
        "not-a-dict",  # skipped
        _entry(167),
    ]
    out = _latest_per_task(q)  # type: ignore[arg-type]
    assert set(out.keys()) == {167}


def test_topic_from_entry_uses_first_error_type():
    assert _topic_from_entry(_entry(1, error_types=["spatial_primitive", "x"])) == (
        "spatial primitive"
    )


def test_topic_from_entry_fallback_when_no_error_types():
    assert _topic_from_entry(_entry(1, error_types=[])) == "stuck task"


def test_title_truncated_and_prefixed():
    long_q = "x " * 100
    title = _title_from_entry(_entry(1, question=long_q), 1)
    assert title.startswith("[gap] ")
    assert len(title) <= 130


def test_title_fallback_without_question():
    title = _title_from_entry(_entry(7, question=None), 7)
    assert title == "[gap] Task 007"


# ── import integration ─────────────────────────────────────────


def test_import_creates_gap_node(tmp_path: Path):
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    _write_queue(q, [_entry(167)])

    rep = import_help_queue(q, graph_path=g)

    assert rep.imported == 1
    assert rep.refreshed == 0
    assert rep.skipped_filled == 0
    assert rep.failed == 0
    node = get_node("sensei_task_167", path=g)
    assert node is not None
    assert node["type"] == "gap"
    assert node["status"] == "active"
    assert node["verified"] is False
    assert node["source"] == "help_queue"
    assert node["body_ref"] is None
    assert node["topic_key"] == "spatial-primitive"
    assert "help:t167" in node["evidence"]
    assert "stuck" in node["tags"]


def test_import_idempotent_on_same_queue(tmp_path: Path):
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    _write_queue(q, [_entry(167)])

    r1 = import_help_queue(q, graph_path=g)
    lines_before = g.read_text().splitlines()
    r2 = import_help_queue(q, graph_path=g)
    lines_after = g.read_text().splitlines()

    assert r1.imported == 1
    assert r2.imported == 0
    assert r2.skipped_up_to_date == 1
    assert lines_before == lines_after  # literally no new writes


def test_import_refreshes_when_queue_entry_is_newer(tmp_path: Path):
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    _write_queue(q, [_entry(167, timestamp="2026-04-18T08:00:00")])
    import_help_queue(q, graph_path=g)

    # New entry, strictly newer
    _write_queue(q, [_entry(167, timestamp="2026-04-19T10:00:00", error_types=["x"])])
    rep = import_help_queue(q, graph_path=g)

    assert rep.refreshed == 1
    assert rep.imported == 0
    node = get_node("sensei_task_167", path=g)
    assert node["topic_key"] == "x"
    # Materialized view stays single node
    assert len(load_latest(path=g)) == 1


def test_import_leaves_filled_nodes_alone(tmp_path: Path):
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    # Seed with a filled node for task 167
    upsert_node(
        id="sensei_task_167",
        type="filled",
        topic="symmetry completion",
        title="Count distinct colors",
        body_ref="wiki/sensei_task_167.md",
        verified=True,
        source="primer",
        path=g,
    )
    _write_queue(q, [_entry(167)])

    rep = import_help_queue(q, graph_path=g)

    assert rep.skipped_filled == 1
    assert rep.imported == 0
    node = get_node("sensei_task_167", path=g)
    assert node["type"] == "filled"  # unchanged
    assert node["verified"] is True


def test_import_handles_missing_queue_file(tmp_path: Path):
    rep = import_help_queue(tmp_path / "nope.json", graph_path=tmp_path / "g.jsonl")
    assert rep.imported == 0
    assert rep.failed == 0


def test_import_handles_malformed_queue(tmp_path: Path):
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    q.write_text("{{not json", encoding="utf-8")
    rep = import_help_queue(q, graph_path=g)
    assert rep.failed == 1
    assert rep.imported == 0


def test_import_accepts_dict_wrapper(tmp_path: Path):
    """Defensively accept ``{"queue": [...]}`` even though arc_scientist writes a list."""
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    q.write_text(json.dumps({"queue": [_entry(167)]}), encoding="utf-8")

    rep = import_help_queue(q, graph_path=g)
    assert rep.imported == 1


def test_import_multiple_tasks_one_node_each(tmp_path: Path):
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    _write_queue(
        q,
        [
            _entry(100, timestamp="2026-04-18T08:00:00"),
            _entry(200, timestamp="2026-04-18T09:00:00"),
            _entry(300, timestamp="2026-04-18T10:00:00"),
            # duplicate of 100, newer — should collapse to single node
            _entry(100, timestamp="2026-04-19T08:00:00", error_types=["different"]),
        ],
    )
    rep = import_help_queue(q, graph_path=g)
    assert rep.imported == 3
    latest = load_latest(path=g)
    assert set(latest.keys()) == {
        "sensei_task_100",
        "sensei_task_200",
        "sensei_task_300",
    }
    # 100 should have the newer error_type
    assert latest["sensei_task_100"]["topic_key"] == "different"


def test_dry_run_writes_nothing(tmp_path: Path):
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    _write_queue(q, [_entry(167), _entry(168)])

    rep = import_help_queue(q, graph_path=g, dry_run=True)

    assert rep.imported == 2
    assert get_node("sensei_task_167", path=g) is None
    assert not g.exists() or g.read_text() == ""


def test_gap_to_filled_e2e(tmp_path: Path):
    """End-to-end: help-queue gap imported, then Primer-style upsert promotes it."""
    q = tmp_path / "queue.json"
    g = tmp_path / "g.jsonl"
    _write_queue(q, [_entry(167, timestamp="2026-04-18T08:00:00")])

    import_help_queue(q, graph_path=g)
    gap = get_node("sensei_task_167", path=g)
    assert gap["type"] == "gap"
    t0_created = gap["created_at"]

    # Simulate Primer publishing after some delay
    time.sleep(0.01)
    upsert_node(
        id="sensei_task_167",
        type="filled",
        topic="spatial primitive",
        title="Task 167 — spatial primitive",
        body_ref="sensei_task_167.md",
        verified=True,
        source="primer",
        evidence=["primer_task:167"],
        path=g,
    )

    filled = get_node("sensei_task_167", path=g)
    assert filled["type"] == "filled"
    assert filled["verified"] is True
    assert filled["created_at"] == t0_created  # preserved across promotion
    # Evidence union from the gap's help:t167 + primer_task:167
    assert "help:t167" in filled["evidence"]
    assert "primer_task:167" in filled["evidence"]
