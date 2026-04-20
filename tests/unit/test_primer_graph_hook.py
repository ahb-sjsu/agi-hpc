"""Unit tests for the Primer → UKG writer hook (Phase 3).

Exercises ``_upsert_graph_node`` directly. Does NOT touch git or the
real Primer loop — the hook is a pure function of (task_num, parsed,
note_path) that delegates to ``agi.knowledge.graph.upsert_node``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agi.knowledge.graph import get_node, load_latest, upsert_node
from agi.primer.service import _upsert_graph_node


@pytest.fixture
def graph_path(tmp_path: Path, monkeypatch) -> Path:
    """Point the knowledge-graph store at a fresh tmp file for the test."""
    p = tmp_path / "graph.jsonl"
    monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(p))
    # graph.py reads DEFAULT_PATH at import time; override by reloading
    # the attribute so tests see the new env value.
    import agi.knowledge.graph as g

    monkeypatch.setattr(g, "DEFAULT_PATH", p)
    return p


@pytest.fixture
def note_path(tmp_path: Path) -> Path:
    p = tmp_path / "sensei_task_167.md"
    p.write_text(
        "# Task 167\n\n## The rule\n\n```python\ndef transform(g): return g\n```\n"
    )
    return p


# ── basic hook behavior ─────────────────────────────────────────


def test_hook_creates_filled_verified_node(graph_path: Path, note_path: Path):
    parsed = {
        "class": "CLASSIFICATION",
        "family": "count-distinct-colors",
    }
    _upsert_graph_node(167, parsed, note_path)

    node = get_node("sensei_task_167", path=graph_path)
    assert node is not None
    assert node["type"] == "filled"
    assert node["verified"] is True
    assert node["status"] == "active"
    assert node["topic_key"] == "count-distinct-colors"
    assert node["source"] == "primer"
    assert "primer_task:167" in node["evidence"]
    assert node["body_ref"] == "sensei_task_167.md"
    assert "classification" in node["tags"]
    assert "count-distinct-colors" in node["tags"]


def test_hook_falls_back_to_class_when_no_family(graph_path: Path, note_path: Path):
    parsed = {"class": "TRANSFORMATION", "family": ""}
    _upsert_graph_node(167, parsed, note_path)
    node = get_node("sensei_task_167", path=graph_path)
    assert node["topic_key"] == "transformation"


def test_hook_handles_empty_parsed_dict(graph_path: Path, note_path: Path):
    """Primer's ``except Exception: parsed = {}`` fallback path."""
    _upsert_graph_node(167, {}, note_path)
    node = get_node("sensei_task_167", path=graph_path)
    # Falls back to default 'transformation' class
    assert node is not None
    assert node["type"] == "filled"
    assert node["verified"] is True


# ── promotion (gap → filled) ────────────────────────────────────


def test_hook_promotes_existing_gap_preserving_created_at(
    graph_path: Path, note_path: Path
):
    # Seed the graph with a gap (simulates Phase 4's help-queue import).
    t0 = int(time.time()) - 3600
    upsert_node(
        id="sensei_task_167",
        type="gap",
        topic="symmetry completion",
        title="open: task 167",
        source="help_queue",
        evidence=["help:t167"],
        now=t0,
        path=graph_path,
    )

    # Primer publishes — hook fires.
    _upsert_graph_node(
        167, {"class": "TRANSFORMATION", "family": "symmetry-completion"}, note_path
    )

    node = get_node("sensei_task_167", path=graph_path)
    assert node["type"] == "filled"
    assert node["verified"] is True
    assert node["created_at"] == t0  # preserved from the gap
    # Evidence union: help:t167 first (from gap), then primer_task:167
    assert node["evidence"][0] == "help:t167"
    assert "primer_task:167" in node["evidence"]


# ── idempotency / robustness ────────────────────────────────────


def test_hook_second_publish_appends_snapshot_but_materializes_same(
    graph_path: Path, note_path: Path
):
    parsed = {"class": "CLASSIFICATION", "family": "count-distinct-colors"}
    _upsert_graph_node(167, parsed, note_path)
    lines1 = graph_path.read_text().splitlines()
    _upsert_graph_node(167, parsed, note_path)
    lines2 = graph_path.read_text().splitlines()

    assert len(lines2) == len(lines1) + 1  # append-only contract
    latest = load_latest(path=graph_path)
    assert len(latest) == 1  # still one node
    assert latest["sensei_task_167"]["verified"] is True


def test_hook_swallows_graph_errors(monkeypatch, note_path: Path, caplog):
    """A graph-write failure must not raise — the publish path is primary."""
    import agi.primer.service as svc

    def _boom(**kwargs):
        raise RuntimeError("disk full")

    # Patch the import site — the hook does a local import, so patch the
    # graph module itself.
    import agi.knowledge.graph as g

    monkeypatch.setattr(g, "upsert_node", _boom)

    # Must not raise
    svc._upsert_graph_node(167, {"class": "TRANSFORMATION", "family": "x"}, note_path)
    assert any("graph upsert failed" in m for m in caplog.messages)
