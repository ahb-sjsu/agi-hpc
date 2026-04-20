"""Unit tests for the Phase 6 context-reader dispatch.

Exercises ``_wiki_context_snippets`` and ``_graph_context_snippets``
directly + ``_context_snippets`` switching on EREBUS_CONTEXT_READER.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from agi.knowledge.graph import upsert_node
from agi.primer.service import (
    Config,
    _context_snippets,
    _graph_context_snippets,
    _wiki_context_snippets,
)

_VERIFIED_TASK_167 = """---
type: sensei_note
task: 167
tags: [classification, count-distinct-colors, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

# Task 167

## The rule
Count distinct colors.
"""

_VERIFIED_META = """---
type: sensei_note
written_by: human
written_at: 2026-04-10
verified_by: hand-reviewed
---

# Meta note

Cross-cutting wisdom about ARC puzzles.
"""

_UNVERIFIED_TASK = """---
type: sensei_note
task: 167
tags: [draft]
written_by: human
---

# Draft, do not cite.
"""


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """Minimal Config pointing wiki_dir at a fresh tmp."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    return Config(
        wiki_dir=wiki,
        memory_path=tmp_path / "memory.json",
        task_dir=tmp_path,
        help_path=tmp_path / "queue.json",
        repo_dir=tmp_path,
        poll_s=300,
        min_attempts=10,
        cooldown_s=21600,
    )


@pytest.fixture
def graph_path(tmp_path: Path, monkeypatch) -> Path:
    """Point the graph store at a fresh tmp JSONL."""
    p = tmp_path / "graph.jsonl"
    monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(p))
    import agi.knowledge.graph as g

    monkeypatch.setattr(g, "DEFAULT_PATH", p)
    return p


# ── wiki reader (legacy) ─────────────────────────────────────────


def test_wiki_reader_returns_verified_task_note(cfg: Config):
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    snips = _wiki_context_snippets(167, cfg)
    assert len(snips) == 1
    assert "Task 167" in snips[0]


def test_wiki_reader_skips_unverified_task_note(cfg: Config):
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_UNVERIFIED_TASK)
    snips = _wiki_context_snippets(167, cfg)
    assert snips == []


def test_wiki_reader_includes_meta_notes(cfg: Config):
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    (cfg.wiki_dir / "sensei_meta_core.md").write_text(_VERIFIED_META)
    snips = _wiki_context_snippets(167, cfg)
    assert len(snips) == 2
    assert any("Meta note" in s for s in snips)


# ── graph reader ─────────────────────────────────────────────────


def test_graph_reader_returns_task_note_when_filled_and_verified(
    cfg: Config, graph_path: Path
):
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    upsert_node(
        id="sensei_task_167",
        type="filled",
        topic="count distinct colors",
        title="Task 167",
        body_ref="sensei_task_167.md",
        verified=True,
        source="primer",
        path=graph_path,
    )
    snips = _graph_context_snippets(167, cfg)
    assert len(snips) == 1
    assert "Task 167" in snips[0]


def test_graph_reader_skips_gap_node(cfg: Config, graph_path: Path, caplog):
    # A gap exists but has no body; eligibility gate rejects it.
    upsert_node(
        id="sensei_task_167",
        type="gap",
        topic="stuck",
        title="[gap] task 167",
        source="help_queue",
        path=graph_path,
    )
    caplog.set_level(logging.WARNING, logger="primer")
    snips = _graph_context_snippets(167, cfg)
    assert snips == []
    assert any("zero eligible snippets" in m for m in caplog.messages)


def test_graph_reader_skips_filled_with_missing_body(
    cfg: Config, graph_path: Path, caplog
):
    # Graph thinks there's a note, but the wiki file is absent —
    # is_context_eligible's body-exists check must reject it.
    upsert_node(
        id="sensei_task_167",
        type="filled",
        topic="count distinct colors",
        title="Task 167",
        body_ref="sensei_task_167.md",  # file not on disk
        verified=True,
        source="primer",
        path=graph_path,
    )
    caplog.set_level(logging.WARNING, logger="knowledge.graph")
    snips = _graph_context_snippets(167, cfg)
    assert snips == []
    assert any("filled_excluded_missing_body" in m for m in caplog.messages)


def test_graph_reader_includes_eligible_meta_notes(cfg: Config, graph_path: Path):
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    (cfg.wiki_dir / "sensei_meta_core.md").write_text(_VERIFIED_META)
    upsert_node(
        id="sensei_task_167",
        type="filled",
        topic="count distinct colors",
        title="Task 167",
        body_ref="sensei_task_167.md",
        verified=True,
        source="primer",
        path=graph_path,
    )
    upsert_node(
        id="sensei_meta_core",
        type="filled",
        topic="meta",
        title="Meta note",
        body_ref="sensei_meta_core.md",
        verified=True,
        source="backfill",
        path=graph_path,
    )
    snips = _graph_context_snippets(167, cfg)
    ids_present = sum(
        1 for s in snips if "sensei_task_167" in s or "sensei_meta_core" in s
    )
    assert ids_present == 2


def test_graph_reader_ignores_filled_non_meta_other_tasks(
    cfg: Config, graph_path: Path
):
    """A filled note for task 999 must NOT leak into task 167's context."""
    (cfg.wiki_dir / "sensei_task_999.md").write_text(
        _VERIFIED_TASK_167.replace("167", "999")
    )
    upsert_node(
        id="sensei_task_999",
        type="filled",
        topic="other",
        title="Task 999",
        body_ref="sensei_task_999.md",
        verified=True,
        source="primer",
        path=graph_path,
    )
    caplog = logging.getLogger("primer")
    snips = _graph_context_snippets(167, cfg)
    # Should be empty — task 167 has no eligible node, and 999 is not meta.
    assert snips == []


def test_graph_reader_warns_on_empty(cfg: Config, graph_path: Path, caplog):
    caplog.set_level(logging.WARNING, logger="primer")
    snips = _graph_context_snippets(167, cfg)
    assert snips == []
    assert any("zero eligible snippets" in m for m in caplog.messages)


# ── dispatch switch ──────────────────────────────────────────────


def test_context_snippets_default_is_wiki(cfg: Config, monkeypatch):
    monkeypatch.delenv("EREBUS_CONTEXT_READER", raising=False)
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    snips = _context_snippets(167, cfg)
    assert len(snips) == 1
    assert "Task 167" in snips[0]


def test_context_snippets_switches_to_graph(cfg: Config, graph_path: Path, monkeypatch):
    monkeypatch.setenv("EREBUS_CONTEXT_READER", "graph")
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    # No wiki glob behavior expected — wipe any helper files that would
    # make the wiki reader accidentally succeed.
    upsert_node(
        id="sensei_task_167",
        type="filled",
        topic="count distinct colors",
        title="Task 167",
        body_ref="sensei_task_167.md",
        verified=True,
        source="primer",
        path=graph_path,
    )
    snips = _context_snippets(167, cfg)
    assert len(snips) == 1


def test_context_snippets_graph_mode_is_empty_when_graph_is_empty(
    cfg: Config, graph_path: Path, monkeypatch, caplog
):
    """A wiki file on disk must NOT be a fallback when mode=graph."""
    monkeypatch.setenv("EREBUS_CONTEXT_READER", "graph")
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    caplog.set_level(logging.WARNING, logger="primer")
    snips = _context_snippets(167, cfg)
    # Graph has no node for task 167 → empty, with warning; no silent
    # wiki fallback (per the spec's rollout-safety note).
    assert snips == []
    assert any("zero eligible snippets" in m for m in caplog.messages)


def test_context_snippets_invalid_mode_falls_back_to_wiki_default(
    cfg: Config, monkeypatch
):
    """context_reader_mode() already validates; an invalid value should
    resolve to the ``wiki`` default and log a warning."""
    monkeypatch.setenv("EREBUS_CONTEXT_READER", "hybrid")
    (cfg.wiki_dir / "sensei_task_167.md").write_text(_VERIFIED_TASK_167)
    snips = _context_snippets(167, cfg)
    assert len(snips) == 1
