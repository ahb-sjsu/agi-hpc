# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for Erebus's Director (Phase A: self-model + reflection).

Dependency-light by design: no torch, no NATS broker, no GPU. The whole
Phase A cycle is file IO + reconciliation, so it runs anywhere pytest does.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agi.director import events, journal, perceive, service
from agi.director.policy import AutonomyTier, PolicyError, gate, is_forbidden
from agi.director.self_model import SelfModel

# ── policy / governance ──────────────────────────────────────────


def test_tier_parse():
    assert AutonomyTier.parse("L2", AutonomyTier.L0) is AutonomyTier.L2
    assert AutonomyTier.parse("3", AutonomyTier.L0) is AutonomyTier.L3
    assert AutonomyTier.parse(None, AutonomyTier.L1) is AutonomyTier.L1


def test_gate_allows_within_ceiling():
    gate(AutonomyTier.L0, ceiling=AutonomyTier.L0)  # no raise
    gate(AutonomyTier.L1, ceiling=AutonomyTier.L2)  # no raise


def test_gate_denies_above_ceiling():
    with pytest.raises(PolicyError):
        gate(AutonomyTier.L2, ceiling=AutonomyTier.L0)


@pytest.mark.parametrize(
    "verb", ["reboot", "systemctl restart atlas-id", "kill -9 1", "rm -rf /"]
)
def test_forbidden_verbs_never_allowed(verb):
    assert is_forbidden(verb)
    # Forbidden even at the highest ceiling.
    with pytest.raises(PolicyError):
        gate(AutonomyTier.L0, verb=verb, ceiling=AutonomyTier.L3)


# ── self-model ───────────────────────────────────────────────────


def test_self_model_roundtrip():
    m = SelfModel(cycle=3, capabilities=["a"], open_problems=["p"])
    m2 = SelfModel.from_dict(m.to_dict())
    assert m2.cycle == 3
    assert m2.capabilities == ["a"]


def test_self_model_from_dict_ignores_unknown_keys():
    m = SelfModel.from_dict({"cycle": 1, "bogus": 42})
    assert m.cycle == 1


def test_self_model_save_load(tmp_path):
    m = SelfModel(cycle=7, values=["honesty"])
    m.save(tmp_path)
    assert (tmp_path / "self_model.json").exists()
    assert (tmp_path / "self_model.md").exists()
    loaded = SelfModel.load(tmp_path)
    assert loaded.cycle == 7
    assert loaded.values == ["honesty"]


def test_self_model_load_missing_returns_default(tmp_path):
    assert SelfModel.load(tmp_path / "nope").cycle == 0


def test_render_md_has_sections_and_honesty():
    goals = [{"title": "solve ARC", "status": "active"}]
    md = SelfModel(active_goals=goals).render_md()
    assert "# Erebus — self-model" in md
    assert "## Capabilities" in md
    assert "## Active goals" in md
    assert "solve ARC" in md


# ── perception ───────────────────────────────────────────────────


def test_perceive_missing_files_degrades(tmp_path):
    p = perceive.Paths(
        memory=tmp_path / "mem.json",
        help_queue=tmp_path / "help.json",
        primer_health=tmp_path / "health.json",
        primer_events=tmp_path / "ev.jsonl",
        disabled_sentinel=tmp_path / ".dis",
        gpu1_maint_sentinel=tmp_path / ".gpu1",
    )
    st = perceive.gather(p)
    assert st.tasks_total == 0
    assert st.faculties_online == []
    # Limitations must always include the honesty + no-reboot statements.
    lims = perceive.derive_limitations(st)
    assert any("not conscious" in item for item in lims)
    assert any("never restart" in item.lower() for item in lims)


def test_perceive_counts_solved_and_stuck(tmp_path):
    mem = {
        "tasks": {
            "1": {"solved": True, "attempts": [{}]},
            "2": {"solved": False, "attempts": [{}] * 12},  # stuck
            "3": {"solved": False, "attempts": [{}] * 3},   # not yet stuck
        }
    }
    health = {"kimi": {"degraded": False}, "qwen3": {"degraded": True}}
    mp = tmp_path / "mem.json"
    mp.write_text(json.dumps(mem))
    hp = tmp_path / "health.json"
    hp.write_text(json.dumps(health))
    p = perceive.Paths(
        memory=mp, primer_health=hp,
        help_queue=tmp_path / "x", primer_events=tmp_path / "y",
        disabled_sentinel=tmp_path / "z", gpu1_maint_sentinel=tmp_path / "w",
    )
    st = perceive.gather(p, min_attempts=10)
    assert st.tasks_total == 3
    assert st.tasks_solved == 1
    assert st.tasks_stuck == 1
    assert "kimi" in st.primer_experts_healthy
    assert "qwen3" in st.primer_experts_degraded


# ── journal ──────────────────────────────────────────────────────


def test_journal_append_tail(tmp_path):
    jp = tmp_path / "journal.jsonl"
    journal.append(ts=1.0, cycle=1, kind="tick", summary="first", path=jp)
    journal.append(ts=2.0, cycle=2, kind="tick", summary="second", path=jp)
    entries = journal.tail(10, path=jp)
    assert [e["summary"] for e in entries] == ["first", "second"]
    assert entries[-1]["cycle"] == 2


# ── reconciliation ───────────────────────────────────────────────


def test_reconcile_increments_and_summarizes():
    prev = SelfModel(cycle=4, self_state={"tasks_solved": 5, "tasks_stuck": 10})
    state = perceive.SelfState(tasks_total=100, tasks_solved=7, tasks_stuck=9,
                               help_queue_len=2)
    new, summary = service.reconcile(prev, state, ts=100.0, tier=AutonomyTier.L0)
    assert new.cycle == 5
    assert new.autonomy_tier == "L0"
    assert "solved +2" in summary  # 7 - 5
    assert "stuck -1" in summary   # 9 - 10
    assert new.recent_history[-1] == summary


# ── full read-only cycle ─────────────────────────────────────────


def _cfg(tmp_path):
    return service.Config(
        directory=tmp_path / "erebus",
        tick_s=3600,
        deep_hour=9,
        nats_servers="nats://127.0.0.1:4222",
        min_attempts=10,
        paths=perceive.Paths(
            memory=tmp_path / "mem.json",
            help_queue=tmp_path / "help.json",
            primer_health=tmp_path / "health.json",
            primer_events=tmp_path / "ev.jsonl",
            disabled_sentinel=tmp_path / ".dis",
            gpu1_maint_sentinel=tmp_path / ".gpu1",
        ),
    )


def test_run_cycle_writes_artifacts_and_chains_proof(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTOR_MAX_TIER", "L0")
    cfg = _cfg(tmp_path)
    node = events.DirectorNode()  # never connected → publishes no-op

    rec1 = asyncio.run(service.run_cycle(cfg, node, deep=False))
    d = cfg.directory
    assert (d / "self_model.json").exists()
    assert (d / "journal.jsonl").exists()
    assert (d / "director_status.json").exists()
    assert (d / "last_proof.txt").exists()

    status = json.loads((d / "director_status.json").read_text())
    assert status["max_tier"] == "L0"
    assert status["summary"]["cycle"] == 1

    # Second cycle: model advances, proof chains on the first.
    rec2 = asyncio.run(service.run_cycle(cfg, node, deep=False))
    assert rec2["cycle"] == 2
    assert rec2["proof"] != rec1["proof"]
    assert SelfModel.load(d).cycle == 2


def test_command_handler_status_pause_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("DIRECTOR_MAX_TIER", "L0")
    cfg = _cfg(tmp_path)
    node = events.DirectorNode()
    handler = service._make_handler(cfg, node)

    r = asyncio.run(handler({"cmd": "status"}))
    assert r["ok"] is True

    r = asyncio.run(handler({"cmd": "pause"}))
    assert r["paused"] is True
    assert cfg.paths.disabled_sentinel.exists()

    r = asyncio.run(handler({"cmd": "resume"}))
    assert r["paused"] is False
    assert not cfg.paths.disabled_sentinel.exists()

    r = asyncio.run(handler({"cmd": "frobnicate"}))
    assert r["ok"] is False


def test_director_node_noop_without_broker():
    node = events.DirectorNode()
    assert node.connected is False
    # Publishing without a connection must be a safe no-op.
    asyncio.run(node.publish_state({"x": 1}))
    asyncio.run(node.publish_journal({"y": 2}))
