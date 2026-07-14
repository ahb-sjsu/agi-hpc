# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the Director's Phase B goal loop.

Dependency-light: no erisml/DEME, no NRP, no NATS, no sensors. Gates and
resource probes are injected, so the whole deliberate→gate→dispatch path is
exercised deterministically.
"""

from __future__ import annotations

import json
import types

import pytest

from agi.director import deliberate as delib
from agi.director import dispatch, goals_phase
from agi.director.charter import Charter, Objective, measure
from agi.director.gate import DemeGate, GateVerdict, NullGate
from agi.director.goals import Goal, GoalTree, Status
from agi.director.govern import Budget, Governor
from agi.director.policy import AutonomyTier, PolicyError

# ── fakes ────────────────────────────────────────────────────────


class FakeGate:
    def __init__(self, allow: bool):
        self._allow = allow

    def gate(self, description, context) -> GateVerdict:
        return GateVerdict(allow=self._allow, score=1.0 if self._allow else 0.0,
                           rationale="fake")


class FakeSafetyGateway:
    """Mimics SafetyGateway.check_action → SafetyResult."""

    def __init__(self, passed: bool):
        self._passed = passed

    def check_action(self, desc, ctx=None):
        return types.SimpleNamespace(
            passed=self._passed, score=1.0 if self._passed else 0.0,
            flags=[], decision_proof={"x": 1},
        )


def _cfg(tmp_path, max_proposals=5):
    d = tmp_path / "erebus"
    d.mkdir()
    paths = types.SimpleNamespace(memory=tmp_path / "mem.json")
    return types.SimpleNamespace(
        directory=d,
        charter_path=d / "charter.json",
        goals_path=d / "goals.json",
        directives_path=d / "directives.json",
        wiki_dir=tmp_path / "wiki",
        paths=paths,
        min_attempts=10,
        max_proposals=max_proposals,
    )


def _write_charter(cfg, objectives):
    cfg.charter_path.write_text(json.dumps({
        "version": 1, "objectives": objectives,
        "limits": {"max_dispatch_per_cycle": 3, "max_dispatch_per_day": 20},
    }))


def _lenient_gov(tmp_path, **budget):
    return Governor(
        Budget(**budget) if budget else Budget(),
        temp_reader=lambda: 50.0,
        gpu1_maint=tmp_path / ".no_gpu1_maint",
        rate_path=tmp_path / "rate.json",
    )


# ── charter ──────────────────────────────────────────────────────


def test_objective_gap_up_down():
    up = Objective(id="o", title="t", metric="m", target=140, direction="up")
    assert up.gap(107) == pytest.approx((140 - 107) / 140)
    assert up.gap(140) == 0.0            # satisfied
    assert up.gap(150) == 0.0            # exceeded → no gap
    down = Objective(id="o", title="t", metric="m", target=80, direction="down")
    assert down.gap(112) == pytest.approx((112 - 80) / 112)
    assert down.gap(80) == 0.0


def test_objective_gap_unmeasured():
    o = Objective(id="o", title="t", metric="m", target=140)
    assert o.gap(None) is None
    assert o.priority(None) is None


def test_measure_unregistered_metric_is_none():
    assert measure("arc_solved", {"tasks_solved": 5}) == 5.0
    assert measure("safety_dossier", {"tasks_solved": 5}) is None


def test_charter_ranked_gaps_orders_measured_first(tmp_path):
    cfg = _cfg(tmp_path)
    _write_charter(cfg, [
        {"id": "a", "title": "solve", "weight": 0.3, "metric": "arc_solved",
         "target": 140, "direction": "up"},
        {"id": "b", "title": "dossier", "weight": 0.9, "metric": "safety_dossier",
         "target": 1, "direction": "up"},  # unmeasured
    ])
    charter = Charter.load(cfg.charter_path)
    ranked = charter.ranked_gaps({"tasks_solved": 107})
    assert ranked[0][0].id == "a"            # measured, positive priority first
    assert ranked[-1][0].id == "b"           # unmeasured last
    assert ranked[-1][2] is None


def test_charter_load_missing_is_none(tmp_path):
    assert Charter.load(tmp_path / "nope.json") is None


# ── goals tree ───────────────────────────────────────────────────


def test_goal_key_dedupe():
    g1 = Goal(id="g1", title="a", action={"type": "teach_task", "args": {"task": 5}})
    g2 = Goal(id="g2", title="b", action={"type": "teach_task", "args": {"task": 5}})
    assert g1.key() == g2.key()


def test_goaltree_add_transition_save_load(tmp_path):
    t = GoalTree()
    g = t.add(Goal(id="", title="x",
                   action={"type": "teach_task", "args": {"task": 1}}))
    assert g.id == "g-0001"
    assert t.has_open_key(g.key())
    t.transition(g.id, Status.GATED.value, cycle=2)
    p = tmp_path / "goals.json"
    t.save(p)
    t2 = GoalTree.load(p)
    assert t2.get("g-0001").status == Status.GATED.value
    # done goals are not "open" for dedupe
    t2.transition("g-0001", Status.DONE.value, cycle=3)
    assert not t2.has_open_key(g.key())


# ── gate ─────────────────────────────────────────────────────────


def test_nullgate_denies():
    assert NullGate().gate("do x", {}).allow is False


def test_demegate_maps_safetyresult():
    assert DemeGate(FakeSafetyGateway(True)).gate("x", {}).allow is True
    assert DemeGate(FakeSafetyGateway(False)).gate("x", {}).allow is False


def test_demegate_error_denies():
    class Boom:
        def check_action(self, *a, **k):
            raise RuntimeError("nope")

    assert DemeGate(Boom()).gate("x", {}).allow is False


# ── governor ─────────────────────────────────────────────────────


def test_governor_thermal_and_unknown_deny(tmp_path):
    hot = Governor(Budget(thermal_c=82), temp_reader=lambda: 90.0,
                   gpu1_maint=tmp_path / "x", rate_path=tmp_path / "r.json")
    assert hot.can_dispatch().ok is False
    unknown = Governor(Budget(), temp_reader=lambda: None,
                       gpu1_maint=tmp_path / "x", rate_path=tmp_path / "r2.json")
    assert unknown.can_dispatch().ok is False  # deny-by-default


def test_governor_gpu1_loan_denies(tmp_path):
    sentinel = tmp_path / ".gpu1_maint"
    sentinel.touch()
    g = Governor(Budget(), temp_reader=lambda: 50.0, gpu1_maint=sentinel,
                 rate_path=tmp_path / "r.json")
    assert g.can_dispatch().ok is False


def test_governor_rate_limit_per_cycle(tmp_path):
    g = _lenient_gov(tmp_path, thermal_c=82, max_per_cycle=2, max_per_day=20)
    g.start_cycle()
    assert g.can_dispatch().ok
    g.record_dispatch()
    assert g.can_dispatch().ok
    g.record_dispatch()
    assert g.can_dispatch().ok is False  # 2 reached


# ── dispatch ─────────────────────────────────────────────────────


def test_dispatch_teach_task_writes_directive(tmp_path):
    g = Goal(id="g1", title="teach", tier="L2",
             action={"type": "teach_task", "args": {"task": 42}},
             provenance={"rationale": "why"})
    dp = tmp_path / "directives.json"
    res = dispatch.dispatch(g, directives_path=dp, cycle=1, ceiling=AutonomyTier.L2)
    assert res.ok and "042" in res.detail
    assert dispatch.open_teach_tasks(dp) == [42]
    # idempotent — second dispatch doesn't duplicate
    dispatch.dispatch(g, directives_path=dp, cycle=2, ceiling=AutonomyTier.L2)
    assert dispatch.open_teach_tasks(dp) == [42]
    # close it
    assert dispatch.close_directive("teach_task:42", dp)
    assert dispatch.open_teach_tasks(dp) == []


def test_dispatch_over_tier_raises(tmp_path):
    g = Goal(id="g1", title="teach", tier="L2",
             action={"type": "teach_task", "args": {"task": 42}})
    with pytest.raises(PolicyError):
        dispatch.dispatch(g, directives_path=tmp_path / "d.json", cycle=1,
                          ceiling=AutonomyTier.L1)


def test_dispatch_unknown_type_raises(tmp_path):
    g = Goal(id="g1", title="x", action={"type": "nope", "args": {}})
    with pytest.raises(ValueError):
        dispatch.dispatch(g, directives_path=tmp_path / "d.json", cycle=1,
                          ceiling=AutonomyTier.L3)


# ── deliberate ───────────────────────────────────────────────────


def test_read_stuck_tasks_ranks_partial_first(tmp_path):
    mem = {"tasks": {
        "1": {"solved": True, "attempts": [{}] * 20},
        "2": {"solved": False, "attempts": [{}] * 15, "best_correct": 0},
        "3": {"solved": False, "attempts": [{}] * 12, "best_correct": 2},  # partial
        "4": {"solved": False, "attempts": [{}] * 3},  # below min
    }}
    p = tmp_path / "mem.json"
    p.write_text(json.dumps(mem))
    ranked = delib.read_stuck_tasks(p, min_attempts=10)
    assert ranked[0] == 3    # partial progress first
    assert 4 not in ranked   # below min_attempts excluded
    assert 1 not in ranked   # solved excluded


def test_deliberate_gates_and_dedupes(tmp_path):
    cfg = _cfg(tmp_path, max_proposals=2)
    _write_charter(cfg, [
        {"id": "obj-solve", "title": "solve", "weight": 0.3, "metric": "arc_solved",
         "target": 140, "direction": "up"},
    ])
    charter = Charter.load(cfg.charter_path)
    tree = GoalTree()
    props = delib.deliberate(
        charter, {"tasks_solved": 107}, tree, FakeGate(True),
        cycle=1, stuck_provider=lambda: [11, 22, 33], max_proposals=2,
    )
    assert len(props) == 2                        # capped
    assert all(g.status == Status.GATED.value for g in props)
    # re-deliberate: same tasks already open → no new proposals
    props2 = delib.deliberate(
        charter, {"tasks_solved": 107}, tree, FakeGate(True),
        cycle=2, stuck_provider=lambda: [11, 22], max_proposals=5,
    )
    assert props2 == []


def test_deliberate_denied_goals_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    _write_charter(cfg, [
        {"id": "obj-solve", "title": "solve", "weight": 0.3, "metric": "arc_solved",
         "target": 140, "direction": "up"},
    ])
    charter = Charter.load(cfg.charter_path)
    tree = GoalTree()
    props = delib.deliberate(charter, {"tasks_solved": 107}, tree, NullGate(),
                             cycle=1, stuck_provider=lambda: [11], max_proposals=5)
    assert props[0].status == Status.REJECTED.value


def test_deliberate_skips_unmeasured_objective(tmp_path):
    cfg = _cfg(tmp_path)
    _write_charter(cfg, [
        {"id": "obj-dossier", "title": "safety", "weight": 0.9,
         "metric": "safety_dossier", "target": 1, "direction": "up"},
    ])
    charter = Charter.load(cfg.charter_path)
    tree = GoalTree()
    props = delib.deliberate(charter, {"tasks_solved": 107}, tree, FakeGate(True),
                             cycle=1, stuck_provider=lambda: [11], max_proposals=5)
    assert props == []  # unmeasured objective yields no goals (no fabrication)


# ── goals_phase (L1 propose-only vs L2 dispatch) ─────────────────


def _arc_charter(cfg):
    _write_charter(cfg, [
        {"id": "obj-solve", "title": "solve", "weight": 0.3, "metric": "arc_solved",
         "target": 140, "direction": "up"},
    ])
    cfg.paths.memory.write_text(json.dumps({"tasks": {
        "11": {"solved": False, "attempts": [{}] * 12, "best_correct": 1},
        "22": {"solved": False, "attempts": [{}] * 11},
    }}))


def test_goals_phase_l1_proposes_no_dispatch(tmp_path):
    cfg = _cfg(tmp_path)
    _arc_charter(cfg)
    note, events, counts = goals_phase.run(
        cfg, {"tasks_solved": 107}, AutonomyTier.L1, cycle=1, gate=FakeGate(True))
    assert counts["proposed"] >= 1
    assert counts["gated"] >= 1
    assert counts["dispatched"] == 0                 # L1 never dispatches
    assert not cfg.directives_path.exists()          # nothing written to Primer
    assert cfg.goals_path.exists()


def test_goals_phase_l2_dispatches_within_budget(tmp_path):
    cfg = _cfg(tmp_path)
    _arc_charter(cfg)
    gov = _lenient_gov(tmp_path, thermal_c=82, max_per_cycle=1, max_per_day=20)
    note, events, counts = goals_phase.run(
        cfg, {"tasks_solved": 107}, AutonomyTier.L2, cycle=1,
        gate=FakeGate(True), governor=gov)
    assert counts["dispatched"] == 1                 # capped at max_per_cycle=1
    assert counts["blocked"] >= 1                    # the rest blocked by rate
    tasks = dispatch.open_teach_tasks(cfg.directives_path)
    assert len(tasks) == 1                            # one directive written


def test_goals_phase_no_charter_is_noop(tmp_path):
    cfg = _cfg(tmp_path)  # no charter written
    note, events, counts = goals_phase.run(
        cfg, {"tasks_solved": 107}, AutonomyTier.L2, cycle=1, gate=FakeGate(True))
    assert (note, events, counts) == ("", [], {})


def test_goals_phase_completes_taught_goal(tmp_path):
    cfg = _cfg(tmp_path)
    _arc_charter(cfg)
    # Pre-seed an active teach goal for task 11, and make it "taught".
    tree = GoalTree()
    tree.add(Goal(id="g-0001", title="teach 011", tier="L2",
                  status=Status.ACTIVE.value,
                  action={"type": "teach_task", "args": {"task": 11}}))
    tree.save(cfg.goals_path)
    cfg.wiki_dir.mkdir()
    (cfg.wiki_dir / "sensei_task_011.md").write_text("taught")
    gov = _lenient_gov(tmp_path, thermal_c=82, max_per_cycle=3, max_per_day=20)
    note, events, counts = goals_phase.run(
        cfg, {"tasks_solved": 108}, AutonomyTier.L2, cycle=5,
        gate=FakeGate(True), governor=gov)
    assert counts["done"] == 1
    assert GoalTree.load(cfg.goals_path).get("g-0001").status == Status.DONE.value
