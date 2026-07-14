"""Unit tests for the validated moral perception lane (xbse → DEME10).

No torch, no GPU, no 2 GB weights: the encoder factory and axis cache are
injected, so we exercise device policy, lazy load + LRU eviction, the
validated-or-absent contract, harm aggregation, and escalation purely in
Python. Real-weight scoring is proven separately on Atlas.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agi.safety.perception import (
    AxisReading,
    MoralPerception,
    PerceptionConfig,
    PerceptionResult,
)
from agi.safety.perception import xbse_perception as xp


# ── injectable stubs ───────────────────────────────────────────────


class _StubScorer:
    """Stands in for xbse.DimensionScorer; returns a fixed valence."""

    def __init__(self, value: float, confidence: float = 0.9):
        self._value = value
        self._confidence = confidence
        self.calls = 0

    def score(self, text: str):
        self.calls += 1
        return SimpleNamespace(value=self._value, confidence=self._confidence, direction=1)


def _perception_with(values: dict, monkeypatch, **cfg_kw):
    """Build a MoralPerception whose _build_scorer yields stub scorers.

    `values` maps axis name -> fixed valence. Any axis present in `values`
    also gets a calibration entry so it is considered validated.
    """
    axis_cache = {name: {"axis": [0.0], "center": 0.0, "scale": 1.0} for name in values}
    cfg = PerceptionConfig(prefer_gpu=False, **cfg_kw)  # default to CPU in tests
    mp = MoralPerception(config=cfg, axis_cache=axis_cache)

    built = {}

    def fake_build(name, ckpt_path, axis_rec):
        s = _StubScorer(values[name])
        built[name] = s
        return s

    # bypass real checkpoint existence + torch load
    monkeypatch.setattr(mp, "_build_scorer", fake_build)
    monkeypatch.setattr(xp.Path, "exists", lambda self: True)
    return mp, built


# ── device policy ──────────────────────────────────────────────────


def test_device_prefers_cpu_when_gpu_disabled(monkeypatch):
    mp, _ = _perception_with({"physical_harm": -1.0}, monkeypatch)
    assert mp._pick_device() == "cpu"


def _fake_sentinel(present: bool):
    return SimpleNamespace(exists=lambda: present)


def test_device_declines_gpu_when_maint_sentinel_present(monkeypatch):
    monkeypatch.setattr(xp, "GPU1_MAINT_SENTINEL", _fake_sentinel(True))
    monkeypatch.setattr(xp, "_gpu_free_mib", lambda idx: 32000)
    cfg = PerceptionConfig(prefer_gpu=True)
    mp = MoralPerception(config=cfg, axis_cache={})
    assert mp._pick_device() == "cpu"  # maint loan wins over free VRAM


def test_device_uses_gpu_when_free_and_no_maint(monkeypatch):
    monkeypatch.setattr(xp, "GPU1_MAINT_SENTINEL", _fake_sentinel(False))
    monkeypatch.setattr(xp, "_gpu_free_mib", lambda idx: 32000)
    cfg = PerceptionConfig(prefer_gpu=True, gpu_index=1, gpu_mem_floor_mib=6000)
    mp = MoralPerception(config=cfg, axis_cache={})
    assert mp._pick_device() == "cuda:1"


def test_device_declines_gpu_when_vram_below_floor(monkeypatch):
    monkeypatch.setattr(xp, "GPU1_MAINT_SENTINEL", _fake_sentinel(False))
    monkeypatch.setattr(xp, "_gpu_free_mib", lambda idx: 512)
    cfg = PerceptionConfig(prefer_gpu=True, gpu_mem_floor_mib=6000)
    mp = MoralPerception(config=cfg, axis_cache={})
    assert mp._pick_device() == "cpu"


# ── validated-or-absent contract ───────────────────────────────────


def test_absent_axis_when_no_calibration(monkeypatch):
    # request an axis with no calibration entry → absent, not guessed
    mp, _ = _perception_with({"physical_harm": -1.0}, monkeypatch)
    res = mp.score("hello", axes=("physical_harm", "fairness_equity"))
    assert res.axes["physical_harm"].present is True
    assert res.axes["fairness_equity"].present is False
    assert res.axes["fairness_equity"].validated is False
    assert res.n_absent == 1
    assert res.n_validated == 1


def test_unvalidated_axes_listed_in_dict(monkeypatch):
    mp, _ = _perception_with({"physical_harm": -1.0}, monkeypatch)
    d = mp.score("x", axes=("physical_harm", "privacy_protection")).to_dict()
    assert "privacy_protection" in d["unvalidated_axes"]
    assert "physical_harm" in d["dimension_scores"]


# ── harm aggregation polarity (xbse valence is uniformly sign-normalised) ──


def test_harm_upheld_is_zero(monkeypatch):
    # +1 = value UPHELD → harm 0.0, on every axis (here physical_harm)
    mp, _ = _perception_with({"physical_harm": 1.0}, monkeypatch)
    res = mp.score("safe", axes=("physical_harm",))
    assert res.harm_aggregate == pytest.approx(0.0)


def test_harm_violated_is_one(monkeypatch):
    # -1 = value VIOLATED → harm 1.0 (physical_harm)
    mp, _ = _perception_with({"physical_harm": -1.0}, monkeypatch)
    res = mp.score("harmful", axes=("physical_harm",))
    assert res.harm_aggregate == pytest.approx(1.0)


def test_harm_violated_is_one_other_axis(monkeypatch):
    # same uniform convention on a virtue-named axis
    mp, _ = _perception_with({"fairness_equity": -1.0}, monkeypatch)
    res = mp.score("unfair", axes=("fairness_equity",))
    assert res.harm_aggregate == pytest.approx(1.0)


# ── escalation ─────────────────────────────────────────────────────


def test_escalates_when_majority_absent(monkeypatch):
    mp, _ = _perception_with({"physical_harm": -1.0}, monkeypatch)
    # 1 validated of 4 requested → below half → escalate
    res = mp.score("x", axes=(
        "physical_harm", "fairness_equity", "privacy_protection", "autonomy_respect",
    ))
    assert res.escalate is True


def test_no_escalation_when_majority_validated(monkeypatch):
    vals = {"physical_harm": -1.0, "fairness_equity": 1.0, "privacy_protection": 1.0}
    mp, _ = _perception_with(vals, monkeypatch)
    res = mp.score("x", axes=tuple(vals) + ("autonomy_respect",))
    assert res.n_validated == 3
    assert res.escalate is False


# ── LRU eviction bounds resident memory ────────────────────────────


def test_lru_eviction_bounds_hot_set(monkeypatch):
    vals = {n: -1.0 for n in (
        "physical_harm", "fairness_equity", "privacy_protection",
        "autonomy_respect", "rights_respect",
    )}
    mp, _ = _perception_with(vals, monkeypatch, max_resident=2)
    for name in vals:
        mp._get_scorer(name)
    assert len(mp._hot) == 2
    # most-recently-used two survive
    assert "autonomy_respect" in mp._hot or "rights_respect" in mp._hot
    assert "physical_harm" not in mp._hot


def test_scorer_cached_not_rebuilt(monkeypatch):
    mp, built = _perception_with({"physical_harm": -1.0}, monkeypatch)
    mp.score("a", axes=("physical_harm",))
    mp.score("b", axes=("physical_harm",))
    # built exactly once; scored twice
    assert built["physical_harm"].calls == 2


# ── status + empty behavior ────────────────────────────────────────


def test_status_reports_calibration(monkeypatch):
    mp, _ = _perception_with({"physical_harm": -1.0, "fairness_equity": 1.0}, monkeypatch)
    st = mp.status()
    assert st["n_calibrated"] == 2
    assert "physical_harm" in st["calibrated_axes"]


def test_all_absent_no_crash(monkeypatch):
    mp = MoralPerception(config=PerceptionConfig(prefer_gpu=False), axis_cache={})
    res = mp.score("nothing calibrated", axes=("physical_harm",))
    assert res.n_validated == 0
    assert res.harm_aggregate == 0.0
    assert res.escalate is True
