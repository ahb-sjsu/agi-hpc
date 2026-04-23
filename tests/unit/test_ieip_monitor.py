# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""Unit tests for ``agi.safety.ieip_monitor``.

Uses the API-passthrough + Ensemble adapters so no PyTorch / model
downloads are required. The HF-backed path is exercised indirectly
through the adapter library's own tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from erisml.ieip.adapters import APIPassthroughAdapter, detect_adapter

from agi.safety.ieip_monitor import (
    DEFAULT_NATS_SUBJECT,
    EventPublisher,
    Monitor,
    Transform,
    _l1_divergence,
    identity_transform,
    monitor_for,
)

# ── Test helpers ---------------------------------------------------------


def _drifting_client(sequence):
    """Return a client whose generate() walks through ``sequence``."""
    it = iter(sequence)

    class _Client:
        def generate(self, inputs):
            try:
                return np.asarray(next(it), dtype=np.float64)
            except StopIteration:
                return np.asarray(sequence[-1], dtype=np.float64)

    return _Client()


class _CapturePublisher(EventPublisher):
    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish(self, subject, payload):
        self.published.append((subject, dict(payload)))


# ── Transform + helpers ---------------------------------------------------


def test_identity_transform_is_noop():
    t = identity_transform()
    assert t.name == "identity"
    assert t.apply("hello") == "hello"


def test_l1_divergence_is_zero_on_identical():
    p = np.array([0.5, 0.5])
    assert _l1_divergence(p, p) == pytest.approx(0.0)


def test_l1_divergence_pads_unequal_length():
    p = np.array([1.0, 0.0])
    q = np.array([1.0, 0.0, 0.0])
    assert _l1_divergence(p, q) == pytest.approx(0.0)


def test_l1_divergence_max_is_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert _l1_divergence(p, q) == pytest.approx(1.0)


# ── monitor_for factory --------------------------------------------------


def test_monitor_for_auto_detects_adapter(tmp_path):
    client_seq = [[0.5, 0.5]]
    mon = monitor_for(
        _drifting_client(client_seq),
        subsystem="test",
        events_path=tmp_path / "events.jsonl",
    )
    assert mon.subsystem == "test"
    assert isinstance(mon.adapter, APIPassthroughAdapter)


def test_monitor_adapter_info_summary(tmp_path):
    mon = monitor_for(
        _drifting_client([[1.0]]),
        subsystem="ego",
        events_path=tmp_path / "e.jsonl",
    )
    info = mon.adapter_info()
    assert info["subsystem"] == "ego"
    assert info["model_family"] == "api-passthrough"
    assert info["calibrated"] == 0
    assert info["event_seq"] == 0


# ── observe_output_distribution (no-hook path) ---------------------------


def test_observe_writes_jsonl_event(tmp_path):
    out = tmp_path / "events.jsonl"
    mon = monitor_for(
        _drifting_client([[0.6, 0.4]]),
        subsystem="test",
        events_path=out,
    )
    event = mon.observe("hello")
    assert event["subsystem"] == "test"
    assert event["mode"] == "output-distribution"
    assert event["seq"] == 1
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["seq"] == 1
    assert loaded["subsystem"] == "test"


def test_observe_respects_event_path_none(tmp_path):
    mon = monitor_for(_drifting_client([[1.0]]), events_path=None)
    event = mon.observe("x")
    # No file written but event still generated in memory.
    assert event["seq"] == 1
    assert mon.recent_events(10) == [event]


def test_observe_produces_zero_error_on_identity_transform():
    client = _drifting_client([[1.0, 0.0]])
    mon = monitor_for(client, subsystem="test", events_path=None)
    event = mon.observe("hi", transforms=[identity_transform()])
    # identity transform → p(x) == p(g·x) → zero error
    assert len(event["sites"]) == 1
    assert event["sites"][0]["error"] == pytest.approx(0.0)


def test_observe_detects_divergence_under_perturbing_transform():
    # Client returns different distributions for different inputs.
    class _KeyedClient:
        def generate(self, inp):
            if inp == "hello":
                return np.array([1.0, 0.0])
            return np.array([0.0, 1.0])

    mon = Monitor(
        adapter=detect_adapter(_KeyedClient()),
        subsystem="test",
        events_path=None,
    )
    upper = Transform(name="upper", apply=lambda s: s.upper())
    event = mon.observe("hello", transforms=[upper])
    assert event["sites"][0]["error"] == pytest.approx(1.0)


def test_observe_publishes_to_publisher(tmp_path):
    pub = _CapturePublisher()
    mon = monitor_for(
        _drifting_client([[0.5, 0.5]]),
        events_path=None,
    )
    mon.set_publisher(pub)
    mon.observe("x")
    assert len(pub.published) == 1
    subject, payload = pub.published[0]
    assert subject == DEFAULT_NATS_SUBJECT
    assert payload["subsystem"] == mon.subsystem


def test_observe_raises_when_output_distribution_unsupported():
    # Build an adapter that declares no capabilities.
    from erisml.ieip.adapters.base import BaseAdapter

    class _Dead(BaseAdapter):
        model_family = "dead"

        def list_sites(self, *, layers=None, kinds=None):
            return []

        def resolve_site(self, site):
            return None

        def num_layers(self):
            return -1

    mon = Monitor(adapter=_Dead(model=object()), events_path=None)
    with pytest.raises(ValueError, match="can't produce"):
        mon.observe_output_distribution("x")


# ── Alert aggregation -----------------------------------------------------


def test_alert_level_normal_on_single_event():
    mon = monitor_for(_drifting_client([[0.5, 0.5]]), events_path=None)
    event = mon.observe("x")
    assert event["alert_level"] == "normal"


# ── Sequence numbers increase monotonically -------------------------------


def test_event_sequence_numbers_monotonic(tmp_path):
    mon = monitor_for(
        _drifting_client([[0.5, 0.5]]),
        events_path=tmp_path / "e.jsonl",
    )
    for _ in range(5):
        mon.observe("x")
    seqs = [ev["seq"] for ev in mon.recent_events(10)]
    assert seqs == [1, 2, 3, 4, 5]


# ── JSONL is valid on flushed append --------------------------------------


def test_jsonl_lines_parse_cleanly(tmp_path):
    out = tmp_path / "events.jsonl"
    mon = monitor_for(_drifting_client([[0.5, 0.5]]), events_path=out)
    for i in range(3):
        mon.observe(f"item-{i}")
    lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        loaded = json.loads(line)
        assert "seq" in loaded
        assert "ts" in loaded
        assert "sites" in loaded


# ── Extra metadata is carried through ------------------------------------


def test_extra_is_attached_to_event(tmp_path):
    mon = monitor_for(_drifting_client([[1.0]]), events_path=None)
    event = mon.observe("x", task_id="arc-042", extra={"stage": "dream"})
    assert event["task_id"] == "arc-042"
    assert event["extra"] == {"stage": "dream"}
