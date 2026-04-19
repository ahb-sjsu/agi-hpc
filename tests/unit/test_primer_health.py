"""Unit tests for agi.primer.health.HealthTracker."""

from __future__ import annotations

import time
from unittest.mock import patch

from agi.primer.health import HealthTracker


def test_fresh_tracker_is_healthy():
    t = HealthTracker()
    assert t.healthy("kimi")
    assert t.healthy("never-seen-name")


def test_degradation_triggers_after_min_failures():
    t = HealthTracker(window=5, slow_s=180.0, min_failures=3)
    # two unhealthy calls — still healthy
    t.record("kimi", latency_s=300.0, ok=False)
    t.record("kimi", latency_s=200.0, ok=False)
    assert t.healthy("kimi")
    # third unhealthy call crosses threshold
    t.record("kimi", latency_s=250.0, ok=False)
    assert not t.healthy("kimi")


def test_slow_but_ok_counts_as_unhealthy():
    t = HealthTracker(window=5, slow_s=100.0, min_failures=2)
    t.record("kimi", latency_s=150.0, ok=True)
    t.record("kimi", latency_s=160.0, ok=True)
    assert not t.healthy("kimi")  # both slow; 2 >= min_failures


def test_fast_ok_calls_stay_healthy():
    t = HealthTracker(window=5, slow_s=100.0, min_failures=3)
    for _ in range(5):
        t.record("qwen3", latency_s=50.0, ok=True)
    assert t.healthy("qwen3")


def test_cooldown_elapses():
    t = HealthTracker(window=3, slow_s=100.0, min_failures=2, cooldown_s=1)
    t.record("kimi", latency_s=200.0, ok=False)
    t.record("kimi", latency_s=200.0, ok=False)
    assert not t.healthy("kimi")
    time.sleep(1.1)
    assert t.healthy("kimi")


def test_recovery_window_rolls_over():
    """Once old unhealthy calls roll out of the window, a re-record that
    brings unhealthy_in_window below min_failures keeps the expert in
    degraded state until cooldown expires; but if cooldown is short and
    a fresh unhealthy burst arrives, we re-mark."""
    t = HealthTracker(window=3, slow_s=100.0, min_failures=2, cooldown_s=1)
    # Mark degraded
    t.record("kimi", latency_s=200.0, ok=False)
    t.record("kimi", latency_s=200.0, ok=False)
    assert not t.healthy("kimi")
    time.sleep(1.1)
    assert t.healthy("kimi")  # cooldown elapsed
    # Two more unhealthy calls re-mark immediately
    t.record("kimi", latency_s=200.0, ok=False)
    t.record("kimi", latency_s=200.0, ok=False)
    assert not t.healthy("kimi")


def test_clear():
    t = HealthTracker(window=2, slow_s=100.0, min_failures=2)
    t.record("kimi", latency_s=200.0, ok=False)
    t.record("kimi", latency_s=200.0, ok=False)
    assert not t.healthy("kimi")
    t.clear("kimi")
    assert t.healthy("kimi")


def test_summary_schema():
    t = HealthTracker(window=3, slow_s=100.0, min_failures=2)
    t.record("kimi", latency_s=50.0, ok=True)
    t.record("kimi", latency_s=200.0, ok=False)
    s = t.summary()
    assert "kimi" in s
    entry = s["kimi"]
    assert set(entry.keys()) >= {
        "healthy",
        "degraded_until_s",
        "window_size",
        "unhealthy_in_window",
        "avg_latency_s",
    }
    assert entry["window_size"] == 2
    assert entry["unhealthy_in_window"] == 1
    assert entry["healthy"] is True  # only 1 unhealthy, below min_failures=2


def test_summary_empty_for_unknown_expert():
    t = HealthTracker()
    assert "kimi" not in t.summary()
