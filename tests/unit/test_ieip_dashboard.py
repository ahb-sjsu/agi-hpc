# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""Tests for the dashboard status builder."""

from __future__ import annotations

import json
import time

import pytest

from agi.safety.ieip_dashboard import (
    build_status,
    load_status,
    read_events,
)

# ── read_events ----------------------------------------------------------


def test_read_events_missing_file_yields_nothing(tmp_path):
    assert list(read_events(tmp_path / "nope.jsonl")) == []


def test_read_events_skips_blank_and_bad_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        "\n".join(
            [
                "",
                '{"seq":1,"ts":1.0,"sites":[]}',
                "not json at all",
                '{"seq":2,"ts":2.0,"sites":[]}',
                "[1,2,3]",  # valid json but wrong shape
            ]
        ),
        encoding="utf-8",
    )
    events = list(read_events(p))
    assert [ev["seq"] for ev in events] == [1, 2]


def test_read_events_filters_by_since_ts(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"seq":1,"ts":100}',
                '{"seq":2,"ts":200}',
                '{"seq":3,"ts":300}',
            ]
        ),
        encoding="utf-8",
    )
    events = list(read_events(p, since_ts=150))
    assert [ev["seq"] for ev in events] == [2, 3]


def test_read_events_limit_tails(tmp_path):
    p = tmp_path / "e.jsonl"
    lines = [json.dumps({"seq": i, "ts": i, "sites": []}) for i in range(10)]
    p.write_text("\n".join(lines), encoding="utf-8")
    events = list(read_events(p, limit=3))
    assert [ev["seq"] for ev in events] == [7, 8, 9]


# ── build_status ---------------------------------------------------------


def _make_event(ts, subsystem="ego", alert="normal", site_events=None):
    return {
        "seq": 1,
        "ts": ts,
        "subsystem": subsystem,
        "model_family": "api-passthrough",
        "alert_level": alert,
        "sites": site_events or [],
    }


def test_build_status_empty_events_has_zero_totals():
    status = build_status([], window_seconds=60, sparkline_buckets=3)
    assert status["total_events"] == 0
    assert status["worst_alert"] == "normal"
    assert status["by_site"] == []
    assert len(status["sparkline"]) == 3


def test_build_status_excludes_events_outside_window():
    now = time.time()
    old = _make_event(now - 10_000)  # well outside default 3600s window
    recent = _make_event(now - 60)
    status = build_status([old, recent], window_seconds=3600, sparkline_buckets=10)
    assert status["total_events"] == 1


def test_build_status_rollups_per_site():
    now = time.time()
    events = [
        _make_event(
            now - 120,
            subsystem="ego",
            alert="elevated",
            site_events=[
                {
                    "site": "L03.residual",
                    "error": 0.5,
                    "drift": 0.1,
                    "alert_level": "elevated",
                },
                {
                    "site": "L14.residual",
                    "error": 0.1,
                    "drift": 0.0,
                    "alert_level": "normal",
                },
            ],
        ),
        _make_event(
            now - 60,
            subsystem="ego",
            alert="elevated",
            site_events=[
                {
                    "site": "L03.residual",
                    "error": 0.7,
                    "drift": 0.3,
                    "alert_level": "elevated",
                }
            ],
        ),
    ]
    status = build_status(events, window_seconds=3600, sparkline_buckets=6)
    # Worst alert across all events should propagate.
    assert status["worst_alert"] == "elevated"
    # L03.residual sorted first because it has higher severity + max_error.
    names = [r["site"] for r in status["by_site"]]
    assert names[0] == "L03.residual"
    l03 = [r for r in status["by_site"] if r["site"] == "L03.residual"][0]
    assert l03["n_events"] == 2
    assert l03["max_error"] == pytest.approx(0.7)
    assert l03["mean_error"] == pytest.approx(0.6)


def test_build_status_worst_alert_critical_wins():
    now = time.time()
    events = [
        _make_event(now - 30, alert="elevated"),
        _make_event(now - 10, alert="critical"),
    ]
    status = build_status(events, window_seconds=3600, sparkline_buckets=3)
    assert status["worst_alert"] == "critical"


def test_build_status_sparkline_has_requested_bucket_count():
    status = build_status([], window_seconds=120, sparkline_buckets=7)
    assert len(status["sparkline"]) == 7
    # Buckets are oldest first, contiguous, equal width.
    widths = [b["ts_end"] - b["ts_start"] for b in status["sparkline"]]
    assert all(abs(w - widths[0]) < 1e-6 for w in widths)


def test_build_status_sparkline_aggregates_per_bucket():
    now = time.time()
    # Bucket width = 60s / 6 buckets = 10s. Place both events within
    # the final 5s so they're solidly in the tail bucket regardless
    # of millisecond jitter between event placement and rollup time.
    events = [
        _make_event(
            now - 3,
            site_events=[
                {"site": "A", "error": 0.1, "alert_level": "normal", "drift": 0}
            ],
        ),
        _make_event(
            now - 1,
            site_events=[
                {"site": "A", "error": 0.3, "alert_level": "normal", "drift": 0}
            ],
        ),
    ]
    status = build_status(events, window_seconds=60, sparkline_buckets=6)
    # Two events in the same tail bucket; max_error captures the larger.
    nonempty = [b for b in status["sparkline"] if b["n_events"] > 0]
    assert len(nonempty) == 1
    assert nonempty[0]["n_events"] == 2
    assert nonempty[0]["max_error"] == pytest.approx(0.3)
    assert nonempty[0]["mean_error"] == pytest.approx(0.2)


def test_build_status_by_subsystem_shape():
    now = time.time()
    events = [
        _make_event(
            now - 5,
            subsystem="ego",
            alert="elevated",
            site_events=[
                {"site": "x", "error": 0.4, "alert_level": "elevated", "drift": 0}
            ],
        ),
        _make_event(
            now - 3,
            subsystem="superego",
            alert="normal",
            site_events=[
                {"site": "y", "error": 0.1, "alert_level": "normal", "drift": 0}
            ],
        ),
    ]
    status = build_status(events, window_seconds=3600, sparkline_buckets=3)
    subs = {s["subsystem"]: s for s in status["by_subsystem"]}
    assert set(subs) == {"ego", "superego"}
    assert subs["ego"]["worst_alert"] == "elevated"
    assert subs["superego"]["worst_alert"] == "normal"


# ── load_status one-shot --------------------------------------------------


def test_load_status_reads_file_end_to_end(tmp_path):
    p = tmp_path / "events.jsonl"
    now = time.time()
    p.write_text(
        json.dumps(
            {
                "seq": 1,
                "ts": now - 30,
                "subsystem": "ego",
                "alert_level": "normal",
                "sites": [
                    {"site": "s", "error": 0.0, "alert_level": "normal", "drift": 0}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    status = load_status(p, window_seconds=3600, sparkline_buckets=3)
    assert status["total_events"] == 1
    assert status["by_site"][0]["site"] == "s"
