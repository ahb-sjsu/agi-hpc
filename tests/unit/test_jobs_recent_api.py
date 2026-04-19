"""Test the /api/jobs/recent lifecycle-tail endpoint on telemetry_server.

Covers the inline-fallback path (no agi package) + correct ordering
(newest first). The happy path (with agi.common.structured_log
importable) is already covered by test_structured_log.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_tsrv():
    path = Path(__file__).resolve().parents[2] / "scripts" / "telemetry_server.py"
    spec = importlib.util.spec_from_file_location("tsrv", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tsrv"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_lifecycle_recent_returns_empty_for_missing_file(tmp_path, monkeypatch):
    tsrv = _load_tsrv()
    monkeypatch.setattr(tsrv, "LIFECYCLE_DIR", tmp_path)
    out = tsrv._get_lifecycle_recent("nonexistent")
    assert out == {"events": []}


def test_lifecycle_recent_returns_events_newest_first(tmp_path, monkeypatch):
    tsrv = _load_tsrv()
    monkeypatch.setattr(tsrv, "LIFECYCLE_DIR", tmp_path)
    # Simulate what structured_log.LifecycleLogger writes
    jsonl = tmp_path / "scientist.jsonl"
    lines = [
        json.dumps(
            {
                "ts": "2026-04-19T10:00:00Z",
                "subsystem": "scientist",
                "event": "cycle_start",
                "seq": 0,
            }
        ),
        json.dumps(
            {
                "ts": "2026-04-19T10:00:01Z",
                "subsystem": "scientist",
                "event": "attempt_end",
                "seq": 1,
                "task": 20,
                "outcome": "pass",
            }
        ),
        json.dumps(
            {
                "ts": "2026-04-19T10:00:02Z",
                "subsystem": "scientist",
                "event": "attempt_end",
                "seq": 2,
                "task": 21,
                "outcome": "fail",
            }
        ),
    ]
    jsonl.write_text("\n".join(lines) + "\n")
    out = tsrv._get_lifecycle_recent("scientist", limit=10)
    assert len(out["events"]) == 3
    assert out["events"][0]["seq"] == 2  # newest first
    assert out["events"][-1]["seq"] == 0


def test_lifecycle_recent_respects_limit(tmp_path, monkeypatch):
    tsrv = _load_tsrv()
    monkeypatch.setattr(tsrv, "LIFECYCLE_DIR", tmp_path)
    jsonl = tmp_path / "primer.jsonl"
    lines = [
        json.dumps(
            {
                "ts": f"2026-04-19T10:{i:02d}:00Z",
                "subsystem": "primer",
                "event": "tick_start",
                "seq": i,
            }
        )
        for i in range(10)
    ]
    jsonl.write_text("\n".join(lines) + "\n")
    out = tsrv._get_lifecycle_recent("primer", limit=3)
    assert len(out["events"]) == 3
    assert out["events"][0]["seq"] == 9
    assert out["events"][-1]["seq"] == 7


def test_lifecycle_recent_handles_malformed_lines(tmp_path, monkeypatch):
    tsrv = _load_tsrv()
    monkeypatch.setattr(tsrv, "LIFECYCLE_DIR", tmp_path)
    jsonl = tmp_path / "scientist.jsonl"
    jsonl.write_text(
        "this is not JSON\n"
        + json.dumps(
            {
                "ts": "2026-04-19T10:00:00Z",
                "subsystem": "scientist",
                "event": "x",
                "seq": 1,
            }
        )
        + "\nanother non-json line\n"
    )
    out = tsrv._get_lifecycle_recent("scientist", limit=10)
    # Only the one valid line is returned; malformed are silently skipped
    assert len(out["events"]) == 1
    assert out["events"][0]["seq"] == 1
