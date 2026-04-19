"""Test the Erebus trends endpoint payload shape."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


def _load_module():
    """Import telemetry_server.py directly by path so we can test the
    pure-function parts without starting the HTTP server."""
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "telemetry_server.py"
    )
    spec = importlib.util.spec_from_file_location("tsrv", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tsrv"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_trends_returns_30_day_window(tmp_path, monkeypatch):
    """by_day must always contain exactly 30 entries, last element = today."""
    tsrv = _load_module()
    tsrv._trends_cache["ts"] = 0.0  # bust cache between tests

    mem = {
        "total_attempts": 5,
        "total_solves": 2,
        "tasks": {
            "1": {
                "solved": True,
                "attempts": [
                    {
                        "timestamp": "2026-04-18T10:00:00",
                        "verified": True,
                        "correct": 1,
                        "total": 1,
                    }
                ],
            },
            "2": {
                "solved": False,
                "attempts": [
                    {
                        "timestamp": "2026-04-18T10:01:00",
                        "verified": False,
                    }
                ],
            },
        },
    }
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(json.dumps(mem))
    monkeypatch.setattr(tsrv, "EREBUS_MEMORY_PATH", str(mem_file))

    trends = tsrv._get_erebus_trends()
    assert trends["window_days"] == 30
    assert len(trends["by_day"]) == 30
    # Today should be the last entry
    today = datetime.now(timezone.utc).date().isoformat()
    assert trends["by_day"][-1]["date"] == today
    # Days sorted ascending
    dates = [e["date"] for e in trends["by_day"]]
    assert dates == sorted(dates)


def test_trends_bins_solves_and_attempts_correctly(tmp_path, monkeypatch):
    tsrv = _load_module()
    tsrv._trends_cache["ts"] = 0.0

    today = datetime.now(timezone.utc).date().isoformat()
    mem = {
        "total_attempts": 3,
        "total_solves": 2,
        "tasks": {
            "5": {
                "attempts": [
                    {"timestamp": today + "T09:00:00", "verified": True},
                    {"timestamp": today + "T10:00:00", "verified": True},
                    {"timestamp": today + "T11:00:00", "verified": False},
                ]
            }
        },
    }
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(json.dumps(mem))
    monkeypatch.setattr(tsrv, "EREBUS_MEMORY_PATH", str(mem_file))

    trends = tsrv._get_erebus_trends()
    last = trends["by_day"][-1]
    assert last["date"] == today
    assert last["solves"] == 2
    assert last["attempts"] == 3


def test_trends_missing_memory_returns_empty_window(tmp_path, monkeypatch):
    tsrv = _load_module()
    tsrv._trends_cache["ts"] = 0.0
    monkeypatch.setattr(tsrv, "EREBUS_MEMORY_PATH", str(tmp_path / "nope.json"))
    trends = tsrv._get_erebus_trends()
    assert trends["by_day"] == []
    assert trends["by_week"] == []
    assert trends["total_solves"] == 0


def test_trends_cache_reuses_recent_result(tmp_path, monkeypatch):
    tsrv = _load_module()
    # Prime the cache with a sentinel
    import time as _t

    tsrv._trends_cache["ts"] = _t.time()
    tsrv._trends_cache["data"] = {"cached": "yes"}
    monkeypatch.setattr(tsrv, "EREBUS_MEMORY_PATH", str(tmp_path / "nope.json"))
    out = tsrv._get_erebus_trends()
    assert out == {"cached": "yes"}
