"""Unit tests for agi.common.structured_log.LifecycleLogger."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from agi.common.structured_log import LifecycleLogger, read_recent


def test_emit_writes_one_jsonline_per_event(tmp_path):
    log = LifecycleLogger("scientist", lifecycle_dir=tmp_path)
    log.emit("cycle_start", cycle=1, total_cycles=5)
    log.emit("attempt_start", task=20, strategy="direct", model="kimi")
    contents = (tmp_path / "scientist.jsonl").read_text().strip().split("\n")
    assert len(contents) == 2
    first = json.loads(contents[0])
    assert first["subsystem"] == "scientist"
    assert first["event"] == "cycle_start"
    assert first["cycle"] == 1
    assert first["total_cycles"] == 5
    assert first["seq"] == 0
    second = json.loads(contents[1])
    assert second["seq"] == 1
    assert second["task"] == 20


def test_emit_timestamp_is_iso_utc_z(tmp_path):
    log = LifecycleLogger("scientist", lifecycle_dir=tmp_path)
    log.emit("ping")
    contents = (tmp_path / "scientist.jsonl").read_text().strip()
    rec = json.loads(contents)
    # 2026-04-19T18:00:00.123456Z
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$", rec["ts"])


def test_seq_monotonic_across_instance(tmp_path):
    log = LifecycleLogger("primer", lifecycle_dir=tmp_path)
    for _ in range(5):
        log.emit("tick_start")
    # New instance picks up where the last left off
    log2 = LifecycleLogger("primer", lifecycle_dir=tmp_path)
    log2.emit("tick_start")
    last = json.loads((tmp_path / "primer.jsonl").read_text().strip().split("\n")[-1])
    assert last["seq"] == 5  # 0..4 then 5


def test_non_jsonsafe_fields_are_coerced(tmp_path):
    log = LifecycleLogger("scientist", lifecycle_dir=tmp_path)
    log.emit(
        "oddball",
        a_tuple=(1, 2),
        a_path=Path("/archive/neurogolf"),
        a_set={1, 2, 3},  # not json-safe; should be stringified
    )
    rec = json.loads((tmp_path / "scientist.jsonl").read_text().strip())
    assert rec["a_tuple"] == [1, 2]
    assert rec["a_path"] == "/archive/neurogolf" or rec["a_path"] == str(
        Path("/archive/neurogolf")
    )
    # a_set ends up as a repr string (order may vary, so just check stringiness)
    assert isinstance(rec["a_set"], str)


def test_read_recent_returns_newest_first(tmp_path):
    log = LifecycleLogger("scientist", lifecycle_dir=tmp_path)
    for i in range(10):
        log.emit("attempt_end", task=i, outcome="fail")
    events = read_recent("scientist", limit=5, lifecycle_dir=tmp_path)
    assert len(events) == 5
    # newest first
    assert events[0]["task"] == 9
    assert events[-1]["task"] == 5


def test_read_recent_handles_missing_file(tmp_path):
    events = read_recent("never_written", limit=5, lifecycle_dir=tmp_path)
    assert events == []


def test_read_recent_respects_limit_above_file_size(tmp_path):
    log = LifecycleLogger("scientist", lifecycle_dir=tmp_path)
    log.emit("attempt_end", task=1)
    log.emit("attempt_end", task=2)
    events = read_recent("scientist", limit=100, lifecycle_dir=tmp_path)
    assert len(events) == 2
    assert events[0]["task"] == 2


def test_explicit_path_overrides_subsystem_default(tmp_path):
    explicit = tmp_path / "weird_name.jsonl"
    log = LifecycleLogger("scientist", path=explicit)
    log.emit("ping")
    assert explicit.exists()
    assert not (tmp_path / "scientist.jsonl").exists()


def test_emit_returns_event_with_populated_seq(tmp_path):
    log = LifecycleLogger("p", lifecycle_dir=tmp_path)
    ev = log.emit("ping", foo="bar")
    assert ev.seq == 0
    assert ev.event == "ping"
    assert ev.subsystem == "p"
    assert ev.fields["foo"] == "bar"


def test_thread_safety_basic(tmp_path):
    """Emit from multiple threads; all events land and seq is unique."""
    import threading

    log = LifecycleLogger("p", lifecycle_dir=tmp_path)

    def worker(start):
        for i in range(20):
            log.emit("work", n=start * 100 + i)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = read_recent("p", limit=200, lifecycle_dir=tmp_path)
    assert len(events) == 80
    seqs = [e["seq"] for e in events]
    assert len(set(seqs)) == 80  # all unique
    assert sorted(seqs) == list(range(80))
