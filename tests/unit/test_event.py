# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""Unit tests for agi.common.event.Event."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from agi.common.event import Event


class TestEventCreate:
    """Tests for Event.create() factory method."""

    def test_create_sets_source_and_type(self) -> None:
        evt = Event.create(source="lh", event_type="lh.request.chat")
        assert evt.source == "lh"
        assert evt.type == "lh.request.chat"

    def test_create_generates_uuid_id(self) -> None:
        evt = Event.create(source="rh", event_type="rh.request.pattern")
        parsed = uuid.UUID(evt.id)
        assert parsed.version == 4

    def test_create_generates_trace_id(self) -> None:
        evt = Event.create(source="meta", event_type="meta.monitor")
        parsed = uuid.UUID(evt.trace_id)
        assert parsed.version == 4

    def test_create_with_explicit_trace_id(self) -> None:
        tid = "custom-trace-123"
        evt = Event.create(source="safety", event_type="safety.check", trace_id=tid)
        assert evt.trace_id == tid

    def test_create_with_payload(self) -> None:
        payload = {"query": "Hello", "temperature": 0.7}
        evt = Event.create(source="lh", event_type="lh.request.chat", payload=payload)
        assert evt.payload == payload

    def test_create_defaults_empty_payload(self) -> None:
        evt = Event.create(source="lh", event_type="lh.request.chat")
        assert evt.payload == {}

    def test_create_timestamp_is_utc(self) -> None:
        evt = Event.create(source="lh", event_type="test")
        assert evt.timestamp.tzinfo is not None


class TestEventToDict:
    """Tests for Event.to_dict() serialisation."""

    def test_to_dict_has_all_fields(self) -> None:
        evt = Event.create(source="safety", event_type="safety.check.input")
        d = evt.to_dict()
        for key in ("id", "timestamp", "source", "type", "payload", "trace_id"):
            assert key in d

    def test_to_dict_timestamp_is_iso_string(self) -> None:
        evt = Event.create(source="lh", event_type="test")
        d = evt.to_dict()
        assert isinstance(d["timestamp"], str)
        datetime.fromisoformat(d["timestamp"])


class TestEventFromDict:
    """Tests for Event.from_dict() deserialisation."""

    def test_round_trip(self) -> None:
        original = Event.create(
            source="memory",
            event_type="memory.store.semantic",
            payload={"text": "hello world", "count": 42},
        )
        d = original.to_dict()
        restored = Event.from_dict(d)
        assert restored.source == original.source
        assert restored.type == original.type
        assert restored.payload == original.payload
        assert restored.id == original.id
        assert restored.trace_id == original.trace_id

    def test_from_dict_drops_unknown_keys(self) -> None:
        d = {
            "id": "abc",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "test",
            "type": "test.event",
            "payload": {},
            "trace_id": "trace-1",
            "stream": "AGI",
            "seq": 42,
            "subject": "agi.test",
        }
        evt = Event.from_dict(d)
        assert evt.source == "test"
        assert evt.id == "abc"

    def test_from_dict_with_string_timestamp(self) -> None:
        ts = "2025-06-15T12:30:00+00:00"
        d = {
            "id": "x",
            "timestamp": ts,
            "source": "test",
            "type": "t",
            "payload": {},
            "trace_id": "tr",
        }
        evt = Event.from_dict(d)
        assert isinstance(evt.timestamp, datetime)
        assert evt.timestamp.year == 2025


class TestEventBytes:
    """Tests for Event.to_bytes() / from_bytes() round-trip."""

    def test_to_bytes_returns_utf8_json(self) -> None:
        evt = Event.create(source="env", event_type="env.sensor.system")
        raw = evt.to_bytes()
        assert isinstance(raw, bytes)
        parsed = json.loads(raw.decode("utf-8"))
        assert parsed["source"] == "env"

    def test_round_trip_bytes(self) -> None:
        original = Event.create(
            source="integration",
            event_type="integration.route",
            payload={"session": "s1", "priority": 1},
        )
        raw = original.to_bytes()
        restored = Event.from_bytes(raw)
        assert restored.source == original.source
        assert restored.type == original.type
        assert restored.payload == original.payload
