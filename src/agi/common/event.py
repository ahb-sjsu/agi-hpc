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

"""
Core Event dataclass for AGI-HPC Event Fabric.

All subsystem communication flows through Event instances published
and consumed via the EventFabric NATS backbone.

Subject hierarchy:
    agi.lh.request.{chat,plan,reason}
    agi.rh.request.{pattern,spatial,creative}
    agi.memory.{store,query}.{semantic,episodic,procedural}
    agi.safety.check.{input,output,action}
    agi.meta.monitor.{latency,quality,confidence}
    agi.env.sensor.{system,repos,network}
    agi.integration.{route,merge,session}
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Canonical event envelope for inter-subsystem messaging.

    Attributes:
        id: Unique event identifier (UUID4).
        timestamp: UTC creation time.
        source: Originating subsystem (e.g. ``"lh"``, ``"safety"``).
        type: Dot-delimited event type matching the NATS subject hierarchy.
        payload: Arbitrary JSON-serialisable payload.
        trace_id: Distributed-tracing correlation token.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    def to_bytes(self) -> bytes:
        """Serialise to UTF-8 JSON bytes for NATS publishing."""
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Deserialise from a dictionary (e.g. parsed JSON).

        Unknown keys (e.g. NATS-injected ``stream``, ``seq``) are
        silently dropped so round-trip through JetStream works cleanly.
        """
        data = dict(data)  # shallow copy to avoid mutating caller
        ts_raw = data.get("timestamp")
        if isinstance(ts_raw, str):
            data["timestamp"] = datetime.fromisoformat(ts_raw)
        # Only keep fields that Event.__init__ accepts
        valid_fields = {f.name for f in __import__("dataclasses").fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)

    @classmethod
    def from_bytes(cls, raw: bytes) -> Event:
        """Deserialise from UTF-8 JSON bytes."""
        return cls.from_dict(json.loads(raw.decode("utf-8")))

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        source: str,
        event_type: str,
        payload: Dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Event:
        """Factory method for quick event creation.

        Args:
            source: Originating subsystem name.
            event_type: Dot-delimited type, e.g. ``"lh.request.chat"``.
            payload: Optional payload dict.
            trace_id: Optional trace id; generated if omitted.
        """
        return cls(
            source=source,
            type=event_type,
            payload=payload or {},
            trace_id=trace_id or str(uuid.uuid4()),
        )
