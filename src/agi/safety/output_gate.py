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
Output Gate: NATS-connected post-LLM safety filter.

Subscribes to ``agi.safety.check.output`` via NATS, runs the full
DEME pipeline on LLM output, and publishes results:

* ``agi.safety.result.output`` -- check result for the router.
* ``agi.safety.veto`` -- published when output is vetoed (with reason).
* ``agi.safety.audit`` -- full decision proof for compliance logging.

Phase 3 (Safety Gateway) -- Atlas integration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from agi.common.event import Event
from agi.safety.deme_gateway import SafetyGateway, SafetyResult


class OutputGate:
    """NATS-connected output safety gate.

    Receives LLM responses on ``agi.safety.check.output``, runs the
    SafetyGateway's ``check_output`` method (full DEME pipeline when
    available), and publishes the result plus an audit log entry.

    Attributes:
        gateway: The SafetyGateway instance performing checks.
        fabric: NatsEventFabric reference (set by the NATS service).
    """

    SUBSCRIBE_SUBJECT = "agi.safety.check.output"
    RESULT_SUBJECT = "agi.safety.result.output"
    VETO_SUBJECT = "agi.safety.veto"
    AUDIT_SUBJECT = "agi.safety.audit"

    def __init__(self, gateway: SafetyGateway) -> None:
        self._gateway = gateway
        self._fabric: Optional[Any] = None
        self._checks_total = 0
        self._vetoes_total = 0

    def set_fabric(self, fabric: Any) -> None:
        """Set the NATS fabric reference (called by NatsService)."""
        self._fabric = fabric

    @property
    def checks_total(self) -> int:
        """Total number of output checks performed."""
        return self._checks_total

    @property
    def vetoes_total(self) -> int:
        """Total number of output vetoes issued."""
        return self._vetoes_total

    async def handle(self, event: Event) -> None:
        """Handle an incoming ``agi.safety.check.output`` event.

        Expected payload:
            response (str): The LLM-generated response text.
            user_message (str, optional): Original user input.
            session_id (str, optional): Session identifier.
            hemisphere (str, optional): Source hemisphere.
            metadata (dict, optional): Additional context.

        Publishes to:
            agi.safety.result.output: SafetyResult as dict.
            agi.safety.veto: If output is vetoed, includes reason.
            agi.safety.audit: Full decision proof for every check.
        """
        trace_id = event.trace_id
        payload = event.payload

        response = payload.get("response", "")
        user_message = payload.get("user_message", "")
        context: Dict[str, Any] = {
            "session_id": payload.get("session_id", "unknown"),
            "hemisphere": payload.get("hemisphere", "lh"),
        }
        if payload.get("metadata"):
            context.update(payload["metadata"])

        # Run the safety check (full DEME pipeline when available)
        result: SafetyResult = self._gateway.check_output(
            response, user_message, context
        )
        self._checks_total += 1

        if self._fabric:
            # Publish result
            result_event = Event.create(
                source="safety",
                event_type="safety.result.output",
                payload={
                    "passed": result.passed,
                    "score": result.score,
                    "flags": result.flags,
                    "gate": "output",
                    "latency_ms": round(result.latency_ms, 2),
                    "decision_proof": result.decision_proof,
                },
                trace_id=trace_id,
            )
            await self._fabric.publish(self.RESULT_SUBJECT, result_event)

            # If vetoed, publish veto event
            if not result.passed:
                self._vetoes_total += 1
                veto_event = Event.create(
                    source="safety",
                    event_type="safety.veto",
                    payload={
                        "gate": "output",
                        "reason": "; ".join(result.flags) or "below_threshold",
                        "score": result.score,
                        "session_id": context.get("session_id", "unknown"),
                        "content_hash": result.decision_proof.get("content_hash", ""),
                    },
                    trace_id=trace_id,
                )
                await self._fabric.publish(self.VETO_SUBJECT, veto_event)
                logger.warning(
                    "[output-gate] VETO trace=%s flags=%s score=%.2f",
                    trace_id[:8],
                    result.flags,
                    result.score,
                )
            else:
                logger.info(
                    "[output-gate] PASS trace=%s score=%.2f latency=%.1fms",
                    trace_id[:8],
                    result.score,
                    result.latency_ms,
                )

            # Always publish audit log entry
            audit_event = Event.create(
                source="safety",
                event_type="safety.audit",
                payload={
                    "gate": "output",
                    "passed": result.passed,
                    "score": result.score,
                    "flags": result.flags,
                    "latency_ms": round(result.latency_ms, 2),
                    "decision_proof": result.decision_proof,
                    "session_id": context.get("session_id", "unknown"),
                },
                trace_id=trace_id,
            )
            await self._fabric.publish(self.AUDIT_SUBJECT, audit_event)
