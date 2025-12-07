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
SafetyClient for the Left Hemisphere (LH).

Implements the LH → Safety subsystem interactions described in:

- Safety Architecture (Sections VII.A–C)
- API Interfaces (XIV.B.3)

SafetyClient performs:
    • Pre-action plan safety checks
    • Hallucination detection (schema/API validity)
    • Constraint vetting (tools, params, forbidden operations)

During early development, if the gRPC SafetyService is unavailable,
the client returns a permissive mock result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import grpc

from agi.proto_gen import safety_pb2, safety_pb2_grpc  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    approved: bool
    issues: Optional[list] = None


class SafetyClient:
    """
    Thin wrapper around the SafetyService gRPC API.

    Expected SafetyService RPC (per API spec in whitepaper):
        rpc CheckPlan (PlanGraphProto) returns (SafetyReview)

    Here we pass internal plan_graph as a Python object; you may create
    a serializer later that maps PlanGraph → PlanGraphProto.
    """

    def __init__(self, address: str = "safety:50200") -> None:
        self._address = address
        try:
            self._channel = grpc.insecure_channel(address)
            self._stub = safety_pb2_grpc.SafetyServiceStub(self._channel)
            logger.info("[LH][SafetyClient] Connected to SafetyService at %s", address)
        except Exception:
            self._stub = None
            logger.exception(
                "[LH][SafetyClient] Could not initialize gRPC stub; "
                "falling back to mock responses"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check_plan(self, plan_graph) -> SafetyResult:
        """
        Submit a plan to the safety subsystem for pre-action analysis.

        The architecture requires:
            • schema/tool validity
            • hallucination detection
            • constraint checks (preconditions, objects, tools)
            • risk classification (passed or rejected)
        """
        if self._stub is None:
            logger.warning("[LH][SafetyClient] SafetyService unavailable; allow plan")
            return SafetyResult(approved=True, issues=["mock_safety"])

        try:
            # TODO: implement real serialization to protobuf PlanGraphProto
            req = safety_pb2.PlanGraphProto()  # type: ignore[attr-defined]
            resp = self._stub.CheckPlan(req)

            return SafetyResult(
                approved=getattr(resp, "approved", True),
                issues=list(getattr(resp, "issues", [])),
            )
        except Exception:
            logger.exception("[LH][SafetyClient] SafetyService RPC failed")
            return SafetyResult(
                approved=False,
                issues=["safety_rpc_failure"],
            )
