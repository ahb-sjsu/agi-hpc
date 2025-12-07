"""
MetacognitionClient for the Left Hemisphere.

Implements the metacognitive pipeline described in:

- Metacognition Layer (Sections VIII.A–E)
- Internal APIs (XIV.B.3)

MetacognitionClient performs:
    • Review of plan structure and reasoning
    • Cross-check consistency with memory and RH simulation
    • Produce a decision: ACCEPT / REVISE / REJECT
    • Optional revised plan request

This stub provides minimal scaffolding for integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import grpc

from agi.proto_gen import meta_pb2, meta_pb2_grpc  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@dataclass
class MetaReviewResult:
    decision: str  # "ACCEPT" | "REVISE" | "REJECT"
    issues: Optional[list] = None
    confidence: float = 1.0


class MetacognitionClient:
    """
    LH → Metacognition Service client.

    Expected service RPCs:

        rpc ReviewPlan(PlanGraphProto) returns (MetaReview)
        rpc RevisePlan(ReviseRequest) returns (RevisedPlan)

    The stub mirrors this API but operates on internal PlanGraph
    objects for now, until PlanGraphProto is defined.
    """

    def __init__(self, address: str = "meta:50300") -> None:
        self._address = address
        try:
            self._channel = grpc.insecure_channel(address)
            self._stub = meta_pb2_grpc.MetacognitionServiceStub(self._channel)
            logger.info("[LH][MetacogClient] Connected to MetaService at %s", address)
        except Exception:
            self._stub = None
            logger.exception("[LH][MetacogClient] Metacognition unavailable; using mock behavior")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def review_plan(self, plan_graph) -> MetaReviewResult:
        """
        Run metacognitive review on a draft plan.

        The architecture requires:
            • reasoning quality check
            • evidence consistency (memory + RH)
            • confidence estimation
        """
        if self._stub is None:
            logger.debug("[LH][MetacogClient] Returning ACCEPT (mock)")
            return MetaReviewResult(decision="ACCEPT", issues=[], confidence=1.0)

        try:
            # TODO: serialize plan_graph → MetaReviewRequest
            req = meta_pb2.MetaReviewRequest()  # type: ignore[attr-defined]
            resp = self._stub.ReviewPlan(req)
            logger.debug(f"[LH][MetacogClient] revise_plan successful: {resp}")
            return MetaReviewResult(
                decision=getattr(resp, "decision", "ACCEPT"),
                issues=list(getattr(resp, "issues", [])),
                confidence=getattr(resp, "confidence", 0.5),
            )
        except Exception:
            logger.exception("[LH][MetacogClient] review_plan RPC failed")
            return MetaReviewResult(
                decision="ACCEPT",
                issues=["meta_rpc_failure"],
                confidence=0.5,
            )

    def revise_plan(self, plan_graph, review: MetaReviewResult):
        """
        Optionally produce a revised plan based on review feedback.

        Implementation note:
            When PlanGraphProto is defined, this will send a ReviseRequest.
        """
        if self._stub is None:
            logger.debug("[LH][MetacogClient] revise_plan (mock): returning original plan")
            return plan_graph

        try:
            req = meta_pb2.ReviseRequest()  # type: ignore[attr-defined]
            resp = self._stub.RevisePlan(req)
            # TODO: deserialize RevisedPlan into PlanGraph
            logger.debug("[LH][MetacogClient] revise_plan successful (stub)")
            return plan_graph
        except Exception:
            logger.exception("[LH][MetacogClient] revise_plan RPC failed")
            return plan_graph
