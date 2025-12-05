"""
Metacognition Service for AGI-HPC.

Responsibilities:
- Evaluate PlanGraph + ReasoningTrace + Simulation summaries
- Compute confidence and detect inconsistencies
- Decide: ACCEPT / REVISE / REJECT
- Provide explanations
- Publishes optional meta events back to LH/RH via EventFabric

This is only the skeleton; full logic gets added later.
"""

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

# Proto-generated imports
# These stubs must exist from generate_protos.py.
from agi.proto_gen.meta_pb2 import MetaReviewResponse, MetaReviewRequest
from agi.proto_gen.meta_pb2_grpc import (
    MetacognitionServiceServicer,
    add_MetacognitionServiceServicer_to_server,
)


class MetacognitionEngine:
    """
    Placeholder meta-engine.

    This component analyzes reasoning traces, plan graphs,
    simulation summaries, memory hits, and computes:
    - confidence   [0.0 → 1.0]
    - issues       [list of strings]
    - decision     {ACCEPT, REVISE, REJECT}
    """

    def evaluate(self, request: MetaReviewRequest):
        issues = []
        confidence = 0.8  # Placeholder baseline

        # ---------------------------------------------------------
        # Inspect reasoning trace
        # ---------------------------------------------------------
        if request.reasoning_trace.steps:
            if "TODO" in request.reasoning_trace.steps[0]:
                issues.append("Reasoning trace contains placeholder TODO")
                confidence -= 0.3

        # ---------------------------------------------------------
        # Inspect plan graph (placeholder)
        # ---------------------------------------------------------
        if len(request.plan.steps) == 0:
            issues.append("Empty plan graph")
            confidence = 0.0

        # ---------------------------------------------------------
        # Simple decision rule for now
        # ---------------------------------------------------------
        if confidence > 0.7 and not issues:
            decision = MetaReviewResponse.ACCEPT
        elif confidence > 0.4:
            decision = MetaReviewResponse.REVISE
        else:
            decision = MetaReviewResponse.REJECT

        return confidence, issues, decision


class MetacognitionServicer(MetacognitionServiceServicer):

    def __init__(self, engine: MetacognitionEngine, fabric: EventFabric):
        self.engine = engine
        self.fabric = fabric

    def ReviewPlan(self, request, context):
        confidence, issues, decision = self.engine.evaluate(request)

        # Publish meta event
        self.fabric.publish(
            "meta.review",
            {
                "confidence": confidence,
                "issues": list(issues),
                "decision": int(decision),
            },
        )

        return MetaReviewResponse(
            confidence=confidence, issues=issues, decision=decision
        )


class MetaService:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.engine = MetacognitionEngine()
        self.grpc_server = GRPCServer(self.config.rpc_port)

    def run(self):
        servicer = MetacognitionServicer(self.engine, self.fabric)
        add_MetacognitionServiceServicer_to_server(servicer, self.grpc_server.server)

        print("[META] Metacognition service running...")
        self.grpc_server.start()
        self.grpc_server.wait()


def main():
    MetaService("configs/meta_config.yaml").run()


if __name__ == "__main__":
    main()
