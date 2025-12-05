
"""
Post-Action Safety Service.

Responsibilities:
- Analyze completed episodes or actions.
- Classify incidents vs near misses vs benign outcomes.
- Detect unmodeled effects.
- Emit structured safety records for learning.

gRPC assumption:
    service PostActionSafety {
        rpc AnalyzeOutcome(PostActionRequest) returns (PostActionResponse);
    }
"""

from typing import Dict, Any

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer
from agi.safety.rules.engine import SafetyRuleEngine, SafetyVerdict

try:
    import agi.proto_gen.safety_pb2 as safety_pb2
    import agi.proto_gen.safety_pb2_grpc as safety_pb2_grpc
    HAVE_PROTO = True
except ImportError:  # pragma: no cover
    safety_pb2 = None
    safety_pb2_grpc = None
    HAVE_PROTO = False


class PostActionSafetyService:
    def __init__(self, config_path: str = "configs/safety_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.engine = SafetyRuleEngine(self.config)
        self.grpc = GRPCServer(self.config.rpc_port + 20)

    def _outcome_to_summary(self, request) -> Dict[str, Any]:
        # TODO: map fields from PostActionRequest prost into dict
        return {
            "incident": getattr(request, "incident", False),
            "near_miss": getattr(request, "near_miss", False),
            "unmodeled_effects": getattr(request, "unmodeled_effects", False),
        }

    def handle_outcome(self, request) -> "safety_pb2.PostActionResponse":
        summary = self._outcome_to_summary(request)
        verdict: SafetyVerdict = self.engine.assess_outcome(summary)

        self.fabric.publish("safety.post_action.analyze", {
            "decision": verdict.decision,
            "risk_score": verdict.risk_score,
            "reasons": verdict.reasons,
        })

        resp = safety_pb2.PostActionResponse()  # type: ignore
        resp.decision = verdict.decision
        resp.risk_score = verdict.risk_score
        resp.reasons.extend(verdict.reasons)
        return resp

    def run(self):
        if not HAVE_PROTO:
            print("[SAFETY-POST] WARNING: safety_pb2_grpc not available. gRPC not wired.")
            self.grpc.start()
            self.grpc.wait()
            return

        class _Servicer(safety_pb2_grpc.PostActionSafetyServicer):  # type: ignore
            def __init__(self, outer: "PostActionSafetyService"):
                self._outer = outer

            def AnalyzeOutcome(self, request, context):  # type: ignore
                return self._outer.handle_outcome(request)

        serv = _Servicer(self)
        safety_pb2_grpc.add_PostActionSafetyServicer_to_server(serv, self.grpc.server)  # type: ignore

        print("[SAFETY-POST] Post-Action Safety service running...")
        self.grpc.start()
        self.grpc.wait()


def main():
    PostActionSafetyService().run()

if __name__ == "__main__":
    main()
