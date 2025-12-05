"""
In-Action Safety Service.

Responsible for:
- Monitoring short-horizon predictions and runtime signals.
- Deciding whether to allow, modify, or veto control actions.
- Publishing interventions as events.

gRPC wiring assumes something like:
    service InActionSafety {
        rpc CheckStep(InActionRequest) returns (InActionResponse);
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


class InActionSafetyService:
    def __init__(self, config_path: str = "configs/safety_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.engine = SafetyRuleEngine(self.config)
        # Use a different port offset if you want separate processes.
        self.grpc = GRPCServer(self.config.rpc_port + 10)

    def _step_to_summary(self, request) -> Dict[str, Any]:
        # TODO: map fields from InActionRequest into dict
        return {
            "predicted_collision": getattr(request, "predicted_collision", False),
            "joint_limit_violations": getattr(request, "joint_limit_violations", 0),
        }

    def handle_step(self, request) -> "safety_pb2.InActionResponse":
        summary = self._step_to_summary(request)
        verdict: SafetyVerdict = self.engine.check_step(summary)

        self.fabric.publish(
            "safety.in_action.check",
            {
                "decision": verdict.decision,
                "risk_score": verdict.risk_score,
                "reasons": verdict.reasons,
            },
        )

        resp = safety_pb2.InActionResponse()  # type: ignore
        resp.decision = verdict.decision
        resp.risk_score = verdict.risk_score
        resp.reasons.extend(verdict.reasons)
        return resp

    def run(self):
        if not HAVE_PROTO:
            print("[SAFETY-IN] WARNING: safety_pb2_grpc not available. gRPC not wired.")
            self.grpc.start()
            self.grpc.wait()
            return

        class _Servicer(safety_pb2_grpc.InActionSafetyServicer):  # type: ignore
            def __init__(self, outer: "InActionSafetyService"):
                self._outer = outer

            def CheckStep(self, request, context):  # type: ignore
                return self._outer.handle_step(request)

        serv = _Servicer(self)
        safety_pb2_grpc.add_InActionSafetyServicer_to_server(serv, self.grpc.server)  # type: ignore

        print("[SAFETY-IN] In-Action Safety service running...")
        self.grpc.start()
        self.grpc.wait()


def main():
    InActionSafetyService().run()


if __name__ == "__main__":
    main()
