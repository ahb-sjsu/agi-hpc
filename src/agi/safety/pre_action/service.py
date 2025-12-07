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
Pre-Action Safety Service.

Responsible for:
- Checking candidate plans/code/tool calls *before* execution.
- Returning ALLOW / BLOCK / REVISE + reasons.
- Emitting events for logging and training.

gRPC wiring assumes your safety.proto defines something like:
    service PreActionSafety {
        rpc CheckPlan(PreActionRequest) returns (PreActionResponse);
    }

If your names differ, adjust the imports and servicer base class below.
"""

from typing import Dict, Any

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer
from agi.safety.rules.engine import SafetyRuleEngine, SafetyVerdict

# Try to import actual gRPC definitions, but remain import-safe if names differ.
try:
    import agi.proto_gen.safety_pb2 as safety_pb2
    import agi.proto_gen.safety_pb2_grpc as safety_pb2_grpc

    HAVE_PROTO = True
except ImportError:  # pragma: no cover
    safety_pb2 = None
    safety_pb2_grpc = None
    HAVE_PROTO = False


class PreActionSafetyService:
    def __init__(self, config_path: str = "configs/safety_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.engine = SafetyRuleEngine(self.config)
        self.grpc = GRPCServer(self.config.rpc_port)

    # Example helper: convert from proto to dict-based summary.
    def _plan_to_summary(self, request) -> Dict[str, Any]:
        # TODO: map fields from your PreActionRequest proto into a dict
        # For now, just a stub structure:
        return {
            "tools": list(getattr(request, "tools", [])),
            "description": getattr(request, "description", ""),
        }

    def handle_plan(self, request) -> "safety_pb2.PreActionResponse":
        summary = self._plan_to_summary(request)
        verdict: SafetyVerdict = self.engine.check_plan(summary)

        # Emit event
        self.fabric.publish(
            "safety.pre_action.check",
            {
                "decision": verdict.decision,
                "risk_score": verdict.risk_score,
                "reasons": verdict.reasons,
            },
        )

        # Map back into proto
        resp = safety_pb2.PreActionResponse()  # type: ignore
        resp.decision = verdict.decision
        resp.risk_score = verdict.risk_score
        resp.reasons.extend(verdict.reasons)
        return resp

    def run(self):
        if not HAVE_PROTO:
            print(
                "[SAFETY-PRE] WARNING: safety_pb2_grpc not available. gRPC not wired."
            )
            self.grpc.start()
            self.grpc.wait()
            return

        # Define Servicer dynamically to avoid hard-coding the proto class names
        class _Servicer(safety_pb2_grpc.PreActionSafetyServicer):  # type: ignore
            def __init__(self, outer: "PreActionSafetyService"):
                self._outer = outer

            def CheckPlan(self, request, context):  # type: ignore
                return self._outer.handle_plan(request)

        serv = _Servicer(self)
        safety_pb2_grpc.add_PreActionSafetyServicer_to_server(serv, self.grpc.server)  # type: ignore

        print("[SAFETY-PRE] Pre-Action Safety service running...")
        self.grpc.start()
        self.grpc.wait()


def main():
    PreActionSafetyService().run()


if __name__ == "__main__":
    main()
