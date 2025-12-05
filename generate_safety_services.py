#!/usr/bin/env python3
"""
generate_safety_services.py

Creates safety subsystem skeletons:

- src/agi/safety/rules/engine.py
- src/agi/safety/pre_action/service.py
- src/agi/safety/in_action/service.py
- src/agi/safety/post_action/service.py

Each service:
- uses config_loader
- uses EventFabric
- uses GRPCServer
- imports safety.proto stubs (with a safe fallback if names differ)

Run from repo root:
    python generate_safety_services.py
"""

from pathlib import Path
from textwrap import dedent

def write(path: Path, content: str, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"[skip] {path}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"[write] {path}")


# =====================================================================
# RULE ENGINE
# =====================================================================

RULE_ENGINE = dedent(r'''
"""
Safety Rule Engine for AGI-HPC.

This is the central place where safety policies are:
- loaded from config / rule files
- applied to plans, runtime signals, and outcomes
- produce structured verdicts: allow / block / revise, with reasons and risk scores
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SafetyVerdict:
    """Result of a safety check."""
    decision: str          # "ALLOW", "BLOCK", "REVISE"
    risk_score: float      # 0.0 (no risk) → 1.0 (max risk)
    reasons: List[str]     # human-readable explanations

class SafetyRuleEngine:
    def __init__(self, config: Any):
        """
        config: ServiceConfig from config_loader.load_config
        """
        self.config = config
        self.rules_config: Dict[str, Any] = config.extra.get("safety_rules", {})
        # TODO: Load additional rule files, model weights, etc.

    # ------------------------------------------------------------------
    # Pre-Action checks (plans, code, tool usage)
    # ------------------------------------------------------------------
    def check_plan(self, plan_summary: Dict[str, Any]) -> SafetyVerdict:
        """
        plan_summary: dict with high-level info about the proposed plan/code.
        Example fields (you can standardize later):
            - tools: List[str]
            - resources: List[str]
            - description: str
            - estimated_impact: str
        """
        reasons = []
        risk = 0.0

        tools = plan_summary.get("tools", [])
        banned_tools = self.rules_config.get("banned_tools", [])

        for t in tools:
            if t in banned_tools:
                reasons.append(f"Tool '{t}' is banned by policy.")
                risk = max(risk, 1.0)

        if not reasons:
            decision = "ALLOW"
        else:
            decision = "BLOCK"

        return SafetyVerdict(decision=decision, risk_score=risk, reasons=reasons)

    # ------------------------------------------------------------------
    # In-Action checks (runtime state, control commands)
    # ------------------------------------------------------------------
    def check_step(self, step_summary: Dict[str, Any]) -> SafetyVerdict:
        """
        step_summary: dict describing one control step or short horizon.
        Example fields:
            - predicted_collision: bool
            - min_distance: float
            - joint_limit_violations: int
        """
        reasons = []
        risk = 0.0

        if step_summary.get("predicted_collision", False):
            reasons.append("Predicted collision in next horizon.")
            risk = max(risk, 0.9)

        if step_summary.get("joint_limit_violations", 0) > 0:
            reasons.append("Joint limits violated in simulation.")
            risk = max(risk, 0.8)

        decision = "ALLOW" if risk < 0.5 else "BLOCK"
        return SafetyVerdict(decision=decision, risk_score=risk, reasons=reasons)

    # ------------------------------------------------------------------
    # Post-Action checks (incident analysis)
    # ------------------------------------------------------------------
    def assess_outcome(self, outcome_summary: Dict[str, Any]) -> SafetyVerdict:
        """
        outcome_summary: high-level result of an episode or action.
        Example fields:
            - incident: bool
            - near_miss: bool
            - unmodeled_effects: bool
        """
        reasons = []
        risk = 0.0

        if outcome_summary.get("incident", False):
            reasons.append("Actual safety incident occurred.")
            risk = max(risk, 1.0)

        if outcome_summary.get("near_miss", False):
            reasons.append("Near miss detected.")
            risk = max(risk, 0.7)

        if outcome_summary.get("unmodeled_effects", False):
            reasons.append("Unmodeled effects observed (model mismatch).")
            risk = max(risk, 0.8)

        decision = "ALLOW" if risk < 0.4 else "REVISE"
        if risk >= 0.9:
            decision = "BLOCK"

        return SafetyVerdict(decision=decision, risk_score=risk, reasons=reasons)
''')


# =====================================================================
# PRE-ACTION SAFETY SERVICE
# =====================================================================

PRE_ACTION_SERVICE = dedent(r'''
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
        self.fabric.publish("safety.pre_action.check", {
            "decision": verdict.decision,
            "risk_score": verdict.risk_score,
            "reasons": verdict.reasons,
        })

        # Map back into proto
        resp = safety_pb2.PreActionResponse()  # type: ignore
        resp.decision = verdict.decision
        resp.risk_score = verdict.risk_score
        resp.reasons.extend(verdict.reasons)
        return resp

    def run(self):
        if not HAVE_PROTO:
            print("[SAFETY-PRE] WARNING: safety_pb2_grpc not available. gRPC not wired.")
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
''')


# =====================================================================
# IN-ACTION SAFETY SERVICE
# =====================================================================

IN_ACTION_SERVICE = dedent(r'''
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

        self.fabric.publish("safety.in_action.check", {
            "decision": verdict.decision,
            "risk_score": verdict.risk_score,
            "reasons": verdict.reasons,
        })

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
''')


# =====================================================================
# POST-ACTION SAFETY SERVICE
# =====================================================================

POST_ACTION_SERVICE = dedent(r'''
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
''')


# =====================================================================
# MAIN: write files
# =====================================================================

def main():
    root = Path("src/agi/safety")

    write(root / "rules" / "engine.py", RULE_ENGINE)
    write(root / "pre_action" / "service.py", PRE_ACTION_SERVICE)
    write(root / "in_action" / "service.py", IN_ACTION_SERVICE)
    write(root / "post_action" / "service.py", POST_ACTION_SERVICE)

    print("\nSafety subsystem skeleton generated.")

if __name__ == "__main__":
    main()
