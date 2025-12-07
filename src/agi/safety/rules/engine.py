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

    decision: str  # "ALLOW", "BLOCK", "REVISE"
    risk_score: float  # 0.0 (no risk) → 1.0 (max risk)
    reasons: List[str]  # human-readable explanations


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
