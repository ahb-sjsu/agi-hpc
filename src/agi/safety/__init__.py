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
Safety Subsystem for AGI-HPC.

Provides a three-layer safety architecture:

Sprint 1-6 (gRPC-based):
- SafetyGateway: Plan/action checking via gRPC + ErisML.
- ErisMLServicer: gRPC service for ethical evaluation.
- PlanStepToEthicalFacts: Converts plan steps to EthicalFacts.

Phase 3 (NATS-connected, DEME-integrated):
- SafetyAdapter: Converts chat interactions into EthicalFacts.
- DemeSafetyGateway: Three-layer check (reflex/tactical/strategic).
- InputGate: NATS-connected pre-LLM safety filter.
- OutputGate: NATS-connected post-LLM safety filter.
- SafetyNatsService: Main service running both gates.
"""

from __future__ import annotations

__all__ = [
    # Sprint 1-6 (gRPC)
    "SafetyGateway",
    "ErisMLServicer",
    "create_erisml_server",
    "PlanStepToEthicalFacts",
    # Phase 3 (NATS + DEME)
    "SafetyAdapter",
    "DemeSafetyGateway",
    "SafetyResult",
    "GatewayConfig",
    "InputGate",
    "OutputGate",
    "SafetyNatsService",
]

# Sprint 1-6 gRPC components
try:
    from agi.safety.gateway import SafetyGateway
except ImportError:
    pass

try:
    from agi.safety.erisml.service import ErisMLServicer, create_erisml_server
except ImportError:
    pass

try:
    from agi.safety.erisml.facts_builder import PlanStepToEthicalFacts
except ImportError:
    pass

# Phase 3: NATS-connected DEME gateway
try:
    from agi.safety.adapter import SafetyAdapter
except ImportError:
    pass

try:
    from agi.safety.deme_gateway import (
        GatewayConfig,
        SafetyResult,
    )
    from agi.safety.deme_gateway import SafetyGateway as DemeSafetyGateway
except ImportError:
    pass

try:
    from agi.safety.input_gate import InputGate
except ImportError:
    pass

try:
    from agi.safety.output_gate import OutputGate
except ImportError:
    pass

try:
    from agi.safety.nats_service import SafetyNatsService
except ImportError:
    pass

# Sprint 6: Safety Learning
try:
    from agi.safety.learning.service import (
        SafetyLearner,
        SafetyLearnerConfig,
        OutcomeFeedback,
        RuleStats,
    )

    __all__.extend(
        [
            "SafetyLearner",
            "SafetyLearnerConfig",
            "OutcomeFeedback",
            "RuleStats",
        ]
    )
except ImportError:
    pass
