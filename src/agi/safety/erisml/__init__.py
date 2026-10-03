# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
ErisML integration module.

Bridges AGI-HPC cognitive architecture to ErisML ethical reasoning framework.
"""

from agi.safety.erisml.service import ErisMLServicer, create_erisml_server
from agi.safety.erisml.facts_builder import PlanStepToEthicalFacts
from agi.safety.erisml.integration import (
    ErisMLIntegration,
    ErisMLConfig,
    IntegratedEvaluation,
    PlanEvaluation,
    SafetyDecision,
    EvaluationSource,
)

# The Hohfeldian structure is erisml-lib's (V4: the correlative s and negation n), not a copy.
from erisml.ethics.hohfeld import (
    HohfeldianState,
    V4Element,
    HohfeldianVerdict,
    compute_bond_index,
    correlative,
    negation,
    v4_multiply,
    v4_inverse,
    v4_apply_to_state,
    v4_between,
    compute_wilson_observable,
)

try:
    from agi.safety.erisml.moral_tensor import (
        MoralTensor,
        SparseCOO,
        MORAL_DIMENSION_NAMES,
        DIMENSION_INDEX,
        DEFAULT_AXIS_NAMES,
    )
except ImportError:
    pass  # numpy not available; tensor features disabled

__all__ = [
    # Service
    "ErisMLServicer",
    "create_erisml_server",
    # Facts builder
    "PlanStepToEthicalFacts",
    # Integration
    "ErisMLIntegration",
    "ErisMLConfig",
    "IntegratedEvaluation",
    "PlanEvaluation",
    "SafetyDecision",
    "EvaluationSource",
    # Hohfeldian V4 gauge structure (erisml-lib)
    "HohfeldianState",
    "V4Element",
    "HohfeldianVerdict",
    "compute_bond_index",
    "correlative",
    "negation",
    "v4_multiply",
    "v4_inverse",
    "v4_apply_to_state",
    "v4_between",
    "compute_wilson_observable",
    # MoralTensor (requires numpy)
    "MoralTensor",
    "SparseCOO",
    "MORAL_DIMENSION_NAMES",
    "DIMENSION_INDEX",
    "DEFAULT_AXIS_NAMES",
]
