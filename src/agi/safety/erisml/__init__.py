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
from agi.safety.erisml.hohfeld import (
    HohfeldianState,
    D4Element,
    HohfeldianVerdict,
    compute_bond_index,
    correlative,
    negation,
    d4_multiply,
    d4_inverse,
    d4_apply_to_state,
    compute_wilson_observable,
    get_klein_four_subgroup,
    is_in_klein_four,
    requires_nonabelian_structure,
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
    # Hohfeldian D4 gauge structure
    "HohfeldianState",
    "D4Element",
    "HohfeldianVerdict",
    "compute_bond_index",
    "correlative",
    "negation",
    "d4_multiply",
    "d4_inverse",
    "d4_apply_to_state",
    "compute_wilson_observable",
    "get_klein_four_subgroup",
    "is_in_klein_four",
    "requires_nonabelian_structure",
    # MoralTensor (requires numpy)
    "MoralTensor",
    "SparseCOO",
    "MORAL_DIMENSION_NAMES",
    "DIMENSION_INDEX",
    "DEFAULT_AXIS_NAMES",
]
