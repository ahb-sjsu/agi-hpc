# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Safety subsystem for AGI-HPC.

Provides three-layer safety architecture:
- Reflex Layer (<100μs): Hardware-level emergency stops
- Tactical Layer (10-100ms): ErisML ethical evaluation
- Strategic Layer: Policy-level governance

Integration with ErisML provides:
- Formal ethical reasoning via DEME pipeline
- Bond Index verification for correlative symmetry
- Hohfeldian consistency checking
- Hash-chained decision proofs for audit
"""

from agi.safety.erisml.service import ErisMLServicer, create_erisml_server
from agi.safety.erisml.facts_builder import PlanStepToEthicalFacts
from agi.safety.gateway import SafetyGateway

__all__ = [
    "ErisMLServicer",
    "create_erisml_server",
    "PlanStepToEthicalFacts",
    "SafetyGateway",
]
