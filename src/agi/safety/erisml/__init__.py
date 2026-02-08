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

__all__ = [
    "ErisMLServicer",
    "create_erisml_server",
    "PlanStepToEthicalFacts",
]
