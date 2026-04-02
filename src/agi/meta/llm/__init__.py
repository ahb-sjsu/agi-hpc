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
LLM integration module for AGI-HPC cognitive architecture.

Provides async client wrappers, inference configuration presets,
and prompt template management for left/right hemisphere services.
"""

from __future__ import annotations

__all__ = [
    "LLMClient",
    "CompletionResult",
    "InferenceConfig",
    "LH_PRESET",
    "RH_PRESET",
    "PromptTemplateRegistry",
]

try:
    from agi.meta.llm.client import LLMClient, CompletionResult
    from agi.meta.llm.config import InferenceConfig, LH_PRESET, RH_PRESET
    from agi.meta.llm.templates import PromptTemplateRegistry
except ImportError:
    pass
