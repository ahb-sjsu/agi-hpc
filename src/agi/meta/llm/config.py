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
Inference configuration for AGI-HPC LLM services.

Provides dataclass-based presets for Left Hemisphere (precise,
analytical) and Right Hemisphere (creative, divergent) inference
parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for a single LLM inference call.

    Attributes:
        temperature: Sampling temperature (0.0 = deterministic, 1.0+ = creative).
        top_p: Nucleus sampling cutoff.
        max_tokens: Maximum tokens to generate.
        system_prompt: System-level instructions prepended to conversation.
        stop_sequences: Sequences that halt generation.
        model: Override model name (uses endpoint default if empty).
        presence_penalty: Penalise tokens already present in context.
        frequency_penalty: Penalise tokens by frequency.
    """

    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    system_prompt: str = ""
    stop_sequences: List[str] = field(default_factory=list)
    model: str = ""
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def to_api_params(self) -> dict:
        """Convert to OpenAI-compatible API parameters."""
        params: dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.stop_sequences:
            params["stop"] = self.stop_sequences
        if self.model:
            params["model"] = self.model
        if self.presence_penalty:
            params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty:
            params["frequency_penalty"] = self.frequency_penalty
        return params


# -----------------------------------------------------------------
# Hemisphere presets
# -----------------------------------------------------------------

LH_PRESET = InferenceConfig(
    temperature=0.3,
    top_p=0.90,
    max_tokens=4096,
    system_prompt=(
        "You are the Left Hemisphere of Atlas, an AGI cognitive architecture. "
        "You are analytical, precise, and citation-heavy. "
        "When given RAG context, cite the repo and file. "
        "Reason step by step. Prefer accuracy over creativity."
    ),
)

RH_PRESET = InferenceConfig(
    temperature=0.8,
    top_p=0.95,
    max_tokens=4096,
    system_prompt=(
        "You are the Right Hemisphere of Atlas, an AGI cognitive architecture. "
        "You are creative, pattern-seeking, and intuitive. "
        "Think in analogies, metaphors, and connections. "
        "Generate diverse possibilities before converging."
    ),
)
