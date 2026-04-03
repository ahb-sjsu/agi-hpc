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
AtlasGym text environments for skill-domain training.
"""

from __future__ import annotations

__all__ = [
    "EthicsEnv",
    "ReasoningEnv",
    "CodingEnv",
    "DebateEnv",
    "MemoryEnv",
]

try:
    from agi.training.envs.ethics_env import EthicsEnv
    from agi.training.envs.reasoning_env import ReasoningEnv
    from agi.training.envs.coding_env import CodingEnv
    from agi.training.envs.debate_env import DebateEnv
    from agi.training.envs.memory_env import MemoryEnv
except Exception:  # pragma: no cover
    pass
