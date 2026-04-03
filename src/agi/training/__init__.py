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
AtlasGym -- Gymnasium-compatible training framework for AGI-HPC.

Provides multi-domain skill environments for training and evaluating the
Atlas cognitive architecture across ethics, reasoning, coding, debate,
and memory recall.
"""

from __future__ import annotations

__all__ = [
    "AtlasGym",
    "CurriculumManager",
    "ResponseScorer",
    "TrainingRunner",
]

try:
    from agi.training.gym_env import AtlasGym
    from agi.training.curriculum import CurriculumManager
    from agi.training.scorer import ResponseScorer
    from agi.training.runner import TrainingRunner
except Exception:  # pragma: no cover
    pass
