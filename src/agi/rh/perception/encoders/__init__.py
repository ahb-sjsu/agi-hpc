# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Vision encoder interfaces and implementations.

Provides pluggable vision encoders for the RH perception pipeline:
- CLIP for semantic features
- DINOv2 for visual features
- ViT for general embeddings
"""

from agi.rh.perception.encoders.base import (
    VisionEncoder,
    BaseVisionEncoder,
    EncoderConfig,
    EncoderOutput,
)
from agi.rh.perception.encoders.clip import CLIPEncoder
from agi.rh.perception.encoders.dino import DINOv2Encoder
from agi.rh.perception.encoders.vit import ViTEncoder

__all__ = [
    # Protocol and base
    "VisionEncoder",
    "BaseVisionEncoder",
    "EncoderConfig",
    "EncoderOutput",
    # Implementations
    "CLIPEncoder",
    "DINOv2Encoder",
    "ViTEncoder",
]
