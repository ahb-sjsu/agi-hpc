# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
RH Perception Module.

Provides the perception pipeline for the Right Hemisphere:
- Perception class for state abstraction
- Vision encoders (CLIP, DINOv2, ViT)
- Object detection (future: YOLO, SAM)
- Depth estimation (future: MiDaS, ZoeDepth)
- State representation building
"""

# Core perception class
from agi.rh.perception.core import Perception

# Vision encoders
from agi.rh.perception.encoders import (
    VisionEncoder,
    BaseVisionEncoder,
    EncoderConfig,
    EncoderOutput,
    CLIPEncoder,
    DINOv2Encoder,
    ViTEncoder,
)
from agi.rh.perception.encoders.base import create_encoder

__all__ = [
    # Core perception
    "Perception",
    # Protocol and base
    "VisionEncoder",
    "BaseVisionEncoder",
    "EncoderConfig",
    "EncoderOutput",
    # Implementations
    "CLIPEncoder",
    "DINOv2Encoder",
    "ViTEncoder",
    # Factory
    "create_encoder",
]
