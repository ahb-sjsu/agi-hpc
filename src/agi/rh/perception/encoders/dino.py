# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
DINOv2 Vision Encoder for AGI-HPC RH Perception.

DINOv2 provides self-supervised visual features that excel at:
- Dense visual features for segmentation
- Object-centric representations
- Depth estimation
- Scene understanding without fine-tuning

This encoder wraps Meta's DINOv2 models.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from agi.rh.perception.encoders.base import (
    BaseVisionEncoder,
    EncoderConfig,
)

logger = logging.getLogger(__name__)


class DINOv2Encoder(BaseVisionEncoder):
    """
    DINOv2-based vision encoder for visual features.

    Provides self-supervised visual embeddings useful for:
    - Dense feature extraction
    - Object segmentation
    - Depth estimation
    - Visual similarity

    Configuration:
        model_name: DINOv2 model variant (e.g., "dinov2_vits14", "dinov2_vitb14")
        device: Compute device (cpu, cuda, mps)
    """

    # DINOv2 embedding dimensions by model
    MODEL_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
        "default": 768,
    }

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        cfg = config or EncoderConfig(model_name="dinov2_vitb14")
        super().__init__(cfg)

        self._model = None
        self._transform = None
        self._dim = self.MODEL_DIMS.get(cfg.model_name, self.MODEL_DIMS["default"])

    @property
    def embedding_dim(self) -> int:
        """Return DINOv2 embedding dimension."""
        return self._dim

    def _load_model(self) -> None:
        """Load DINOv2 model from torch hub."""
        try:
            import torch

            # Load DINOv2 from torch hub
            self._model = torch.hub.load(
                "facebookresearch/dinov2",
                self._config.model_name,
            )
            self._model = self._model.to(self._config.device)
            self._model.eval()

            # Standard ImageNet normalization
            self._mean = np.array([0.485, 0.456, 0.406])
            self._std = np.array([0.229, 0.224, 0.225])

            logger.info(
                "[RH][DINOv2] Loaded model=%s dim=%d device=%s",
                self._config.model_name,
                self._dim,
                self._config.device,
            )
        except ImportError:
            logger.warning("[RH][DINOv2] torch not installed, using stub mode")
            self._model = None
        except Exception as e:
            logger.warning("[RH][DINOv2] Model load failed: %s, using stub", e)
            self._model = None

    def _encode_impl(self, images: np.ndarray) -> np.ndarray:
        """
        Encode images using DINOv2.

        Args:
            images: Preprocessed images (B, H, W, C) normalized to [0, 1]

        Returns:
            Feature embeddings (B, embedding_dim)
        """
        if self._model is None:
            # Stub mode: return random embeddings
            batch_size = images.shape[0]
            return np.random.randn(batch_size, self._dim).astype(np.float32)

        try:
            import torch
            import torch.nn.functional as F

            device = self._config.device

            # Ensure BHWC format
            if images.shape[1] in (1, 3, 4) and len(images.shape) == 4:
                # Already BCHW, transpose to BHWC first
                images = np.transpose(images, (0, 2, 3, 1))

            # Resize to model input size (224 or 518 for DINOv2)
            target_size = 518 if "g14" in self._config.model_name else 224

            batch_tensors = []
            for i in range(images.shape[0]):
                img = images[i]

                # Ensure float32 and [0, 1] range
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                elif img.max() > 1.0:
                    img = img / 255.0

                # Normalize
                img = (img - self._mean) / self._std

                # Convert to BCHW tensor
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.float().to(device)

                # Resize
                tensor = F.interpolate(
                    tensor,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                batch_tensors.append(tensor)

            batch = torch.cat(batch_tensors, dim=0)

            with torch.no_grad():
                features = self._model(batch)
                features = features.cpu().numpy()

            return features.astype(np.float32)

        except Exception as e:
            logger.warning("[RH][DINOv2] Encoding failed: %s", e)
            batch_size = images.shape[0]
            return np.random.randn(batch_size, self._dim).astype(np.float32)

    def encode_dense(self, images: np.ndarray) -> np.ndarray:
        """
        Extract dense (patch-level) features from images.

        Useful for segmentation and dense prediction tasks.

        Args:
            images: Input images (B, H, W, C)

        Returns:
            Dense features (B, num_patches, embedding_dim)
        """
        if self._model is None:
            batch_size = images.shape[0]
            num_patches = (224 // 14) ** 2  # 256 patches for 224x224
            return np.random.randn(batch_size, num_patches, self._dim).astype(
                np.float32
            )

        try:
            import torch
            import torch.nn.functional as F

            device = self._config.device

            # Preprocess
            if images.shape[1] in (1, 3, 4) and len(images.shape) == 4:
                images = np.transpose(images, (0, 2, 3, 1))

            target_size = 518 if "g14" in self._config.model_name else 224

            batch_tensors = []
            for i in range(images.shape[0]):
                img = images[i]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                img = (img - self._mean) / self._std
                tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.float().to(device)
                tensor = F.interpolate(
                    tensor,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                batch_tensors.append(tensor)

            batch = torch.cat(batch_tensors, dim=0)

            with torch.no_grad():
                # Get intermediate features
                features = self._model.get_intermediate_layers(batch, n=1)[0]
                features = features.cpu().numpy()

            return features.astype(np.float32)

        except Exception as e:
            logger.warning("[RH][DINOv2] Dense encoding failed: %s", e)
            batch_size = images.shape[0]
            num_patches = (224 // 14) ** 2
            return np.random.randn(batch_size, num_patches, self._dim).astype(
                np.float32
            )
