# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Vision Transformer (ViT) Encoder for AGI-HPC RH Perception.

ViT provides general-purpose visual embeddings trained on large-scale
image classification tasks. Useful for:
- General visual feature extraction
- Image classification
- Transfer learning base
- Multi-modal fusion

This encoder wraps HuggingFace transformers ViT models.
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


class ViTEncoder(BaseVisionEncoder):
    """
    Vision Transformer encoder for general visual features.

    Provides ImageNet-pretrained visual embeddings useful for:
    - General feature extraction
    - Classification
    - Transfer learning

    Configuration:
        model_name: ViT model variant (e.g., "google/vit-base-patch16-224")
        device: Compute device (cpu, cuda, mps)
    """

    # ViT embedding dimensions by model
    MODEL_DIMS = {
        "google/vit-base-patch16-224": 768,
        "google/vit-large-patch16-224": 1024,
        "google/vit-huge-patch14-224-in21k": 1280,
        "vit-base": 768,
        "vit-large": 1024,
        "default": 768,
    }

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        cfg = config or EncoderConfig(model_name="google/vit-base-patch16-224")
        super().__init__(cfg)

        self._model = None
        self._processor = None
        self._dim = self.MODEL_DIMS.get(cfg.model_name, self.MODEL_DIMS["default"])

    @property
    def embedding_dim(self) -> int:
        """Return ViT embedding dimension."""
        return self._dim

    def _load_model(self) -> None:
        """Load ViT model from HuggingFace."""
        try:
            from transformers import ViTModel, ViTImageProcessor

            self._model = ViTModel.from_pretrained(self._config.model_name)
            self._processor = ViTImageProcessor.from_pretrained(self._config.model_name)

            if self._config.device != "cpu":
                import torch

                self._model = self._model.to(self._config.device)

            self._model.eval()

            logger.info(
                "[RH][ViT] Loaded model=%s dim=%d device=%s",
                self._config.model_name,
                self._dim,
                self._config.device,
            )
        except ImportError:
            logger.warning("[RH][ViT] transformers not installed, using stub mode")
            self._model = None
        except Exception as e:
            logger.warning("[RH][ViT] Model load failed: %s, using stub", e)
            self._model = None

    def _encode_impl(self, images: np.ndarray) -> np.ndarray:
        """
        Encode images using ViT.

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
            from PIL import Image

            device = self._config.device

            # Convert to PIL images for processor
            pil_images = []
            for i in range(images.shape[0]):
                img = images[i]

                # Ensure HWC format
                if img.shape[0] in (1, 3, 4) and len(img.shape) == 3:
                    img = np.transpose(img, (1, 2, 0))

                # Ensure uint8 for PIL
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).clip(0, 255).astype(np.uint8)

                pil_img = Image.fromarray(img)
                pil_images.append(pil_img)

            # Process images
            inputs = self._processor(
                images=pil_images,
                return_tensors="pt",
            )

            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use CLS token as image embedding
                features = outputs.last_hidden_state[:, 0, :]
                features = features.cpu().numpy()

            return features.astype(np.float32)

        except Exception as e:
            logger.warning("[RH][ViT] Encoding failed: %s", e)
            batch_size = images.shape[0]
            return np.random.randn(batch_size, self._dim).astype(np.float32)

    def encode_all_tokens(self, images: np.ndarray) -> np.ndarray:
        """
        Extract all patch token features (not just CLS).

        Useful for dense prediction tasks.

        Args:
            images: Input images (B, H, W, C)

        Returns:
            All token features (B, num_tokens, embedding_dim)
        """
        if self._model is None:
            batch_size = images.shape[0]
            # 197 tokens for 224x224 with patch size 16 (196 patches + 1 CLS)
            num_tokens = 197
            return np.random.randn(batch_size, num_tokens, self._dim).astype(np.float32)

        try:
            import torch
            from PIL import Image

            device = self._config.device

            pil_images = []
            for i in range(images.shape[0]):
                img = images[i]
                if img.shape[0] in (1, 3, 4) and len(img.shape) == 3:
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                pil_images.append(pil_img)

            inputs = self._processor(images=pil_images, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                features = outputs.last_hidden_state.cpu().numpy()

            return features.astype(np.float32)

        except Exception as e:
            logger.warning("[RH][ViT] All-token encoding failed: %s", e)
            batch_size = images.shape[0]
            num_tokens = 197
            return np.random.randn(batch_size, num_tokens, self._dim).astype(np.float32)
