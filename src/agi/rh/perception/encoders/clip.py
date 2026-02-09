# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
CLIP Vision Encoder for AGI-HPC RH Perception.

CLIP (Contrastive Language-Image Pre-training) provides semantic feature
embeddings that are aligned with text representations, enabling:
- Open-vocabulary object recognition
- Semantic scene understanding
- Text-guided visual search
- Cross-modal reasoning

This encoder wraps OpenAI's CLIP or open-source alternatives.
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


class CLIPEncoder(BaseVisionEncoder):
    """
    CLIP-based vision encoder for semantic features.

    Provides text-aligned visual embeddings useful for:
    - Semantic object detection
    - Scene classification
    - Visual question answering
    - Image-text similarity

    Configuration:
        model_name: CLIP model variant (e.g., "ViT-B/32", "ViT-L/14")
        device: Compute device (cpu, cuda, mps)
    """

    # Standard CLIP embedding dimensions by model
    MODEL_DIMS = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
        "RN50": 1024,
        "RN101": 512,
        "default": 512,
    }

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        cfg = config or EncoderConfig(model_name="ViT-B/32")
        super().__init__(cfg)

        self._model = None
        self._preprocess_fn = None
        self._dim = self.MODEL_DIMS.get(cfg.model_name, self.MODEL_DIMS["default"])

    @property
    def embedding_dim(self) -> int:
        """Return CLIP embedding dimension."""
        return self._dim

    def _load_model(self) -> None:
        """Load CLIP model."""
        try:
            import torch
            import clip

            self._model, self._preprocess_fn = clip.load(
                self._config.model_name,
                device=self._config.device,
            )
            self._model.eval()

            logger.info(
                "[RH][CLIP] Loaded model=%s dim=%d device=%s",
                self._config.model_name,
                self._dim,
                self._config.device,
            )
        except ImportError:
            logger.warning("[RH][CLIP] clip package not installed, using stub mode")
            self._model = None
        except Exception as e:
            logger.warning("[RH][CLIP] Model load failed: %s, using stub", e)
            self._model = None

    def _encode_impl(self, images: np.ndarray) -> np.ndarray:
        """
        Encode images using CLIP.

        Args:
            images: Preprocessed images (B, H, W, C) or (B, C, H, W)

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

            # Convert numpy to PIL and apply CLIP preprocessing
            batch_tensors = []
            for i in range(images.shape[0]):
                img = images[i]
                # Ensure HWC format and uint8
                if img.shape[0] in (1, 3, 4) and len(img.shape) == 3:
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                tensor = self._preprocess_fn(pil_img)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors).to(device)

            with torch.no_grad():
                features = self._model.encode_image(batch)
                features = features.cpu().numpy()

            # Normalize features
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

            return features.astype(np.float32)

        except Exception as e:
            logger.warning("[RH][CLIP] Encoding failed: %s", e)
            batch_size = images.shape[0]
            return np.random.randn(batch_size, self._dim).astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode text using CLIP text encoder.

        Args:
            texts: List of text strings to encode

        Returns:
            Text embeddings (N, embedding_dim)
        """
        if self._model is None:
            return np.random.randn(len(texts), self._dim).astype(np.float32)

        try:
            import torch
            import clip

            device = self._config.device
            tokens = clip.tokenize(texts).to(device)

            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features.cpu().numpy()

            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            return features.astype(np.float32)

        except Exception as e:
            logger.warning("[RH][CLIP] Text encoding failed: %s", e)
            return np.random.randn(len(texts), self._dim).astype(np.float32)

    def similarity(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
    ) -> np.ndarray:
        """
        Compute image-text similarity scores.

        Args:
            image_features: Image embeddings (N, D)
            text_features: Text embeddings (M, D)

        Returns:
            Similarity matrix (N, M)
        """
        return np.dot(image_features, text_features.T)
