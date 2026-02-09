# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Vision Encoder Protocol and Base Implementation.

Defines the interface for vision encoders used in the RH perception pipeline.
Encoders convert raw visual input (images, video frames) into feature embeddings
that can be used for downstream tasks like object detection, semantic search,
and world model grounding.

Sprint 4 Implementation:
- VisionEncoder protocol for encoder interface
- BaseVisionEncoder with common functionality
- EncoderConfig for configuration
- EncoderOutput for standardized output
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    """
    Configuration for vision encoders.

    Environment variables:
        AGI_RH_ENCODER_MODEL: Model name/identifier
        AGI_RH_ENCODER_DEVICE: Device to use (cpu, cuda, cuda:0, mps)
        AGI_RH_ENCODER_BATCH_SIZE: Batch size for inference
    """

    model_name: str = "default"
    device: str = "cpu"
    batch_size: int = 1
    image_size: int = 224
    normalize: bool = True
    cache_embeddings: bool = False
    dtype: str = "float32"
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Output Container
# ---------------------------------------------------------------------------


@dataclass
class EncoderOutput:
    """
    Standardized output from vision encoders.

    Attributes:
        features: Feature embedding vector(s)
        shape: Shape of input image(s)
        model_name: Name of encoder model used
        latency_ms: Inference time in milliseconds
        extra: Additional model-specific outputs
    """

    features: np.ndarray
    shape: tuple
    model_name: str
    latency_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        if len(self.features.shape) == 1:
            return self.features.shape[0]
        return self.features.shape[-1]

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        if len(self.features.shape) == 1:
            return 1
        return self.features.shape[0]

    def to_list(self) -> List[float]:
        """Convert features to list for serialization."""
        return self.features.flatten().tolist()


# ---------------------------------------------------------------------------
# Vision Encoder Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VisionEncoder(Protocol):
    """
    Protocol for vision encoders.

    All vision encoders must implement this interface to be used
    with the RH perception pipeline.

    The encode method accepts various input formats:
    - numpy.ndarray: HWC or BHWC image array
    - bytes: Raw image bytes (JPEG, PNG)
    - PIL.Image: PIL Image object
    - torch.Tensor: PyTorch tensor
    """

    @property
    def name(self) -> str:
        """Return the encoder name/identifier."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        ...

    def encode(
        self,
        images: Union[np.ndarray, List[np.ndarray], Any],
    ) -> EncoderOutput:
        """
        Encode image(s) to feature embeddings.

        Args:
            images: Input image(s) in various formats

        Returns:
            EncoderOutput containing feature embeddings
        """
        ...

    def encode_batch(
        self,
        images: List[Any],
    ) -> EncoderOutput:
        """
        Encode a batch of images.

        Args:
            images: List of images to encode

        Returns:
            EncoderOutput with batched features
        """
        ...

    def is_available(self) -> bool:
        """
        Check if the encoder is available and ready.

        Returns:
            True if the encoder can be used
        """
        ...


# ---------------------------------------------------------------------------
# Base Vision Encoder
# ---------------------------------------------------------------------------


class BaseVisionEncoder(ABC):
    """
    Base class for vision encoders with common functionality.

    Provides:
    - Configuration management
    - Input preprocessing
    - Output postprocessing
    - Caching support
    """

    def __init__(self, config: Optional[EncoderConfig] = None) -> None:
        self._config = config or EncoderConfig()
        self._cache: Dict[str, EncoderOutput] = {}
        self._initialized = False

        logger.info(
            "[RH][Encoder] Initializing %s on device=%s",
            self.__class__.__name__,
            self._config.device,
        )

    @property
    def name(self) -> str:
        """Return the encoder name."""
        return self._config.model_name

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality."""
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """Load the encoder model. Called lazily on first encode."""
        pass

    @abstractmethod
    def _encode_impl(self, images: np.ndarray) -> np.ndarray:
        """
        Internal encoding implementation.

        Args:
            images: Preprocessed images as numpy array (BCHW or BHWC)

        Returns:
            Feature embeddings as numpy array
        """
        pass

    def encode(
        self,
        images: Union[np.ndarray, List[np.ndarray], Any],
    ) -> EncoderOutput:
        """
        Encode image(s) to feature embeddings.

        Args:
            images: Input image(s) in various formats

        Returns:
            EncoderOutput containing feature embeddings
        """
        import time

        if not self._initialized:
            self._load_model()
            self._initialized = True

        start = time.time()

        # Preprocess input
        processed, original_shape = self._preprocess(images)

        # Encode
        features = self._encode_impl(processed)

        latency_ms = (time.time() - start) * 1000

        return EncoderOutput(
            features=features,
            shape=original_shape,
            model_name=self.name,
            latency_ms=latency_ms,
        )

    def encode_batch(
        self,
        images: List[Any],
    ) -> EncoderOutput:
        """
        Encode a batch of images.

        Args:
            images: List of images to encode

        Returns:
            EncoderOutput with batched features
        """
        import time

        if not self._initialized:
            self._load_model()
            self._initialized = True

        start = time.time()

        # Preprocess each image
        processed_list = []
        shapes = []
        for img in images:
            proc, shape = self._preprocess(img)
            processed_list.append(proc)
            shapes.append(shape)

        # Stack into batch
        batch = np.concatenate(processed_list, axis=0)

        # Encode batch
        features = self._encode_impl(batch)

        latency_ms = (time.time() - start) * 1000

        return EncoderOutput(
            features=features,
            shape=shapes[0] if shapes else (0,),
            model_name=self.name,
            latency_ms=latency_ms,
            extra={"batch_shapes": shapes},
        )

    def is_available(self) -> bool:
        """
        Check if the encoder is available.

        Returns:
            True if the encoder can be used
        """
        try:
            if not self._initialized:
                self._load_model()
                self._initialized = True
            return True
        except Exception as e:
            logger.warning("[RH][Encoder] %s not available: %s", self.name, e)
            return False

    def _preprocess(
        self,
        images: Union[np.ndarray, Any],
    ) -> tuple[np.ndarray, tuple]:
        """
        Preprocess input images.

        Args:
            images: Input in various formats

        Returns:
            Tuple of (preprocessed array, original shape)
        """
        # Convert to numpy if needed
        if hasattr(images, "numpy"):
            # torch.Tensor
            arr = images.cpu().numpy()
        elif hasattr(images, "__array__"):
            arr = np.asarray(images)
        elif isinstance(images, bytes):
            # Raw bytes - would need PIL
            arr = self._decode_bytes(images)
        elif isinstance(images, np.ndarray):
            arr = images
        else:
            # Assume it's a PIL Image or similar
            arr = np.asarray(images)

        original_shape = arr.shape

        # Ensure batch dimension
        if len(arr.shape) == 3:
            arr = arr[np.newaxis, ...]

        # Normalize if configured
        if self._config.normalize and arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0

        return arr, original_shape

    def _decode_bytes(self, data: bytes) -> np.ndarray:
        """Decode image bytes to numpy array."""
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(data))
            return np.asarray(img)
        except ImportError:
            # Fallback: return dummy array
            logger.warning("[RH][Encoder] PIL not available for byte decoding")
            return np.zeros((224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_encoder(
    encoder_type: str = "vit",
    config: Optional[EncoderConfig] = None,
) -> VisionEncoder:
    """
    Create a vision encoder by type.

    Args:
        encoder_type: Type of encoder (clip, dino, vit)
        config: Optional encoder configuration

    Returns:
        VisionEncoder instance

    Raises:
        ValueError: If encoder type is not supported
    """
    encoder_type = encoder_type.lower()

    if encoder_type == "clip":
        from agi.rh.perception.encoders.clip import CLIPEncoder

        return CLIPEncoder(config)
    elif encoder_type in ("dino", "dinov2"):
        from agi.rh.perception.encoders.dino import DINOv2Encoder

        return DINOv2Encoder(config)
    elif encoder_type == "vit":
        from agi.rh.perception.encoders.vit import ViTEncoder

        return ViTEncoder(config)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
