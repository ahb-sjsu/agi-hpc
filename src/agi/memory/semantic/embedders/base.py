# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Base embedding model protocol and utilities.

Defines the interface for text embedding models used by semantic memory.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    @property
    def dimension(self) -> int:
        """Get embedding vector dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Get model name/identifier."""
        ...

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text list.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        ...


class BaseEmbedder(ABC):
    """Base class for embedding models."""

    def __init__(self, model_name: str, dimension: int):
        self._model_name = model_name
        self._dimension = dimension
        logger.info(
            "[embedding] initialized model=%s dimension=%d",
            model_name,
            dimension,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @abstractmethod
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Implementation of batch embedding."""
        pass

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text list."""
        if not texts:
            return []

        embeddings = await self._embed_batch(texts)

        logger.debug(
            "[embedding][%s] embedded %d texts",
            self._model_name,
            len(texts),
        )

        return embeddings

    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        results = await self.embed([text])
        return results[0] if results else []


@dataclass
class EmbeddingResult:
    """Result container for embedding operations."""

    embeddings: List[List[float]]
    model: str
    dimension: int
    token_count: int = 0


def compute_text_hash(text: str) -> str:
    """Compute hash for cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
