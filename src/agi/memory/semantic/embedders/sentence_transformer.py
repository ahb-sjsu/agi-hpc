# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Sentence Transformer embedding model.

Uses sentence-transformers library for local embedding generation.
Supports models like all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from agi.memory.semantic.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


# Default models with their dimensions
DEFAULT_MODELS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
}


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedding model using sentence-transformers.

    Runs locally, no API key required.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        # Load model to determine dimension
        self._model = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_sentence_embedding_dimension()

        super().__init__(model_name, dimension)

        self._batch_size = batch_size
        self._device = device or self._model.device

        logger.info(
            "[embedding][sentence-transformer] loaded %s on %s",
            model_name,
            self._device,
        )

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        # sentence-transformers is synchronous, but we keep async interface
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding for convenience."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
