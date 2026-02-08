# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
OpenAI embedding model.

Uses OpenAI's text-embedding API for embedding generation.
Supports text-embedding-3-small, text-embedding-3-large, etc.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from agi.memory.semantic.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


# Model dimensions
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding model using OpenAI API.

    Requires OPENAI_API_KEY environment variable.
    """

    API_URL = "https://api.openai.com/v1/embeddings"

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            logger.warning(
                "[embedding][openai] No API key. Set OPENAI_API_KEY."
            )

        # Determine dimension
        dimension = dimensions or MODEL_DIMENSIONS.get(model_name, 1536)

        super().__init__(model_name, dimension)

        self._custom_dimensions = dimensions

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        if not self._api_key:
            raise ValueError("OpenAI API key not configured")

        if httpx is None:
            return await self._embed_urllib(texts)

        return await self._embed_httpx(texts)

    async def _embed_httpx(self, texts: List[str]) -> List[List[float]]:
        """Embed using httpx."""
        payload = {
            "input": texts,
            "model": self._model_name,
        }

        if self._custom_dimensions:
            payload["dimensions"] = self._custom_dimensions

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

        # Sort by index to ensure correct order
        data = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    async def _embed_urllib(self, texts: List[str]) -> List[List[float]]:
        """Embed using urllib (fallback)."""
        import urllib.request

        payload = {
            "input": texts,
            "model": self._model_name,
        }

        if self._custom_dimensions:
            payload["dimensions"] = self._custom_dimensions

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.API_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30.0) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        data = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data]
