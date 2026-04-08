# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Embedding compression codec for NATS event payloads and memory storage.

Provides transparent compression/decompression of embedding vectors in
Event payloads using TurboQuant 3-bit quantization. Reduces embedding
payload size by ~10x (4096 bytes -> ~392 bytes for 1024-dim BGE-M3).

When PCA rotation is available, additionally projects 1024 -> 384 dims
before quantization, achieving ~27x compression.

Usage in a NATS service::

    from agi.common.embedding_codec import EmbeddingCodec

    codec = EmbeddingCodec()

    # Compress before publishing
    payload["embedding"] = codec.compress(embedding_array)

    # Decompress after receiving
    embedding = codec.decompress(payload["embedding"])

The codec auto-detects whether a payload field is compressed (dict with
"_tq" key) or raw (list of floats), so mixed old/new services work.
"""

from __future__ import annotations

import base64
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from turboquant_pro import TurboQuantPGVector
    from turboquant_pro.pgvector import CompressedEmbedding

    _HAS_TQ = True
except ImportError:
    _HAS_TQ = False
    logger.debug("turboquant_pro not available; embedding codec in passthrough mode")


class EmbeddingCodec:
    """Compress/decompress embeddings for NATS transport and storage.

    Supports three modes:
    - **passthrough**: No compression (turboquant_pro not installed).
    - **tq_only**: TurboQuant 3-bit quantization (~10x compression).
    - **pca_tq**: PCA-384 projection + TurboQuant 3-bit (~27x compression).

    The mode is auto-selected based on available dependencies and PCA model.

    Args:
        dim: Embedding dimension (default 1024 for BGE-M3).
        bits: Quantization bit width (2, 3, or 4).
        seed: TurboQuant rotation matrix seed.
        pca_path: Path to PCA rotation matrix (.pkl file). If None, uses
            TQ-only mode. If the file doesn't exist, falls back to TQ-only.
    """

    def __init__(
        self,
        dim: int = 1024,
        bits: int = 3,
        seed: int = 42,
        pca_path: Optional[str] = None,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self._pca_data: Optional[Dict[str, Any]] = None
        self._tq: Optional[Any] = None
        self._tq_pca: Optional[Any] = None  # TQ for PCA-reduced dim
        self._mode = "passthrough"

        if not _HAS_TQ:
            logger.info("[embedding-codec] passthrough mode (no turboquant_pro)")
            return

        # Try to load PCA model
        if pca_path is None:
            pca_path = os.environ.get(
                "AGI_PCA_MODEL",
                "/home/claude/agi-hpc/data/pca_rotation_384.pkl",
            )

        if pca_path and os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                self._pca_data = pickle.load(f)
            pca_dim = self._pca_data["n_components"]
            self._pca_components = self._pca_data["components"].T.astype(
                np.float32
            )  # (1024, pca_dim)
            self._pca_mean = self._pca_data["mean"].astype(np.float32)
            self._tq_pca = TurboQuantPGVector(dim=pca_dim, bits=bits, seed=seed)
            self._mode = "pca_tq"
            logger.info(
                "[embedding-codec] pca_tq mode: %dd -> PCA-%dd -> TQ%d-bit",
                dim,
                pca_dim,
                bits,
            )
        else:
            self._tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
            self._mode = "tq_only"
            logger.info("[embedding-codec] tq_only mode: %dd -> TQ%d-bit", dim, bits)

    @property
    def mode(self) -> str:
        """Current compression mode: 'passthrough', 'tq_only', or 'pca_tq'."""
        return self._mode

    def compress(self, embedding: Union[np.ndarray, List[float]]) -> Dict[str, Any]:
        """Compress an embedding vector for transport.

        Args:
            embedding: 1D float array or list of shape (dim,).

        Returns:
            Dict suitable for JSON serialization in an Event payload.
            Contains "_tq" marker key for auto-detection on decompress.
        """
        embedding = np.asarray(embedding, dtype=np.float32).ravel()

        if self._mode == "passthrough":
            return {"v": embedding.tolist()}

        if self._mode == "pca_tq":
            # PCA project + L2 normalize
            centered = embedding - self._pca_mean
            projected = centered @ self._pca_components
            norm = float(np.linalg.norm(projected))
            if norm > 1e-10:
                projected = projected / norm

            compressed = self._tq_pca.compress_embedding(projected)
            return {
                "_tq": "pca",
                "b": base64.b85encode(compressed.packed_bytes).decode("ascii"),
                "n": round(norm, 6),
                "d": compressed.dim,
                "bits": compressed.bits,
            }

        # tq_only
        compressed = self._tq.compress_embedding(embedding)
        return {
            "_tq": "raw",
            "b": base64.b85encode(compressed.packed_bytes).decode("ascii"),
            "n": round(compressed.norm, 6),
            "d": compressed.dim,
            "bits": compressed.bits,
        }

    def decompress(self, data: Union[Dict[str, Any], List[float]]) -> np.ndarray:
        """Decompress an embedding from transport format.

        Handles both compressed (dict with "_tq") and raw (list) formats,
        so services can interop during rolling upgrades.

        Args:
            data: Compressed dict or raw float list.

        Returns:
            Float32 numpy array of shape (dim,) or (pca_dim,).
        """
        # Raw float list (uncompressed legacy format)
        if isinstance(data, list):
            return np.array(data, dtype=np.float32)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict or list, got {type(data)}")

        # Uncompressed dict format (passthrough mode)
        if "_tq" not in data:
            return np.array(data.get("v", data.get("embedding", [])), dtype=np.float32)

        packed_bytes = base64.b85decode(data["b"])
        norm = data["n"]
        dim = data["d"]
        bits = data["bits"]
        mode = data["_tq"]

        if not _HAS_TQ:
            raise RuntimeError(
                "Received compressed embedding but turboquant_pro is not installed"
            )

        compressed = CompressedEmbedding(
            packed_bytes=packed_bytes, norm=norm, dim=dim, bits=bits
        )

        if mode == "pca":
            if self._tq_pca is None:
                raise RuntimeError(
                    "Received PCA-compressed embedding but no PCA model loaded"
                )
            return self._tq_pca.decompress_embedding(compressed)

        if self._tq is None:
            # Create on-the-fly if we only have PCA mode but receive raw TQ
            tq = TurboQuantPGVector(dim=dim, bits=bits, seed=42)
            return tq.decompress_embedding(compressed)

        return self._tq.decompress_embedding(compressed)

    def payload_size(self, embedding: Union[np.ndarray, List[float]]) -> Dict[str, int]:
        """Compare compressed vs raw payload sizes.

        Args:
            embedding: Sample embedding.

        Returns:
            Dict with raw_bytes, compressed_bytes, ratio.
        """
        import json

        raw = json.dumps({"embedding": list(map(float, embedding))}).encode()
        compressed = json.dumps(
            {"embedding": self.compress(np.asarray(embedding))}
        ).encode()

        return {
            "raw_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "ratio": round(len(raw) / max(len(compressed), 1), 1),
        }

    def __repr__(self) -> str:
        return f"EmbeddingCodec(mode={self._mode}, dim={self.dim}, bits={self.bits})"
