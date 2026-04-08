# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Shared embedding service for AGI-HPC.

Consolidates BGE-M3 model loading into a single process, avoiding
duplicate ~2.5 GB model copies across RAG server, memory service,
and metacognition. Other services encode text via HTTP or NATS request.

Can be used in two modes:
- **In-process**: Import and call ``SharedEmbedder.encode()`` directly.
- **HTTP microservice**: Run as a standalone Flask server on port 8083.

The service also provides PCA-384 projection for compressed search.

Usage (in-process)::

    from agi.common.embedding_service import SharedEmbedder

    embedder = SharedEmbedder.instance()
    embedding = embedder.encode("How does consensus work?")
    pca_embedding = embedder.encode_pca("How does consensus work?")

Usage (HTTP client)::

    import requests
    resp = requests.post("http://localhost:8083/encode", json={
        "texts": ["Hello world"],
        "pca": True
    })
    embeddings = resp.json()["embeddings"]
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore


class SharedEmbedder:
    """Singleton shared embedding model.

    Loads BGE-M3 once and provides encoding + PCA projection.
    Thread-safe via a lock on the model.

    Args:
        model_name: Sentence-transformer model name.
        device: Device for inference ('cpu', 'cuda:0', etc.).
        pca_path: Path to PCA rotation matrix for PCA-384 projection.
    """

    _instance: Optional[SharedEmbedder] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        pca_path: Optional[str] = None,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        logger.info("[embedder] loading %s on %s...", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._device = device
        self._encode_lock = threading.Lock()

        # PCA projection
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_dim: int = 0

        if pca_path is None:
            pca_path = os.environ.get(
                "AGI_PCA_MODEL",
                "/home/claude/agi-hpc/data/pca_rotation_384.pkl",
            )

        if pca_path and os.path.exists(pca_path):
            with open(pca_path, "rb") as f:
                pca_data = pickle.load(f)
            self._pca_components = pca_data["components"].T.astype(np.float32)
            self._pca_mean = pca_data["mean"].astype(np.float32)
            self._pca_dim = pca_data["n_components"]
            logger.info(
                "[embedder] PCA-%d loaded (%.1f%% variance)",
                self._pca_dim,
                pca_data["variance_captured"] * 100,
            )

        logger.info("[embedder] ready (%s)", model_name)

    @classmethod
    def instance(
        cls,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        pca_path: Optional[str] = None,
    ) -> SharedEmbedder:
        """Get or create the singleton embedder instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_name=model_name,
                        device=device,
                        pca_path=pca_path,
                    )
        return cls._instance

    @property
    def dim(self) -> int:
        """Full embedding dimension (e.g. 1024)."""
        return self._model.get_sentence_embedding_dimension()

    @property
    def pca_dim(self) -> int:
        """PCA-projected dimension (e.g. 384), or 0 if no PCA."""
        return self._pca_dim

    @property
    def has_pca(self) -> bool:
        """Whether PCA projection is available."""
        return self._pca_components is not None

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text(s) to full-dimension embeddings.

        Args:
            texts: Single text or list of texts.
            normalize: L2-normalize embeddings (default True for cosine).

        Returns:
            Array of shape (dim,) for single text, (n, dim) for list.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        with self._encode_lock:
            embeddings = self._model.encode(texts, normalize_embeddings=normalize)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings[0] if single else embeddings

    def encode_pca(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode text(s) and project to PCA-384 space.

        Args:
            texts: Single text or list of texts.
            normalize: L2-normalize the PCA-projected vectors.

        Returns:
            Array of shape (pca_dim,) for single text, (n, pca_dim) for list.

        Raises:
            RuntimeError: If no PCA model is loaded.
        """
        if not self.has_pca:
            raise RuntimeError("No PCA model loaded")

        embeddings = self.encode(texts, normalize=False)
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[np.newaxis, :]

        # PCA project
        centered = embeddings - self._pca_mean
        projected = centered @ self._pca_components

        if normalize:
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            projected = projected / (norms + 1e-10)

        return projected[0] if single else projected

    def __repr__(self) -> str:
        pca = f", PCA-{self._pca_dim}" if self.has_pca else ""
        return f"SharedEmbedder({self._model_name}, {self._device}{pca})"


# ---------------------------------------------------------------------------
# HTTP microservice (optional)
# ---------------------------------------------------------------------------


def create_app():
    """Create a Flask app for the shared embedding service."""
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    embedder = SharedEmbedder.instance()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": embedder._model_name,
                "dim": embedder.dim,
                "pca_dim": embedder.pca_dim,
            }
        )

    @app.route("/encode", methods=["POST"])
    def encode():
        data = request.get_json()
        texts = data.get("texts", [])
        use_pca = data.get("pca", False)
        normalize = data.get("normalize", True)

        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        if use_pca and embedder.has_pca:
            embeddings = embedder.encode_pca(texts, normalize=normalize)
        else:
            embeddings = embedder.encode(texts, normalize=normalize)

        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        return jsonify(
            {
                "embeddings": embeddings.tolist(),
                "dim": embeddings.shape[1],
                "n": len(texts),
                "pca": use_pca and embedder.has_pca,
            }
        )

    return app


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8083
    app = create_app()
    print(f"Embedding service on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
