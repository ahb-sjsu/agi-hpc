# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
TurboQuant model weight compression.

Three compression methods:

**Beam** (``method="beam"``): Direct PolarQuant on weight matrix rows.
Each row is treated as a beam vector — rotated, scalar-quantized, and
bit-packed with its L2 norm preserved.  No decomposition; one error
source.  Adapted from Theory Radar's TurboBeam search compression.

**Beam Mixed** (``method="beam_mixed"``): Variance-based per-coordinate
mixed precision after PolarQuant rotation.  High-variance coordinates
get 4-bit, medium 3-bit, low 2-bit.  Achieves near-4-bit quality at
~3.3 average bits.  Inspired by Reddit u/FabulousExample4605's
per-weight precision selection (r/LocalLLaMA, 2026-04) and the
eigenvalue-weighted mixed precision design from TurboQuant Pro v0.9.0.

**SVD** (``method="svd"``): Truncated SVD followed by PolarQuant on the
factors.  Higher compression (~10-20x) but two error sources (rank
truncation + quantization) that compound across layers.

Theory:
    - Zandieh et al. (ICLR 2026): random rotation decorrelates
      dimensions for near-optimal scalar quantization (all methods).
    - Eckart-Young-Mirsky theorem: truncated SVD is optimal rank-k
      approximation (SVD method).
    - Theory Radar TurboBeam: direction-preserving vector quantization
      maintains similarity structure (beam methods).
    - Per-coordinate variance analysis after rotation reveals which
      dimensions carry signal vs noise (beam_mixed method).

Reuses :class:`TurboQuantKV` from ``turboquant_kv.py`` for the
rotation + quantization + bit-packing pipeline (shape-agnostic).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agi.meta.llm.turboquant_kv import _CODEBOOKS, CompressedKV, TurboQuantKV

logger = logging.getLogger(__name__)

try:
    import scipy.linalg

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #

_DEFAULT_SKIP_PATTERNS: List[str] = [
    r".*embed.*",
    r".*wte.*",
    r".*wpe.*",
    r".*lm_head.*",
    r".*norm.*",
    r".*ln_.*",
    r".*bias$",
]


@dataclass
class WeightCompressionConfig:
    """Configuration for TurboQuant weight compression.

    Attributes:
        method: Compression method — ``"beam"`` (direct PolarQuant,
            ~5x compression, high quality) or ``"svd"`` (SVD + PolarQuant,
            ~10-20x compression, lower quality).
        energy_threshold: Fraction of Frobenius-norm energy to retain
            when selecting the SVD truncation rank (SVD method only).
        bits: TurboQuant bit width for quantization (2, 3, or 4).
        min_rank: Floor on SVD rank per layer (SVD method only).
        max_rank_ratio: Ceiling as fraction of min(d_out, d_in) (SVD only).
        pack_factors: Bit-pack quantized data for max compression.
        skip_patterns: Regex patterns for layer names to skip.
        min_matrix_elements: Minimum weight elements to bother compressing.
        use_gpu: Use CuPy GPU kernels if available.
        seed: Random seed for TurboQuant rotation matrices.
    """

    method: str = "beam"
    energy_threshold: float = 0.95
    bits: int = 3
    min_rank: int = 16
    max_rank_ratio: float = 0.5
    pack_factors: bool = True
    skip_patterns: List[str] = field(
        default_factory=lambda: list(_DEFAULT_SKIP_PATTERNS)
    )
    min_matrix_elements: int = 4096
    use_gpu: bool = False
    seed: int = 42


# ------------------------------------------------------------------ #
# Compressed data structures                                          #
# ------------------------------------------------------------------ #


@dataclass
class CompressedWeight:
    """A single compressed weight matrix.

    For ``method="svd"``: stores truncated SVD factors U_k, S_k, Vt_k.
    For ``method="beam"``: stores the full weight via U_compressed with
    S=[1.0] and Vt_compressed as a dummy empty array.
    """

    U_compressed: Union[CompressedKV, np.ndarray]
    S: np.ndarray  # (k,) singular values (svd) or [1.0] (beam)
    Vt_compressed: Union[CompressedKV, np.ndarray]
    rank: int
    original_shape: Tuple[int, int]
    original_dtype: np.dtype
    energy_retained: float
    quantized: bool  # True if TQ was applied
    method: str = "svd"  # "svd" or "beam"

    # Shapes needed for decompression reshape
    U_shape: Tuple[int, int] = (0, 0)  # (d_out, k) or (d_out, d_in) for beam
    Vt_shape: Tuple[int, int] = (0, 0)  # (k, d_in) or (0, 0) for beam

    def nbytes(self) -> int:
        """Total bytes of the compressed representation."""
        s_bytes = self.S.nbytes

        def _size(obj: Union[CompressedKV, np.ndarray]) -> int:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            return obj.nbytes()  # CompressedKV

        if self.method == "beam":
            return _size(self.U_compressed) + s_bytes
        if self.quantized:
            return _size(self.U_compressed) + s_bytes + _size(self.Vt_compressed)
        return self.U_compressed.nbytes + s_bytes + self.Vt_compressed.nbytes

    def original_nbytes(self) -> int:
        """Bytes of the original uncompressed weight."""
        return int(np.prod(self.original_shape)) * self.original_dtype.itemsize

    def compression_ratio(self) -> float:
        """Ratio of original size to compressed size."""
        return self.original_nbytes() / max(self.nbytes(), 1)


@dataclass
class CompressedModel:
    """An entire model with compressed weight matrices."""

    weights: Dict[str, CompressedWeight]
    uncompressed: Dict[str, np.ndarray]
    config: WeightCompressionConfig
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_original_bytes(self) -> int:
        """Total bytes if all weights were uncompressed."""
        total = sum(cw.original_nbytes() for cw in self.weights.values())
        total += sum(v.nbytes for v in self.uncompressed.values())
        return total

    def total_compressed_bytes(self) -> int:
        """Total bytes of the compressed model."""
        total = sum(cw.nbytes() for cw in self.weights.values())
        total += sum(v.nbytes for v in self.uncompressed.values())
        return total

    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        return self.total_original_bytes() / max(self.total_compressed_bytes(), 1)

    def summary(self) -> Dict[str, Any]:
        """Human-readable compression summary."""
        return {
            "compressed_layers": len(self.weights),
            "uncompressed_layers": len(self.uncompressed),
            "original_mb": self.total_original_bytes() / (1024 * 1024),
            "compressed_mb": self.total_compressed_bytes() / (1024 * 1024),
            "ratio": self.compression_ratio(),
            "savings_pct": (1 - 1 / self.compression_ratio()) * 100,
        }


# ------------------------------------------------------------------ #
# Main engine                                                         #
# ------------------------------------------------------------------ #


class TurboQuantWeights:
    """SVD + TurboQuant weight compression engine.

    Compresses weight matrices by truncated SVD factorization followed
    by TurboQuant quantization of the SVD factors.

    Args:
        config: Compression configuration.
    """

    def __init__(self, config: Optional[WeightCompressionConfig] = None) -> None:
        self.config = config or WeightCompressionConfig()
        self._compiled_skip = [re.compile(p) for p in self.config.skip_patterns]
        # Cache TurboQuantKV instances to avoid O(d^3) QR per forward pass
        self._tq_cache: Dict[Tuple, TurboQuantKV] = {}

    # ------------------------------------------------------------------ #
    # Rank selection                                                      #
    # ------------------------------------------------------------------ #

    def select_rank(
        self,
        singular_values: np.ndarray,
        energy_threshold: Optional[float] = None,
    ) -> int:
        """Select SVD truncation rank from cumulative energy.

        Args:
            singular_values: Singular values in descending order.
            energy_threshold: Override config energy threshold.

        Returns:
            Rank k such that sum(S[:k]^2) / sum(S^2) >= threshold.
        """
        threshold = energy_threshold or self.config.energy_threshold
        S2 = singular_values.astype(np.float64) ** 2
        total = S2.sum()
        if total < 1e-30:
            return self.config.min_rank

        cumulative = np.cumsum(S2) / total
        k = int(np.searchsorted(cumulative, threshold)) + 1

        max_rank = int(self.config.max_rank_ratio * len(singular_values))
        k = max(self.config.min_rank, min(k, max_rank))
        return k

    # ------------------------------------------------------------------ #
    # Single weight compression                                           #
    # ------------------------------------------------------------------ #

    def compress_weight(
        self,
        weight: np.ndarray,
        name: str = "",
    ) -> CompressedWeight:
        """Compress a single weight matrix.

        Uses the method specified in config: ``"beam"`` (direct PolarQuant)
        or ``"svd"`` (truncated SVD + PolarQuant on factors).

        Args:
            weight: 2-D weight matrix (d_out, d_in).
            name: Layer name (for logging).

        Returns:
            CompressedWeight.
        """
        if weight.ndim != 2:
            raise ValueError(f"Expected 2-D weight, got shape {weight.shape}")

        if self.config.method == "beam":
            return self._compress_beam(weight, name)
        if self.config.method == "beam_mixed":
            return self._compress_beam_mixed(weight, name)
        return self._compress_svd(weight, name)

    def _compress_beam(
        self,
        weight: np.ndarray,
        name: str = "",
    ) -> CompressedWeight:
        """Beam compression: direct PolarQuant on weight rows.

        Each row is treated as a beam vector — the rotation decorrelates
        dimensions, the Lloyd-Max codebook quantizes, and the L2 norm
        is stored exactly.  No SVD, no rank truncation.
        """
        d_out, d_in = weight.shape
        original_dtype = weight.dtype
        W = weight.astype(np.float32)

        tq = self._get_tq(d_in, self.config.seed)

        # Treat rows as vectors: (1, 1, d_out, d_in)
        W_4d = W.reshape(1, 1, d_out, d_in)
        W_comp = tq.compress(W_4d, packed=self.config.pack_factors)

        cw = CompressedWeight(
            U_compressed=W_comp,
            S=np.array([1.0], dtype=np.float32),
            Vt_compressed=np.empty(0, dtype=np.float32),
            rank=d_in,  # full rank — no truncation
            original_shape=(d_out, d_in),
            original_dtype=np.dtype(original_dtype),
            energy_retained=1.0,
            quantized=True,
            method="beam",
            U_shape=(d_out, d_in),
            Vt_shape=(0, 0),
        )

        logger.info(
            "[tq-weights] %s: (%d, %d) beam %d-bit, %.1fx compression",
            name or "weight",
            d_out,
            d_in,
            self.config.bits,
            cw.compression_ratio(),
        )

        return cw

    def _compress_beam_mixed(
        self,
        weight: np.ndarray,
        name: str = "",
    ) -> CompressedWeight:
        """Mixed-precision beam: per-coordinate bit allocation after rotation.

        After PolarQuant rotation, measures variance of each coordinate
        across all weight rows. High-variance coordinates get 4-bit,
        medium get 3-bit, low get 2-bit. Average ~3.0-3.3 bits.

        Inspired by eigenvalue-weighted mixed precision (TurboQuant Pro
        v0.9.0 roadmap) and per-weight precision selection (Reddit
        r/LocalLLaMA FabulousExample4605).
        """
        import math

        d_out, d_in = weight.shape
        original_dtype = weight.dtype
        W = weight.astype(np.float32)

        # --- Step 1: Rotation (reuse cached TQ for the rotation matrix) ---
        tq = self._get_tq(d_in, self.config.seed)

        # Compute norms and unit vectors
        norms = np.linalg.norm(W, axis=-1)  # (d_out,)
        safe_norms = np.maximum(norms, 1e-30)[:, np.newaxis]
        W_unit = W / safe_norms

        # Apply rotation (access TQ internals)
        W_rot = tq._rotate(tq._to_device(W_unit))
        if hasattr(W_rot, "get"):  # CuPy → numpy
            W_rot = W_rot.get()

        # --- Step 2: Per-coordinate variance analysis ---
        coord_var = np.var(W_rot, axis=0)  # (d_in,) variance per coord
        total_var = coord_var.sum()

        if total_var < 1e-30:
            # Degenerate — fall back to uniform
            return self._compress_beam(weight, name)

        # Sort coordinates by variance (descending)
        cum_var = np.cumsum(np.sort(coord_var)[::-1]) / total_var

        # Bit bands: top 60% variance->4bit, next 30%->3bit, bottom 10%->2bit
        var_threshold_4 = 0.60
        var_threshold_3 = 0.90  # cumulative: 60% + 30% = 90%

        n_4bit = int(np.searchsorted(cum_var, var_threshold_4)) + 1
        n_3bit = int(np.searchsorted(cum_var, var_threshold_3)) + 1 - n_4bit
        n_2bit = d_in - n_4bit - n_3bit

        # Map coordinates to bit bands based on their variance rank
        coord_order = np.argsort(coord_var)[::-1]  # highest variance first
        bit_map = np.zeros(d_in, dtype=np.uint8)
        bit_map[coord_order[:n_4bit]] = 4
        bit_map[coord_order[n_4bit : n_4bit + n_3bit]] = 3
        bit_map[coord_order[n_4bit + n_3bit :]] = 2

        avg_bits = (n_4bit * 4 + n_3bit * 3 + n_2bit * 2) / d_in

        # --- Step 3: Quantize each coordinate with its assigned codebook ---
        scale = 1.0 / math.sqrt(d_in)
        indices = np.empty_like(W_rot, dtype=np.uint8)

        for bits_val in [2, 3, 4]:
            mask = bit_map == bits_val
            if not mask.any():
                continue
            raw_centroids = _CODEBOOKS[bits_val]
            centroids = (raw_centroids * scale).astype(np.float32)
            boundaries = (centroids[:-1] + centroids[1:]) / 2.0
            indices[:, mask] = np.searchsorted(boundaries, W_rot[:, mask]).astype(
                np.uint8
            )

        # --- Step 4: Pack into CompressedKV-compatible format ---
        # Store as unpacked uint8 indices (mixed bit widths can't use standard packing)
        comp = CompressedKV(
            indices=indices,
            norms=norms.astype(np.float32),
            bits=0,  # 0 signals mixed precision
            original_dtype=np.dtype(original_dtype),
            packed=False,
            n_values=int(np.prod(indices.shape)),
            shape=tuple(int(s) for s in indices.shape),
        )

        # Store the bit_map and coord stats in a side array
        cw = CompressedWeight(
            U_compressed=comp,
            S=np.array([1.0], dtype=np.float32),
            Vt_compressed=bit_map,  # store bit assignments as numpy array
            rank=d_in,
            original_shape=(d_out, d_in),
            original_dtype=np.dtype(original_dtype),
            energy_retained=1.0,
            quantized=True,
            method="beam_mixed",
            U_shape=(d_out, d_in),
            Vt_shape=(0, 0),
        )

        logger.info(
            "[tq-weights] %s: (%d, %d) beam_mixed avg %.1f-bit "
            "(4b:%d 3b:%d 2b:%d), %.1fx compression",
            name or "weight",
            d_out,
            d_in,
            avg_bits,
            n_4bit,
            n_3bit,
            n_2bit,
            cw.compression_ratio(),
        )

        return cw

    def _decompress_beam_mixed(self, cw: CompressedWeight) -> np.ndarray:
        """Decompress a mixed-precision beam-compressed weight matrix."""
        import math

        d_out, d_in = cw.original_shape
        indices = cw.U_compressed.indices  # (d_out, d_in) uint8
        norms = cw.U_compressed.norms  # (d_out,)
        bit_map = cw.Vt_compressed  # (d_in,) uint8 bit assignments

        tq = self._get_tq(d_in, self.config.seed)
        scale = 1.0 / math.sqrt(d_in)

        # Reconstruct rotated coordinates from indices + codebooks
        W_rot = np.empty((d_out, d_in), dtype=np.float32)
        for bits_val in [2, 3, 4]:
            mask = bit_map == bits_val
            if not mask.any():
                continue
            centroids = (_CODEBOOKS[bits_val] * scale).astype(np.float32)
            W_rot[:, mask] = centroids[indices[:, mask]]

        # Inverse rotation
        W_rot_dev = tq._to_device(W_rot)
        W_unit = tq._unrotate(W_rot_dev)
        if hasattr(W_unit, "get"):
            W_unit = W_unit.get()

        # Scale by norms
        return W_unit * norms[:, np.newaxis]

    def _compress_svd(
        self,
        weight: np.ndarray,
        name: str = "",
    ) -> CompressedWeight:
        """SVD compression: truncated SVD + PolarQuant on factors."""
        d_out, d_in = weight.shape
        original_dtype = weight.dtype
        W = weight.astype(np.float32)

        # --- SVD ---
        if _HAS_SCIPY:
            U, S, Vt = scipy.linalg.svd(W, full_matrices=False, lapack_driver="gesdd")
        else:
            U, S, Vt = np.linalg.svd(W, full_matrices=False)

        # --- Rank selection ---
        k = self.select_rank(S)
        U_k = np.ascontiguousarray(U[:, :k])
        S_k = S[:k].astype(np.float32)
        Vt_k = np.ascontiguousarray(Vt[:k, :])

        energy = float(np.sum(S_k**2) / max(np.sum(S**2), 1e-30))

        # --- TurboQuant on factors ---
        tq_u = self._get_tq(k, self.config.seed)
        tq_vt = self._get_tq(d_in, self.config.seed + 1)

        U_4d = U_k.reshape(1, 1, d_out, k)
        Vt_4d = Vt_k.reshape(1, 1, k, d_in)

        U_comp = tq_u.compress(U_4d, packed=self.config.pack_factors)
        Vt_comp = tq_vt.compress(Vt_4d, packed=self.config.pack_factors)

        cw = CompressedWeight(
            U_compressed=U_comp,
            S=S_k,
            Vt_compressed=Vt_comp,
            rank=k,
            original_shape=(d_out, d_in),
            original_dtype=np.dtype(original_dtype),
            energy_retained=energy,
            quantized=True,
            method="svd",
            U_shape=(d_out, k),
            Vt_shape=(k, d_in),
        )

        logger.info(
            "[tq-weights] %s: (%d, %d) -> rank %d (%.1f%% energy), "
            "%.1fx compression",
            name or "weight",
            d_out,
            d_in,
            k,
            energy * 100,
            cw.compression_ratio(),
        )

        return cw

    # ------------------------------------------------------------------ #
    # Decompression                                                       #
    # ------------------------------------------------------------------ #

    def decompress_weight(self, cw: CompressedWeight) -> np.ndarray:
        """Reconstruct an approximate weight matrix from compressed form.

        Args:
            cw: Compressed weight from :meth:`compress_weight`.

        Returns:
            Reconstructed weight matrix (d_out, d_in) in float32.
        """
        if cw.method == "beam":
            return self._decompress_beam(cw)
        if cw.method == "beam_mixed":
            return self._decompress_beam_mixed(cw)
        U_k, Vt_k = self._decompress_factors(cw)
        W_approx = (U_k * cw.S[np.newaxis, :]) @ Vt_k
        return W_approx

    def _decompress_beam(self, cw: CompressedWeight) -> np.ndarray:
        """Decompress a beam-compressed weight matrix."""
        tq = self._get_tq(cw.original_shape[1], self.config.seed)
        return tq.decompress(cw.U_compressed).reshape(cw.original_shape)

    def compressed_linear(
        self,
        x: np.ndarray,
        cw: CompressedWeight,
    ) -> np.ndarray:
        """Compute a linear transform using compressed weights.

        For beam: decompresses full W, computes ``x @ W.T``.
        For SVD: factored matmul ``x @ Vt.T @ diag(S) @ U.T``.

        Args:
            x: Input tensor (..., d_in).
            cw: Compressed weight with shape (d_out, d_in).

        Returns:
            Output tensor (..., d_out).
        """
        if cw.method in ("beam", "beam_mixed"):
            W = self.decompress_weight(cw)
            return x @ W.T

        U_k, Vt_k = self._decompress_factors(cw)
        h = x @ Vt_k.T  # (..., k)
        h = h * cw.S  # (..., k)
        return h @ U_k.T  # (..., d_out)

    def _get_tq(self, head_dim: int, seed: int) -> TurboQuantKV:
        """Get a cached TurboQuantKV instance (avoids O(d^3) QR per call)."""
        key = (head_dim, self.config.bits, seed)
        if key not in self._tq_cache:
            self._tq_cache[key] = TurboQuantKV(
                head_dim=head_dim,
                n_heads=1,
                bits=self.config.bits,
                use_gpu=self.config.use_gpu,
                seed=seed,
            )
        return self._tq_cache[key]

    def _decompress_factors(
        self, cw: CompressedWeight
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompress U and Vt factors from a CompressedWeight.

        For beam: returns (W_decompressed, empty_array).
        For SVD: returns (U_k, Vt_k).
        """
        if cw.method == "beam":
            W = self._decompress_beam(cw)
            return W, np.empty(0, dtype=np.float32)

        if cw.method == "beam_mixed":
            W = self._decompress_beam_mixed(cw)
            return W, np.empty(0, dtype=np.float32)

        if cw.quantized:
            tq_u = self._get_tq(cw.rank, self.config.seed)
            tq_vt = self._get_tq(cw.original_shape[1], self.config.seed + 1)
            U_k = tq_u.decompress(cw.U_compressed).reshape(cw.U_shape)
            Vt_k = tq_vt.decompress(cw.Vt_compressed).reshape(cw.Vt_shape)
        else:
            U_k = cw.U_compressed.reshape(cw.U_shape)
            Vt_k = cw.Vt_compressed.reshape(cw.Vt_shape)
        return U_k, Vt_k

    def precompute_factors(
        self, compressed: CompressedModel
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Decompress all factors once for maximum inference speed.

        For beam: stores (W_decompressed, empty).
        For SVD: stores (U_k, Vt_k).

        Args:
            compressed: CompressedModel to precompute.

        Returns:
            Dict mapping layer name to (factor1, factor2) numpy arrays.
        """
        factors = {}
        for name, cw in compressed.weights.items():
            factors[name] = self._decompress_factors(cw)
        return factors

    # ------------------------------------------------------------------ #
    # Model-level compression                                             #
    # ------------------------------------------------------------------ #

    def _should_skip(self, name: str, tensor: np.ndarray) -> bool:
        """Check if a layer should be skipped."""
        if tensor.ndim != 2:
            return True
        if tensor.size < self.config.min_matrix_elements:
            return True
        for pat in self._compiled_skip:
            if pat.match(name):
                return True
        return False

    def compress_state_dict(
        self,
        state_dict: Dict[str, np.ndarray],
    ) -> CompressedModel:
        """Compress all eligible weight matrices in a state dict.

        Args:
            state_dict: Mapping of layer names to weight arrays.
                Values can be numpy arrays or (if torch is available)
                torch tensors which will be converted automatically.

        Returns:
            CompressedModel with compressed and uncompressed layers.
        """
        weights: Dict[str, CompressedWeight] = {}
        uncompressed: Dict[str, np.ndarray] = {}

        for name, tensor in state_dict.items():
            # Convert torch tensors if present
            arr = self._to_numpy(tensor)

            if self._should_skip(name, arr):
                uncompressed[name] = arr
                logger.debug("[tq-weights] Skipping %s (%s)", name, arr.shape)
                continue

            weights[name] = self.compress_weight(arr, name=name)

        model = CompressedModel(
            weights=weights,
            uncompressed=uncompressed,
            config=self.config,
        )

        summary = model.summary()
        logger.info(
            "[tq-weights] Model compressed: %d layers, %.1f MB -> %.1f MB (%.1fx)",
            summary["compressed_layers"],
            summary["original_mb"],
            summary["compressed_mb"],
            summary["ratio"],
        )

        return model

    # ------------------------------------------------------------------ #
    # PyTorch model patching                                              #
    # ------------------------------------------------------------------ #

    def patch_torch_model(
        self,
        model: Any,
        compressed: CompressedModel,
    ) -> Any:
        """Replace nn.Linear / Conv1D layers with CompressedLinear modules.

        Handles both nn.Linear (weight shape d_out x d_in, forward = x @ W.T)
        and Conv1D (weight shape d_in x d_out, forward = x @ W) used by GPT-2.

        Args:
            model: A torch.nn.Module.
            compressed: CompressedModel from :meth:`compress_state_dict`.

        Returns:
            The patched model (modified in place).
        """
        import torch
        import torch.nn as nn

        for name, cw in compressed.weights.items():
            # Navigate to parent module
            parts = name.replace(".weight", "").split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            old_module = getattr(parent, attr)

            # Skip non-linear modules (Embedding, LayerNorm, etc.)
            is_linear = isinstance(old_module, nn.Linear)
            is_conv1d = type(old_module).__name__ == "Conv1D"
            if not (is_linear or is_conv1d):
                logger.debug(
                    "[tq-weights] Skipping %s (type=%s)",
                    name,
                    type(old_module).__name__,
                )
                continue

            bias = None
            bias_name = name.replace(".weight", ".bias")
            if bias_name in compressed.uncompressed:
                bias = torch.from_numpy(compressed.uncompressed[bias_name].copy())

            new_module = _make_compressed_linear(
                cw, self, bias=bias, transposed=is_conv1d
            )
            setattr(parent, attr, new_module)
            logger.debug("[tq-weights] Patched %s (Conv1D=%s)", name, is_conv1d)

        return model

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #

    def save(self, compressed: CompressedModel, path: Union[str, Path]) -> None:
        """Save a compressed model to disk.

        Args:
            compressed: CompressedModel to save.
            path: Output file path (.npz).
        """
        import json

        arrays: Dict[str, np.ndarray] = {}
        meta: Dict[str, Any] = {
            "config": {
                "method": compressed.config.method,
                "energy_threshold": compressed.config.energy_threshold,
                "bits": compressed.config.bits,
                "min_rank": compressed.config.min_rank,
                "max_rank_ratio": compressed.config.max_rank_ratio,
                "pack_factors": compressed.config.pack_factors,
                "seed": compressed.config.seed,
            },
            "compressed_layers": {},
            "uncompressed_layers": list(compressed.uncompressed.keys()),
        }

        for name, cw in compressed.weights.items():
            prefix = f"c_{name}"
            arrays[f"{prefix}_S"] = cw.S
            if cw.method == "beam":
                arrays[f"{prefix}_U_indices"] = cw.U_compressed.indices
                arrays[f"{prefix}_U_norms"] = cw.U_compressed.norms
            elif cw.quantized:
                arrays[f"{prefix}_U_indices"] = cw.U_compressed.indices
                arrays[f"{prefix}_U_norms"] = cw.U_compressed.norms
                arrays[f"{prefix}_Vt_indices"] = cw.Vt_compressed.indices
                arrays[f"{prefix}_Vt_norms"] = cw.Vt_compressed.norms
            else:
                arrays[f"{prefix}_U"] = cw.U_compressed
                arrays[f"{prefix}_Vt"] = cw.Vt_compressed

            has_tq_u = isinstance(cw.U_compressed, CompressedKV)
            has_tq_vt = isinstance(cw.Vt_compressed, CompressedKV)
            meta["compressed_layers"][name] = {
                "method": cw.method,
                "rank": cw.rank,
                "original_shape": list(cw.original_shape),
                "original_dtype": str(cw.original_dtype),
                "energy_retained": cw.energy_retained,
                "quantized": cw.quantized,
                "U_shape": list(cw.U_shape),
                "Vt_shape": list(cw.Vt_shape),
                "bits": cw.U_compressed.bits if has_tq_u else 0,
                "packed": cw.U_compressed.packed if has_tq_u else False,
                "U_n_values": cw.U_compressed.n_values if has_tq_u else 0,
                "U_idx_shape": list(cw.U_compressed.shape) if has_tq_u else [],
                "Vt_n_values": cw.Vt_compressed.n_values if has_tq_vt else 0,
                "Vt_idx_shape": list(cw.Vt_compressed.shape) if has_tq_vt else [],
            }

        for name, arr in compressed.uncompressed.items():
            arrays[f"u_{name}"] = arr

        arrays["__metadata__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )

        np.savez_compressed(str(path), **arrays)
        logger.info("[tq-weights] Saved to %s", path)

    def load(self, path: Union[str, Path]) -> CompressedModel:
        """Load a compressed model from disk.

        Args:
            path: Path to .npz file saved by :meth:`save`.

        Returns:
            CompressedModel.
        """
        import json

        data = np.load(str(path), allow_pickle=False)
        meta = json.loads(bytes(data["__metadata__"]))

        cfg = WeightCompressionConfig(**meta["config"])

        weights: Dict[str, CompressedWeight] = {}
        for name, layer_meta in meta["compressed_layers"].items():
            prefix = f"c_{name}"
            S = data[f"{prefix}_S"]
            method = layer_meta.get("method", "svd")

            U_comp: Union[CompressedKV, np.ndarray]
            Vt_comp: Union[CompressedKV, np.ndarray]

            if method == "beam":
                U_comp = CompressedKV(
                    indices=data[f"{prefix}_U_indices"],
                    norms=data[f"{prefix}_U_norms"],
                    bits=layer_meta["bits"],
                    original_dtype=np.dtype(np.float32),
                    packed=layer_meta["packed"],
                    n_values=layer_meta["U_n_values"],
                    shape=tuple(layer_meta["U_idx_shape"]),
                )
                Vt_comp = np.empty(0, dtype=np.float32)
            elif layer_meta["quantized"]:
                U_comp = CompressedKV(
                    indices=data[f"{prefix}_U_indices"],
                    norms=data[f"{prefix}_U_norms"],
                    bits=layer_meta["bits"],
                    original_dtype=np.dtype(np.float32),
                    packed=layer_meta["packed"],
                    n_values=layer_meta["U_n_values"],
                    shape=tuple(layer_meta["U_idx_shape"]),
                )
                Vt_comp = CompressedKV(
                    indices=data[f"{prefix}_Vt_indices"],
                    norms=data[f"{prefix}_Vt_norms"],
                    bits=layer_meta["bits"],
                    original_dtype=np.dtype(np.float32),
                    packed=layer_meta["packed"],
                    n_values=layer_meta["Vt_n_values"],
                    shape=tuple(layer_meta["Vt_idx_shape"]),
                )
            else:
                U_comp = data[f"{prefix}_U"]
                Vt_comp = data[f"{prefix}_Vt"]

            weights[name] = CompressedWeight(
                U_compressed=U_comp,
                S=S,
                Vt_compressed=Vt_comp,
                rank=layer_meta["rank"],
                original_shape=tuple(layer_meta["original_shape"]),
                original_dtype=np.dtype(layer_meta["original_dtype"]),
                energy_retained=layer_meta["energy_retained"],
                quantized=layer_meta["quantized"],
                method=method,
                U_shape=tuple(layer_meta["U_shape"]),
                Vt_shape=tuple(layer_meta["Vt_shape"]),
            )

        uncompressed: Dict[str, np.ndarray] = {}
        for name in meta["uncompressed_layers"]:
            uncompressed[name] = data[f"u_{name}"]

        return CompressedModel(
            weights=weights,
            uncompressed=uncompressed,
            config=cfg,
        )

    # ------------------------------------------------------------------ #
    # Static estimation                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def estimate_compression(
        d_out: int,
        d_in: int,
        rank: int,
        bits: int = 3,
        original_dtype: str = "float16",
    ) -> Dict[str, float]:
        """Estimate compression ratio without running SVD.

        Args:
            d_out: Output dimension.
            d_in: Input dimension.
            rank: SVD truncation rank.
            bits: TurboQuant bit width.
            original_dtype: Original weight dtype.

        Returns:
            Dict with original_bytes, compressed_bytes, ratio, savings_pct.
        """
        elem_size = 2 if original_dtype == "float16" else 4
        original = d_out * d_in * elem_size

        # S: always fp32
        s_bytes = rank * 4

        # U factors: d_out vectors of dim k
        u_bits = d_out * rank * bits
        u_packed = (u_bits + 7) // 8
        u_norms = d_out * 4  # fp32 per vector

        # Vt factors: k vectors of dim d_in
        vt_bits = rank * d_in * bits
        vt_packed = (vt_bits + 7) // 8
        vt_norms = rank * 4

        compressed = s_bytes + u_packed + u_norms + vt_packed + vt_norms

        return {
            "original_bytes": original,
            "compressed_bytes": compressed,
            "ratio": original / max(compressed, 1),
            "savings_pct": (1 - compressed / original) * 100,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        """Convert a tensor to numpy, handling torch tensors."""
        if isinstance(tensor, np.ndarray):
            return tensor
        # torch.Tensor
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)


# ------------------------------------------------------------------ #
# PyTorch nn.Module for compressed inference                          #
# ------------------------------------------------------------------ #


def _make_compressed_linear(
    cw: CompressedWeight,
    engine: TurboQuantWeights,
    bias: Any = None,
    transposed: bool = False,
) -> Any:
    """Create a CompressedLinear nn.Module (requires torch).

    Args:
        cw: Compressed weight.
        engine: TurboQuantWeights engine.
        bias: Optional bias tensor.
        transposed: If True, the original layer used ``x @ W`` (Conv1D)
            instead of ``x @ W.T`` (nn.Linear).  The SVD was computed
            on W as-is, so the factored matmul direction must match.
    """
    import torch
    import torch.nn as nn

    class CompressedLinear(nn.Module):
        """Drop-in nn.Linear/Conv1D replacement using SVD + TurboQuant.

        Decompresses factors on first forward pass and caches them.
        """

        def __init__(
            self,
            compressed_weight: CompressedWeight,
            tq_engine: TurboQuantWeights,
            bias_tensor: Any = None,
            is_transposed: bool = False,
        ) -> None:
            super().__init__()
            self._cw = compressed_weight
            self._engine = tq_engine
            self._transposed = is_transposed
            self._U_k: Optional[torch.Tensor] = None
            self._Vt_k: Optional[torch.Tensor] = None
            self._S: Optional[torch.Tensor] = None
            if bias_tensor is not None:
                self.register_buffer("bias", bias_tensor.float())
            else:
                self.bias = None

        def _ensure_factors(self) -> None:
            if self._U_k is not None:
                return
            if self._cw.method in ("beam", "beam_mixed"):
                W = self._engine.decompress_weight(self._cw)
                self._U_k = torch.from_numpy(W)
                self._Vt_k = torch.empty(0)
                self._S = torch.empty(0)
            else:
                U_k, Vt_k = self._engine._decompress_factors(self._cw)
                self._U_k = torch.from_numpy(U_k)
                self._Vt_k = torch.from_numpy(Vt_k)
                self._S = torch.from_numpy(self._cw.S)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self._ensure_factors()
            assert self._U_k is not None

            if self._cw.method in ("beam", "beam_mixed"):
                W = self._U_k.to(x.device, x.dtype)
                if self._transposed:
                    out = x @ W
                else:
                    out = x @ W.T
            else:
                U = self._U_k.to(x.device, x.dtype)
                Vt = self._Vt_k.to(x.device, x.dtype)
                S = self._S.to(x.device, x.dtype)
                if self._transposed:
                    h = x @ U
                    h = h * S
                    out = h @ Vt
                else:
                    h = x @ Vt.T
                    h = h * S
                    out = h @ U.T

            if self.bias is not None:
                out = out + self.bias.to(x.dtype)
            return out

    return CompressedLinear(cw, engine, bias, transposed)


class CompressedLinearNumpy:
    """NumPy-only compressed linear (no torch dependency).

    For use in pure-numpy inference or testing without PyTorch.
    """

    def __init__(
        self,
        compressed_weight: CompressedWeight,
        engine: TurboQuantWeights,
        bias: Any = None,
    ) -> None:
        self.cw = compressed_weight
        self.engine = engine
        self._bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.engine.compressed_linear(x, self.cw)
        if self._bias is not None:
            bias = self.engine._to_numpy(self._bias)
            out = out + bias
        return out
