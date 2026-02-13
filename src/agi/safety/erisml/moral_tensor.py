# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
MoralTensor: Multi-rank tensor representation of ethical assessment.

DEME V3 introduces multi-rank moral tensors that extend MoralVector for
multi-agent ethics. This provides:

1. Rank-1 (9,): V2-compatible MoralVector equivalent
2. Rank-2 (9, n): Per-party distributional ethics
3. Rank-3 (9, n, tau): Temporal evolution
4. Rank-4 (9, n, a, c): Coalition actions
5. Rank-5 (9, n, tau, s): Uncertainty samples
6. Rank-6 (9, n, tau, a, c, s): Full multi-agent context

The 9 ethical dimensions are:
    physical_harm, rights_respect, fairness_equity, autonomy_respect,
    privacy_protection, societal_environmental, virtue_care,
    legitimacy_trust, epistemic_quality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

if TYPE_CHECKING:
    from agi.safety.erisml.service import MoralVector

logger = logging.getLogger(__name__)

# Standard dimension names (from Nine Dimensions paper)
MORAL_DIMENSION_NAMES: Tuple[str, ...] = (
    "physical_harm",  # 0: Consequences/Welfare
    "rights_respect",  # 1: Rights/Duties
    "fairness_equity",  # 2: Justice/Fairness
    "autonomy_respect",  # 3: Autonomy/Agency
    "privacy_protection",  # 4: Privacy/Data
    "societal_environmental",  # 5: Societal/Environmental
    "virtue_care",  # 6: Virtue/Care
    "legitimacy_trust",  # 7: Procedural Legitimacy
    "epistemic_quality",  # 8: Epistemic Status
)

# Dimension index mapping
DIMENSION_INDEX: Dict[str, int] = {
    name: i for i, name in enumerate(MORAL_DIMENSION_NAMES)
}

# Standard axis names by position for each rank
DEFAULT_AXIS_NAMES: Dict[int, Tuple[str, ...]] = {
    1: ("k",),
    2: ("k", "n"),
    3: ("k", "n", "tau"),
    4: ("k", "n", "a", "c"),
    5: ("k", "n", "tau", "a", "s"),
    6: ("k", "n", "tau", "a", "c", "s"),
}


def _require_numpy() -> None:
    """Raise RuntimeError if numpy is not available."""
    if not HAS_NUMPY:
        raise RuntimeError(
            "numpy is required for MoralTensor operations. "
            "Install with: pip install numpy"
        )


@dataclass
class SparseCOO:
    """
    COO sparse tensor storage for memory efficiency.

    Uses coordinate format (COO) for efficient storage of sparse tensors
    where most values are a constant fill value.
    """

    coords: Any  # np.ndarray — (nnz, rank) array of coordinates
    values: Any  # np.ndarray — (nnz,) array of values
    shape: Tuple[int, ...]
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        """Validate sparse tensor structure."""
        _require_numpy()
        if self.coords.ndim != 2:
            raise ValueError(f"coords must be 2D, got shape {self.coords.shape}")
        if self.values.ndim != 1:
            raise ValueError(f"values must be 1D, got shape {self.values.shape}")
        if len(self.coords) != len(self.values):
            raise ValueError(
                f"coords and values must have same length, "
                f"got {len(self.coords)} and {len(self.values)}"
            )
        if self.coords.shape[1] != len(self.shape):
            raise ValueError(
                f"coords columns ({self.coords.shape[1]}) must match "
                f"rank ({len(self.shape)})"
            )

    @property
    def nnz(self) -> int:
        """Number of non-fill values."""
        return len(self.values)

    @property
    def rank(self) -> int:
        """Tensor rank."""
        return len(self.shape)

    def to_dense(self) -> Any:
        """Convert to dense NumPy array."""
        dense = np.full(self.shape, self.fill_value, dtype=np.float64)
        if self.nnz > 0:
            idx = tuple(self.coords[:, i] for i in range(self.rank))
            dense[idx] = self.values
        return dense

    @classmethod
    def from_dense(
        cls, data: Any, fill_value: float = 0.0, tol: float = 1e-10
    ) -> SparseCOO:
        """Create sparse tensor from dense array."""
        _require_numpy()
        if fill_value == 0.0:
            mask = np.abs(data) > tol
        else:
            mask = np.abs(data - fill_value) > tol

        coords = np.argwhere(mask)
        values = data[mask]

        return cls(
            coords=coords.astype(np.int32),
            values=values.astype(np.float64),
            shape=data.shape,
            fill_value=fill_value,
        )


@dataclass
class MoralTensor:
    """
    Multi-rank tensor for ethical assessment (ranks 1-6).

    Provides a unified representation for single-agent and multi-agent
    ethical assessments with support for temporal evolution, coalitions,
    and uncertainty quantification.
    """

    _data: Any  # Union[np.ndarray, SparseCOO]
    shape: Tuple[int, ...]
    rank: int
    axis_names: Tuple[str, ...] = field(default_factory=lambda: ("k",))
    axis_labels: Dict[str, List[str]] = field(default_factory=dict)
    veto_flags: List[str] = field(default_factory=list)
    veto_locations: List[Tuple[int, ...]] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)
    is_sparse: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tensor structure after initialization."""
        _require_numpy()
        self._validate_rank()
        self._validate_shape()
        self._validate_first_dimension()
        self._validate_bounds()
        self._validate_axis_names()
        self._validate_veto_locations()

    def _validate_rank(self) -> None:
        if not 1 <= self.rank <= 6:
            raise ValueError(f"Rank must be 1-6, got {self.rank}")
        if self.rank != len(self.shape):
            raise ValueError(
                f"Rank ({self.rank}) must match shape dimensions ({len(self.shape)})"
            )

    def _validate_shape(self) -> None:
        if len(self.shape) == 0:
            raise ValueError("Shape cannot be empty")
        for i, dim in enumerate(self.shape):
            if dim <= 0:
                raise ValueError(f"Dimension {i} must be positive, got {dim}")

    def _validate_first_dimension(self) -> None:
        if self.shape[0] != 9:
            raise ValueError(
                f"First dimension must be 9 (moral dimensions), got {self.shape[0]}"
            )

    def _validate_bounds(self) -> None:
        data = self.to_dense()
        if np.any(data < 0.0) or np.any(data > 1.0):
            raise ValueError("All tensor values must be in [0, 1]")

    def _validate_axis_names(self) -> None:
        if len(self.axis_names) != self.rank:
            raise ValueError(
                f"axis_names length ({len(self.axis_names)}) must match rank ({self.rank})"
            )

    def _validate_veto_locations(self) -> None:
        non_k_shape = self.shape[1:]
        for loc in self.veto_locations:
            if len(loc) > len(non_k_shape):
                raise ValueError(
                    f"Veto location {loc} has more dimensions than "
                    f"non-k axes ({len(non_k_shape)})"
                )
            for i, idx in enumerate(loc):
                if idx < 0 or idx >= non_k_shape[i]:
                    raise ValueError(
                        f"Veto location {loc} index {idx} out of bounds "
                        f"for axis {i + 1} (size {non_k_shape[i]})"
                    )

    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------

    def to_dense(self) -> Any:
        """Get dense NumPy array representation."""
        if isinstance(self._data, SparseCOO):
            return self._data.to_dense()
        return np.array(self._data, dtype=np.float64)

    def to_sparse(self, fill_value: float = 0.0) -> SparseCOO:
        """Get sparse COO representation."""
        if isinstance(self._data, SparseCOO):
            return self._data
        return SparseCOO.from_dense(self._data, fill_value=fill_value)

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_dense(
        cls,
        data: Any,
        axis_names: Optional[Tuple[str, ...]] = None,
        axis_labels: Optional[Dict[str, List[str]]] = None,
        veto_flags: Optional[List[str]] = None,
        veto_locations: Optional[List[Tuple[int, ...]]] = None,
        reason_codes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extensions: Optional[Dict[str, Any]] = None,
    ) -> MoralTensor:
        """Create MoralTensor from dense NumPy array."""
        _require_numpy()
        data = np.asarray(data, dtype=np.float64)
        rank = data.ndim
        shape = data.shape

        if axis_names is None:
            axis_names = DEFAULT_AXIS_NAMES.get(
                rank, tuple(f"dim{i}" for i in range(rank))
            )

        return cls(
            _data=data,
            shape=shape,
            rank=rank,
            axis_names=axis_names,
            axis_labels=axis_labels or {},
            veto_flags=veto_flags or [],
            veto_locations=veto_locations or [],
            reason_codes=reason_codes or [],
            is_sparse=False,
            metadata=metadata or {},
            extensions=extensions or {},
        )

    @classmethod
    def from_sparse(
        cls,
        coords: Any,
        values: Any,
        shape: Tuple[int, ...],
        fill_value: float = 0.0,
        axis_names: Optional[Tuple[str, ...]] = None,
        axis_labels: Optional[Dict[str, List[str]]] = None,
        veto_flags: Optional[List[str]] = None,
        veto_locations: Optional[List[Tuple[int, ...]]] = None,
        reason_codes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        extensions: Optional[Dict[str, Any]] = None,
    ) -> MoralTensor:
        """Create MoralTensor from sparse COO format."""
        _require_numpy()
        sparse_data = SparseCOO(
            coords=np.asarray(coords, dtype=np.int32),
            values=np.asarray(values, dtype=np.float64),
            shape=shape,
            fill_value=fill_value,
        )
        rank = len(shape)

        if axis_names is None:
            axis_names = DEFAULT_AXIS_NAMES.get(
                rank, tuple(f"dim{i}" for i in range(rank))
            )

        return cls(
            _data=sparse_data,
            shape=shape,
            rank=rank,
            axis_names=axis_names,
            axis_labels=axis_labels or {},
            veto_flags=veto_flags or [],
            veto_locations=veto_locations or [],
            reason_codes=reason_codes or [],
            is_sparse=True,
            metadata=metadata or {},
            extensions=extensions or {},
        )

    @classmethod
    def from_moral_vector(cls, vec: MoralVector) -> MoralTensor:
        """
        Create rank-1 tensor from MoralVector (backward compatibility).

        Args:
            vec: AGI-HPC MoralVector instance from erisml service.

        Returns:
            Rank-1 MoralTensor equivalent to the MoralVector.
        """
        _require_numpy()
        data = np.array(
            [
                vec.physical_harm,
                vec.rights_respect,
                vec.fairness_equity,
                vec.autonomy_respect,
                vec.privacy_protection,
                vec.societal_environmental,
                vec.virtue_care,
                vec.legitimacy_trust,
                vec.epistemic_quality,
            ],
            dtype=np.float64,
        )

        return cls(
            _data=data,
            shape=(9,),
            rank=1,
            axis_names=("k",),
            axis_labels={"k": list(MORAL_DIMENSION_NAMES)},
            veto_flags=list(vec.veto_flags),
            veto_locations=[],
            reason_codes=list(vec.reason_codes),
            is_sparse=False,
            metadata={},
            extensions={},
        )

    @classmethod
    def from_moral_vectors(
        cls,
        vectors: Dict[str, MoralVector],
        axis_name: str = "n",
    ) -> MoralTensor:
        """
        Stack multiple MoralVectors into a rank-2 tensor.

        Args:
            vectors: Dict mapping party/entity names to MoralVectors.
            axis_name: Name for the stacked axis (default "n").

        Returns:
            Rank-2 MoralTensor of shape (9, n).
        """
        _require_numpy()
        if not vectors:
            raise ValueError("vectors dict cannot be empty")

        names = list(vectors.keys())
        n = len(names)

        data = np.zeros((9, n), dtype=np.float64)
        all_veto_flags: List[str] = []
        all_veto_locations: List[Tuple[int, ...]] = []
        all_reason_codes: List[str] = []

        for j, name in enumerate(names):
            vec = vectors[name]
            data[0, j] = vec.physical_harm
            data[1, j] = vec.rights_respect
            data[2, j] = vec.fairness_equity
            data[3, j] = vec.autonomy_respect
            data[4, j] = vec.privacy_protection
            data[5, j] = vec.societal_environmental
            data[6, j] = vec.virtue_care
            data[7, j] = vec.legitimacy_trust
            data[8, j] = vec.epistemic_quality

            for veto in vec.veto_flags:
                if veto not in all_veto_flags:
                    all_veto_flags.append(veto)
                all_veto_locations.append((j,))

            for code in vec.reason_codes:
                if code not in all_reason_codes:
                    all_reason_codes.append(code)

        return cls(
            _data=data,
            shape=(9, n),
            rank=2,
            axis_names=("k", axis_name),
            axis_labels={"k": list(MORAL_DIMENSION_NAMES), axis_name: names},
            veto_flags=all_veto_flags,
            veto_locations=all_veto_locations,
            reason_codes=all_reason_codes,
            is_sparse=False,
            metadata={},
            extensions={},
        )

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> MoralTensor:
        """Create worst-case tensor (harm=1, others=0)."""
        _require_numpy()
        if shape[0] != 9:
            raise ValueError(f"First dimension must be 9, got {shape[0]}")

        data = np.zeros(shape, dtype=np.float64)
        data[0, ...] = 1.0  # physical_harm = worst case

        rank = len(shape)
        axis_names = DEFAULT_AXIS_NAMES.get(rank, tuple(f"dim{i}" for i in range(rank)))

        return cls(
            _data=data,
            shape=shape,
            rank=rank,
            axis_names=axis_names,
            axis_labels={},
            veto_flags=[],
            veto_locations=[],
            reason_codes=[],
            is_sparse=False,
            metadata={},
            extensions={},
        )

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> MoralTensor:
        """Create ideal tensor (harm=0, others=1)."""
        _require_numpy()
        if shape[0] != 9:
            raise ValueError(f"First dimension must be 9, got {shape[0]}")

        data = np.ones(shape, dtype=np.float64)
        data[0, ...] = 0.0  # physical_harm = best case

        rank = len(shape)
        axis_names = DEFAULT_AXIS_NAMES.get(rank, tuple(f"dim{i}" for i in range(rank)))

        return cls(
            _data=data,
            shape=shape,
            rank=rank,
            axis_names=axis_names,
            axis_labels={},
            veto_flags=[],
            veto_locations=[],
            reason_codes=[],
            is_sparse=False,
            metadata={},
            extensions={},
        )

    # -------------------------------------------------------------------------
    # Conversion to MoralVector
    # -------------------------------------------------------------------------

    def to_moral_vector(self) -> MoralVector:
        """
        Convert rank-1 tensor to MoralVector.

        Raises:
            ValueError: If tensor rank > 1.
        """
        if self.rank != 1:
            raise ValueError(
                f"Can only convert rank-1 tensor to MoralVector, got rank {self.rank}"
            )

        from agi.safety.erisml.service import MoralVector

        data = self.to_dense()
        return MoralVector(
            physical_harm=float(data[0]),
            rights_respect=float(data[1]),
            fairness_equity=float(data[2]),
            autonomy_respect=float(data[3]),
            privacy_protection=float(data[4]),
            societal_environmental=float(data[5]),
            virtue_care=float(data[6]),
            legitimacy_trust=float(data[7]),
            epistemic_quality=float(data[8]),
            veto_flags=list(self.veto_flags),
            reason_codes=list(self.reason_codes),
        )

    # -------------------------------------------------------------------------
    # Indexing and Slicing
    # -------------------------------------------------------------------------

    def slice_axis(
        self, axis: str, index: Union[int, slice]
    ) -> Union[float, MoralTensor]:
        """
        Slice tensor by named axis.

        Args:
            axis: Axis name to slice.
            index: Index or slice for that axis.

        Returns:
            Sliced MoralTensor (or float for scalar result).
        """
        if axis not in self.axis_names:
            raise ValueError(f"Axis '{axis}' not found in {self.axis_names}")

        axis_idx = self.axis_names.index(axis)
        data = self.to_dense()

        slices: List[Any] = [slice(None)] * self.rank
        slices[axis_idx] = index
        result = data[tuple(slices)]

        if isinstance(index, int):
            new_axis_names = tuple(
                n for i, n in enumerate(self.axis_names) if i != axis_idx
            )
        else:
            new_axis_names = self.axis_names

        if result.ndim == 0:
            return float(result)

        return MoralTensor.from_dense(
            result,
            axis_names=new_axis_names if new_axis_names else None,
            veto_flags=list(self.veto_flags),
            reason_codes=list(self.reason_codes),
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    def slice_party(self, index: Union[int, str]) -> Union[float, MoralTensor]:
        """Slice tensor by party index or label."""
        if "n" not in self.axis_names:
            raise ValueError("Tensor does not have party axis 'n'")

        if isinstance(index, str):
            labels = self.axis_labels.get("n", [])
            if index not in labels:
                raise ValueError(f"Party label '{index}' not found in {labels}")
            idx = labels.index(index)
        else:
            idx = index

        return self.slice_axis("n", idx)

    def slice_time(self, index: Union[int, slice, str]) -> Union[float, MoralTensor]:
        """Slice tensor by time step."""
        if "tau" not in self.axis_names:
            raise ValueError("Tensor does not have time axis 'tau'")

        if isinstance(index, str):
            labels = self.axis_labels.get("tau", [])
            if index not in labels:
                raise ValueError(f"Time label '{index}' not found in {labels}")
            idx: Union[int, slice] = labels.index(index)
        else:
            idx = index

        return self.slice_axis("tau", idx)

    def slice_dimension(self, dim_name: str) -> Any:
        """
        Extract values for a single ethical dimension.

        Returns a numpy array (not a MoralTensor) since the result no
        longer has the 9 ethical dimensions as the first axis.
        """
        if dim_name not in DIMENSION_INDEX:
            raise ValueError(
                f"Dimension '{dim_name}' not found. "
                f"Valid: {list(DIMENSION_INDEX.keys())}"
            )

        idx = DIMENSION_INDEX[dim_name]
        data = self.to_dense()
        return data[idx, ...]

    # -------------------------------------------------------------------------
    # Reduction Operations
    # -------------------------------------------------------------------------

    def reduce(
        self,
        axis: str,
        method: str = "mean",
        keepdims: bool = False,
    ) -> MoralTensor:
        """
        Reduce tensor along named axis.

        Args:
            axis: Axis name to reduce.
            method: Reduction method - "mean", "sum", "min", "max".
            keepdims: Whether to keep reduced dimension.
        """
        if axis not in self.axis_names:
            raise ValueError(f"Axis '{axis}' not found in {self.axis_names}")

        axis_idx = self.axis_names.index(axis)
        data = self.to_dense()

        if method == "mean":
            result = np.mean(data, axis=axis_idx, keepdims=keepdims)
        elif method == "sum":
            result = np.clip(np.sum(data, axis=axis_idx, keepdims=keepdims), 0.0, 1.0)
        elif method == "min":
            result = np.min(data, axis=axis_idx, keepdims=keepdims)
        elif method == "max":
            result = np.max(data, axis=axis_idx, keepdims=keepdims)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        if keepdims:
            new_axis_names = self.axis_names
        else:
            new_axis_names = tuple(
                n for i, n in enumerate(self.axis_names) if i != axis_idx
            )

        return MoralTensor.from_dense(
            result,
            axis_names=new_axis_names if new_axis_names else None,
            veto_flags=list(self.veto_flags),
            reason_codes=list(self.reason_codes),
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    def contract(
        self,
        axis: str,
        weights: Optional[Any] = None,
        normalize: bool = True,
    ) -> MoralTensor:
        """
        Contract tensor along axis using optional weights.

        Args:
            axis: Named axis to contract.
            weights: Weight array matching axis dimension (default: uniform).
            normalize: If True, normalize weights to sum to 1.
        """
        if axis not in self.axis_names:
            raise ValueError(f"Axis '{axis}' not found in {self.axis_names}")

        axis_idx = self.axis_names.index(axis)
        data = self.to_dense()
        axis_size = data.shape[axis_idx]

        if weights is None:
            w = np.ones(axis_size, dtype=np.float64) / axis_size
        else:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != axis_size:
                raise ValueError(
                    f"Weights length ({len(w)}) must match axis size ({axis_size})"
                )
            if normalize:
                w_sum = w.sum()
                if w_sum > 0:
                    w = w / w_sum
                else:
                    w = np.ones(axis_size, dtype=np.float64) / axis_size

        result = np.tensordot(data, w, axes=([axis_idx], [0]))
        result = np.clip(result, 0.0, 1.0)

        new_axis_names = tuple(
            n for i, n in enumerate(self.axis_names) if i != axis_idx
        )
        new_axis_labels = {k: v for k, v in self.axis_labels.items() if k != axis}

        return MoralTensor.from_dense(
            result,
            axis_names=new_axis_names if new_axis_names else None,
            axis_labels=new_axis_labels,
            veto_flags=list(self.veto_flags),
            reason_codes=list(self.reason_codes),
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    # -------------------------------------------------------------------------
    # Conversion Operations
    # -------------------------------------------------------------------------

    def to_vector(
        self,
        strategy: str = "mean",
        weights: Optional[Dict[str, Any]] = None,
        party_idx: Optional[int] = None,
    ) -> MoralVector:
        """
        Collapse tensor to MoralVector using specified strategy.

        Args:
            strategy: "mean", "max", "min", "weighted", or "party".
            weights: Dict mapping axis names to weight arrays (for "weighted").
            party_idx: Party index for "party" strategy.
        """
        from agi.safety.erisml.service import MoralVector

        if self.rank == 1:
            return self.to_moral_vector()

        data = self.to_dense()

        if strategy == "mean":
            result = data
            for _ in range(1, self.rank):
                result = np.mean(result, axis=-1)
        elif strategy == "max":
            result = data
            for _ in range(1, self.rank):
                result = np.max(result, axis=-1)
        elif strategy == "min":
            result = data
            for _ in range(1, self.rank):
                result = np.min(result, axis=-1)
        elif strategy == "weighted":
            if weights is None:
                raise ValueError("'weighted' strategy requires weights dict")
            tensor = self
            for axis_name in reversed(list(self.axis_names[1:])):
                w = weights.get(axis_name)
                tensor = tensor.contract(axis_name, weights=w)
            result = tensor.to_dense()
        elif strategy == "party":
            if party_idx is None:
                raise ValueError("'party' strategy requires party_idx")
            if "n" not in self.axis_names:
                raise ValueError("Tensor does not have party axis 'n'")
            tensor = self.slice_party(party_idx)
            if isinstance(tensor, (int, float)):
                raise ValueError("Cannot convert scalar to MoralVector")
            if tensor.rank == 1:
                return tensor.to_moral_vector()
            return tensor.to_vector(strategy="mean")
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                "Valid: 'mean', 'max', 'min', 'weighted', 'party'"
            )

        return MoralVector(
            physical_harm=float(result[0]),
            rights_respect=float(result[1]),
            fairness_equity=float(result[2]),
            autonomy_respect=float(result[3]),
            privacy_protection=float(result[4]),
            societal_environmental=float(result[5]),
            virtue_care=float(result[6]),
            legitimacy_trust=float(result[7]),
            epistemic_quality=float(result[8]),
            veto_flags=list(self.veto_flags),
            reason_codes=list(self.reason_codes),
        )

    def promote_rank(
        self,
        target_rank: int,
        axis_sizes: Optional[Dict[str, int]] = None,
    ) -> MoralTensor:
        """
        Expand tensor to higher rank by adding dimensions.

        New dimensions are added by broadcasting (replicating) values.

        Args:
            target_rank: Target rank (must be > current rank, max 6).
            axis_sizes: Sizes for new axes {axis_name: size}.
        """
        if target_rank <= self.rank:
            raise ValueError(
                f"Target rank ({target_rank}) must be > current rank ({self.rank})"
            )
        if target_rank > 6:
            raise ValueError(f"Target rank cannot exceed 6, got {target_rank}")

        target_axis_names = DEFAULT_AXIS_NAMES.get(
            target_rank, tuple(f"dim{i}" for i in range(target_rank))
        )

        new_axes = [name for name in target_axis_names if name not in self.axis_names]

        if axis_sizes is None:
            axis_sizes = {}

        for axis in new_axes:
            if axis not in axis_sizes:
                raise ValueError(f"Missing size for new axis '{axis}' in axis_sizes")

        data = self.to_dense()
        new_shape: List[int] = []

        old_axis_idx = 0
        for name in target_axis_names:
            if name in self.axis_names:
                new_shape.append(self.shape[old_axis_idx])
                old_axis_idx += 1
            else:
                new_shape.append(axis_sizes[name])

        reshaped = data
        for i, name in enumerate(target_axis_names):
            if name not in self.axis_names:
                reshaped = np.expand_dims(reshaped, axis=i)

        result = np.broadcast_to(reshaped, tuple(new_shape)).copy()

        new_axis_labels = dict(self.axis_labels)
        for axis in new_axes:
            new_axis_labels[axis] = []

        return MoralTensor.from_dense(
            result,
            axis_names=target_axis_names,
            axis_labels=new_axis_labels,
            veto_flags=list(self.veto_flags),
            reason_codes=list(self.reason_codes),
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    # -------------------------------------------------------------------------
    # Arithmetic Operations
    # -------------------------------------------------------------------------

    def __add__(self, other: Union[MoralTensor, float]) -> MoralTensor:
        """Element-wise addition with clamping to [0, 1]."""
        data = self.to_dense()

        if isinstance(other, MoralTensor):
            other_data = other.to_dense()
            if data.shape != other_data.shape:
                raise ValueError(f"Shape mismatch: {data.shape} vs {other_data.shape}")
            result = np.clip(data + other_data, 0.0, 1.0)
            merged_vetoes = list(set(self.veto_flags) | set(other.veto_flags))
            merged_reasons = list(set(self.reason_codes) | set(other.reason_codes))
        else:
            result = np.clip(data + float(other), 0.0, 1.0)
            merged_vetoes = list(self.veto_flags)
            merged_reasons = list(self.reason_codes)

        return MoralTensor.from_dense(
            result,
            axis_names=self.axis_names,
            axis_labels=dict(self.axis_labels),
            veto_flags=merged_vetoes,
            reason_codes=merged_reasons,
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    def __radd__(self, other: float) -> MoralTensor:
        return self.__add__(other)

    def __mul__(self, other: Union[MoralTensor, float]) -> MoralTensor:
        """Element-wise or scalar multiplication with clamping to [0, 1]."""
        data = self.to_dense()

        if isinstance(other, MoralTensor):
            other_data = other.to_dense()
            if data.shape != other_data.shape:
                raise ValueError(f"Shape mismatch: {data.shape} vs {other_data.shape}")
            result = np.clip(data * other_data, 0.0, 1.0)
            merged_vetoes = list(set(self.veto_flags) | set(other.veto_flags))
            merged_reasons = list(set(self.reason_codes) | set(other.reason_codes))
        else:
            result = np.clip(data * float(other), 0.0, 1.0)
            merged_vetoes = list(self.veto_flags)
            merged_reasons = list(self.reason_codes)

        return MoralTensor.from_dense(
            result,
            axis_names=self.axis_names,
            axis_labels=dict(self.axis_labels),
            veto_flags=merged_vetoes,
            reason_codes=merged_reasons,
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    def __rmul__(self, other: float) -> MoralTensor:
        return self.__mul__(other)

    def __sub__(self, other: Union[MoralTensor, float]) -> MoralTensor:
        """Element-wise subtraction with clamping to [0, 1]."""
        data = self.to_dense()

        if isinstance(other, MoralTensor):
            other_data = other.to_dense()
            if data.shape != other_data.shape:
                raise ValueError(f"Shape mismatch: {data.shape} vs {other_data.shape}")
            result = np.clip(data - other_data, 0.0, 1.0)
            merged_vetoes = list(set(self.veto_flags) | set(other.veto_flags))
            merged_reasons = list(set(self.reason_codes) | set(other.reason_codes))
        else:
            result = np.clip(data - float(other), 0.0, 1.0)
            merged_vetoes = list(self.veto_flags)
            merged_reasons = list(self.reason_codes)

        return MoralTensor.from_dense(
            result,
            axis_names=self.axis_names,
            axis_labels=dict(self.axis_labels),
            veto_flags=merged_vetoes,
            reason_codes=merged_reasons,
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    def __truediv__(self, other: Union[MoralTensor, float]) -> MoralTensor:
        """Element-wise division with clamping to [0, 1]."""
        data = self.to_dense()

        if isinstance(other, MoralTensor):
            other_data = other.to_dense()
            if data.shape != other_data.shape:
                raise ValueError(f"Shape mismatch: {data.shape} vs {other_data.shape}")
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(
                    np.abs(other_data) < 1e-10,
                    1.0,
                    data / other_data,
                )
            result = np.clip(result, 0.0, 1.0)
            merged_vetoes = list(set(self.veto_flags) | set(other.veto_flags))
            merged_reasons = list(set(self.reason_codes) | set(other.reason_codes))
        else:
            divisor = float(other)
            if abs(divisor) < 1e-10:
                result = np.ones_like(data)
            else:
                result = np.clip(data / divisor, 0.0, 1.0)
            merged_vetoes = list(self.veto_flags)
            merged_reasons = list(self.reason_codes)

        return MoralTensor.from_dense(
            result,
            axis_names=self.axis_names,
            axis_labels=dict(self.axis_labels),
            veto_flags=merged_vetoes,
            reason_codes=merged_reasons,
            metadata=dict(self.metadata),
            extensions=dict(self.extensions),
        )

    # -------------------------------------------------------------------------
    # Comparison Operations
    # -------------------------------------------------------------------------

    def dominates(self, other: MoralTensor) -> bool:
        """
        Check if this tensor Pareto-dominates another.

        Note: physical_harm (dim 0) is inverted (lower is better).
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        self_data = self.to_dense()
        other_data = other.to_dense()

        harm_self = self_data[0, ...]
        harm_other = other_data[0, ...]

        if np.any(harm_self > harm_other):
            return False

        for k in range(1, 9):
            if np.any(self_data[k, ...] < other_data[k, ...]):
                return False

        harm_strictly_better = bool(np.any(harm_self < harm_other))
        other_strictly_better = any(
            bool(np.any(self_data[k, ...] > other_data[k, ...])) for k in range(1, 9)
        )

        return harm_strictly_better or other_strictly_better

    def distance(
        self,
        other: MoralTensor,
        metric: str = "frobenius",
    ) -> float:
        """
        Compute distance to another MoralTensor.

        Args:
            other: MoralTensor to compare against.
            metric: "frobenius", "euclidean", "max", or "mean_abs".
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        diff = self.to_dense() - other.to_dense()

        if metric in ("frobenius", "euclidean"):
            return float(np.linalg.norm(diff))
        elif metric == "max":
            return float(np.max(np.abs(diff)))
        elif metric == "mean_abs":
            return float(np.mean(np.abs(diff)))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # -------------------------------------------------------------------------
    # Veto Handling
    # -------------------------------------------------------------------------

    def has_veto(self) -> bool:
        """Check if any veto flags are set."""
        return len(self.veto_flags) > 0

    def has_veto_at(self, **coords: int) -> bool:
        """
        Check if a veto applies at specific coordinates.

        Args:
            **coords: Named coordinates (e.g., n=2, tau=5).
        """
        if not self.veto_flags:
            return False

        target: List[Optional[int]] = []
        for axis_name in self.axis_names[1:]:
            if axis_name in coords:
                target.append(coords[axis_name])
            else:
                target.append(None)

        for loc in self.veto_locations:
            if not loc:
                return True

            matches = True
            for i, (loc_idx, tgt_idx) in enumerate(zip(loc, target, strict=False)):
                if tgt_idx is not None and loc_idx != tgt_idx:
                    matches = False
                    break
            if matches:
                return True

        if self.veto_flags and not self.veto_locations:
            return True

        return False

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        result: Dict[str, Any] = {
            "version": "3.0.0",
            "shape": list(self.shape),
            "rank": self.rank,
            "axis_names": list(self.axis_names),
            "axis_labels": self.axis_labels,
            "veto_flags": self.veto_flags,
            "veto_locations": [list(loc) for loc in self.veto_locations],
            "reason_codes": self.reason_codes,
            "is_sparse": self.is_sparse,
            "metadata": self.metadata,
            "extensions": self.extensions,
        }

        if self.is_sparse:
            sparse = self.to_sparse()
            result["sparse_coords"] = sparse.coords.tolist()
            result["sparse_values"] = sparse.values.tolist()
            result["fill_value"] = sparse.fill_value
        else:
            result["data"] = self.to_dense().tolist()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MoralTensor:
        """Deserialize from dict."""
        _require_numpy()
        shape = tuple(data["shape"])
        axis_names = tuple(data["axis_names"])
        axis_labels = data.get("axis_labels", {})
        veto_flags = data.get("veto_flags", [])
        veto_locations = [tuple(loc) for loc in data.get("veto_locations", [])]
        reason_codes = data.get("reason_codes", [])
        is_sparse = data.get("is_sparse", False)
        metadata = data.get("metadata", {})
        extensions = data.get("extensions", {})

        if is_sparse:
            return cls.from_sparse(
                coords=np.array(data["sparse_coords"], dtype=np.int32),
                values=np.array(data["sparse_values"], dtype=np.float64),
                shape=shape,
                fill_value=data.get("fill_value", 0.0),
                axis_names=axis_names,
                axis_labels=axis_labels,
                veto_flags=veto_flags,
                veto_locations=veto_locations,
                reason_codes=reason_codes,
                metadata=metadata,
                extensions=extensions,
            )
        else:
            return cls.from_dense(
                data=np.array(data["data"], dtype=np.float64),
                axis_names=axis_names,
                axis_labels=axis_labels,
                veto_flags=veto_flags,
                veto_locations=veto_locations,
                reason_codes=reason_codes,
                metadata=metadata,
                extensions=extensions,
            )

    # -------------------------------------------------------------------------
    # Special Methods
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        sparse_str = ", sparse" if self.is_sparse else ""
        veto_str = f", vetoes={len(self.veto_flags)}" if self.veto_flags else ""
        return (
            f"MoralTensor(rank={self.rank}, shape={self.shape}{sparse_str}{veto_str})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MoralTensor):
            return False
        if self.shape != other.shape:
            return False
        if not np.allclose(self.to_dense(), other.to_dense(), rtol=1e-9, atol=1e-9):
            return False
        if set(self.veto_flags) != set(other.veto_flags):
            return False
        return True


__all__ = [
    "MoralTensor",
    "SparseCOO",
    "MORAL_DIMENSION_NAMES",
    "DIMENSION_INDEX",
    "DEFAULT_AXIS_NAMES",
    "HAS_NUMPY",
]
