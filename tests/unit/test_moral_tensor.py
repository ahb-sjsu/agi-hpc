# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for MoralTensor multi-rank ethical assessment."""

from __future__ import annotations

import numpy as np
import pytest

from agi.safety.erisml.moral_tensor import (
    DEFAULT_AXIS_NAMES,
    DIMENSION_INDEX,
    MORAL_DIMENSION_NAMES,
    MoralTensor,
    SparseCOO,
)
from agi.safety.erisml.service import MoralVector

# =============================================================================
# CONSTANTS
# =============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_dimension_names_count(self) -> None:
        assert len(MORAL_DIMENSION_NAMES) == 9

    def test_dimension_index_mapping(self) -> None:
        for i, name in enumerate(MORAL_DIMENSION_NAMES):
            assert DIMENSION_INDEX[name] == i

    def test_default_axis_names(self) -> None:
        assert DEFAULT_AXIS_NAMES[1] == ("k",)
        assert DEFAULT_AXIS_NAMES[2] == ("k", "n")
        assert DEFAULT_AXIS_NAMES[3] == ("k", "n", "tau")
        assert DEFAULT_AXIS_NAMES[4] == ("k", "n", "a", "c")
        assert DEFAULT_AXIS_NAMES[5] == ("k", "n", "tau", "a", "s")
        assert DEFAULT_AXIS_NAMES[6] == ("k", "n", "tau", "a", "c", "s")


# =============================================================================
# RANK CREATION AND VALIDATION
# =============================================================================


class TestCreation:
    """Test tensor creation and shape validation."""

    def test_rank1_creation(self) -> None:
        data = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.5])
        t = MoralTensor.from_dense(data)
        assert t.rank == 1
        assert t.shape == (9,)
        assert t.axis_names == ("k",)

    def test_rank2_creation(self) -> None:
        data = np.random.rand(9, 3)
        t = MoralTensor.from_dense(data)
        assert t.rank == 2
        assert t.shape == (9, 3)
        assert t.axis_names == ("k", "n")

    def test_rank3_creation(self) -> None:
        data = np.random.rand(9, 3, 5)
        t = MoralTensor.from_dense(data)
        assert t.rank == 3
        assert t.shape == (9, 3, 5)

    def test_rank4_creation(self) -> None:
        data = np.random.rand(9, 2, 3, 4)
        t = MoralTensor.from_dense(data)
        assert t.rank == 4
        assert t.shape == (9, 2, 3, 4)
        assert t.axis_names == ("k", "n", "a", "c")

    def test_rank5_creation(self) -> None:
        data = np.random.rand(9, 2, 3, 4, 5)
        t = MoralTensor.from_dense(data)
        assert t.rank == 5
        assert t.shape == (9, 2, 3, 4, 5)

    def test_rank6_creation(self) -> None:
        data = np.random.rand(9, 2, 3, 4, 5, 6)
        t = MoralTensor.from_dense(data)
        assert t.rank == 6
        assert t.shape == (9, 2, 3, 4, 5, 6)

    def test_first_dim_must_be_9(self) -> None:
        with pytest.raises(ValueError, match="First dimension must be 9"):
            MoralTensor.from_dense(np.random.rand(5, 3))

    def test_values_must_be_in_range(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MoralTensor.from_dense(np.full((9,), 1.5))

    def test_negative_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MoralTensor.from_dense(np.full((9,), -0.1))


# =============================================================================
# FACTORY METHODS
# =============================================================================


class TestFactoryMethods:
    """Test zeros, ones, from_moral_vector, from_moral_vectors."""

    def test_zeros(self) -> None:
        t = MoralTensor.zeros((9, 3))
        data = t.to_dense()
        # physical_harm should be 1.0 (worst case)
        assert np.all(data[0, :] == 1.0)
        # Others should be 0.0
        for k in range(1, 9):
            assert np.all(data[k, :] == 0.0)

    def test_ones(self) -> None:
        t = MoralTensor.ones((9, 3))
        data = t.to_dense()
        # physical_harm should be 0.0 (best case)
        assert np.all(data[0, :] == 0.0)
        # Others should be 1.0
        for k in range(1, 9):
            assert np.all(data[k, :] == 1.0)

    def test_from_moral_vector(self) -> None:
        mv = MoralVector(
            physical_harm=0.1,
            rights_respect=0.9,
            fairness_equity=0.8,
            autonomy_respect=0.7,
            privacy_protection=0.6,
            societal_environmental=0.5,
            virtue_care=0.4,
            legitimacy_trust=0.3,
            epistemic_quality=0.5,
            veto_flags=["TEST_VETO"],
            reason_codes=["test_reason"],
        )
        t = MoralTensor.from_moral_vector(mv)
        assert t.rank == 1
        assert t.shape == (9,)
        data = t.to_dense()
        assert data[0] == pytest.approx(0.1)
        assert data[1] == pytest.approx(0.9)
        assert "TEST_VETO" in t.veto_flags
        assert "test_reason" in t.reason_codes

    def test_from_moral_vectors(self) -> None:
        mv_a = MoralVector(physical_harm=0.1, rights_respect=0.9)
        mv_b = MoralVector(physical_harm=0.5, rights_respect=0.5)
        t = MoralTensor.from_moral_vectors({"agent": mv_a, "affected": mv_b})
        assert t.rank == 2
        assert t.shape == (9, 2)
        assert t.axis_labels.get("n") == ["agent", "affected"]

    def test_from_moral_vectors_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            MoralTensor.from_moral_vectors({})


# =============================================================================
# SPARSE STORAGE
# =============================================================================


class TestSparseStorage:
    """Test SparseCOO and sparse/dense roundtrip."""

    def test_sparse_roundtrip(self) -> None:
        data = np.zeros((9, 3), dtype=np.float64)
        data[0, 1] = 0.5
        data[3, 2] = 0.8
        t = MoralTensor.from_dense(data)
        sparse = t.to_sparse()
        assert isinstance(sparse, SparseCOO)
        assert sparse.nnz == 2
        dense_back = sparse.to_dense()
        np.testing.assert_allclose(dense_back, data)

    def test_sparse_creation(self) -> None:
        coords = np.array([[0, 0], [3, 1]], dtype=np.int32)
        values = np.array([0.5, 0.8], dtype=np.float64)
        t = MoralTensor.from_sparse(
            coords=coords, values=values, shape=(9, 2), fill_value=0.0
        )
        assert t.is_sparse
        assert t.shape == (9, 2)
        data = t.to_dense()
        assert data[0, 0] == pytest.approx(0.5)
        assert data[3, 1] == pytest.approx(0.8)


# =============================================================================
# SLICING
# =============================================================================


class TestSlicing:
    """Test axis-based slicing operations."""

    def test_slice_party(self) -> None:
        mv_a = MoralVector(physical_harm=0.1)
        mv_b = MoralVector(physical_harm=0.5)
        t = MoralTensor.from_moral_vectors({"a": mv_a, "b": mv_b})
        sliced = t.slice_party("a")
        assert isinstance(sliced, MoralTensor)
        assert sliced.rank == 1
        assert sliced.to_dense()[0] == pytest.approx(0.1)

    def test_slice_party_by_index(self) -> None:
        data = np.random.rand(9, 3)
        t = MoralTensor.from_dense(data)
        sliced = t.slice_party(1)
        assert isinstance(sliced, MoralTensor)
        np.testing.assert_allclose(sliced.to_dense(), data[:, 1])

    def test_slice_dimension(self) -> None:
        data = np.random.rand(9, 3)
        t = MoralTensor.from_dense(data)
        harm = t.slice_dimension("physical_harm")
        np.testing.assert_allclose(harm, data[0, :])

    def test_slice_invalid_dimension(self) -> None:
        t = MoralTensor.from_dense(np.random.rand(9, 3))
        with pytest.raises(ValueError, match="not found"):
            t.slice_dimension("nonexistent")

    def test_slice_no_party_axis(self) -> None:
        t = MoralTensor.from_dense(
            np.random.rand(
                9,
            )
        )
        with pytest.raises(ValueError, match="party axis"):
            t.slice_party(0)


# =============================================================================
# REDUCTION
# =============================================================================


class TestReduction:
    """Test reduce and contract operations."""

    def test_reduce_mean(self) -> None:
        data = np.random.rand(9, 4)
        t = MoralTensor.from_dense(data)
        reduced = t.reduce("n", method="mean")
        assert reduced.rank == 1
        assert reduced.shape == (9,)
        np.testing.assert_allclose(reduced.to_dense(), data.mean(axis=1))

    def test_reduce_max(self) -> None:
        data = np.random.rand(9, 4)
        t = MoralTensor.from_dense(data)
        reduced = t.reduce("n", method="max")
        np.testing.assert_allclose(reduced.to_dense(), data.max(axis=1))

    def test_reduce_min(self) -> None:
        data = np.random.rand(9, 4)
        t = MoralTensor.from_dense(data)
        reduced = t.reduce("n", method="min")
        np.testing.assert_allclose(reduced.to_dense(), data.min(axis=1))

    def test_reduce_invalid_axis(self) -> None:
        t = MoralTensor.from_dense(np.random.rand(9, 3))
        with pytest.raises(ValueError, match="not found"):
            t.reduce("nonexistent")

    def test_reduce_invalid_method(self) -> None:
        t = MoralTensor.from_dense(np.random.rand(9, 3))
        with pytest.raises(ValueError, match="Unknown reduction"):
            t.reduce("n", method="median")

    def test_contract_uniform(self) -> None:
        data = np.random.rand(9, 3)
        t = MoralTensor.from_dense(data)
        contracted = t.contract("n")
        assert contracted.rank == 1
        expected = np.clip(data.mean(axis=1), 0.0, 1.0)
        np.testing.assert_allclose(contracted.to_dense(), expected, atol=1e-10)

    def test_contract_weighted(self) -> None:
        data = np.random.rand(9, 3)
        t = MoralTensor.from_dense(data)
        weights = np.array([0.5, 0.3, 0.2])
        contracted = t.contract("n", weights=weights)
        assert contracted.rank == 1


# =============================================================================
# PROMOTE RANK
# =============================================================================


class TestPromoteRank:
    """Test rank promotion via broadcasting."""

    def test_rank1_to_rank2(self) -> None:
        data = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.5])
        t = MoralTensor.from_dense(data)
        promoted = t.promote_rank(2, axis_sizes={"n": 3})
        assert promoted.rank == 2
        assert promoted.shape == (9, 3)
        for j in range(3):
            np.testing.assert_allclose(promoted.to_dense()[:, j], data)

    def test_rank_must_increase(self) -> None:
        t = MoralTensor.from_dense(np.random.rand(9, 3))
        with pytest.raises(ValueError, match="must be >"):
            t.promote_rank(2)

    def test_rank_cannot_exceed_6(self) -> None:
        t = MoralTensor.from_dense(
            np.random.rand(
                9,
            )
        )
        with pytest.raises(ValueError, match="cannot exceed 6"):
            t.promote_rank(
                7, axis_sizes={"n": 2, "tau": 3, "a": 2, "c": 2, "s": 2, "x": 2}
            )

    def test_missing_axis_size_raises(self) -> None:
        t = MoralTensor.from_dense(
            np.random.rand(
                9,
            )
        )
        with pytest.raises(ValueError, match="Missing size"):
            t.promote_rank(2)


# =============================================================================
# ARITHMETIC
# =============================================================================


class TestArithmetic:
    """Test element-wise arithmetic with clamping."""

    def test_addition_clamped(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.7))
        b = MoralTensor.from_dense(np.full((9,), 0.5))
        result = a + b
        # Should be clamped to 1.0
        np.testing.assert_allclose(result.to_dense(), np.ones(9))

    def test_subtraction_clamped(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.3))
        b = MoralTensor.from_dense(np.full((9,), 0.5))
        result = a - b
        # Should be clamped to 0.0
        np.testing.assert_allclose(result.to_dense(), np.zeros(9))

    def test_multiplication(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5))
        b = MoralTensor.from_dense(np.full((9,), 0.6))
        result = a * b
        np.testing.assert_allclose(result.to_dense(), np.full(9, 0.3))

    def test_scalar_multiplication(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5))
        result = a * 0.5
        np.testing.assert_allclose(result.to_dense(), np.full(9, 0.25))

    def test_division(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.6))
        result = a / 2.0
        np.testing.assert_allclose(result.to_dense(), np.full(9, 0.3))

    def test_division_by_zero(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5))
        result = a / 0.0
        # Should be 1.0 (max)
        np.testing.assert_allclose(result.to_dense(), np.ones(9))

    def test_shape_mismatch_raises(self) -> None:
        a = MoralTensor.from_dense(np.random.rand(9, 2))
        b = MoralTensor.from_dense(np.random.rand(9, 3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            _ = a + b

    def test_veto_flags_merged(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5), veto_flags=["VETO_A"])
        b = MoralTensor.from_dense(np.full((9,), 0.3), veto_flags=["VETO_B"])
        result = a + b
        assert "VETO_A" in result.veto_flags
        assert "VETO_B" in result.veto_flags


# =============================================================================
# COMPARISON
# =============================================================================


class TestComparison:
    """Test dominates and distance."""

    def test_dominates(self) -> None:
        # Better: less harm, more everything else
        better = MoralTensor.ones((9,))
        worse = MoralTensor.zeros((9,))
        assert better.dominates(worse)
        assert not worse.dominates(better)

    def test_self_does_not_dominate(self) -> None:
        t = MoralTensor.from_dense(np.full((9,), 0.5))
        assert not t.dominates(t)

    def test_distance_frobenius(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5))
        b = MoralTensor.from_dense(np.full((9,), 0.6))
        d = a.distance(b, metric="frobenius")
        expected = np.linalg.norm(np.full(9, -0.1))
        assert d == pytest.approx(expected)

    def test_distance_max(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5))
        b = MoralTensor.from_dense(np.full((9,), 0.7))
        d = a.distance(b, metric="max")
        assert d == pytest.approx(0.2)

    def test_distance_mean_abs(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5))
        b = MoralTensor.from_dense(np.full((9,), 0.7))
        d = a.distance(b, metric="mean_abs")
        assert d == pytest.approx(0.2)

    def test_distance_shape_mismatch(self) -> None:
        a = MoralTensor.from_dense(np.random.rand(9, 2))
        b = MoralTensor.from_dense(np.random.rand(9, 3))
        with pytest.raises(ValueError, match="Shape mismatch"):
            a.distance(b)


# =============================================================================
# VETO HANDLING
# =============================================================================


class TestVetoHandling:
    """Test veto flag and location tracking."""

    def test_has_veto(self) -> None:
        t = MoralTensor.from_dense(np.full((9,), 0.5), veto_flags=["HARM"])
        assert t.has_veto()

    def test_no_veto(self) -> None:
        t = MoralTensor.from_dense(np.full((9,), 0.5))
        assert not t.has_veto()

    def test_has_veto_at_global(self) -> None:
        t = MoralTensor.from_dense(
            np.random.rand(9, 3),
            veto_flags=["GLOBAL"],
            veto_locations=[()],
        )
        assert t.has_veto_at(n=0)
        assert t.has_veto_at(n=2)

    def test_has_veto_at_specific(self) -> None:
        t = MoralTensor.from_dense(
            np.random.rand(9, 3),
            veto_flags=["PARTY_HARM"],
            veto_locations=[(1,)],
        )
        assert not t.has_veto_at(n=0)
        assert t.has_veto_at(n=1)
        assert not t.has_veto_at(n=2)


# =============================================================================
# SERIALIZATION
# =============================================================================


class TestSerialization:
    """Test to_dict/from_dict roundtrip."""

    def test_dense_roundtrip(self) -> None:
        data = np.random.rand(9, 3)
        t = MoralTensor.from_dense(data, veto_flags=["TEST"], reason_codes=["code"])
        d = t.to_dict()
        t2 = MoralTensor.from_dict(d)
        assert t == t2
        assert t2.veto_flags == ["TEST"]
        assert t2.reason_codes == ["code"]

    def test_sparse_roundtrip(self) -> None:
        data = np.zeros((9, 3), dtype=np.float64)
        data[0, 1] = 0.5
        data[3, 2] = 0.8
        t = MoralTensor.from_dense(data)
        # Convert to sparse for serialization
        sparse = t.to_sparse()
        t_sparse = MoralTensor.from_sparse(sparse.coords, sparse.values, sparse.shape)
        d = t_sparse.to_dict()
        t2 = MoralTensor.from_dict(d)
        np.testing.assert_allclose(t2.to_dense(), data)


# =============================================================================
# MORAL VECTOR CONVERSION
# =============================================================================


class TestMoralVectorConversion:
    """Test conversion between MoralVector and MoralTensor."""

    def test_to_moral_vector(self) -> None:
        data = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.5])
        t = MoralTensor.from_dense(data)
        mv = t.to_moral_vector()
        assert mv.physical_harm == pytest.approx(0.1)
        assert mv.rights_respect == pytest.approx(0.9)
        assert mv.epistemic_quality == pytest.approx(0.5)

    def test_to_moral_vector_rank2_raises(self) -> None:
        t = MoralTensor.from_dense(np.random.rand(9, 3))
        with pytest.raises(ValueError, match="rank-1"):
            t.to_moral_vector()

    def test_to_vector_mean(self) -> None:
        data = np.random.rand(9, 4)
        t = MoralTensor.from_dense(data)
        mv = t.to_vector(strategy="mean")
        expected = data.mean(axis=1)
        assert mv.physical_harm == pytest.approx(expected[0])

    def test_to_vector_max(self) -> None:
        data = np.random.rand(9, 4)
        t = MoralTensor.from_dense(data)
        mv = t.to_vector(strategy="max")
        expected = data.max(axis=1)
        assert mv.physical_harm == pytest.approx(expected[0])

    def test_mv_to_tensor_roundtrip(self) -> None:
        mv = MoralVector(
            physical_harm=0.2,
            rights_respect=0.8,
            fairness_equity=0.7,
            autonomy_respect=0.6,
            privacy_protection=0.5,
            societal_environmental=0.4,
            virtue_care=0.3,
            legitimacy_trust=0.2,
            epistemic_quality=0.9,
        )
        t = MoralTensor.from_moral_vector(mv)
        mv2 = t.to_moral_vector()
        assert mv2.physical_harm == pytest.approx(mv.physical_harm)
        assert mv2.rights_respect == pytest.approx(mv.rights_respect)
        assert mv2.epistemic_quality == pytest.approx(mv.epistemic_quality)

    def test_mv_to_tensor_method(self) -> None:
        """Test the MoralVector.to_tensor() convenience method."""
        mv = MoralVector(physical_harm=0.3, rights_respect=0.7)
        t = mv.to_tensor()
        assert t.rank == 1
        assert t.to_dense()[0] == pytest.approx(0.3)


# =============================================================================
# REPR AND EQUALITY
# =============================================================================


class TestReprEquality:
    """Test __repr__ and __eq__."""

    def test_repr(self) -> None:
        t = MoralTensor.from_dense(np.random.rand(9, 3))
        s = repr(t)
        assert "rank=2" in s
        assert "shape=(9, 3)" in s

    def test_repr_with_vetoes(self) -> None:
        t = MoralTensor.from_dense(
            np.random.rand(
                9,
            ),
            veto_flags=["V1", "V2"],
        )
        s = repr(t)
        assert "vetoes=2" in s

    def test_equality(self) -> None:
        data = np.random.rand(9, 3)
        a = MoralTensor.from_dense(data.copy())
        b = MoralTensor.from_dense(data.copy())
        assert a == b

    def test_inequality_shape(self) -> None:
        a = MoralTensor.from_dense(np.random.rand(9, 2))
        b = MoralTensor.from_dense(np.random.rand(9, 3))
        assert a != b

    def test_inequality_data(self) -> None:
        a = MoralTensor.from_dense(np.full((9,), 0.5))
        b = MoralTensor.from_dense(np.full((9,), 0.6))
        assert a != b

    def test_not_equal_to_other_type(self) -> None:
        t = MoralTensor.from_dense(
            np.random.rand(
                9,
            )
        )
        assert t != "not a tensor"
