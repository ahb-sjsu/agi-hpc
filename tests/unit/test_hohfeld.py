# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for D4 dihedral group and Hohfeldian normative positions."""

from __future__ import annotations

import pytest

from agi.safety.erisml.hohfeld import (
    D4Element,
    HohfeldianState,
    HohfeldianVerdict,
    compute_bond_index,
    compute_wilson_observable,
    correlative,
    d4_apply_to_state,
    d4_inverse,
    d4_multiply,
    get_klein_four_subgroup,
    is_in_klein_four,
    negation,
    requires_nonabelian_structure,
)

ALL_ELEMENTS = list(D4Element)
ALL_STATES = list(HohfeldianState)


# =============================================================================
# D4 GROUP AXIOMS
# =============================================================================


class TestD4GroupAxioms:
    """Verify D4 satisfies group axioms."""

    def test_closure(self) -> None:
        """Product of any two elements is in D4."""
        for a in ALL_ELEMENTS:
            for b in ALL_ELEMENTS:
                result = d4_multiply(a, b)
                assert result in ALL_ELEMENTS

    def test_identity(self) -> None:
        """e is the identity element."""
        for g in ALL_ELEMENTS:
            assert d4_multiply(D4Element.E, g) == g
            assert d4_multiply(g, D4Element.E) == g

    def test_inverse(self) -> None:
        """Every element has an inverse."""
        for g in ALL_ELEMENTS:
            inv = d4_inverse(g)
            assert d4_multiply(g, inv) == D4Element.E
            assert d4_multiply(inv, g) == D4Element.E

    def test_associativity(self) -> None:
        """(a * b) * c == a * (b * c) for all elements."""
        for a in ALL_ELEMENTS:
            for b in ALL_ELEMENTS:
                for c in ALL_ELEMENTS:
                    left = d4_multiply(d4_multiply(a, b), c)
                    right = d4_multiply(a, d4_multiply(b, c))
                    assert left == right, f"({a}*{b})*{c} != {a}*({b}*{c})"

    def test_group_order(self) -> None:
        """D4 has exactly 8 elements."""
        assert len(ALL_ELEMENTS) == 8

    def test_rotation_order_4(self) -> None:
        """r^4 = e."""
        r = D4Element.R
        result = r
        for _ in range(3):
            result = d4_multiply(result, r)
        assert result == D4Element.E

    def test_reflection_order_2(self) -> None:
        """s^2 = e."""
        assert d4_multiply(D4Element.S, D4Element.S) == D4Element.E

    def test_non_abelian(self) -> None:
        """D4 is non-abelian: r*s != s*r."""
        rs = d4_multiply(D4Element.R, D4Element.S)
        sr = d4_multiply(D4Element.S, D4Element.R)
        assert rs != sr


# =============================================================================
# CORRELATIVE AND NEGATION
# =============================================================================


class TestCorrelativeNegation:
    """Test correlative (s) and negation (r^2) operations."""

    def test_correlative_o_c(self) -> None:
        assert correlative(HohfeldianState.O) == HohfeldianState.C

    def test_correlative_c_o(self) -> None:
        assert correlative(HohfeldianState.C) == HohfeldianState.O

    def test_correlative_l_n(self) -> None:
        assert correlative(HohfeldianState.L) == HohfeldianState.N

    def test_correlative_n_l(self) -> None:
        assert correlative(HohfeldianState.N) == HohfeldianState.L

    def test_correlative_involution(self) -> None:
        """Correlative applied twice returns to original."""
        for state in ALL_STATES:
            assert correlative(correlative(state)) == state

    def test_negation_o_l(self) -> None:
        assert negation(HohfeldianState.O) == HohfeldianState.L

    def test_negation_l_o(self) -> None:
        assert negation(HohfeldianState.L) == HohfeldianState.O

    def test_negation_c_n(self) -> None:
        assert negation(HohfeldianState.C) == HohfeldianState.N

    def test_negation_n_c(self) -> None:
        assert negation(HohfeldianState.N) == HohfeldianState.C

    def test_negation_involution(self) -> None:
        """Negation applied twice returns to original."""
        for state in ALL_STATES:
            assert negation(negation(state)) == state


# =============================================================================
# D4 ACTION ON STATES
# =============================================================================


class TestD4Action:
    """Test D4 group action on Hohfeldian states."""

    def test_identity_preserves(self) -> None:
        for state in ALL_STATES:
            assert d4_apply_to_state(D4Element.E, state) == state

    def test_rotation(self) -> None:
        """r: O->C->L->N->O."""
        assert d4_apply_to_state(D4Element.R, HohfeldianState.O) == HohfeldianState.C
        assert d4_apply_to_state(D4Element.R, HohfeldianState.C) == HohfeldianState.L
        assert d4_apply_to_state(D4Element.R, HohfeldianState.L) == HohfeldianState.N
        assert d4_apply_to_state(D4Element.R, HohfeldianState.N) == HohfeldianState.O

    def test_r2_is_negation(self) -> None:
        """r^2: O<->L, C<->N."""
        for state in ALL_STATES:
            assert d4_apply_to_state(D4Element.R2, state) == negation(state)

    def test_s_is_correlative(self) -> None:
        """s: O<->C, L<->N."""
        for state in ALL_STATES:
            assert d4_apply_to_state(D4Element.S, state) == correlative(state)

    def test_r3_is_reverse_rotation(self) -> None:
        """r^3: O->N->L->C->O."""
        assert d4_apply_to_state(D4Element.R3, HohfeldianState.O) == HohfeldianState.N
        assert d4_apply_to_state(D4Element.R3, HohfeldianState.N) == HohfeldianState.L
        assert d4_apply_to_state(D4Element.R3, HohfeldianState.L) == HohfeldianState.C
        assert d4_apply_to_state(D4Element.R3, HohfeldianState.C) == HohfeldianState.O

    def test_all_elements_map_states(self) -> None:
        """Every element maps every state to a valid state."""
        for element in ALL_ELEMENTS:
            for state in ALL_STATES:
                result = d4_apply_to_state(element, state)
                assert result in ALL_STATES


# =============================================================================
# BOND INDEX
# =============================================================================


class TestBondIndex:
    """Test Bond Index computation."""

    def test_perfect_symmetry(self) -> None:
        """Perfect correlative symmetry yields 0."""
        a = [
            HohfeldianVerdict("A", HohfeldianState.O),
            HohfeldianVerdict("A", HohfeldianState.O),
            HohfeldianVerdict("A", HohfeldianState.L),
        ]
        b = [
            HohfeldianVerdict("B", HohfeldianState.C),
            HohfeldianVerdict("B", HohfeldianState.C),
            HohfeldianVerdict("B", HohfeldianState.N),
        ]
        assert compute_bond_index(a, b) == 0.0

    def test_full_asymmetry(self) -> None:
        """Complete mismatch yields 1/tau."""
        a = [
            HohfeldianVerdict("A", HohfeldianState.O),
            HohfeldianVerdict("A", HohfeldianState.L),
        ]
        b = [
            HohfeldianVerdict("B", HohfeldianState.L),  # Expected C
            HohfeldianVerdict("B", HohfeldianState.O),  # Expected N
        ]
        assert compute_bond_index(a, b) == 1.0

    def test_partial_asymmetry(self) -> None:
        """Mixed results give fractional index."""
        a = [
            HohfeldianVerdict("A", HohfeldianState.O),
            HohfeldianVerdict("A", HohfeldianState.O),
        ]
        b = [
            HohfeldianVerdict("B", HohfeldianState.C),  # Correct
            HohfeldianVerdict("B", HohfeldianState.L),  # Wrong
        ]
        assert compute_bond_index(a, b) == 0.5

    def test_tau_scaling(self) -> None:
        """tau parameter scales the result."""
        a = [HohfeldianVerdict("A", HohfeldianState.O)]
        b = [HohfeldianVerdict("B", HohfeldianState.L)]  # Wrong
        assert compute_bond_index(a, b, tau=1.0) == 1.0
        assert compute_bond_index(a, b, tau=2.0) == 0.5

    def test_empty_lists(self) -> None:
        """Empty lists return 0."""
        assert compute_bond_index([], []) == 0.0

    def test_unequal_lengths_raises(self) -> None:
        """Mismatched list lengths raise ValueError."""
        a = [HohfeldianVerdict("A", HohfeldianState.O)]
        with pytest.raises(ValueError, match="equal length"):
            compute_bond_index(a, [])


# =============================================================================
# WILSON OBSERVABLE
# =============================================================================


class TestWilsonObservable:
    """Test Wilson loop / holonomy computation."""

    def test_trivial_path(self) -> None:
        """Empty path has identity holonomy."""
        holonomy, matched = compute_wilson_observable(
            [], HohfeldianState.O, HohfeldianState.O
        )
        assert holonomy == D4Element.E
        assert matched is True

    def test_s_path(self) -> None:
        """Single s reflection: O -> C."""
        holonomy, matched = compute_wilson_observable(
            [D4Element.S], HohfeldianState.O, HohfeldianState.C
        )
        assert holonomy == D4Element.S
        assert matched is True

    def test_r2_path(self) -> None:
        """r^2 negation: O -> L."""
        holonomy, matched = compute_wilson_observable(
            [D4Element.R2], HohfeldianState.O, HohfeldianState.L
        )
        assert holonomy == D4Element.R2
        assert matched is True

    def test_mismatch(self) -> None:
        """Wrong observation yields matched=False."""
        holonomy, matched = compute_wilson_observable(
            [D4Element.S], HohfeldianState.O, HohfeldianState.L
        )
        assert holonomy == D4Element.S
        assert matched is False

    def test_closed_loop(self) -> None:
        """s followed by s returns to start."""
        holonomy, matched = compute_wilson_observable(
            [D4Element.S, D4Element.S], HohfeldianState.O, HohfeldianState.O
        )
        assert holonomy == D4Element.E
        assert matched is True


# =============================================================================
# KLEIN FOUR SUBGROUP
# =============================================================================


class TestKleinFourSubgroup:
    """Test abelian subgroup analysis."""

    def test_klein_four_elements(self) -> None:
        v4 = get_klein_four_subgroup()
        assert set(v4) == {D4Element.E, D4Element.R2, D4Element.S, D4Element.SR2}

    def test_is_in_klein_four(self) -> None:
        assert is_in_klein_four(D4Element.E)
        assert is_in_klein_four(D4Element.R2)
        assert is_in_klein_four(D4Element.S)
        assert is_in_klein_four(D4Element.SR2)
        assert not is_in_klein_four(D4Element.R)
        assert not is_in_klein_four(D4Element.R3)

    def test_klein_four_is_subgroup(self) -> None:
        """V4 is closed under multiplication."""
        v4 = get_klein_four_subgroup()
        for a in v4:
            for b in v4:
                assert d4_multiply(a, b) in v4

    def test_requires_nonabelian(self) -> None:
        assert not requires_nonabelian_structure(
            [D4Element.E, D4Element.S, D4Element.R2]
        )
        assert requires_nonabelian_structure([D4Element.R])
        assert requires_nonabelian_structure([D4Element.S, D4Element.SR])


# =============================================================================
# HOHFELDIAN VERDICT
# =============================================================================


class TestHohfeldianVerdict:
    """Test HohfeldianVerdict dataclass."""

    def test_is_correct_when_matches(self) -> None:
        v = HohfeldianVerdict("A", HohfeldianState.O, expected=HohfeldianState.O)
        assert v.is_correct is True

    def test_is_correct_when_wrong(self) -> None:
        v = HohfeldianVerdict("A", HohfeldianState.O, expected=HohfeldianState.C)
        assert v.is_correct is False

    def test_is_correct_when_unknown(self) -> None:
        v = HohfeldianVerdict("A", HohfeldianState.O)
        assert v.is_correct is None

    def test_is_correlative_consistent(self) -> None:
        for state in ALL_STATES:
            v = HohfeldianVerdict("A", state)
            assert v.is_correlative_consistent is True
