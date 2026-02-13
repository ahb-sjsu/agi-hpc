# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Hohfeldian normative positions with D4 dihedral group structure.

This module implements Wesley Hohfeld's four fundamental normative positions
(Obligation, Claim, Liberty, No-claim) and the D4 dihedral group that acts
on them as symmetry transformations.

The key insight is that moral reasoning exhibits gauge symmetries:
- Correlative symmetry (s): O<->C, L<->N - perspective swap between parties
- Negation symmetry (r^2): O<->L, C<->N - logical negation of normative status

The D4 group (symmetries of a square) provides the complete mathematical
structure for these transformations.

References:
    Hohfeld, W.N. (1917). "Fundamental Legal Conceptions as Applied in
    Judicial Reasoning." Yale Law Journal, 26(8), 710-770.

    Bond, A.H. & Claude (2026). "SQND-Probe: A Gamified Instrument for
    Measuring Dihedral Gauge Structure in Human Moral Reasoning."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# HOHFELDIAN NORMATIVE POSITIONS
# =============================================================================


class HohfeldianState(str, Enum):
    """
    The four Hohfeldian normative positions.

    These form the vertices of a square on which D4 acts:

        O -------- C
        |          |
        |          |
        L -------- N

    Correlative pairs (perspective swap):
    - O <-> C: If A has obligation to B, then B has claim against A
    - L <-> N: If A has liberty against B, then B has no-claim against A

    Negation pairs (logical opposites):
    - O <-> L: Obligation is the negation of liberty
    - C <-> N: Claim is the negation of no-claim
    """

    O = "O"  # noqa: E741 - Obligation: MUST do something
    C = "C"  # Claim: OWED something / has a RIGHT to demand
    L = "L"  # Liberty: FREE to choose / no obligation
    N = "N"  # No-claim: CANNOT demand / no right against other


# =============================================================================
# D4 DIHEDRAL GROUP
# =============================================================================


class D4Element(str, Enum):
    """
    The 8 elements of the dihedral group D4 (symmetries of a square).

    Generators:
    - r: 90 degree clockwise rotation
    - s: reflection (horizontal axis, swapping O<->C and L<->N)

    Elements:
    - e: identity
    - r: 90 degree rotation
    - r2: 180 degree rotation (negation)
    - r3: 270 degree rotation (= r^-1)
    - s: reflection (correlative)
    - sr: reflection composed with rotation
    - sr2: reflection composed with 180 degree rotation
    - sr3: reflection composed with 270 degree rotation

    Group relations:
    - r^4 = e
    - s^2 = e
    - srs = r^-1 (= r3)
    """

    E = "e"  # Identity
    R = "r"  # 90 degree rotation: O->C->L->N->O
    R2 = "r2"  # 180 degree rotation (negation): O<->L, C<->N
    R3 = "r3"  # 270 degree rotation (= r^-1): O->N->L->C->O
    S = "s"  # Reflection (correlative): O<->C, L<->N
    SR = "sr"  # s composed with r
    SR2 = "sr2"  # s composed with r^2
    SR3 = "sr3"  # s composed with r^3


# Complete multiplication table for D4
_D4_MULT_TABLE: Dict[Tuple[D4Element, D4Element], D4Element] = {
    # Identity row and column
    (D4Element.E, D4Element.E): D4Element.E,
    (D4Element.E, D4Element.R): D4Element.R,
    (D4Element.E, D4Element.R2): D4Element.R2,
    (D4Element.E, D4Element.R3): D4Element.R3,
    (D4Element.E, D4Element.S): D4Element.S,
    (D4Element.E, D4Element.SR): D4Element.SR,
    (D4Element.E, D4Element.SR2): D4Element.SR2,
    (D4Element.E, D4Element.SR3): D4Element.SR3,
    # R row
    (D4Element.R, D4Element.E): D4Element.R,
    (D4Element.R, D4Element.R): D4Element.R2,
    (D4Element.R, D4Element.R2): D4Element.R3,
    (D4Element.R, D4Element.R3): D4Element.E,
    (D4Element.R, D4Element.S): D4Element.SR3,
    (D4Element.R, D4Element.SR): D4Element.S,
    (D4Element.R, D4Element.SR2): D4Element.SR,
    (D4Element.R, D4Element.SR3): D4Element.SR2,
    # R2 row
    (D4Element.R2, D4Element.E): D4Element.R2,
    (D4Element.R2, D4Element.R): D4Element.R3,
    (D4Element.R2, D4Element.R2): D4Element.E,
    (D4Element.R2, D4Element.R3): D4Element.R,
    (D4Element.R2, D4Element.S): D4Element.SR2,
    (D4Element.R2, D4Element.SR): D4Element.SR3,
    (D4Element.R2, D4Element.SR2): D4Element.S,
    (D4Element.R2, D4Element.SR3): D4Element.SR,
    # R3 row
    (D4Element.R3, D4Element.E): D4Element.R3,
    (D4Element.R3, D4Element.R): D4Element.E,
    (D4Element.R3, D4Element.R2): D4Element.R,
    (D4Element.R3, D4Element.R3): D4Element.R2,
    (D4Element.R3, D4Element.S): D4Element.SR,
    (D4Element.R3, D4Element.SR): D4Element.SR2,
    (D4Element.R3, D4Element.SR2): D4Element.SR3,
    (D4Element.R3, D4Element.SR3): D4Element.S,
    # S row
    (D4Element.S, D4Element.E): D4Element.S,
    (D4Element.S, D4Element.R): D4Element.SR,
    (D4Element.S, D4Element.R2): D4Element.SR2,
    (D4Element.S, D4Element.R3): D4Element.SR3,
    (D4Element.S, D4Element.S): D4Element.E,
    (D4Element.S, D4Element.SR): D4Element.R,
    (D4Element.S, D4Element.SR2): D4Element.R2,
    (D4Element.S, D4Element.SR3): D4Element.R3,
    # SR row
    (D4Element.SR, D4Element.E): D4Element.SR,
    (D4Element.SR, D4Element.R): D4Element.SR2,
    (D4Element.SR, D4Element.R2): D4Element.SR3,
    (D4Element.SR, D4Element.R3): D4Element.S,
    (D4Element.SR, D4Element.S): D4Element.R3,
    (D4Element.SR, D4Element.SR): D4Element.E,
    (D4Element.SR, D4Element.SR2): D4Element.R,
    (D4Element.SR, D4Element.SR3): D4Element.R2,
    # SR2 row
    (D4Element.SR2, D4Element.E): D4Element.SR2,
    (D4Element.SR2, D4Element.R): D4Element.SR3,
    (D4Element.SR2, D4Element.R2): D4Element.S,
    (D4Element.SR2, D4Element.R3): D4Element.SR,
    (D4Element.SR2, D4Element.S): D4Element.R2,
    (D4Element.SR2, D4Element.SR): D4Element.R3,
    (D4Element.SR2, D4Element.SR2): D4Element.E,
    (D4Element.SR2, D4Element.SR3): D4Element.R,
    # SR3 row
    (D4Element.SR3, D4Element.E): D4Element.SR3,
    (D4Element.SR3, D4Element.R): D4Element.S,
    (D4Element.SR3, D4Element.R2): D4Element.SR,
    (D4Element.SR3, D4Element.R3): D4Element.SR2,
    (D4Element.SR3, D4Element.S): D4Element.R,
    (D4Element.SR3, D4Element.SR): D4Element.R2,
    (D4Element.SR3, D4Element.SR2): D4Element.R3,
    (D4Element.SR3, D4Element.SR3): D4Element.E,
}

# Inverse table for D4
_D4_INVERSE: Dict[D4Element, D4Element] = {
    D4Element.E: D4Element.E,
    D4Element.R: D4Element.R3,
    D4Element.R2: D4Element.R2,
    D4Element.R3: D4Element.R,
    D4Element.S: D4Element.S,  # Reflections are self-inverse
    D4Element.SR: D4Element.SR,
    D4Element.SR2: D4Element.SR2,
    D4Element.SR3: D4Element.SR3,
}

# Rotation action on Hohfeldian states
_ROTATION: Dict[HohfeldianState, HohfeldianState] = {
    HohfeldianState.O: HohfeldianState.C,
    HohfeldianState.C: HohfeldianState.L,
    HohfeldianState.L: HohfeldianState.N,
    HohfeldianState.N: HohfeldianState.O,
}

# Reflection (correlative) action on Hohfeldian states
_REFLECTION: Dict[HohfeldianState, HohfeldianState] = {
    HohfeldianState.O: HohfeldianState.C,
    HohfeldianState.C: HohfeldianState.O,
    HohfeldianState.L: HohfeldianState.N,
    HohfeldianState.N: HohfeldianState.L,
}


# =============================================================================
# D4 GROUP OPERATIONS
# =============================================================================


def d4_multiply(a: D4Element, b: D4Element) -> D4Element:
    """
    Compute a * b in the D4 group.

    Uses right-to-left composition: (a * b)(x) = a(b(x))
    """
    return _D4_MULT_TABLE[(a, b)]


def d4_inverse(a: D4Element) -> D4Element:
    """Compute the inverse of element a in D4."""
    return _D4_INVERSE[a]


def d4_apply_to_state(element: D4Element, state: HohfeldianState) -> HohfeldianState:
    """
    Apply a D4 group element to a Hohfeldian state.

    This is the fundamental group action that maps:
    - r: O->C->L->N->O (rotation)
    - s: O<->C, L<->N (correlative/reflection)
    - r^2: O<->L, C<->N (negation)
    """
    result = state
    match element:
        case D4Element.E:
            pass
        case D4Element.R:
            result = _ROTATION[result]
        case D4Element.R2:
            result = _ROTATION[_ROTATION[result]]
        case D4Element.R3:
            result = _ROTATION[_ROTATION[_ROTATION[result]]]
        case D4Element.S:
            result = _REFLECTION[result]
        case D4Element.SR:
            result = _ROTATION[_REFLECTION[result]]
        case D4Element.SR2:
            result = _ROTATION[_ROTATION[_REFLECTION[result]]]
        case D4Element.SR3:
            result = _ROTATION[_ROTATION[_ROTATION[_REFLECTION[result]]]]
    return result


def correlative(state: HohfeldianState) -> HohfeldianState:
    """
    Get the correlative state (s-reflection): O<->C, L<->N.

    The correlative is the perspective swap:
    - If A has obligation to B, then B has claim against A
    - If A has liberty against B, then B has no-claim against A
    """
    return _REFLECTION[state]


def negation(state: HohfeldianState) -> HohfeldianState:
    """
    Get the negation state (r^2): O<->L, C<->N.

    The negation is the logical opposite:
    - Obligation is the absence of liberty
    - Claim is the absence of no-claim
    """
    return d4_apply_to_state(D4Element.R2, state)


# =============================================================================
# VERDICT AND MEASUREMENT
# =============================================================================


@dataclass
class HohfeldianVerdict:
    """
    A classification of a party's normative position.

    This is the "measurement" in the gauge theory sense.
    """

    party_name: str
    state: HohfeldianState
    expected: Optional[HohfeldianState] = None
    confidence: float = 1.0

    @property
    def is_correct(self) -> Optional[bool]:
        """Check if verdict matches expected state (if known)."""
        if self.expected is None:
            return None
        return self.state == self.expected

    @property
    def is_correlative_consistent(self) -> bool:
        """Check if state is one of the correlative pair."""
        return self.state in (HohfeldianState.O, HohfeldianState.C) or self.state in (
            HohfeldianState.L,
            HohfeldianState.N,
        )


# =============================================================================
# BOND INDEX (CORRELATIVE SYMMETRY MEASURE)
# =============================================================================


def compute_bond_index(
    verdicts_a: List[HohfeldianVerdict],
    verdicts_b: List[HohfeldianVerdict],
    tau: float = 1.0,
) -> float:
    """
    Compute the bond index measuring deviation from correlative symmetry.

    The bond index quantifies how consistently a reasoner applies the
    correlative transformation when shifting perspective from party A to B.

    Args:
        verdicts_a: Classifications of party A's positions
        verdicts_b: Classifications of party B's positions (same scenarios)
        tau: Temperature/scaling parameter

    Returns:
        Bond index in [0, 1/tau]:
        - 0: Perfect correlative symmetry (all B = s(A))
        - 1/tau: Complete antisymmetry

    The correlative gauge principle requires:
        verdict_b = correlative(verdict_a)

    Violations indicate systematic asymmetries in moral reasoning.
    """
    if len(verdicts_a) != len(verdicts_b):
        raise ValueError("Verdict lists must have equal length")

    if not verdicts_a:
        return 0.0

    defects = 0
    total = 0

    for va, vb in zip(verdicts_a, verdicts_b, strict=False):
        expected_b = correlative(va.state)
        if vb.state != expected_b:
            defects += 1
        total += 1

    return (defects / total) / tau


def compute_wilson_observable(
    path: List[D4Element],
    initial_state: HohfeldianState,
    observed_final: HohfeldianState,
) -> Tuple[D4Element, bool]:
    """
    Compute Wilson observable for a closed path of transformations.

    In discrete gauge theory, the Wilson loop is the holonomy around
    a closed path. For our D4 gauge structure, this measures whether
    the observed final state matches the predicted holonomy.

    Args:
        path: Sequence of D4 group elements (transformations applied)
        initial_state: Starting Hohfeldian position
        observed_final: Actually observed final state

    Returns:
        (holonomy, matched): The predicted group element and whether
        the observation matched the prediction.
    """
    # Compute holonomy (product of path elements)
    holonomy = D4Element.E
    for g in path:
        holonomy = d4_multiply(holonomy, g)

    # Predicted final state
    predicted_final = d4_apply_to_state(holonomy, initial_state)
    matched = observed_final == predicted_final

    return holonomy, matched


# =============================================================================
# ABELIAN SUBGROUP ANALYSIS
# =============================================================================


def get_klein_four_subgroup() -> List[D4Element]:
    """
    Return the Klein four-group V4 = {e, r^2, s, sr^2}.

    This abelian subgroup is generated by {r^2, s} and is important
    because if only negation (r^2) and correlative (s) operations are
    empirically observed, we have NOT demonstrated non-abelian structure.

    Non-abelian structure requires demonstrating operations from
    {r, r^3, sr, sr^3} - the "quarter-turn" elements.
    """
    return [D4Element.E, D4Element.R2, D4Element.S, D4Element.SR2]


def is_in_klein_four(element: D4Element) -> bool:
    """Check if an element is in the abelian Klein four subgroup."""
    return element in get_klein_four_subgroup()


def requires_nonabelian_structure(elements: List[D4Element]) -> bool:
    """
    Check if a set of operations requires non-abelian group structure.

    Returns True if the elements include any from {r, r^3, sr, sr^3},
    which don't commute with s and thus demonstrate D4's non-abelian nature.
    """
    non_abelian_elements = {D4Element.R, D4Element.R3, D4Element.SR, D4Element.SR3}
    return any(e in non_abelian_elements for e in elements)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "HohfeldianState",
    "D4Element",
    # Group operations
    "d4_multiply",
    "d4_inverse",
    "d4_apply_to_state",
    "correlative",
    "negation",
    # Verdict
    "HohfeldianVerdict",
    # Bond index
    "compute_bond_index",
    "compute_wilson_observable",
    # Subgroup analysis
    "get_klein_four_subgroup",
    "is_in_klein_four",
    "requires_nonabelian_structure",
]
