"""agi-hpc's Hohfeldian structure is erisml-lib's, not a copy (V4: correlative s, negation n).

Its group laws are tested in erisml-lib (tests/test_hohfeld_v4.py) and proved in
formal/HohfeldV4.lean; here it is enough that agi-hpc re-exports those very objects and that
the safety service's correlative check and bond index behave as V4 says.
"""

import erisml.ethics.hohfeld as lib

import agi.safety.erisml as safety
from agi.safety.erisml import (
    HohfeldianState,
    HohfeldianVerdict,
    V4Element,
    compute_bond_index,
    correlative,
)


def test_the_package_re_exports_erisml_libs_objects_not_copies():
    for name in (
        "HohfeldianState",
        "V4Element",
        "HohfeldianVerdict",
        "correlative",
        "negation",
        "compute_bond_index",
        "compute_wilson_observable",
        "v4_between",
    ):
        assert getattr(safety, name) is getattr(lib, name), name


def test_there_is_no_d4_left():
    assert not any(n.startswith(("D4", "d4_")) for n in safety.__all__)


def test_correlative_and_bond_index_as_the_service_uses_them():
    o, c, l, n = (
        HohfeldianState.O,
        HohfeldianState.C,
        HohfeldianState.L,
        HohfeldianState.N,
    )
    assert [correlative(x) for x in (o, c, l, n)] == [c, o, n, l]
    a = [HohfeldianVerdict("A", o), HohfeldianVerdict("A", l)]
    assert (
        compute_bond_index(a, [HohfeldianVerdict("B", c), HohfeldianVerdict("B", n)])
        == 0.0
    )
    assert (
        compute_bond_index(a, [HohfeldianVerdict("B", c), HohfeldianVerdict("B", o)])
        == 0.5
    )
    assert len(list(V4Element)) == 4
