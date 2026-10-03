"""agi-hpc's Hohfeldian structure is erisml-lib's, not a copy (V4: correlative s, negation n).

Its group laws are tested in erisml-lib (tests/test_hohfeld_v4.py) and proved in
formal/HohfeldV4.lean; here it is enough that agi-hpc re-exports those very objects and that
the safety service's correlative check and bond index behave as V4 says.
"""

import pytest

import erisml.ethics.hohfeld as lib

USED = (
    "HohfeldianState",
    "V4Element",
    "HohfeldianVerdict",
    "correlative",
    "negation",
    "compute_bond_index",
    "compute_wilson_observable",
    "v4_between",
)


def test_the_erisml_lib_dependency_provides_what_agi_hpc_uses():
    for name in USED:
        assert hasattr(lib, name), name
    o, c, l, n = (
        lib.HohfeldianState.O,
        lib.HohfeldianState.C,
        lib.HohfeldianState.L,
        lib.HohfeldianState.N,
    )
    assert [lib.correlative(x) for x in (o, c, l, n)] == [c, o, n, l]
    a = [lib.HohfeldianVerdict("A", o), lib.HohfeldianVerdict("A", l)]
    assert (
        lib.compute_bond_index(
            a, [lib.HohfeldianVerdict("B", c), lib.HohfeldianVerdict("B", n)]
        )
        == 0.0
    )
    assert (
        lib.compute_bond_index(
            a, [lib.HohfeldianVerdict("B", c), lib.HohfeldianVerdict("B", o)]
        )
        == 0.5
    )
    assert len(list(lib.V4Element)) == 4


def test_the_package_re_exports_erisml_libs_objects_not_copies():
    pytest.importorskip("grpc")  # agi.safety.erisml imports the gRPC service
    import agi.safety.erisml as safety

    for name in USED:
        assert getattr(safety, name) is getattr(lib, name), name
    assert not any(x.startswith(("D4", "d4_")) for x in safety.__all__)
