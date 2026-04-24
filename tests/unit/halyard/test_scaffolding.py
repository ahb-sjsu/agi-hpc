# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Sprint-0 scaffolding tests.

These tests exist so the Halyard Table package shape is verified by
CI from day one. Each subsequent sprint adds its own test module;
these tests stay green across all of them as a shape guarantee.

If one of these starts failing, something in the package skeleton
changed without a matching doc update.
"""

from __future__ import annotations

import importlib

import pytest

# ─────────────────────────────────────────────────────────────────
# Package import shape
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "module_name",
    [
        "agi.halyard",
        "agi.halyard.sigma4",
        "agi.halyard.state",
        "agi.halyard.keeper",
    ],
)
def test_packages_importable(module_name: str) -> None:
    """Every Halyard Table subpackage must be importable at rest.

    Sprint 0 scaffolding: empty subpackages with ``__init__.py``
    files. This check catches accidental deletion of a subpackage
    or a renaming typo.
    """
    mod = importlib.import_module(module_name)
    assert mod is not None


# ─────────────────────────────────────────────────────────────────
# NATS subject constants
# ─────────────────────────────────────────────────────────────────


def test_halyard_namespace_constant() -> None:
    """The Halyard namespace must be the documented value.

    Every NATS subject for this system is built from this constant;
    changing it is a cross-cutting protocol change, not a refactor.
    """
    from agi.halyard import NAMESPACE

    assert NAMESPACE == "agi.rh.halyard"


def test_session_subjects() -> None:
    """Session subjects match the HALYARD_TABLE.md §5.1 contract."""
    from agi import halyard

    assert halyard.SUBJECT_SESSION_START == "agi.rh.halyard.session.start"
    assert halyard.SUBJECT_SESSION_END == "agi.rh.halyard.session.end"
    assert halyard.SUBJECT_SESSION_SILENCE == "agi.rh.halyard.session.silence"
    assert halyard.SUBJECT_SESSION_RESUME == "agi.rh.halyard.session.resume"
    assert halyard.SUBJECT_SESSION_TICK == "agi.rh.halyard.session.tick"


def test_sheet_subject_formatters() -> None:
    """Sheet subjects are formatted per-pc_id.

    Format strings live in the package so callers don't fabricate
    subject strings with an f-string at the call site — that is
    exactly how subject formats drift across a codebase.
    """
    from agi import halyard

    pc_id = "imogen-roth"
    assert (
        halyard.SUBJECT_SHEET_PATCH_FMT.format(pc_id=pc_id)
        == "agi.rh.halyard.sheet.imogen-roth.patch"
    )
    assert (
        halyard.SUBJECT_SHEET_UPDATE_FMT.format(pc_id=pc_id)
        == "agi.rh.halyard.sheet.imogen-roth.update"
    )


def test_scene_subjects() -> None:
    """Scene-cue subjects per the contract."""
    from agi import halyard

    assert halyard.SUBJECT_SCENE_TRIGGER == "agi.rh.halyard.scene.trigger"
    assert halyard.SUBJECT_SCENE_CUE == "agi.rh.halyard.scene.cue"


def test_keeper_subjects() -> None:
    """Keeper-action subjects per the contract."""
    from agi import halyard

    assert halyard.SUBJECT_KEEPER_APPROVE == "agi.rh.halyard.keeper.approve"
    assert halyard.SUBJECT_KEEPER_REJECT == "agi.rh.halyard.keeper.reject"
    assert halyard.SUBJECT_KEEPER_OVERRIDE == "agi.rh.halyard.keeper.override"
    assert halyard.SUBJECT_KEEPER_DICE == "agi.rh.halyard.keeper.dice"


# ─────────────────────────────────────────────────────────────────
# SIGMA-4 identity constants
# ─────────────────────────────────────────────────────────────────


def test_sigma4_nats_subjects() -> None:
    """SIGMA-4 uses the ``agi.rh.sigma4.*`` lane (parallel to artemis).

    Kept as a flat two-subject contract like ARTEMIS; the halyard
    subjects are for session / sheet / scene / keeper, not for the
    AIs themselves.
    """
    from agi.halyard import sigma4

    assert sigma4.SUBJECT_HEARD == "agi.rh.sigma4.heard"
    assert sigma4.SUBJECT_SAY == "agi.rh.sigma4.say"


def test_sigma4_livekit_identity() -> None:
    """SIGMA-4's LiveKit identity is distinct from ARTEMIS's."""
    from agi.halyard import sigma4

    assert sigma4.LIVEKIT_IDENTITY == "sigma-4"
    assert sigma4.LIVEKIT_DISPLAY_NAME == "SIGMA-4"


# ─────────────────────────────────────────────────────────────────
# Archive root
# ─────────────────────────────────────────────────────────────────


def test_session_archive_root() -> None:
    """Character-sheet archives and session records live here.

    The path is overridable at service startup via env var, but
    the default must match what the runbook documents so operators
    don't end up with records scattered across paths.
    """
    from agi.halyard import state

    assert state.SESSION_ARCHIVE_ROOT == "/archive/halyard"
