# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Sprint-3 tests — schema + access-control + patch application.

Covers the three modules under ``src/agi/halyard/state/schema/``:

- ``schema/__init__.py`` — JSON-Schema validation of whole sheets.
- ``schema/access.py`` — path-to-tier classification and
  per-author authz decision.
- ``schema/patch.py`` — minimal RFC 6902 applier with authz gate.

The store and API layers have their own test files; this file is
deliberately the "schema layer" suite.
"""

from __future__ import annotations

import copy

import jsonschema
import pytest

from agi.halyard.state.schema import load_schema, validate_sheet
from agi.halyard.state.schema.access import (
    AccessTier,
    Author,
    can_write,
    tier_for_path,
)
from agi.halyard.state.schema.patch import (
    AuthorizationError,
    InvalidPatchError,
    PatchRequest,
    apply_patch,
)

# ─────────────────────────────────────────────────────────────────
# Minimal valid sheet fixture
# ─────────────────────────────────────────────────────────────────


def _minimal_sheet() -> dict:
    """The smallest well-formed sheet. Used as the base for positive
    tests and mutated for negative tests.

    Values picked to satisfy the schema's minimums without
    resembling any real PC.
    """
    return {
        "schema_version": "1.0",
        "session_id": "halyard-s01",
        "pc_id": "test-pc-01",
        "identity": {
            "name": "Test Investigator",
            "age": 35,
            "origin": "Luna, Copernicus",
            "role": "medical_officer",
            "chassis": "baseline_human",
            "credit_rating": 45,
        },
        "characteristics": {
            "str": 55, "con": 60, "siz": 55, "dex": 65,
            "app": 55, "int": 70, "pow": 60, "edu": 75,
        },
        "derived": {
            "hp_max": 11, "mp_max": 12, "san_starting": 60,
            "san_max": 65, "luck_max": 50, "move": 8,
            "build": 0, "damage_bonus": "0", "dodge_base": 32,
        },
        "skills": {
            "medicine": {"value": 75, "base": 1},
            "first aid": {"value": 60, "base": 30},
        },
        "bonds": [
            {
                "id": "sister-care-home",
                "tier": 1,
                "name": "Sister in care-home on Luna",
                "detail": "Advanced dementia; recognizes voice only.",
                "status": "intact",
            }
        ],
        "status": {
            "hp_current": 11,
            "mp_current": 12,
            "san_current": 60,
            "luck_current": 50,
        },
        "campaign": {
            "faction_loyalty": "clean",
            "personal_hook": "A recurring nightmare about drowning.",
            "why_this_contract": "Pays off sister's care-home bills.",
        },
    }


@pytest.fixture
def good_sheet() -> dict:
    return _minimal_sheet()


# ─────────────────────────────────────────────────────────────────
# Schema load + basic validation
# ─────────────────────────────────────────────────────────────────


def test_load_schema_returns_dict() -> None:
    schema = load_schema()
    assert isinstance(schema, dict)
    assert schema["$schema"].startswith("https://json-schema.org/draft/2020-12")


def test_load_schema_is_cached() -> None:
    """load_schema uses lru_cache — repeated calls return the same object."""
    assert load_schema() is load_schema()


def test_minimal_sheet_is_valid(good_sheet: dict) -> None:
    validate_sheet(good_sheet)


# ─────────────────────────────────────────────────────────────────
# Negative schema tests
# ─────────────────────────────────────────────────────────────────


def test_unknown_role_rejected(good_sheet: dict) -> None:
    good_sheet["identity"]["role"] = "xenotheologian"
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_unknown_chassis_rejected(good_sheet: dict) -> None:
    good_sheet["identity"]["chassis"] = "cyborg"
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_san_out_of_range_rejected(good_sheet: dict) -> None:
    good_sheet["status"]["san_current"] = 999
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_characteristic_below_minimum_rejected(good_sheet: dict) -> None:
    good_sheet["characteristics"]["str"] = 1
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_extra_top_level_field_rejected(good_sheet: dict) -> None:
    """additionalProperties=false keeps typo'd fields out of the sheet."""
    good_sheet["not_a_real_field"] = True
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_missing_session_id_rejected(good_sheet: dict) -> None:
    del good_sheet["session_id"]
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_schema_version_must_be_1_0(good_sheet: dict) -> None:
    good_sheet["schema_version"] = "2.0"
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_unknown_faction_loyalty_rejected(good_sheet: dict) -> None:
    good_sheet["campaign"]["faction_loyalty"] = "cult_of_the_torch"
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_bonds_status_enum_enforced(good_sheet: dict) -> None:
    good_sheet["bonds"][0]["status"] = "uncertain"
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


def test_condition_code_enum_enforced(good_sheet: dict) -> None:
    good_sheet["status"]["conditions"] = [
        {"code": "haunted", "note": "by gran"}
    ]
    with pytest.raises(jsonschema.ValidationError):
        validate_sheet(good_sheet)


# ─────────────────────────────────────────────────────────────────
# Access tier classification
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path, expected",
    [
        # Immutable public fields.
        ("/schema_version", AccessTier.PUBLIC),
        ("/session_id", AccessTier.PUBLIC),
        ("/pc_id", AccessTier.PUBLIC),
        ("/identity/name", AccessTier.PUBLIC),
        ("/identity/age", AccessTier.PUBLIC),
        ("/identity/role", AccessTier.PUBLIC),
        ("/identity/chassis", AccessTier.PUBLIC),
        ("/characteristics/str", AccessTier.PUBLIC),
        ("/characteristics/pow", AccessTier.PUBLIC),
        ("/derived/hp_max", AccessTier.PUBLIC),
        ("/derived/san_max", AccessTier.PUBLIC),
        ("/skills/medicine/base", AccessTier.PUBLIC),
        # Player-writable fields.
        ("/status/hp_current", AccessTier.PLAYER),
        ("/status/san_current", AccessTier.PLAYER),
        ("/status/luck_current", AccessTier.PLAYER),
        ("/status/conditions", AccessTier.PLAYER),
        ("/status/conditions/0", AccessTier.PLAYER),
        ("/status/conditions/0/note", AccessTier.PLAYER),
        ("/skills/medicine/value", AccessTier.PLAYER),
        ("/skills/medicine/improvement_check", AccessTier.PLAYER),
        ("/bonds/0/status", AccessTier.PLAYER),
        ("/bonds/0/detail", AccessTier.PLAYER),
        ("/equipment/0/qty", AccessTier.PLAYER),
        # Keeper-restricted.
        ("/campaign/faction_loyalty", AccessTier.KEEPER),
        ("/campaign/keeper_hook", AccessTier.KEEPER),
        ("/campaign/rem_log_cSv", AccessTier.KEEPER),
        ("/bonds", AccessTier.KEEPER),
        ("/bonds/-", AccessTier.KEEPER),
        ("/skills", AccessTier.KEEPER),
        # Unclassified fallback → KEEPER (defensive default).
        ("/unknown/future/field", AccessTier.KEEPER),
    ],
)
def test_tier_for_path(path: str, expected: AccessTier) -> None:
    assert tier_for_path(path) is expected


# ─────────────────────────────────────────────────────────────────
# can_write authorization decision
# ─────────────────────────────────────────────────────────────────


def test_keeper_may_write_keeper_path() -> None:
    ok, _ = can_write(
        path="/campaign/faction_loyalty",
        author=Author.KEEPER,
        author_pc_id=None,
        target_pc_id="any",
    )
    assert ok is True


def test_keeper_may_write_player_path() -> None:
    ok, _ = can_write(
        path="/status/hp_current",
        author=Author.KEEPER,
        author_pc_id=None,
        target_pc_id="any",
    )
    assert ok is True


def test_keeper_cannot_write_public_path() -> None:
    """Public paths are immutable — even the Keeper must go through
    a dedicated admin endpoint to touch identity, not a patch."""
    ok, reason = can_write(
        path="/identity/name",
        author=Author.KEEPER,
        author_pc_id=None,
        target_pc_id="any",
    )
    assert ok is False
    assert "immutable" in reason


def test_system_may_write_like_keeper() -> None:
    """System-origin writes (rule engine, dice effects) map to
    keeper-level authorization."""
    ok, _ = can_write(
        path="/campaign/rem_log_cSv",
        author=Author.SYSTEM,
        author_pc_id=None,
        target_pc_id="any",
    )
    assert ok is True


def test_player_may_write_own_player_path() -> None:
    ok, _ = can_write(
        path="/status/san_current",
        author=Author.PLAYER,
        author_pc_id="cross",
        target_pc_id="cross",
    )
    assert ok is True


def test_player_cannot_write_other_pcs_sheet() -> None:
    ok, reason = can_write(
        path="/status/san_current",
        author=Author.PLAYER,
        author_pc_id="cross",
        target_pc_id="halverson",
    )
    assert ok is False
    assert "may not patch sheet" in reason


def test_player_cannot_write_keeper_path() -> None:
    ok, reason = can_write(
        path="/campaign/faction_loyalty",
        author=Author.PLAYER,
        author_pc_id="cross",
        target_pc_id="cross",
    )
    assert ok is False
    assert "Keeper-restricted" in reason


def test_player_requires_author_pc_id() -> None:
    """A player-origin patch with no author_pc_id is structurally
    invalid — defense in depth against a malformed envelope."""
    ok, reason = can_write(
        path="/status/hp_current",
        author=Author.PLAYER,
        author_pc_id=None,
        target_pc_id="cross",
    )
    assert ok is False
    assert "requires author_pc_id" in reason


# ─────────────────────────────────────────────────────────────────
# Patch application — happy path
# ─────────────────────────────────────────────────────────────────


def test_keeper_replace_san_current(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/status/san_current", "value": 54}],
        reason="SAN 1D6 — Mi-go reveal",
    )
    out = apply_patch(good_sheet, req)
    assert out["status"]["san_current"] == 54
    # Original untouched (copy-on-write).
    assert good_sheet["status"]["san_current"] == 60


def test_player_replace_own_hp(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.PLAYER,
        author_pc_id="test-pc-01",
        patch=[{"op": "replace", "path": "/status/hp_current", "value": 7}],
    )
    out = apply_patch(good_sheet, req)
    assert out["status"]["hp_current"] == 7


def test_keeper_add_bond(good_sheet: dict) -> None:
    new_bond = {
        "id": "ark-friend",
        "tier": 2,
        "name": "Ark Vorstov",
        "status": "intact",
    }
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "add", "path": "/bonds/-", "value": new_bond}],
        reason="Ark becomes significant in Ch.3",
    )
    out = apply_patch(good_sheet, req)
    assert len(out["bonds"]) == 2
    assert out["bonds"][-1]["id"] == "ark-friend"


def test_keeper_remove_skill(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "remove", "path": "/skills/first aid"}],
    )
    out = apply_patch(good_sheet, req)
    assert "first aid" not in out["skills"]


def test_player_toggle_improvement_check(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.PLAYER,
        author_pc_id="test-pc-01",
        patch=[
            {
                "op": "add",
                "path": "/skills/medicine/improvement_check",
                "value": True,
            }
        ],
    )
    out = apply_patch(good_sheet, req)
    assert out["skills"]["medicine"]["improvement_check"] is True


def test_multi_op_patch_applied_in_order(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[
            {"op": "replace", "path": "/status/hp_current", "value": 6},
            {"op": "replace", "path": "/status/san_current", "value": 52},
        ],
    )
    out = apply_patch(good_sheet, req)
    assert out["status"]["hp_current"] == 6
    assert out["status"]["san_current"] == 52


# ─────────────────────────────────────────────────────────────────
# Patch application — authz failures
# ─────────────────────────────────────────────────────────────────


def test_player_patch_to_keeper_field_rejected(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.PLAYER,
        author_pc_id="test-pc-01",
        patch=[
            {
                "op": "replace",
                "path": "/campaign/faction_loyalty",
                "value": "hollow_hand",
            }
        ],
    )
    with pytest.raises(AuthorizationError):
        apply_patch(good_sheet, req)


def test_cross_pc_player_patch_rejected(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="halverson",  # target other PC
        author=Author.PLAYER,
        author_pc_id="cross",      # author is someone else
        patch=[{"op": "replace", "path": "/status/hp_current", "value": 1}],
    )
    with pytest.raises(AuthorizationError):
        apply_patch(good_sheet, req)


def test_public_path_patch_always_rejected(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/identity/name", "value": "X"}],
    )
    with pytest.raises(AuthorizationError):
        apply_patch(good_sheet, req)


def test_multi_op_authz_fails_atomically(good_sheet: dict) -> None:
    """If any op in a patch is rejected, none is applied."""
    original = copy.deepcopy(good_sheet)
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.PLAYER,
        author_pc_id="test-pc-01",
        patch=[
            {"op": "replace", "path": "/status/hp_current", "value": 3},
            {
                "op": "replace",
                "path": "/campaign/faction_loyalty",
                "value": "hollow_hand",
            },
        ],
    )
    with pytest.raises(AuthorizationError):
        apply_patch(good_sheet, req)
    # Input unchanged.
    assert good_sheet == original


# ─────────────────────────────────────────────────────────────────
# Patch application — malformed input
# ─────────────────────────────────────────────────────────────────


def test_unsupported_op_rejected(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "move", "from": "/a", "path": "/b"}],
    )
    with pytest.raises(InvalidPatchError):
        apply_patch(good_sheet, req)


def test_missing_path_rejected(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "value": 1}],
    )
    with pytest.raises(InvalidPatchError):
        apply_patch(good_sheet, req)


def test_replace_missing_value_rejected(good_sheet: dict) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/status/hp_current"}],
    )
    with pytest.raises(InvalidPatchError):
        apply_patch(good_sheet, req)


def test_bad_pointer_rejected(good_sheet: dict) -> None:
    """Pointers must start with '/' (or be empty)."""
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "status.hp_current", "value": 1}],
    )
    with pytest.raises(InvalidPatchError):
        apply_patch(good_sheet, req)


def test_replace_missing_key_rejected(good_sheet: dict) -> None:
    """RFC 6902: replace requires the target to exist."""
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="test-pc-01",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/campaign/does_not_exist", "value": 1}],
    )
    with pytest.raises(InvalidPatchError):
        apply_patch(good_sheet, req)


def test_pointer_escapes_honored(good_sheet: dict) -> None:
    """``first aid`` has a space; no escape needed there. But if a
    future field had a ``/`` or ``~`` we'd rely on RFC 6901 escapes.
    Exercise the escape path with a synthetic stub."""
    from agi.halyard.state.schema.patch import _parse_pointer

    assert _parse_pointer("/a~1b/c") == ["a/b", "c"]
    assert _parse_pointer("/a~0b") == ["a~b"]
    # Combined: ~01 should decode to ~1 (not //) — i.e. the ~0 is
    # decoded first-by-ordering, then ~1 is literal.
    assert _parse_pointer("/~01") == ["~1"]
