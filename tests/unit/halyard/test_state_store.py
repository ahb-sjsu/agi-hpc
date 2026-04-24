# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Sprint-3 tests — halyard-state on-disk Store.

Covers :class:`agi.halyard.state.store.Store`:

- Creation and retrieval of sheets.
- Patch application through the store (authz + schema checked).
- Append-only log records each patch.
- Multiple sessions and PCs isolated on disk.
- Error paths (duplicate create, not-found get, bad patch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agi.halyard.state.schema.access import Author
from agi.halyard.state.schema.patch import (
    AuthorizationError,
    InvalidPatchError,
    PatchRequest,
)
from agi.halyard.state.store import (
    AlreadyExistsError,
    NotFoundError,
    SchemaError,
    Store,
)

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _minimal_sheet(session_id: str = "halyard-s01", pc_id: str = "cross") -> dict:
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "pc_id": pc_id,
        "identity": {
            "name": "Halden Cross",
            "age": 41,
            "origin": "Luna, Apennine",
            "role": "security_officer",
            "chassis": "baseline_human",
            "credit_rating": 35,
        },
        "characteristics": {
            "str": 75, "con": 75, "siz": 65, "dex": 70,
            "app": 55, "int": 65, "pow": 55, "edu": 60,
        },
        "derived": {
            "hp_max": 14, "mp_max": 11, "san_starting": 55,
            "san_max": 65, "luck_max": 55, "move": 8,
            "build": 1, "damage_bonus": "+1D4", "dodge_base": 35,
        },
        "skills": {
            "firearms (handgun)": {"value": 70, "base": 20},
            "spot hidden": {"value": 60, "base": 25},
        },
        "bonds": [],
        "status": {
            "hp_current": 14,
            "mp_current": 11,
            "san_current": 55,
            "luck_current": 55,
        },
        "campaign": {"faction_loyalty": "clean"},
    }


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """Fresh Store rooted at a pytest tmp_path — no /archive touches."""
    return Store(archive_root=tmp_path)


# ─────────────────────────────────────────────────────────────────
# Create + get
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_returns_sheet(store: Store) -> None:
    sheet = await store.create("halyard-s01", "cross", _minimal_sheet())
    assert sheet["pc_id"] == "cross"
    assert sheet["identity"]["name"] == "Halden Cross"


@pytest.mark.asyncio
async def test_create_writes_to_disk(store: Store, tmp_path: Path) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    p = tmp_path / "sheets" / "halyard-s01" / "cross.json"
    assert p.is_file()
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["pc_id"] == "cross"


@pytest.mark.asyncio
async def test_create_writes_log_entry(
    store: Store, tmp_path: Path
) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    log_p = tmp_path / "sheets" / "halyard-s01" / "log.jsonl"
    assert log_p.is_file()
    lines = log_p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "create"
    assert record["pc_id"] == "cross"


@pytest.mark.asyncio
async def test_create_normalizes_session_and_pc(
    store: Store, tmp_path: Path
) -> None:
    """The store enforces that the on-disk ``session_id`` and
    ``pc_id`` match the path-side identifiers. A caller that
    accidentally submits an inconsistent sheet has the sheet's
    fields overwritten — the URL-style identity is authoritative.
    """
    inconsistent = _minimal_sheet(session_id="wrong", pc_id="other")
    out = await store.create("halyard-s01", "cross", inconsistent)
    assert out["session_id"] == "halyard-s01"
    assert out["pc_id"] == "cross"


@pytest.mark.asyncio
async def test_create_duplicate_rejected(store: Store) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    with pytest.raises(AlreadyExistsError):
        await store.create("halyard-s01", "cross", _minimal_sheet())


@pytest.mark.asyncio
async def test_create_invalid_sheet_rejected(store: Store) -> None:
    bad = _minimal_sheet()
    bad["identity"]["role"] = "xenotheologian"
    with pytest.raises(SchemaError):
        await store.create("halyard-s01", "cross", bad)


@pytest.mark.asyncio
async def test_get_returns_copy(store: Store) -> None:
    """Mutating the returned dict should NOT affect the store —
    defensive copy on read is load-bearing for correctness."""
    await store.create("halyard-s01", "cross", _minimal_sheet())
    s1 = await store.get("halyard-s01", "cross")
    s1["status"]["hp_current"] = 0
    s2 = await store.get("halyard-s01", "cross")
    assert s2["status"]["hp_current"] == 14


@pytest.mark.asyncio
async def test_get_not_found(store: Store) -> None:
    with pytest.raises(NotFoundError):
        await store.get("halyard-s01", "ghost")


@pytest.mark.asyncio
async def test_get_hits_disk_after_cold_start(
    tmp_path: Path,
) -> None:
    """A new Store instance should be able to load a sheet written
    by an earlier instance. This is the normal restart case."""
    s1 = Store(archive_root=tmp_path)
    await s1.create("halyard-s01", "cross", _minimal_sheet())
    # Throw away cache, build a fresh store, re-read.
    s2 = Store(archive_root=tmp_path)
    sheet = await s2.get("halyard-s01", "cross")
    assert sheet["pc_id"] == "cross"


# ─────────────────────────────────────────────────────────────────
# Listing
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_sheets_sorted(store: Store) -> None:
    for pc in ("cross", "halverson", "bond"):
        await store.create("halyard-s01", pc, _minimal_sheet(pc_id=pc))
    ids = await store.list_sheets("halyard-s01")
    assert ids == ["bond", "cross", "halverson"]


@pytest.mark.asyncio
async def test_list_sheets_unknown_session(store: Store) -> None:
    assert await store.list_sheets("no-such") == []


# ─────────────────────────────────────────────────────────────────
# Patch
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_applies_and_persists(
    store: Store, tmp_path: Path
) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="cross",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/status/hp_current", "value": 9}],
        reason="took 5 HP from a fall on the ladder",
    )
    out = await store.patch(req)
    assert out["status"]["hp_current"] == 9

    # Persists to disk.
    on_disk = json.loads(
        (tmp_path / "sheets" / "halyard-s01" / "cross.json").read_text(
            encoding="utf-8"
        )
    )
    assert on_disk["status"]["hp_current"] == 9


@pytest.mark.asyncio
async def test_patch_appends_log_record(
    store: Store, tmp_path: Path
) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="cross",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/status/san_current", "value": 48}],
        reason="SAN 1D6 from Mi-go reveal",
    )
    await store.patch(req)
    log_p = tmp_path / "sheets" / "halyard-s01" / "log.jsonl"
    lines = log_p.read_text(encoding="utf-8").strip().splitlines()
    # First line is "create", second is the patch.
    assert len(lines) == 2
    record = json.loads(lines[1])
    assert record["kind"] == "patch"
    assert record["reason"] == "SAN 1D6 from Mi-go reveal"
    assert record["author"] == "keeper"


@pytest.mark.asyncio
async def test_patch_player_wrong_pc_rejected(store: Store) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    await store.create(
        "halyard-s01", "halverson",
        _minimal_sheet(pc_id="halverson"),
    )
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="halverson",
        author=Author.PLAYER,
        author_pc_id="cross",
        patch=[{"op": "replace", "path": "/status/hp_current", "value": 1}],
    )
    with pytest.raises(AuthorizationError):
        await store.patch(req)


@pytest.mark.asyncio
async def test_patch_to_missing_sheet_raises(store: Store) -> None:
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="ghost",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/status/hp_current", "value": 1}],
    )
    with pytest.raises(NotFoundError):
        await store.patch(req)


@pytest.mark.asyncio
async def test_patch_that_invalidates_schema_rejected(
    store: Store,
) -> None:
    """A patch whose *result* violates the schema is rejected and
    the sheet is not persisted."""
    await store.create("halyard-s01", "cross", _minimal_sheet())
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="cross",
        author=Author.KEEPER,
        author_pc_id=None,
        # 999 exceeds the schema's max for san_current.
        patch=[{"op": "replace", "path": "/status/san_current", "value": 999}],
    )
    with pytest.raises(SchemaError):
        await store.patch(req)
    # Sheet unchanged on retry.
    sheet = await store.get("halyard-s01", "cross")
    assert sheet["status"]["san_current"] == 55


@pytest.mark.asyncio
async def test_malformed_patch_raises_invalid(store: Store) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="cross",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "move", "from": "/a", "path": "/b"}],
    )
    with pytest.raises(InvalidPatchError):
        await store.patch(req)


# ─────────────────────────────────────────────────────────────────
# Isolation
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_sessions_isolated(store: Store) -> None:
    await store.create("halyard-s01", "cross", _minimal_sheet())
    await store.create(
        "halyard-s02", "cross",
        _minimal_sheet(session_id="halyard-s02"),
    )
    s1 = await store.get("halyard-s01", "cross")
    s2 = await store.get("halyard-s02", "cross")
    assert s1["session_id"] == "halyard-s01"
    assert s2["session_id"] == "halyard-s02"


# ─────────────────────────────────────────────────────────────────
# Digest reproducibility
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_digests_reflect_state(
    store: Store, tmp_path: Path
) -> None:
    """Two identical patches produce different log digests only if
    the prior state differs — a weak but useful tamper-detection
    check."""
    from agi.halyard.state.store import _sheet_digest

    await store.create("halyard-s01", "cross", _minimal_sheet())
    req = PatchRequest(
        session_id="halyard-s01",
        target_pc_id="cross",
        author=Author.KEEPER,
        author_pc_id=None,
        patch=[{"op": "replace", "path": "/status/hp_current", "value": 9}],
    )
    out = await store.patch(req)
    expected = _sheet_digest(out)

    log_lines = (
        (tmp_path / "sheets" / "halyard-s01" / "log.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    latest = json.loads(log_lines[-1])
    assert latest["digest"] == expected
