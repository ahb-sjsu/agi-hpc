# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Sprint-3 tests — halyard-state NATS bridge.

Exercises :class:`agi.halyard.state.bridge.StateBridge` using a
fake NATS client and the real :class:`Store` (rooted under a
tmp_path). Covers:

- Happy-path: patch subject → apply → update subject published.
- Bad subject: rejected without touching the store.
- Malformed JSON: rejected with structured error.
- Authz denial: error published; store unchanged.
- Envelope pc_id must match subject pc_id.
- WS fan-out hook invoked on success.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from agi.halyard.state.bridge import StateBridge
from agi.halyard.state.store import Store

# ─────────────────────────────────────────────────────────────────
# Fake NATS client
# ─────────────────────────────────────────────────────────────────


class _FakeNats:
    """Records every publish. subscribe() is unused in these tests
    (we call handle_patch_message directly)."""

    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))

    async def subscribe(self, subject: str, *, cb: Any = None) -> None:
        pass

    def update_subjects(self) -> list[str]:
        return [s for s, _ in self.published if s.endswith(".update")]

    def error_subjects(self) -> list[str]:
        return [s for s, _ in self.published if s.endswith(".error")]


# ─────────────────────────────────────────────────────────────────
# Sample sheet
# ─────────────────────────────────────────────────────────────────


def _sheet(pc_id: str = "cross") -> dict:
    return {
        "schema_version": "1.0",
        "session_id": "halyard-s01",
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


@pytest_asyncio.fixture
async def seeded_store(tmp_path: Path) -> Store:
    s = Store(archive_root=tmp_path)
    await s.create("halyard-s01", "cross", _sheet())
    return s


# ─────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_applied_and_update_published(
    seeded_store: Store,
) -> None:
    fake = _FakeNats()
    bridge = StateBridge(store=seeded_store, nats_client=fake)

    envelope = {
        "session_id": "halyard-s01",
        "pc_id": "cross",
        "author": "keeper",
        "patch": [{"op": "replace", "path": "/status/hp_current", "value": 7}],
        "reason": "shrapnel",
    }
    result = await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        json.dumps(envelope).encode("utf-8"),
    )

    assert result.ok is True
    assert result.pc_id == "cross"
    assert result.updated_sheet["status"]["hp_current"] == 7

    # Update published.
    updates = fake.update_subjects()
    assert "agi.rh.halyard.sheet.cross.update" in updates


@pytest.mark.asyncio
async def test_update_payload_includes_full_sheet(
    seeded_store: Store,
) -> None:
    fake = _FakeNats()
    bridge = StateBridge(store=seeded_store, nats_client=fake)
    envelope = {
        "session_id": "halyard-s01",
        "pc_id": "cross",
        "author": "keeper",
        "patch": [{"op": "replace", "path": "/status/luck_current", "value": 30}],
    }
    await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        json.dumps(envelope).encode("utf-8"),
    )
    update_payloads = [p for s, p in fake.published if s.endswith(".update")]
    assert len(update_payloads) == 1
    decoded = json.loads(update_payloads[0])
    assert decoded["sheet"]["status"]["luck_current"] == 30


# ─────────────────────────────────────────────────────────────────
# Bad subjects / malformed envelopes
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bad_subject_structured_error(seeded_store: Store) -> None:
    fake = _FakeNats()
    bridge = StateBridge(store=seeded_store, nats_client=fake)
    result = await bridge.handle_patch_message(
        "agi.rh.wrong.subject",
        b'{"anything": true}',
    )
    assert result.ok is False
    assert result.error_code == "bad_subject"


@pytest.mark.asyncio
async def test_bad_json_structured_error(seeded_store: Store) -> None:
    fake = _FakeNats()
    bridge = StateBridge(store=seeded_store, nats_client=fake)
    result = await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        b"not json",
    )
    assert result.ok is False
    assert result.error_code == "bad_json"


@pytest.mark.asyncio
async def test_envelope_pc_mismatch_rejected(seeded_store: Store) -> None:
    """If the envelope's pc_id disagrees with the subject's pc_id,
    the bridge refuses the patch. Trust boundary check."""
    fake = _FakeNats()
    bridge = StateBridge(store=seeded_store, nats_client=fake)
    envelope = {
        "session_id": "halyard-s01",
        "pc_id": "halverson",  # subject is .cross.
        "author": "keeper",
        "patch": [{"op": "replace", "path": "/status/hp_current", "value": 1}],
    }
    result = await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        json.dumps(envelope).encode("utf-8"),
    )
    assert result.ok is False
    assert result.error_code == "bad_envelope"


@pytest.mark.asyncio
async def test_player_without_author_pc_id_rejected(
    seeded_store: Store,
) -> None:
    fake = _FakeNats()
    bridge = StateBridge(store=seeded_store, nats_client=fake)
    envelope = {
        "session_id": "halyard-s01",
        "pc_id": "cross",
        "author": "player",
        # author_pc_id deliberately missing
        "patch": [{"op": "replace", "path": "/status/hp_current", "value": 5}],
    }
    result = await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        json.dumps(envelope).encode("utf-8"),
    )
    assert result.ok is False
    assert result.error_code == "bad_envelope"


# ─────────────────────────────────────────────────────────────────
# Authz denial publishes an error subject
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_authz_denied_error_published(seeded_store: Store) -> None:
    fake = _FakeNats()
    bridge = StateBridge(store=seeded_store, nats_client=fake)
    envelope = {
        "session_id": "halyard-s01",
        "pc_id": "cross",
        "author": "player",
        "author_pc_id": "cross",
        "patch": [
            {
                "op": "replace",
                "path": "/campaign/faction_loyalty",
                "value": "hollow_hand",
            }
        ],
    }
    result = await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        json.dumps(envelope).encode("utf-8"),
    )
    assert result.ok is False
    assert result.error_code == "authz_denied"
    assert "agi.rh.halyard.sheet.cross.error" in fake.error_subjects()


# ─────────────────────────────────────────────────────────────────
# WS fan-out hook
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ws_broadcast_invoked_on_success(seeded_store: Store) -> None:
    fake = _FakeNats()
    received: list[tuple[str, dict]] = []

    async def _ws_fanout(session_id: str, payload: dict) -> None:
        received.append((session_id, payload))

    bridge = StateBridge(
        store=seeded_store, nats_client=fake, ws_broadcast=_ws_fanout
    )
    envelope = {
        "session_id": "halyard-s01",
        "pc_id": "cross",
        "author": "keeper",
        "patch": [{"op": "replace", "path": "/status/hp_current", "value": 3}],
    }
    await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        json.dumps(envelope).encode("utf-8"),
    )
    assert len(received) == 1
    session_id, msg = received[0]
    assert session_id == "halyard-s01"
    assert msg["kind"] == "sheet.update"
    assert msg["pc_id"] == "cross"
    assert msg["sheet"]["status"]["hp_current"] == 3


@pytest.mark.asyncio
async def test_ws_broadcast_not_invoked_on_error(
    seeded_store: Store,
) -> None:
    """WS listeners get only successful updates — errors are the
    author's problem, not the table's."""
    fake = _FakeNats()
    received: list[tuple[str, dict]] = []

    async def _ws_fanout(session_id: str, payload: dict) -> None:
        received.append((session_id, payload))

    bridge = StateBridge(
        store=seeded_store, nats_client=fake, ws_broadcast=_ws_fanout
    )
    envelope = {
        "session_id": "halyard-s01",
        "pc_id": "cross",
        "author": "keeper",
        "patch": [
            {"op": "replace", "path": "/identity/name", "value": "X"}
        ],  # public field — rejected
    }
    result = await bridge.handle_patch_message(
        "agi.rh.halyard.sheet.cross.patch",
        json.dumps(envelope).encode("utf-8"),
    )
    assert result.ok is False
    assert received == []
