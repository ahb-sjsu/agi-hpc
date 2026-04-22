# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""End-to-end round-trip test: fake bot → nats-server → ArtemisService.

Spins up an ephemeral ``nats-server`` subprocess on a random high port,
runs the ARTEMIS service against it, publishes 20 ``heard`` events as
a simulated bot, and verifies that each addressed event produces a
valid ``say`` reply with a correctly-chained DecisionProof.

Skipped entirely if the ``nats-server`` binary is not on PATH. On
Atlas this is provided by the ``atlas-nats.service`` install; in CI
it must be present for this test to run. Keep unit tests
(``test_artemis_nats.py``) as the primary regression guard; this
test is the "does it actually work on a real broker" sanity check.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import time

import pytest

import agi.primer.artemis.context as _ctx
import agi.primer.artemis.mode as _mode
from agi.primer.artemis.context import Bible
from agi.primer.artemis.nats_handler import (
    SUBJECT_HEARD,
    SUBJECT_SAY,
    SUBJECT_SILENCE,
    ArtemisService,
)
from agi.primer.artemis.validator import DecisionProof

_NATS_SERVER = shutil.which("nats-server")
_REQUIRES_NATS = pytest.mark.skipif(
    _NATS_SERVER is None,
    reason="nats-server not on PATH; install it to run this e2e test",
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _free_port() -> int:
    """Grab an OS-assigned free port. Racy but adequate for a short test."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def nats_server():
    """Start nats-server on a free port and tear down after."""
    port = _free_port()
    proc = subprocess.Popen(
        [_NATS_SERVER, "-p", str(port), "-a", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait up to 3s for the port to open.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("nats-server did not open port within 3s")
    try:
        yield f"nats://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def tmp_artemis_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(_ctx, "_SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(_mode, "_PROOFS_DIR", tmp_path / "proofs")
    return tmp_path


class ScriptedLLM:
    """Deterministic LLM that always returns the Nth scripted reply."""

    def __init__(self, template: str = "Reading {n} confirmed."):
        self.template = template
        self.calls = 0

    async def __call__(self, system, messages):
        self.calls += 1
        return (self.template.format(n=self.calls), "qwen3", 0.05)


# ─────────────────────────────────────────────────────────────────
# The test
# ─────────────────────────────────────────────────────────────────


@_REQUIRES_NATS
@pytest.mark.asyncio
async def test_twenty_turn_round_trip(tmp_artemis_dirs, nats_server):
    """Publish 20 heard events as a fake bot; collect 20 valid say replies."""
    import nats  # lazy; skip marker above guards the case where it's absent

    # Collect reply messages from the service.
    received: list[dict] = []
    reply_received = asyncio.Event()

    bot_nc = await nats.connect(nats_server)

    async def on_say(msg):
        received.append(json.loads(msg.data))
        if len(received) >= 20:
            reply_received.set()

    await bot_nc.subscribe(SUBJECT_SAY, cb=on_say)

    # Start the service against the same broker.
    service = ArtemisService(
        nats_url=nats_server,
        bible=Bible(),
        llm=ScriptedLLM(),
        keeper_approval_required=False,
    )
    await service.start()

    try:
        # Publish 20 addressed heard events.
        for i in range(1, 21):
            payload = {
                "session_id": "e2e",
                "turn_id": f"t-{i:03d}",
                "speaker": "player:imogen",
                "text": f"ARTEMIS, reading {i}?",
                "ts": time.time(),
                "partial": False,
                "meta": {},
            }
            await bot_nc.publish(SUBJECT_HEARD, json.dumps(payload).encode("utf-8"))

        # Wait for all 20 replies, with a generous 10s ceiling.
        try:
            await asyncio.wait_for(reply_received.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail(f"only {len(received)}/20 replies received in 10s")
    finally:
        await bot_nc.close()
        await service.stop()

    # ── assertions ──────────────────────────────────────────────
    assert len(received) == 20
    # Each reply carries the matching turn_id and a proof_hash.
    for i, reply in enumerate(received, start=1):
        assert reply["session_id"] == "e2e"
        assert reply["turn_id"] == f"t-{i:03d}"
        assert reply["text"].startswith("Reading ")
        assert reply["proof_hash"]

    # Reconstruct the proof chain from the on-disk session chain and
    # verify it links end-to-end.
    chain_path = tmp_artemis_dirs / "proofs" / "e2e.chain"
    assert chain_path.exists()
    proofs: list[DecisionProof] = []
    for line in chain_path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        proofs.append(
            DecisionProof(
                turn_id=d["turn_id"],
                session_id=d["session_id"],
                ts=d["ts"],
                reply_sha=d["reply_sha"],
                prev_hash=d["prev_hash"],
                check_results=tuple(),  # unused by verify_chain hash recomputation
                verdict=d["verdict"],
                self_hash=d["self_hash"],
            )
        )
    # Note: we can't call verify_chain here because we stripped the
    # check_results payload above; re-hashing would disagree. Instead,
    # verify the links explicitly.
    assert proofs[0].prev_hash == "genesis"
    for i in range(1, len(proofs)):
        assert (
            proofs[i].prev_hash == proofs[i - 1].self_hash
        ), f"chain broken at turn {i}"


@_REQUIRES_NATS
@pytest.mark.asyncio
async def test_silence_message_suppresses_replies(tmp_artemis_dirs, nats_server):
    """Silence the session, then a heard event should produce no say reply."""
    import nats

    received: list[dict] = []
    bot_nc = await nats.connect(nats_server)

    async def on_say(msg):
        received.append(json.loads(msg.data))

    await bot_nc.subscribe(SUBJECT_SAY, cb=on_say)

    service = ArtemisService(
        nats_url=nats_server,
        bible=Bible(),
        llm=ScriptedLLM(),
        keeper_approval_required=False,
    )
    await service.start()

    try:
        await bot_nc.publish(
            SUBJECT_SILENCE,
            json.dumps({"session_id": "muted", "silenced": True}).encode(),
        )
        # Give the silence a moment to propagate.
        await asyncio.sleep(0.1)

        await bot_nc.publish(
            SUBJECT_HEARD,
            json.dumps(
                {
                    "session_id": "muted",
                    "turn_id": "t-1",
                    "speaker": "player:imogen",
                    "text": "ARTEMIS, anything?",
                    "ts": time.time(),
                    "partial": False,
                    "meta": {},
                }
            ).encode(),
        )
        # Wait a moment for any (non-)reply.
        await asyncio.sleep(0.5)
    finally:
        await bot_nc.close()
        await service.stop()

    assert received == [], f"expected silence, got {received}"
