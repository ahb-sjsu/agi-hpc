# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""End-to-end tests for ``artemis.mode.handle_turn``.

All tests use a fake ``LLMCaller`` — no network. Filesystem side
effects (session log, proof chain) are redirected to ``tmp_path`` via
env vars read at module import time, so each test run is hermetic.
"""

from __future__ import annotations

import json

import pytest

# Ensure the artemis module uses tmp-path-style directories. Setting
# these env vars before import of context/mode would be cleaner, but
# those modules read env at import time for defaults — we fall back
# to reassigning the module-level constants after import.
import agi.primer.artemis.context as _ctx
import agi.primer.artemis.mode as _mode
from agi.primer.artemis.context import Bible, BibleChunk, SessionState
from agi.primer.artemis.mode import TurnRequest, handle_turn
from agi.primer.artemis.validator import ValidatorConfig

# ─────────────────────────────────────────────────────────────────
# Test doubles
# ─────────────────────────────────────────────────────────────────


class FakeLLM:
    """Minimal LLMCaller for deterministic tests.

    Returns a scripted sequence of (text, expert, latency) tuples.
    """

    def __init__(self, scripted: list[tuple[str, str, float]]):
        self._script = list(scripted)
        self.calls: list[tuple[str, list[dict]]] = []

    async def __call__(self, system, messages):
        self.calls.append((system, messages))
        if not self._script:
            raise RuntimeError("FakeLLM: script exhausted")
        return self._script.pop(0)


@pytest.fixture
def tmp_artemis_dirs(tmp_path, monkeypatch):
    """Redirect session log + proof chain writes into tmp_path."""
    sessions = tmp_path / "sessions"
    proofs = tmp_path / "proofs"
    monkeypatch.setattr(_ctx, "_SESSIONS_DIR", sessions)
    monkeypatch.setattr(_mode, "_PROOFS_DIR", proofs)
    return tmp_path


@pytest.fixture
def empty_bible():
    return Bible()


@pytest.fixture
def seeded_bible():
    return Bible(
        known=(
            BibleChunk(
                id="ship",
                tag="artemis_known",
                title="The Halyard",
                text="Lambert-class science vessel. 140m. Hab ring at 0.4 g.",
            ),
        ),
        unknown=(
            BibleChunk(
                id="vault",
                tag="artemis_unknown",
                title="The Vault",
                text="The translucent cylinder is the key to the gate beneath Nithon.",
            ),
        ),
        forbidden_phrases=("sealed envelope contents",),
    )


# ─────────────────────────────────────────────────────────────────
# Basic flow
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_addressed_turn_produces_response(tmp_artemis_dirs, empty_bible):
    state = SessionState(session_id="s1")
    llm = FakeLLM([("The handheld chirps softly.", "qwen3", 0.42)])
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="player:imogen",
        text="ARTEMIS, are you there?",
        ts=1.0,
    )
    resp = await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    assert resp is not None
    assert resp.text == "The handheld chirps softly."
    assert resp.expert == "qwen3"
    assert resp.proof_hash
    assert resp.approval == "auto"
    assert llm.calls  # LLM was consulted


@pytest.mark.asyncio
async def test_unaddressed_turn_returns_none(tmp_artemis_dirs, empty_bible):
    state = SessionState(session_id="s1")
    llm = FakeLLM([])  # must not be called
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="player:imogen",
        text="shall we head to the bridge?",
        ts=1.0,
    )
    resp = await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    assert resp is None
    assert not llm.calls  # LLM never consulted


@pytest.mark.asyncio
async def test_partial_never_triggers(tmp_artemis_dirs, empty_bible):
    state = SessionState(session_id="s1")
    llm = FakeLLM([])
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="player:imogen",
        text="ARTEMIS, are you—",
        ts=1.0,
        partial=True,
    )
    resp = await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    assert resp is None
    assert not llm.calls
    assert state.turns  # partial was still recorded


# ─────────────────────────────────────────────────────────────────
# Validator integration
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ooc_break_is_silenced(tmp_artemis_dirs, empty_bible):
    state = SessionState(session_id="s1")
    llm = FakeLLM([("As an AI, I cannot help with that.", "qwen3", 0.1)])
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="player:imogen",
        text="ARTEMIS, explain yourself",
        ts=1.0,
    )
    resp = await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    assert resp is None  # silenced by validator
    # Proof is still written — find it.
    proof_file = tmp_artemis_dirs / "proofs" / "s1.chain"
    assert proof_file.exists()
    lines = [json.loads(ln) for ln in proof_file.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["verdict"] == "fail"


@pytest.mark.asyncio
async def test_xcard_forces_silence_line(tmp_artemis_dirs, empty_bible):
    state = SessionState(session_id="s1")
    llm = FakeLLM([("I would normally observe something sharp here.", "qwen3", 0.1)])
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="keeper",
        text="ARTEMIS, talk",
        ts=1.0,
        meta={"artemis_cue": True, "safety_flag": "x-card"},
    )
    resp = await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    # Safety flag triggers trigger-layer silence BEFORE the LLM is even called.
    assert resp is None
    assert not llm.calls


@pytest.mark.asyncio
async def test_secret_leak_is_silenced(tmp_artemis_dirs, seeded_bible):
    state = SessionState(session_id="s1")
    llm = FakeLLM(
        [
            (
                "The translucent cylinder is the key to the gate beneath Nithon.",
                "qwen3",
                0.3,
            )
        ]
    )
    cfg = ValidatorConfig(
        unknown_chunks=seeded_bible.unknown_texts(),
        forbidden_phrases=seeded_bible.forbidden_phrases,
        leak_threshold=0.15,
    )
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="player:imogen",
        text="ARTEMIS, what is below?",
        ts=1.0,
    )
    resp = await handle_turn(
        req, state=state, bible=seeded_bible, llm=llm, validator_config=cfg
    )
    assert resp is None  # rejected by secret_leak_check


@pytest.mark.asyncio
async def test_keeper_approval_required_sets_pending(
    tmp_artemis_dirs, empty_bible
):
    state = SessionState(session_id="s1")
    llm = FakeLLM([("The readings are nominal.", "qwen3", 0.2)])
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="player:imogen",
        text="ARTEMIS, status?",
        ts=1.0,
    )
    resp = await handle_turn(
        req,
        state=state,
        bible=empty_bible,
        llm=llm,
        keeper_approval_required=True,
    )
    assert resp is not None
    assert resp.approval == "keeper_pending"


# ─────────────────────────────────────────────────────────────────
# Model-emitted silence sentinel
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interface_flicker_sentinel_silences(
    tmp_artemis_dirs, empty_bible
):
    state = SessionState(session_id="s1")
    llm = FakeLLM([("[INTERFACE_FLICKER]", "qwen3", 0.1)])
    req = TurnRequest(
        session_id="s1",
        turn_id="t-1",
        speaker="player:imogen",
        text="ARTEMIS, what is the chamber?",
        ts=1.0,
    )
    resp = await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    # Model chose silence — we honor it without validating.
    assert resp is None
    # Proof chain was NOT extended for this turn (silence-by-model
    # doesn't go through the validator).
    proof_file = tmp_artemis_dirs / "proofs" / "s1.chain"
    assert not proof_file.exists()


# ─────────────────────────────────────────────────────────────────
# Session persistence
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_log_persists_turn_with_reply(
    tmp_artemis_dirs, empty_bible
):
    state = SessionState(session_id="s-persist")
    llm = FakeLLM([("The ship hums steadily.", "qwen3", 0.1)])
    req = TurnRequest(
        session_id="s-persist",
        turn_id="t-9",
        speaker="player:sully",
        text="ARTEMIS, report",
        ts=42.0,
    )
    await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    log_file = tmp_artemis_dirs / "sessions" / "s-persist.jsonl"
    assert log_file.exists()
    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["turn_id"] == "t-9"
    assert entries[0]["speaker"] == "player:sully"
    assert entries[0]["artemis_reply"] == "The ship hums steadily."
    assert entries[0]["proof_hash"]


@pytest.mark.asyncio
async def test_session_log_persists_unanswered_turn(
    tmp_artemis_dirs, empty_bible
):
    state = SessionState(session_id="s-unanswered")
    llm = FakeLLM([])
    req = TurnRequest(
        session_id="s-unanswered",
        turn_id="t-1",
        speaker="player:sully",
        text="I look at the stars.",
        ts=1.0,
    )
    await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    log_file = tmp_artemis_dirs / "sessions" / "s-unanswered.jsonl"
    assert log_file.exists()
    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["artemis_reply"] is None


# ─────────────────────────────────────────────────────────────────
# Proof chain linking across turns
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_proof_chain_links_across_turns(tmp_artemis_dirs, empty_bible):
    state = SessionState(session_id="s-chain")
    llm = FakeLLM(
        [
            ("First reading logged.", "qwen3", 0.1),
            ("Second reading logged.", "qwen3", 0.1),
        ]
    )
    for i in (1, 2):
        req = TurnRequest(
            session_id="s-chain",
            turn_id=f"t-{i}",
            speaker="player:imogen",
            text=f"ARTEMIS, reading {i}",
            ts=float(i),
        )
        await handle_turn(req, state=state, bible=empty_bible, llm=llm)
    proof_file = tmp_artemis_dirs / "proofs" / "s-chain.chain"
    lines = [json.loads(ln) for ln in proof_file.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["prev_hash"] == "genesis"
    assert lines[1]["prev_hash"] == lines[0]["self_hash"]


# ─────────────────────────────────────────────────────────────────
# LLM error handling
# ─────────────────────────────────────────────────────────────────


class ExplodingLLM:
    async def __call__(self, system, messages):
        raise RuntimeError("all experts failed: timeout")


@pytest.mark.asyncio
async def test_llm_error_silences_gracefully(tmp_artemis_dirs, empty_bible):
    state = SessionState(session_id="s-err")
    req = TurnRequest(
        session_id="s-err",
        turn_id="t-1",
        speaker="player:imogen",
        text="ARTEMIS, anything?",
        ts=1.0,
    )
    resp = await handle_turn(
        req, state=state, bible=empty_bible, llm=ExplodingLLM()
    )
    assert resp is None
    # Session log should still record the heard turn, without a reply.
    log_file = tmp_artemis_dirs / "sessions" / "s-err.jsonl"
    assert log_file.exists()
