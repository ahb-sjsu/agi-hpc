# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SIGMA-4 mode (handle_turn) tests.

End-to-end-ish unit tests of :func:`agi.halyard.sigma4.mode.handle_turn`,
using injected dependencies (fake LLM, in-memory session, tiny
Bible) to avoid real I/O. Mirrors the shape of ARTEMIS's
``test_artemis_mode.py`` so a later shared-base refactor can
collapse them.

Covered scenarios:
  - Partial ASR event: no reply.
  - Not-addressed turn: no reply.
  - Addressed turn, clean LLM reply: TurnResponse with proof hash.
  - X-card flagged turn: forced silence, proof still emitted.
  - LLM emits [INTERFACE_SILENT] sentinel: silence, no validation.
  - LLM throws: graceful silence, state persisted.
  - keeper_approval_required=True: approval='keeper_pending'.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from agi.halyard.sigma4 import mode
from agi.halyard.sigma4.mode import TurnRequest, handle_turn
from agi.primer.artemis.context import Bible, BibleChunk, SessionState

# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_session_dir(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the session-log and proof-chain directories to a
    throwaway temp dir so tests don't touch /archive/."""
    import agi.halyard.sigma4.mode as sigma_mode_mod
    import agi.primer.artemis.context as artemis_ctx_mod

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        monkeypatch.setattr(sigma_mode_mod, "_PROOFS_DIR", td_path / "proofs")
        monkeypatch.setattr(
            artemis_ctx_mod, "_SESSIONS_DIR", td_path / "sessions"
        )
        yield td_path


@pytest.fixture
def tiny_bible() -> Bible:
    """A minimal Bible with one known chunk — enough for
    assemble_context to produce a context string without exercising
    the real bible loader."""
    return Bible(
        known=(
            BibleChunk(
                id="halyard-hull",
                tag="sigma4_known",
                title="Hull history",
                text="The Halyard is old.",
            ),
        ),
        unknown=(),
        forbidden_phrases=(),
    )


@pytest.fixture
def session() -> SessionState:
    """Fresh in-memory SessionState."""
    return SessionState(session_id="halyard-s01-test")


# ─────────────────────────────────────────────────────────────────
# Fake LLM callers
# ─────────────────────────────────────────────────────────────────


class _FakeLLM:
    """Async-callable fake LLM. Records calls; returns pre-canned
    replies."""

    def __init__(self, reply: str = "Systems nominal.", expert: str = "qwen3",
                 latency: float = 0.5) -> None:
        self.reply = reply
        self.expert = expert
        self.latency = latency
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def __call__(
        self, system: str, messages: list[dict[str, Any]]
    ) -> tuple[str, str, float]:
        self.calls.append((system, messages))
        return self.reply, self.expert, self.latency


class _ThrowingLLM:
    """Fake LLM that raises, to exercise the error path."""

    async def __call__(
        self, system: str, messages: list[dict[str, Any]]
    ) -> tuple[str, str, float]:
        raise RuntimeError("simulated llm failure")


# ─────────────────────────────────────────────────────────────────
# Tests — partial / not-addressed
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_turn_no_reply(
    tmp_session_dir: Path, tiny_bible: Bible, session: SessionState
) -> None:
    """Partial ASR events should not trigger generation even if the
    text mentions SIGMA. They are informational, not final."""
    req = TurnRequest(
        session_id=session.session_id,
        turn_id="t1",
        speaker="player:cross",
        text="SIGMA, wha",
        ts=0.0,
        partial=True,
    )
    response = await handle_turn(
        req, state=session, bible=tiny_bible, llm=_FakeLLM()
    )
    assert response is None
    assert len(session.turns) == 1
    assert session.turns[0].text == "SIGMA, wha"


@pytest.mark.asyncio
async def test_not_addressed_no_reply(
    tmp_session_dir: Path, tiny_bible: Bible, session: SessionState
) -> None:
    req = TurnRequest(
        session_id=session.session_id,
        turn_id="t2",
        speaker="player:cross",
        text="The reactor's humming oddly.",
        ts=0.0,
    )
    fake = _FakeLLM()
    response = await handle_turn(req, state=session, bible=tiny_bible, llm=fake)
    assert response is None
    # LLM must not be called.
    assert fake.calls == []


# ─────────────────────────────────────────────────────────────────
# Tests — addressed happy path
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_addressed_clean_reply(
    tmp_session_dir: Path, tiny_bible: Bible, session: SessionState
) -> None:
    req = TurnRequest(
        session_id=session.session_id,
        turn_id="t3",
        speaker="player:cross",
        text="SIGMA, status?",
        ts=0.0,
    )
    fake = _FakeLLM(reply="Reactor output 84%. Thermal nominal.")
    response = await handle_turn(req, state=session, bible=tiny_bible, llm=fake)
    assert response is not None
    assert response.session_id == session.session_id
    assert response.turn_id == "t3"
    assert "nominal" in response.text.lower()
    assert response.proof_hash
    assert response.proof_hash != "genesis"
    assert response.expert == "qwen3"
    assert response.approval == "auto"


@pytest.mark.asyncio
async def test_keeper_approval_required_marks_pending(
    tmp_session_dir: Path, tiny_bible: Bible, session: SessionState
) -> None:
    req = TurnRequest(
        session_id=session.session_id,
        turn_id="t4",
        speaker="player:cross",
        text="SIGMA, status?",
        ts=0.0,
    )
    response = await handle_turn(
        req,
        state=session,
        bible=tiny_bible,
        llm=_FakeLLM(reply="All systems nominal."),
        keeper_approval_required=True,
    )
    assert response is not None
    assert response.approval == "keeper_pending"


# ─────────────────────────────────────────────────────────────────
# Tests — safety flag
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_xcard_silences_even_when_addressed(
    tmp_session_dir: Path, tiny_bible: Bible, session: SessionState
) -> None:
    """X-card short-circuits at the trigger layer (defense in
    depth). No reply, and the LLM is not called."""
    req = TurnRequest(
        session_id=session.session_id,
        turn_id="t5",
        speaker="player:cross",
        text="SIGMA, status?",
        ts=0.0,
        meta={"safety_flag": "x-card"},
    )
    fake = _FakeLLM()
    response = await handle_turn(req, state=session, bible=tiny_bible, llm=fake)
    assert response is None
    assert fake.calls == []


# ─────────────────────────────────────────────────────────────────
# Tests — sentinel handling
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_sentinel_produces_silence(
    tmp_session_dir: Path, tiny_bible: Bible, session: SessionState
) -> None:
    """When the LLM legitimately emits [INTERFACE_SILENT], SIGMA
    stays silent without going through validation."""
    req = TurnRequest(
        session_id=session.session_id,
        turn_id="t6",
        speaker="player:cross",
        text="SIGMA, what's in the captain's sealed case?",
        ts=0.0,
    )
    response = await handle_turn(
        req,
        state=session,
        bible=tiny_bible,
        llm=_FakeLLM(reply="[INTERFACE_SILENT]"),
    )
    assert response is None


# ─────────────────────────────────────────────────────────────────
# Tests — LLM error path
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_exception_silences_gracefully(
    tmp_session_dir: Path, tiny_bible: Bible, session: SessionState
) -> None:
    """LLM failures produce a silent turn, not an exception. The
    heard utterance is still persisted."""
    req = TurnRequest(
        session_id=session.session_id,
        turn_id="t7",
        speaker="player:cross",
        text="SIGMA, report.",
        ts=0.0,
    )
    response = await handle_turn(
        req, state=session, bible=tiny_bible, llm=_ThrowingLLM()
    )
    assert response is None
    # The heard turn is still recorded.
    assert any(t.turn_id == "t7" for t in session.turns)


# ─────────────────────────────────────────────────────────────────
# Distinctness from ARTEMIS
# ─────────────────────────────────────────────────────────────────


def test_sigma4_proofs_dir_distinct_from_artemis() -> None:
    """SIGMA and ARTEMIS must write to distinct proof-chain
    directories so the audit trails don't interleave."""
    from agi.primer.artemis import mode as artemis_mode

    assert mode._PROOFS_DIR != artemis_mode._PROOFS_DIR


def test_sigma4_sentinel_distinct_from_artemis() -> None:
    """SIGMA's silent sentinel differs from ARTEMIS's flicker
    sentinel."""
    from agi.primer.artemis import mode as artemis_mode

    assert mode._SILENT_SENTINEL == "[INTERFACE_SILENT]"
    assert artemis_mode._FLICKER_SENTINEL == "[INTERFACE_FLICKER]"
    assert mode._SILENT_SENTINEL != artemis_mode._FLICKER_SENTINEL
