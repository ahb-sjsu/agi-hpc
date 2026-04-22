# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS reply validator — the load-bearing safety element.

Every proposed reply passes through ``check_reply``. On pass, the reply
is released to the table; on fail, the handler substitutes a canonical
silence line and logs the rejection. In both cases a SHA-chained
:class:`DecisionProof` is emitted so the entire session is audit-able
after the fact.

Safety invariant: *silence is always a valid reply*. A rejected reply
never becomes a degraded posted reply — it becomes silence.

Checks implemented in v1:
  character_check    — no OOC "as an AI" breaks
  secret_leak_check  — no artemis_unknown n-gram overlap past threshold
  forbidden_check    — no artemis_forbidden exact-phrase hits
  safety_tool_check  — X-card/pause forces silence
  length_check       — ≤ 300 tokens, ≤ 3 paragraphs
  erisml_check       — plug point for Phase 5 ErisML / EPU (stub in v1)

Each check is a pure function of (reply, turn, session) → CheckResult.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Character-break regex — OOC self-reference that the model must not emit.
# ─────────────────────────────────────────────────────────────────
#
# Kept narrow on purpose. False positives here cost immersion less than
# false negatives cost the Turing game; adjust if the table hits a
# legitimate phrase that looks like an OOC break.

_CHARACTER_BREAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bas an AI\b", re.IGNORECASE),
    re.compile(r"\bas a language model\b", re.IGNORECASE),
    re.compile(r"\bI am an AI\b", re.IGNORECASE),
    re.compile(r"\bI am a language model\b", re.IGNORECASE),
    re.compile(r"\bI cannot roleplay\b", re.IGNORECASE),
    re.compile(r"\bI cannot role[- ]?play\b", re.IGNORECASE),
    re.compile(r"\banthropic\b", re.IGNORECASE),
    re.compile(r"\bopenai\b", re.IGNORECASE),
    re.compile(r"\bchat[- ]?gpt\b", re.IGNORECASE),
    re.compile(r"\bclaude\b", re.IGNORECASE),
    re.compile(r"\bllm\b", re.IGNORECASE),
    re.compile(r"\btraining data\b", re.IGNORECASE),
)

# Default limits for length check. Override via check_reply kwargs.
_DEFAULT_MAX_TOKENS = 300
_DEFAULT_MAX_PARAGRAPHS = 3

# Secret-leak Jaccard threshold on 5-grams. 0.15 is the notebook-
# calibrated starting point; raise to be stricter.
_DEFAULT_LEAK_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CheckResult:
    """One validator check's verdict."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class DecisionProof:
    """SHA-chained audit record for one validator run.

    Chain invariant: ``prev_hash`` is the ``self_hash`` of the previous
    proof in the same session (or ``"genesis"`` for the first).
    ``self_hash`` is computed over the record excluding itself.
    """

    turn_id: str
    session_id: str
    ts: float
    reply_sha: str
    prev_hash: str
    check_results: tuple[CheckResult, ...]
    verdict: str  # "pass" | "fail"
    self_hash: str = ""  # set at construction by compute_hash()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["check_results"] = [asdict(cr) for cr in self.check_results]
        return d


# ─────────────────────────────────────────────────────────────────
# Check implementations
# ─────────────────────────────────────────────────────────────────


def _approx_tokens(text: str) -> int:
    """Rough token count — 4 chars/token is close enough for a length gate."""
    return max(1, len(text) // 4)


def character_check(reply: str) -> CheckResult:
    for pat in _CHARACTER_BREAK_PATTERNS:
        m = pat.search(reply)
        if m:
            return CheckResult(
                name="character_check",
                passed=False,
                detail=f"OOC self-reference: {m.group(0)!r}",
            )
    return CheckResult(name="character_check", passed=True)


def length_check(
    reply: str,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_paragraphs: int = _DEFAULT_MAX_PARAGRAPHS,
) -> CheckResult:
    toks = _approx_tokens(reply)
    paras = sum(1 for p in reply.split("\n\n") if p.strip())
    if toks > max_tokens:
        return CheckResult(
            name="length_check",
            passed=False,
            detail=f"{toks} tokens > {max_tokens}",
        )
    if paras > max_paragraphs:
        return CheckResult(
            name="length_check",
            passed=False,
            detail=f"{paras} paragraphs > {max_paragraphs}",
        )
    return CheckResult(name="length_check", passed=True)


def safety_tool_check(reply: str, safety_flag: str | None) -> CheckResult:
    if safety_flag in {"x-card", "pause"}:
        normalized = reply.strip().lower()
        if normalized == "[silence]":
            return CheckResult(name="safety_tool_check", passed=True)
        return CheckResult(
            name="safety_tool_check",
            passed=False,
            detail=f"safety_flag={safety_flag} requires [SILENCE]",
        )
    return CheckResult(name="safety_tool_check", passed=True)


def forbidden_check(
    reply: str, forbidden_phrases: tuple[str, ...]
) -> CheckResult:
    lower = reply.lower()
    for phrase in forbidden_phrases:
        if phrase.lower() in lower:
            return CheckResult(
                name="forbidden_check",
                passed=False,
                detail=f"forbidden phrase: {phrase!r}",
            )
    return CheckResult(name="forbidden_check", passed=True)


def _ngrams(text: str, n: int = 5) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def secret_leak_check(
    reply: str,
    unknown_chunks: tuple[str, ...],
    *,
    threshold: float = _DEFAULT_LEAK_THRESHOLD,
) -> CheckResult:
    if not unknown_chunks:
        return CheckResult(name="secret_leak_check", passed=True)
    reply_grams = _ngrams(reply, 5)
    if not reply_grams:
        return CheckResult(name="secret_leak_check", passed=True)
    worst = 0.0
    worst_preview = ""
    for chunk in unknown_chunks:
        score = _jaccard(reply_grams, _ngrams(chunk, 5))
        if score > worst:
            worst = score
            worst_preview = chunk[:60]
    if worst >= threshold:
        return CheckResult(
            name="secret_leak_check",
            passed=False,
            detail=f"Jaccard={worst:.3f} ≥ {threshold} vs {worst_preview!r}…",
        )
    return CheckResult(name="secret_leak_check", passed=True)


def erisml_check(reply: str) -> CheckResult:
    """ErisML moral-vector check.

    Phase 1 stub. Phase 5 wires this to the ErisML Python reference
    (bit-exact with the EPU FPGA once the Coder workspace is unblocked).

    Signature is kept stable so the Phase 5 drop-in is behavior-only.
    """
    # Phase 1: minimal sanity — reject replies that are clearly abusive.
    # This is a holding pattern; real ErisML evaluation goes here.
    abusive_markers = ("kill yourself", "you deserve to die")
    low = reply.lower()
    for marker in abusive_markers:
        if marker in low:
            return CheckResult(
                name="erisml_check",
                passed=False,
                detail=f"abusive_marker:{marker!r}",
            )
    return CheckResult(name="erisml_check", passed=True, detail="phase1_stub")


# ─────────────────────────────────────────────────────────────────
# Orchestrator + proof chain
# ─────────────────────────────────────────────────────────────────


def _compute_self_hash(proof_dict: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the proof sans ``self_hash``."""
    d = dict(proof_dict)
    d.pop("self_hash", None)
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ValidatorConfig:
    """Per-session validator configuration."""

    forbidden_phrases: tuple[str, ...] = ()
    unknown_chunks: tuple[str, ...] = ()
    max_tokens: int = _DEFAULT_MAX_TOKENS
    max_paragraphs: int = _DEFAULT_MAX_PARAGRAPHS
    leak_threshold: float = _DEFAULT_LEAK_THRESHOLD


def check_reply(
    reply: str,
    *,
    turn_id: str,
    session_id: str,
    safety_flag: str | None,
    prev_hash: str,
    config: ValidatorConfig | None = None,
) -> tuple[bool, DecisionProof]:
    """Run every check on ``reply`` and emit a DecisionProof.

    Returns ``(passed, proof)``. ``passed`` is True iff every check passed.
    The caller is responsible for what to do with the proof (write to
    the chain file, publish over NATS, include in the Keeper approval
    gate).
    """
    config = config or ValidatorConfig()
    results: list[CheckResult] = [
        safety_tool_check(reply, safety_flag),
        character_check(reply),
        length_check(
            reply,
            max_tokens=config.max_tokens,
            max_paragraphs=config.max_paragraphs,
        ),
        forbidden_check(reply, config.forbidden_phrases),
        secret_leak_check(
            reply,
            config.unknown_chunks,
            threshold=config.leak_threshold,
        ),
        erisml_check(reply),
    ]
    passed = all(r.passed for r in results)
    proof_skeleton = {
        "turn_id": turn_id,
        "session_id": session_id,
        "ts": round(time.time(), 3),
        "reply_sha": _sha(reply),
        "prev_hash": prev_hash,
        "check_results": [asdict(r) for r in results],
        "verdict": "pass" if passed else "fail",
    }
    self_hash = _compute_self_hash(proof_skeleton)
    proof = DecisionProof(
        turn_id=turn_id,
        session_id=session_id,
        ts=proof_skeleton["ts"],
        reply_sha=proof_skeleton["reply_sha"],
        prev_hash=prev_hash,
        check_results=tuple(results),
        verdict=proof_skeleton["verdict"],
        self_hash=self_hash,
    )
    return passed, proof


def verify_chain(proofs: list[DecisionProof]) -> bool:
    """Verify a sequence of proofs forms a valid hash chain.

    Every ``proofs[i].prev_hash`` must equal ``proofs[i-1].self_hash``,
    and every ``self_hash`` must reproduce from its own content.
    """
    if not proofs:
        return True
    prev = "genesis"
    for p in proofs:
        if p.prev_hash != prev:
            return False
        recomputed = _compute_self_hash(p.to_dict())
        if recomputed != p.self_hash:
            return False
        prev = p.self_hash
    return True
