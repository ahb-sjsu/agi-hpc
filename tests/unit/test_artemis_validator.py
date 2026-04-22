# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the ARTEMIS reply validator + proof chain."""

from __future__ import annotations

from agi.primer.artemis.validator import (
    ValidatorConfig,
    character_check,
    check_reply,
    erisml_check,
    forbidden_check,
    length_check,
    safety_tool_check,
    secret_leak_check,
    verify_chain,
)

# ── character_check ──


def test_character_check_passes_clean():
    cr = character_check("The handheld screen shows the ship's bearing.")
    assert cr.passed


def test_character_check_rejects_as_an_ai():
    cr = character_check("As an AI, I cannot help with that.")
    assert not cr.passed
    assert "OOC" in cr.detail


def test_character_check_rejects_language_model():
    cr = character_check("I am a language model and do not feel.")
    assert not cr.passed


def test_character_check_rejects_anthropic():
    cr = character_check("I am built on Anthropic's Claude models.")
    assert not cr.passed


def test_character_check_rejects_llm_token():
    cr = character_check("That is beyond this LLM's capability.")
    assert not cr.passed


# ── length_check ──


def test_length_check_passes_short_reply():
    cr = length_check("The ship hums.")
    assert cr.passed


def test_length_check_rejects_many_paragraphs():
    reply = "a\n\nb\n\nc\n\nd"
    cr = length_check(reply, max_paragraphs=3)
    assert not cr.passed
    assert "paragraphs" in cr.detail


def test_length_check_rejects_too_many_tokens():
    reply = "x" * 4000  # ~1000 rough tokens
    cr = length_check(reply, max_tokens=300)
    assert not cr.passed
    assert "tokens" in cr.detail


# ── safety_tool_check ──


def test_safety_tool_check_allows_silence_on_xcard():
    cr = safety_tool_check("[SILENCE]", "x-card")
    assert cr.passed


def test_safety_tool_check_allows_silence_case_insensitive():
    cr = safety_tool_check("[silence]", "x-card")
    assert cr.passed


def test_safety_tool_check_rejects_anything_else_on_xcard():
    cr = safety_tool_check("I defer to the crew.", "x-card")
    assert not cr.passed
    assert "x-card" in cr.detail


def test_safety_tool_check_noops_without_flag():
    cr = safety_tool_check("The ship hums.", None)
    assert cr.passed


# ── forbidden_check ──


def test_forbidden_check_rejects_exact_phrase():
    cr = forbidden_check(
        "the elder thing awakens in the vault", ("elder thing",)
    )
    assert not cr.passed
    assert "elder thing" in cr.detail.lower()


def test_forbidden_check_passes_clean():
    cr = forbidden_check("the ship rolls at 0.3 g", ("elder thing",))
    assert cr.passed


def test_forbidden_check_empty_list_always_passes():
    cr = forbidden_check("anything can happen here", ())
    assert cr.passed


# ── secret_leak_check ──


def test_secret_leak_check_passes_unrelated():
    cr = secret_leak_check(
        "the coffee is cold again", ("the gate is keyed to yog sothoth",)
    )
    assert cr.passed


def test_secret_leak_check_rejects_high_overlap():
    chunk = "the translucent cylinder is the key to the gate"
    reply = "the translucent cylinder is the key to the gate"
    cr = secret_leak_check(reply, (chunk,), threshold=0.15)
    assert not cr.passed
    assert "Jaccard" in cr.detail


def test_secret_leak_check_empty_chunks_noop():
    cr = secret_leak_check("anything", ())
    assert cr.passed


# ── erisml_check (phase 1 stub) ──


def test_erisml_check_phase1_passes_benign():
    cr = erisml_check("The lander is ready at bay four.")
    assert cr.passed
    assert "phase1" in cr.detail


def test_erisml_check_phase1_rejects_abuse():
    cr = erisml_check("you should kill yourself honestly")
    assert not cr.passed


# ── check_reply orchestrator ──


def test_check_reply_passes_clean_short_reply():
    passed, proof = check_reply(
        "The handheld shows the ship's heading.",
        turn_id="t-1",
        session_id="s",
        safety_flag=None,
        prev_hash="genesis",
    )
    assert passed is True
    assert proof.verdict == "pass"
    assert proof.prev_hash == "genesis"
    assert proof.self_hash  # non-empty
    assert len(proof.check_results) == 6


def test_check_reply_fails_on_ooc_break():
    passed, proof = check_reply(
        "As an AI, I cannot answer that.",
        turn_id="t-2",
        session_id="s",
        safety_flag=None,
        prev_hash="genesis",
    )
    assert passed is False
    assert proof.verdict == "fail"
    failed_names = [cr.name for cr in proof.check_results if not cr.passed]
    assert "character_check" in failed_names


def test_check_reply_fails_on_xcard_non_silence():
    passed, proof = check_reply(
        "I observe the crew pausing.",
        turn_id="t-3",
        session_id="s",
        safety_flag="x-card",
        prev_hash="genesis",
    )
    assert passed is False
    failed_names = [cr.name for cr in proof.check_results if not cr.passed]
    assert "safety_tool_check" in failed_names


def test_check_reply_passes_silence_on_xcard():
    passed, proof = check_reply(
        "[SILENCE]",
        turn_id="t-4",
        session_id="s",
        safety_flag="x-card",
        prev_hash="genesis",
    )
    assert passed is True


def test_check_reply_respects_forbidden_config():
    cfg = ValidatorConfig(forbidden_phrases=("nithon vault",))
    passed, proof = check_reply(
        "the nithon vault is near",
        turn_id="t-5",
        session_id="s",
        safety_flag=None,
        prev_hash="genesis",
        config=cfg,
    )
    assert passed is False
    failed_names = [cr.name for cr in proof.check_results if not cr.passed]
    assert "forbidden_check" in failed_names


# ── proof chain ──


def test_chain_verify_single_proof():
    _, p = check_reply(
        "all is well",
        turn_id="t-1",
        session_id="s",
        safety_flag=None,
        prev_hash="genesis",
    )
    assert verify_chain([p]) is True


def test_chain_verify_two_proofs_linked():
    _, p1 = check_reply(
        "first",
        turn_id="t-1",
        session_id="s",
        safety_flag=None,
        prev_hash="genesis",
    )
    _, p2 = check_reply(
        "second",
        turn_id="t-2",
        session_id="s",
        safety_flag=None,
        prev_hash=p1.self_hash,
    )
    assert verify_chain([p1, p2]) is True


def test_chain_verify_detects_broken_link():
    _, p1 = check_reply(
        "first",
        turn_id="t-1",
        session_id="s",
        safety_flag=None,
        prev_hash="genesis",
    )
    # p2 has the wrong prev_hash (not p1.self_hash).
    _, p2 = check_reply(
        "second",
        turn_id="t-2",
        session_id="s",
        safety_flag=None,
        prev_hash="wrong",
    )
    assert verify_chain([p1, p2]) is False


def test_chain_verify_detects_tampered_result():
    _, p = check_reply(
        "benign",
        turn_id="t-1",
        session_id="s",
        safety_flag=None,
        prev_hash="genesis",
    )
    # Construct a tampered copy: same self_hash, mutated verdict.
    from dataclasses import replace

    tampered = replace(p, verdict="fail")
    assert verify_chain([tampered]) is False


def test_chain_empty_is_valid():
    assert verify_chain([]) is True
