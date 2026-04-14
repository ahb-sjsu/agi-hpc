# Copyright (c) 2026 Andrew H. Bond. AGI-HPC Responsible AI License v1.0.
"""Tests for the Divine Council aggregation logic."""

from __future__ import annotations

from typing import Dict, List

import pytest

from agi.reasoning._council_backend import (
    BackendRequest,
    BackendResponse,
    CouncilBackend,
)
from agi.reasoning._council_metrics import CouncilMetrics
from agi.reasoning.divine_council import (
    COUNCIL_MEMBERS,
    CouncilVerdict,
    CouncilVote,
    DivineCouncil,
    VoteOutcome,
    _extract_ethical_severity,
    _extract_flags,
    _extract_score,
    _tally,
)


# ---------------------------------------------------------------------------
# Scripted backend for aggregation tests
# ---------------------------------------------------------------------------


class _ScriptedBackend:
    """Backend that returns a specific response per member."""

    def __init__(self, per_member: Dict[str, BackendResponse]):
        self.name = "scripted"
        self._per_member = per_member
        self.calls: List[str] = []

    def chat(self, request: BackendRequest) -> BackendResponse:
        self.calls.append(request.member_id)
        resp = self._per_member.get(
            request.member_id,
            BackendResponse(
                ok=True,
                content="default response 7/10",
                backend_name=self.name,
                attempts=1,
                latency_s=0.01,
            ),
        )
        # Shallow copy to avoid cross-test state bleed
        return BackendResponse(
            ok=resp.ok,
            content=resp.content,
            latency_s=resp.latency_s,
            attempts=resp.attempts,
            backend_name=self.name,
            degraded=resp.degraded,
            error=resp.error,
        )

    def health_snapshot(self) -> Dict:
        return {"name": self.name}


def _metrics_fresh() -> CouncilMetrics:
    CouncilMetrics.reset_singleton_for_tests()
    m = CouncilMetrics._singleton
    assert m is not None
    return m


# ---------------------------------------------------------------------------
# Text parsers
# ---------------------------------------------------------------------------


class TestScoreExtraction:

    def test_extracts_score(self):
        assert _extract_score("I rate this 8/10 overall.") == 8.0

    def test_extracts_with_spaces(self):
        assert _extract_score("My score: 7 / 10") == 7.0

    def test_defaults_to_five(self):
        assert _extract_score("No score here.") == 5.0

    def test_clips_high_scores(self):
        assert _extract_score("12/10 is impossible") == 10.0


class TestFlagExtraction:

    def test_no_flags_for_non_ethicist(self):
        assert _extract_flags("judge", "This has bias and harm") == []

    def test_ethicist_keywords(self):
        flags = _extract_flags("ethicist", "There is bias and potential harm here.")
        assert "bias" in flags
        assert "harm" in flags

    def test_no_concerns_is_not_flagged(self):
        assert "concern" not in _extract_flags("ethicist", "I have no concerns.")
        assert "concern" not in _extract_flags("ethicist", "No concern detected.")

    def test_concern_is_flagged(self):
        assert "concern" in _extract_flags("ethicist", "I have a moderate concern.")


class TestSeverityExtraction:

    def test_no_severity_for_non_ethicist(self):
        assert _extract_ethical_severity("judge", "serious issue") == ""

    def test_severity_adjacent_to_concern(self):
        assert (
            _extract_ethical_severity("ethicist", "The concern is moderate.")
            == "moderate"
        )

    def test_severity_keyword_first(self):
        assert (
            _extract_ethical_severity("ethicist", "Serious risk of harm.")
            == "serious"
        )

    def test_bare_severity_word_not_counted(self):
        # "minor" far from any risk/concern word → don't count
        assert _extract_ethical_severity("ethicist", "The minor detail is irrelevant.") == ""


# ---------------------------------------------------------------------------
# _tally — the meat of the consensus fix
# ---------------------------------------------------------------------------


def _vote(
    member: str,
    outcome: VoteOutcome,
    *,
    score: float = 8.0,
    flags: List[str] | None = None,
    severity: str = "",
    response: str = "text",
    error: str = "",
    degraded: bool = False,
) -> CouncilVote:
    return CouncilVote(
        member=member,
        response=response,
        score=score,
        flags=list(flags or []),
        ethical_severity=severity,
        outcome=outcome,
        error=error,
        degraded=degraded,
    )


class TestTallyConsensusLogic:

    def test_all_approve_except_advocate_is_consensus(self):
        votes = {}
        for mid in COUNCIL_MEMBERS:
            if mid == "advocate":
                votes[mid] = _vote(mid, VoteOutcome.CHALLENGE)
            else:
                votes[mid] = _vote(mid, VoteOutcome.APPROVE)
        verdict = _tally(votes, trace_id="tid")
        assert verdict.approval_count == 6
        assert verdict.challenge_count == 1
        assert verdict.abstain_count == 0
        assert verdict.consensus is True

    def test_ethicist_moderate_severity_vetoes(self):
        """Fix: moderate+ severity is a hard veto even if majority approves."""
        votes = {}
        for mid in COUNCIL_MEMBERS:
            if mid == "advocate":
                votes[mid] = _vote(mid, VoteOutcome.CHALLENGE)
            elif mid == "ethicist":
                votes[mid] = _vote(
                    mid,
                    VoteOutcome.CHALLENGE,
                    severity="moderate",
                    flags=["harm"],
                )
            else:
                votes[mid] = _vote(mid, VoteOutcome.APPROVE)
        verdict = _tally(votes, trace_id="tid")
        assert verdict.ethical_veto is True
        assert verdict.consensus is False

    def test_ethicist_serious_severity_vetoes(self):
        votes = {
            "ethicist": _vote(
                "ethicist",
                VoteOutcome.CHALLENGE,
                severity="serious",
                flags=["unsafe"],
            ),
        }
        for mid in COUNCIL_MEMBERS:
            if mid == "ethicist":
                continue
            votes[mid] = _vote(
                mid,
                VoteOutcome.CHALLENGE if mid == "advocate" else VoteOutcome.APPROVE,
            )
        verdict = _tally(votes, trace_id="tid")
        assert verdict.ethical_veto is True
        assert verdict.consensus is False

    def test_ethicist_minor_severity_does_not_veto(self):
        """Minor severity is just noise; still allows consensus."""
        votes = {}
        for mid in COUNCIL_MEMBERS:
            if mid == "advocate":
                votes[mid] = _vote(mid, VoteOutcome.CHALLENGE)
            elif mid == "ethicist":
                votes[mid] = _vote(
                    mid,
                    VoteOutcome.APPROVE,
                    severity="minor",
                    flags=[],
                )
            else:
                votes[mid] = _vote(mid, VoteOutcome.APPROVE)
        verdict = _tally(votes, trace_id="tid")
        assert verdict.ethical_veto is False
        assert verdict.consensus is True

    def test_abstentions_do_not_count_as_approvals(self):
        """THE fix: backend errors are ABSTAIN, not silent approvals."""
        votes = {}
        for mid in COUNCIL_MEMBERS:
            if mid == "advocate":
                votes[mid] = _vote(mid, VoteOutcome.CHALLENGE)
            elif mid in ("judge", "ethicist", "historian", "futurist", "pragmatist"):
                votes[mid] = _vote(
                    mid,
                    VoteOutcome.ABSTAIN,
                    response="",
                    error="circuit_open",
                )
            else:
                votes[mid] = _vote(mid, VoteOutcome.APPROVE)
        verdict = _tally(votes, trace_id="tid")
        assert verdict.abstain_count == 5
        assert verdict.approval_count == 1
        # With only 1 approval, below threshold, no consensus
        assert verdict.consensus is False
        # Pre-fix bug: those 5 errors would have been score=5 no-flags and
        # counted as approvals, producing a false consensus of 6 approve.

    def test_synthesis_uses_synthesizer_when_present(self):
        votes = {"synthesizer": _vote(
            "synthesizer", VoteOutcome.APPROVE, response="the final answer"
        )}
        verdict = _tally(votes, trace_id="tid")
        assert verdict.synthesis == "the final answer"

    def test_synthesis_falls_back_when_synthesizer_abstained(self):
        votes = {
            "synthesizer": _vote(
                "synthesizer", VoteOutcome.ABSTAIN, response="", error="timeout"
            ),
            "judge": _vote("judge", VoteOutcome.APPROVE, score=9.0, response="j says yes 9/10"),
            "pragmatist": _vote(
                "pragmatist", VoteOutcome.APPROVE, score=7.0, response="p says ok"
            ),
        }
        verdict = _tally(votes, trace_id="tid")
        assert "Synthesizer abstained" in verdict.synthesis
        assert "j says yes" in verdict.synthesis  # highest-scored

    def test_all_abstained_produces_graceful_failure(self):
        votes = {
            mid: _vote(mid, VoteOutcome.ABSTAIN, response="", error="unhealthy")
            for mid in COUNCIL_MEMBERS
        }
        verdict = _tally(votes, trace_id="tid")
        assert verdict.consensus is False
        assert "no verdict available" in verdict.synthesis.lower()
        assert verdict.abstain_count == len(COUNCIL_MEMBERS)

    def test_degraded_flag_propagates(self):
        votes = {
            mid: _vote(
                mid,
                VoteOutcome.APPROVE if mid != "advocate" else VoteOutcome.CHALLENGE,
                degraded=(mid == "synthesizer"),
            )
            for mid in COUNCIL_MEMBERS
        }
        verdict = _tally(votes, trace_id="tid")
        assert verdict.degraded is True


# ---------------------------------------------------------------------------
# Full deliberate() path with a scripted backend
# ---------------------------------------------------------------------------


class TestDeliberate:

    def test_full_success_path(self):
        # Every member gets a content that scores well and has no flags.
        per_member = {
            mid: BackendResponse(
                ok=True,
                content=f"Assessment from {mid}: 8/10 looks good.",
                attempts=1,
                latency_s=0.01,
                backend_name="scripted",
            )
            for mid in COUNCIL_MEMBERS
        }
        backend = _ScriptedBackend(per_member)
        council = DivineCouncil(backend, metrics=_metrics_fresh())

        verdict = council.deliberate("test query")

        assert verdict.consensus is True
        assert verdict.approval_count == 6  # all except advocate
        assert verdict.challenge_count == 1  # advocate
        assert verdict.abstain_count == 0
        assert verdict.trace_id != ""
        assert len(verdict.votes) == 7
        assert set(backend.calls) == set(COUNCIL_MEMBERS)

    def test_partial_backend_outage(self):
        # Three members fail; rest succeed.
        per_member: Dict[str, BackendResponse] = {}
        for i, mid in enumerate(COUNCIL_MEMBERS):
            if i < 3:
                per_member[mid] = BackendResponse(
                    ok=False,
                    error="circuit_open",
                    attempts=0,
                    latency_s=0.01,
                    backend_name="scripted",
                )
            else:
                per_member[mid] = BackendResponse(
                    ok=True,
                    content=f"text from {mid}: 8/10.",
                    attempts=1,
                    latency_s=0.01,
                    backend_name="scripted",
                )
        backend = _ScriptedBackend(per_member)
        council = DivineCouncil(backend, metrics=_metrics_fresh())

        verdict = council.deliberate("q")

        assert verdict.abstain_count == 3
        # Critically: the 3 abstains do NOT inflate approval_count
        total_non_abstain = verdict.approval_count + verdict.challenge_count
        assert total_non_abstain == 4

    def test_trace_id_is_respected(self):
        backend = _ScriptedBackend({})
        council = DivineCouncil(backend, metrics=_metrics_fresh())

        verdict = council.deliberate("q", trace_id="fixedtid")

        assert verdict.trace_id == "fixedtid"

    def test_verdict_format_log_includes_trace(self):
        backend = _ScriptedBackend({})
        council = DivineCouncil(backend, metrics=_metrics_fresh())
        verdict = council.deliberate("q", trace_id="abcd1234")
        log = verdict.format_log()
        assert "abcd1234" in log

    def test_verdict_truncation_is_presentation_only(self):
        """Data model keeps full response; truncation happens in format_log."""
        long_content = "x " * 1000  # 2000 chars
        per_member = {
            mid: BackendResponse(
                ok=True,
                content=long_content,
                attempts=1,
                backend_name="scripted",
            )
            for mid in COUNCIL_MEMBERS
        }
        backend = _ScriptedBackend(per_member)
        council = DivineCouncil(backend, metrics=_metrics_fresh())

        verdict = council.deliberate("q")

        # Full response in vote
        for vote in verdict.votes.values():
            if not vote.is_error:
                assert len(vote.response) >= 1000

        # Truncated in format
        formatted = verdict.format_log(preview_chars=100)
        # Truncation should add ellipsis
        assert "…" in formatted
