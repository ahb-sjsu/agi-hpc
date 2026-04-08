# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for the Divine Council multi-agent deliberation.

Tests parallel deliberation, vote aggregation, consensus detection,
ethical flagging, and synthesis.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agi.reasoning.divine_council import (
    COUNCIL_MEMBERS,
    CouncilVerdict,
    CouncilVote,
    DivineCouncil,
)


def _mock_response(content):
    mock = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    return mock


class TestCouncilMembers:
    """Tests for council member definitions."""

    def test_four_members(self) -> None:
        assert len(COUNCIL_MEMBERS) == 4

    def test_required_members(self) -> None:
        assert "judge" in COUNCIL_MEMBERS
        assert "advocate" in COUNCIL_MEMBERS
        assert "synthesizer" in COUNCIL_MEMBERS
        assert "ethicist" in COUNCIL_MEMBERS

    def test_each_has_system_prompt(self) -> None:
        for mid, info in COUNCIL_MEMBERS.items():
            assert "system_prompt" in info
            assert len(info["system_prompt"]) > 50


class TestDeliberation:
    """Tests for the full deliberation pipeline."""

    def test_produces_verdict(self) -> None:
        council = DivineCouncil()
        responses = [
            _mock_response("The response is accurate. Score: 8/10"),
            _mock_response("I challenge this — the reasoning is weak."),
            _mock_response("Combining both views, the answer is..."),
            _mock_response("No ethical concerns. The response is fair."),
        ]

        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = responses
            verdict = council.deliberate(
                "Is AI safe?",
                superego_response="AI needs regulation.",
                id_response="AI empowers creativity.",
            )

        assert isinstance(verdict, CouncilVerdict)
        assert verdict.method == "divine_council"
        assert len(verdict.votes) == 4
        assert verdict.synthesis != ""
        assert verdict.total_latency_s >= 0

    def test_consensus_when_approved(self) -> None:
        council = DivineCouncil()
        responses = [
            _mock_response("Excellent answer. 9/10"),
            _mock_response("Minor quibble but overall good."),
            _mock_response("Well-integrated synthesis."),
            _mock_response("Ethically sound. No concerns."),
        ]

        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = responses
            verdict = council.deliberate("Test query")

        assert verdict.consensus is True
        assert verdict.approval_count >= 2

    def test_no_consensus_with_ethical_flag(self) -> None:
        council = DivineCouncil()
        responses = [
            _mock_response("Good. 8/10"),
            _mock_response("Acceptable."),
            _mock_response("Synthesis looks good."),
            _mock_response("CONCERN: potential bias detected. Harm risk."),
        ]

        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = responses
            verdict = council.deliberate("Test query")

        assert len(verdict.ethical_flags) > 0
        assert verdict.consensus is False


class TestScoreParsing:
    """Tests for score extraction from responses."""

    def test_parses_score(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response("Score: 8/10. Good answer.")
            vote = council._call_member("judge", "test")

        assert vote.score == 8.0

    def test_default_score_without_number(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response("This is a reasonable answer.")
            vote = council._call_member("judge", "test")

        assert vote.score == 5.0


class TestEthicalFlags:
    """Tests for ethicist flag detection."""

    def test_detects_bias(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                "There is potential bias in the response."
            )
            vote = council._call_member("ethicist", "test")

        assert "bias" in vote.flags

    def test_detects_harm(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response(
                "This could cause harm to vulnerable groups."
            )
            vote = council._call_member("ethicist", "test")

        assert "harm" in vote.flags

    def test_clean_passes(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.return_value = _mock_response("Ethically sound. No issues found.")
            vote = council._call_member("ethicist", "test")

        assert len(vote.flags) == 0


class TestFormatLog:
    """Tests for the UI debate log formatting."""

    def test_includes_all_members(self) -> None:
        verdict = CouncilVerdict(
            consensus=True,
            synthesis="The answer is...",
            votes={
                "judge": CouncilVote("judge", "Good. 8/10", 8, latency_s=5),
                "advocate": CouncilVote("advocate", "I challenge...", 5, latency_s=4),
                "synthesizer": CouncilVote("synthesizer", "Merged...", 7, latency_s=6),
                "ethicist": CouncilVote("ethicist", "No concerns.", 8, latency_s=3),
            },
            approval_count=3,
            challenge_count=1,
        )
        log = verdict.format_log()
        assert "Judge" in log
        assert "Advocate" in log
        assert "Synthesizer" in log
        assert "Ethicist" in log
        assert "Yes" in log and "approve" in log


class TestVerdictDataclass:
    """Tests for the CouncilVerdict structure."""

    def test_defaults(self) -> None:
        v = CouncilVerdict(consensus=False, synthesis="")
        assert v.method == "divine_council"
        assert v.approval_count == 0
        assert v.votes == {}

    def test_handles_error_gracefully(self) -> None:
        council = DivineCouncil()
        with patch("agi.reasoning.divine_council.requests.post") as mock_post:
            mock_post.side_effect = Exception("all down")
            verdict = council.deliberate("test")

        # Should still produce a verdict, just with error messages
        assert isinstance(verdict, CouncilVerdict)
        assert len(verdict.votes) == 4
