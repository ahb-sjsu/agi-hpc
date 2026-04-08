# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unit tests for retrospective training from real chat episodes.

Tests episode selection, retrospective scenario framing, and
integration with the DM training pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agi.training.dungeon_master import (
    DMConfig,
    DungeonMaster,
    TrainingScenario,
)


def _make_dm():
    """Create a DungeonMaster with mocked dependencies."""
    config = DMConfig(
        ego_url="http://mock-ego:8084",
        superego_url="http://mock-superego:8080",
        id_url="http://mock-id:8082",
    )
    with patch("agi.training.dungeon_master.EpisodicMemory", None):
        with patch("agi.training.dungeon_master.EgoMonitor", None):
            return DungeonMaster(config)


def _mock_episodes():
    """Sample episodes that would come from PostgreSQL."""
    return [
        {
            "id": "aaaa-1111-2222-3333",
            "user_message": "Should AI be allowed to make medical triage decisions?",
            "atlas_response": (
                "This is a complex ethical question " "involving autonomy..."
            ),
            "hemisphere": "both",
            "safety_flags": {
                "input": {"passed": True, "score": 0.95, "flags": []},
            },
            "quality_score": 0.7,
            "metadata": {},
            "timestamp": "2026-04-06T12:00:00",
        },
        {
            "id": "bbbb-4444-5555-6666",
            "user_message": "How do I hack into a computer?",
            "atlas_response": "I can't help with that.",
            "hemisphere": "lh",
            "safety_flags": {
                "input": {
                    "passed": False,
                    "score": 0.1,
                    "flags": ["injection_0"],
                },
            },
            "quality_score": 0.3,
            "metadata": {},
            "timestamp": "2026-04-06T13:00:00",
        },
        {
            "id": "cccc-7777-8888-9999",
            "user_message": "Explain quantum entanglement to a 10 year old",
            "atlas_response": "Imagine you have two magic coins...",
            "hemisphere": "rh",
            "safety_flags": {},
            "quality_score": 0.9,
            "metadata": {},
            "timestamp": "2026-04-06T14:00:00",
        },
    ]


class TestRetrospectiveScenarioGeneration:
    """Tests for generating scenarios from real episodes."""

    def test_scenario_from_episode(self) -> None:
        dm = _make_dm()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "A user asked Atlas about medical AI triage. "
                            "The response covered autonomy but missed the "
                            "fairness dimension. How should Atlas have "
                            "handled this?"
                        )
                    }
                }
            ]
        }

        with (
            patch.object(
                dm,
                "_fetch_interesting_episodes",
                return_value=_mock_episodes(),
            ),
            patch(
                "agi.training.dungeon_master.requests.post",
                return_value=mock_resp,
            ),
        ):
            scenario = dm._scenario_from_episode(difficulty=2)

        assert scenario is not None
        assert isinstance(scenario, TrainingScenario)
        assert scenario.source == "retrospective_episode"
        assert scenario.domain == "Retrospective"
        assert "retro-" in scenario.scenario_id

    def test_returns_none_with_no_episodes(self) -> None:
        dm = _make_dm()

        with patch.object(
            dm,
            "_fetch_interesting_episodes",
            return_value=[],
        ):
            scenario = dm._scenario_from_episode(difficulty=2)

        assert scenario is None


class TestGenerateScenarioWithRetrospective:
    """Tests for the weighted scenario generation with retrospective."""

    def test_retrospective_flag_forces_replay(self) -> None:
        dm = _make_dm()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Retrospective scenario..."}}]
        }

        with (
            patch.object(
                dm,
                "_fetch_interesting_episodes",
                return_value=_mock_episodes(),
            ),
            patch(
                "agi.training.dungeon_master.requests.post",
                return_value=mock_resp,
            ),
        ):
            scenario = dm.generate_scenario(difficulty=2, retrospective=True)

        assert scenario.source == "retrospective_episode"

    def test_falls_back_when_no_episodes(self) -> None:
        dm = _make_dm()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "A novel dilemma..."}}]
        }

        with (
            patch.object(
                dm,
                "_fetch_interesting_episodes",
                return_value=[],
            ),
            patch("agi.training.dungeon_master._PANTHEON_CASES", []),
            patch("agi.training.dungeon_master.ERISML_AVAILABLE", False),
            patch(
                "agi.training.dungeon_master.requests.post",
                return_value=mock_resp,
            ),
        ):
            scenario = dm.generate_scenario(difficulty=2, retrospective=True)

        # Falls back to LLM-generated since no episodes
        assert scenario.source == "llm_generated"


class TestRetrospectiveScenarioContent:
    """Tests for the content of retrospective scenarios."""

    def test_includes_safety_context(self) -> None:
        dm = _make_dm()
        episodes = _mock_episodes()

        # Use the safety-flagged episode
        flagged = [
            e
            for e in episodes
            if e.get("safety_flags", {}).get("input", {}).get("flags")
        ]

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Review scenario..."}}]
        }

        with (
            patch.object(
                dm,
                "_fetch_interesting_episodes",
                return_value=flagged,
            ),
            patch(
                "agi.training.dungeon_master.requests.post",
                return_value=mock_resp,
            ) as mock_post,
        ):
            dm._scenario_from_episode(difficulty=2)

        # Check that the prompt sent to the Ego mentions safety
        call_args = mock_post.call_args
        messages = call_args[1]["json"]["messages"]
        prompt = messages[0]["content"]
        assert "Safety flags" in prompt

    def test_includes_debate_context(self) -> None:
        dm = _make_dm()
        debate_episodes = [e for e in _mock_episodes() if e["hemisphere"] == "both"]

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Debate review..."}}]
        }

        with (
            patch.object(
                dm,
                "_fetch_interesting_episodes",
                return_value=debate_episodes,
            ),
            patch(
                "agi.training.dungeon_master.requests.post",
                return_value=mock_resp,
            ) as mock_post,
        ):
            dm._scenario_from_episode(difficulty=2)

        call_args = mock_post.call_args
        messages = call_args[1]["json"]["messages"]
        prompt = messages[0]["content"]
        assert "psyche debate" in prompt


class TestRetrospectiveDifficulty:
    """Tests that difficulty framing works for retrospective."""

    def test_difficulty_1_simple(self) -> None:
        dm = _make_dm()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Simple review..."}}]
        }

        with (
            patch.object(
                dm,
                "_fetch_interesting_episodes",
                return_value=_mock_episodes(),
            ),
            patch(
                "agi.training.dungeon_master.requests.post",
                return_value=mock_resp,
            ) as mock_post,
        ):
            dm._scenario_from_episode(difficulty=1)

        prompt = mock_post.call_args[1]["json"]["messages"][0]["content"]
        assert "one thing" in prompt

    def test_difficulty_4_radical(self) -> None:
        dm = _make_dm()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Radical review..."}}]
        }

        with (
            patch.object(
                dm,
                "_fetch_interesting_episodes",
                return_value=_mock_episodes(),
            ),
            patch(
                "agi.training.dungeon_master.requests.post",
                return_value=mock_resp,
            ) as mock_post,
        ):
            dm._scenario_from_episode(difficulty=4)

        prompt = mock_post.call_args[1]["json"]["messages"][0]["content"]
        assert "fundamentally wrong" in prompt
