# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""Unit tests for agi.training.curriculum -- CurriculumManager."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

from agi.training.curriculum import (
    CurriculumConfig,
    CurriculumManager,
    EnvCurriculumState,
)


class TestEnvCurriculumState:
    """Tests for EnvCurriculumState dataclass."""

    def test_success_rate_empty(self) -> None:
        state = EnvCurriculumState(env_name="test")
        assert state.success_rate == 0.0

    def test_success_rate_all_high(self) -> None:
        state = EnvCurriculumState(env_name="test")
        state.recent_scores = deque([0.8, 0.9, 1.0, 0.75, 0.7])
        # All >= 0.7 -> success_rate = 1.0
        assert state.success_rate == 1.0

    def test_success_rate_mixed(self) -> None:
        state = EnvCurriculumState(env_name="test")
        state.recent_scores = deque([0.9, 0.3, 0.8, 0.2, 0.7])
        # 3 of 5 >= 0.7 -> 0.6
        assert abs(state.success_rate - 0.6) < 1e-6

    def test_avg_score(self) -> None:
        state = EnvCurriculumState(env_name="test")
        state.recent_scores = deque([0.5, 0.7, 0.9])
        assert abs(state.avg_score - 0.7) < 1e-6

    def test_to_dict(self) -> None:
        state = EnvCurriculumState(
            env_name="ethics",
            current_level=2,
            total_episodes=50,
            promotions=1,
            demotions=0,
        )
        d = state.to_dict()
        assert d["env_name"] == "ethics"
        assert d["current_level"] == 2
        assert d["total_episodes"] == 50


class TestCurriculumManagerBasic:
    """Tests for basic CurriculumManager operations."""

    def _make_manager(self, **kwargs) -> CurriculumManager:
        """Create a CurriculumManager with NATS and procedural disabled."""
        cfg = CurriculumConfig(
            enable_nats=False,
            enable_procedural=False,
            **kwargs,
        )
        return CurriculumManager(config=cfg)

    def test_initial_level(self) -> None:
        mgr = self._make_manager()
        assert mgr.get_level("ethics") == 1

    def test_set_level(self) -> None:
        mgr = self._make_manager()
        mgr.set_level("ethics", 3)
        assert mgr.get_level("ethics") == 3

    def test_set_level_clamped_high(self) -> None:
        mgr = self._make_manager(max_level=4)
        mgr.set_level("ethics", 10)
        assert mgr.get_level("ethics") == 4

    def test_set_level_clamped_low(self) -> None:
        mgr = self._make_manager(min_level=1)
        mgr.set_level("ethics", 0)
        assert mgr.get_level("ethics") == 1

    def test_record_score_before_window_full(self) -> None:
        mgr = self._make_manager(window_size=20)
        # Record fewer than 20 scores -> no promotion
        for _ in range(10):
            result = mgr.record_score("ethics", 0.9)
        assert result is None
        assert mgr.get_level("ethics") == 1


class TestCurriculumPromotion:
    """Tests for promotion logic."""

    def _make_manager(self, **kwargs) -> CurriculumManager:
        defaults = {
            "enable_nats": False,
            "enable_procedural": False,
            "window_size": 20,
            "promote_threshold": 0.80,
            "demote_threshold": 0.40,
            "max_level": 4,
        }
        defaults.update(kwargs)
        cfg = CurriculumConfig(**defaults)
        return CurriculumManager(config=cfg)

    def test_promote_at_high_success_rate(self) -> None:
        mgr = self._make_manager()
        # Fill window with high scores (all >= 0.7 -> 100% success rate)
        for i in range(19):
            mgr.record_score("ethics", 0.9)
        # The 20th score triggers evaluation
        result = mgr.record_score("ethics", 0.9)
        assert result == "promoted"
        assert mgr.get_level("ethics") == 2

    def test_no_promote_below_threshold(self) -> None:
        mgr = self._make_manager()
        # Mix of high and low scores -> ~50% success rate
        for i in range(20):
            score = 0.9 if i % 2 == 0 else 0.3
            result = mgr.record_score("ethics", score)
        # 50% success rate < 80% threshold -> no promotion
        assert result is None
        assert mgr.get_level("ethics") == 1

    def test_promote_clears_window(self) -> None:
        mgr = self._make_manager()
        for _ in range(20):
            mgr.record_score("ethics", 0.9)
        # After promotion, window should be cleared
        # Recording another score should not immediately promote again
        result = mgr.record_score("ethics", 0.9)
        assert result is None  # Window not full yet

    def test_promote_at_max_level_stays(self) -> None:
        mgr = self._make_manager(max_level=2)
        mgr.set_level("ethics", 2)
        for _ in range(20):
            result = mgr.record_score("ethics", 0.9)
        # Already at max_level, should not promote
        assert result is None
        assert mgr.get_level("ethics") == 2


class TestCurriculumDemotion:
    """Tests for demotion logic."""

    def _make_manager(self, **kwargs) -> CurriculumManager:
        defaults = {
            "enable_nats": False,
            "enable_procedural": False,
            "window_size": 20,
            "promote_threshold": 0.80,
            "demote_threshold": 0.40,
            "max_level": 4,
        }
        defaults.update(kwargs)
        cfg = CurriculumConfig(**defaults)
        return CurriculumManager(config=cfg)

    def test_demote_at_low_success_rate(self) -> None:
        mgr = self._make_manager()
        mgr.set_level("ethics", 3)
        # All low scores -> 0% success rate < 40% threshold
        for i in range(19):
            mgr.record_score("ethics", 0.2)
        result = mgr.record_score("ethics", 0.2)
        assert result == "demoted"
        assert mgr.get_level("ethics") == 2

    def test_no_demote_above_threshold(self) -> None:
        mgr = self._make_manager()
        mgr.set_level("ethics", 2)
        # 50% success rate > 40% threshold -> no demotion
        for i in range(20):
            score = 0.9 if i % 2 == 0 else 0.3
            result = mgr.record_score("ethics", score)
        assert result is None
        assert mgr.get_level("ethics") == 2

    def test_demote_at_min_level_stays(self) -> None:
        mgr = self._make_manager(min_level=1)
        assert mgr.get_level("ethics") == 1
        for _ in range(20):
            result = mgr.record_score("ethics", 0.1)
        # Already at min_level, should not demote further
        assert result is None
        assert mgr.get_level("ethics") == 1


class TestCurriculumStatus:
    """Tests for get_status() and get_training_metrics()."""

    def _make_manager(self) -> CurriculumManager:
        cfg = CurriculumConfig(enable_nats=False, enable_procedural=False)
        return CurriculumManager(config=cfg)

    def test_get_status_empty(self) -> None:
        mgr = self._make_manager()
        status = mgr.get_status()
        assert status == {}

    def test_get_status_after_scores(self) -> None:
        mgr = self._make_manager()
        mgr.record_score("ethics", 0.8)
        mgr.record_score("coding", 0.5)
        status = mgr.get_status()
        assert "ethics" in status
        assert "coding" in status

    def test_get_training_metrics(self) -> None:
        mgr = self._make_manager()
        mgr.record_score("ethics", 0.8)
        metrics = mgr.get_training_metrics()
        assert "environments" in metrics
        assert "total_episodes" in metrics
        assert metrics["total_episodes"] == 1
