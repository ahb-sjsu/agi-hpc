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

"""Unit tests for agi.metacognition.disagreement -- DisagreementMetric."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from agi.metacognition.disagreement import (
    DisagreementMetric,
    DisagreementResult,
)


class TestConfidenceMapping:
    """Tests for DisagreementMetric.map_confidence() static method."""

    def test_high_similarity_high_confidence(self) -> None:
        conf = DisagreementMetric.map_confidence(0.90)
        assert conf == 0.9

    def test_boundary_high(self) -> None:
        conf = DisagreementMetric.map_confidence(0.86)
        assert conf == 0.9

    def test_medium_similarity_medium_confidence(self) -> None:
        conf = DisagreementMetric.map_confidence(0.65)
        assert 0.4 <= conf <= 0.85

    def test_low_similarity_low_confidence(self) -> None:
        conf = DisagreementMetric.map_confidence(0.3)
        assert conf == 0.3

    def test_zero_similarity(self) -> None:
        conf = DisagreementMetric.map_confidence(0.0)
        assert conf == 0.3

    def test_boundary_at_0_85(self) -> None:
        conf = DisagreementMetric.map_confidence(0.85)
        # 0.85 is the boundary; (0.85 - 0.5)/0.35 = 1.0 -> 0.4 + 0.45 = 0.85
        assert abs(conf - 0.85) < 1e-6

    def test_boundary_at_0_5(self) -> None:
        conf = DisagreementMetric.map_confidence(0.5)
        assert abs(conf - 0.4) < 1e-6

    def test_perfect_similarity(self) -> None:
        conf = DisagreementMetric.map_confidence(1.0)
        assert conf == 0.9


class TestCosineSimilarity:
    """Tests for the static _cosine_similarity method."""

    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        sim = DisagreementMetric._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = DisagreementMetric._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        sim = DisagreementMetric._cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        sim = DisagreementMetric._cosine_similarity(a, b)
        assert sim == 0.0


class TestDisagreementMetricCompute:
    """Tests for DisagreementMetric.compute() with mocked embedding model."""

    def _make_metric_with_mock(self, embeddings: np.ndarray) -> DisagreementMetric:
        """Create a DisagreementMetric with a mocked embedding model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = embeddings
        return DisagreementMetric(embed_model=mock_model)

    def test_identical_responses_high_confidence(self) -> None:
        # Same embedding for both -> similarity=1.0 -> confidence=0.9
        embs = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        metric = self._make_metric_with_mock(embs)
        result = metric.compute("response A", "response A")
        assert abs(result.similarity - 1.0) < 1e-6
        assert result.confidence == 0.9

    def test_orthogonal_responses_low_confidence(self) -> None:
        embs = np.array([[1.0, 0.0], [0.0, 1.0]])
        metric = self._make_metric_with_mock(embs)
        result = metric.compute("analytical answer", "creative answer")
        assert abs(result.similarity) < 1e-6
        assert result.confidence == 0.3

    def test_result_fields_populated(self) -> None:
        embs = np.array([[0.8, 0.6], [0.7, 0.7]])
        metric = self._make_metric_with_mock(embs)
        result = metric.compute("spock text", "kirk text")
        assert result.spock_text == "spock text"
        assert result.kirk_text == "kirk text"
        assert result.compute_time_ms >= 0

    def test_leader_determination_with_query(self) -> None:
        # spock=[1,0], kirk=[0,1], query=[1,0] -> spock closer -> "lh"
        embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        metric = self._make_metric_with_mock(embs)
        result = metric.compute("spock", "kirk", query="analytical question")
        assert result.hemisphere_that_led == "lh"

    def test_leader_rh_when_kirk_closer(self) -> None:
        # spock=[1,0], kirk=[0,1], query=[0,1] -> kirk closer -> "rh"
        embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        metric = self._make_metric_with_mock(embs)
        result = metric.compute("spock", "kirk", query="creative question")
        assert result.hemisphere_that_led == "rh"

    def test_default_leader_lh_without_query(self) -> None:
        embs = np.array([[0.5, 0.5], [0.5, 0.5]])
        metric = self._make_metric_with_mock(embs)
        result = metric.compute("spock", "kirk")
        assert result.hemisphere_that_led == "lh"


class TestCalibrationECE:
    """Tests for Expected Calibration Error computation."""

    def test_ece_insufficient_data(self) -> None:
        metric = DisagreementMetric()
        # Fewer than 5 observations -> returns 0.0
        metric.record_feedback(0.8, True)
        metric.record_feedback(0.7, True)
        assert metric.compute_ece() == 0.0

    def test_ece_perfectly_calibrated(self) -> None:
        metric = DisagreementMetric()
        # All predictions at 1.0 and all accepted -> ECE should be ~0
        for _ in range(20):
            metric.record_feedback(1.0, True)
        ece = metric.compute_ece()
        assert ece < 0.1

    def test_ece_poorly_calibrated(self) -> None:
        metric = DisagreementMetric()
        # Predict 0.9 but all rejected -> high ECE
        for _ in range(20):
            metric.record_feedback(0.9, False)
        ece = metric.compute_ece()
        assert ece > 0.5

    def test_record_feedback_bounded_by_window(self) -> None:
        metric = DisagreementMetric(calibration_window=10)
        for i in range(20):
            metric.record_feedback(0.5, True)
        # Window is 10, so only last 10 should be kept
        assert len(metric._calibration_history) == 10


class TestDisagreementResult:
    """Tests for DisagreementResult dataclass."""

    def test_fields(self) -> None:
        r = DisagreementResult(
            spock_text="a",
            kirk_text="b",
            similarity=0.75,
            confidence=0.65,
            hemisphere_that_led="lh",
            compute_time_ms=1.5,
        )
        assert r.spock_text == "a"
        assert r.similarity == 0.75
        assert r.confidence == 0.65
        assert r.hemisphere_that_led == "lh"
