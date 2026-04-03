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

"""Unit tests for agi.training.scorer -- scoring strategies."""

from __future__ import annotations

from agi.training.scorer import (
    ResponseScorer,
    ScorerConfig,
    score_code_execution,
    score_exact_match,
    score_keyword_presence,
    score_numeric_match,
    score_structure,
)


class TestScoreKeywordPresence:
    """Tests for score_keyword_presence()."""

    def test_all_keywords_present(self) -> None:
        score = score_keyword_presence(
            "Python is a programming language",
            ["python", "programming", "language"],
        )
        assert score == 1.0

    def test_some_keywords_present(self) -> None:
        score = score_keyword_presence(
            "Python is great",
            ["python", "programming", "language"],
        )
        assert abs(score - 1.0 / 3.0) < 1e-6

    def test_no_keywords_present(self) -> None:
        score = score_keyword_presence(
            "Hello world",
            ["python", "programming"],
        )
        assert score == 0.0

    def test_empty_keywords(self) -> None:
        score = score_keyword_presence("Hello", [])
        assert score == 0.0

    def test_case_insensitive_default(self) -> None:
        score = score_keyword_presence(
            "PYTHON is Great",
            ["python", "great"],
        )
        assert score == 1.0

    def test_case_sensitive(self) -> None:
        score = score_keyword_presence(
            "PYTHON is Great",
            ["python"],
            case_sensitive=True,
        )
        assert score == 0.0


class TestScoreStructure:
    """Tests for score_structure()."""

    def test_long_response_scores_high(self) -> None:
        words = " ".join(["word"] * 100)
        paragraphs = words + "\n\n" + words + "\n\n" + words
        score = score_structure(paragraphs, min_words=50, require_paragraphs=2)
        assert score > 0.7

    def test_short_response_scores_low(self) -> None:
        score = score_structure("Short.", min_words=50, require_paragraphs=2)
        assert score < 0.5

    def test_empty_response(self) -> None:
        score = score_structure("", min_words=50, require_paragraphs=2)
        assert score <= 0.3


class TestScoreExactMatch:
    """Tests for score_exact_match()."""

    def test_exact_match(self) -> None:
        assert score_exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive_match(self) -> None:
        assert score_exact_match("PARIS", "paris") == 1.0

    def test_whitespace_stripped(self) -> None:
        assert score_exact_match("  Paris  ", "Paris") == 1.0

    def test_no_match(self) -> None:
        assert score_exact_match("London", "Paris") == 0.0


class TestScoreNumericMatch:
    """Tests for score_numeric_match()."""

    def test_exact_numeric_match(self) -> None:
        score = score_numeric_match("The answer is 42", 42.0)
        assert score == 1.0

    def test_close_match(self) -> None:
        score = score_numeric_match("The answer is 42.000005", 42.0, tolerance=1e-4)
        assert score == 1.0

    def test_within_10x_tolerance(self) -> None:
        score = score_numeric_match("The answer is 42.005", 42.0, tolerance=1e-3)
        assert score == 0.5

    def test_no_numbers_in_response(self) -> None:
        score = score_numeric_match("No numbers here", 42.0)
        assert score == 0.0

    def test_uses_last_number(self) -> None:
        score = score_numeric_match("Step 1: 10, Step 2: 20, Final: 42", 42.0)
        assert score == 1.0

    def test_far_off_value(self) -> None:
        score = score_numeric_match("The answer is 100", 42.0)
        assert score == 0.0


class TestScoreCodeExecution:
    """Tests for score_code_execution()."""

    def test_correct_code_all_pass(self) -> None:
        code = "def add(a, b): return a + b"
        test_cases = [
            {"call": "add(1, 2)", "expected": 3},
            {"call": "add(0, 0)", "expected": 0},
            {"call": "add(-1, 1)", "expected": 0},
        ]
        score, details = score_code_execution(code, test_cases, timeout=5)
        assert score == 1.0
        assert details["passed"] == 3

    def test_wrong_code_partial(self) -> None:
        code = "def add(a, b): return a + b + 1"
        test_cases = [
            {"call": "add(1, 2)", "expected": 3},
            {"call": "add(0, 0)", "expected": 1},
        ]
        score, details = score_code_execution(code, test_cases, timeout=5)
        # First fails (returns 4 not 3), second passes (returns 1)
        assert 0.0 < score < 1.0

    def test_syntax_error_zero(self) -> None:
        code = "def add(a, b) return a + b"  # missing colon
        test_cases = [{"call": "add(1, 2)", "expected": 3}]
        score, details = score_code_execution(code, test_cases, timeout=5)
        assert score == 0.0

    def test_empty_test_cases(self) -> None:
        code = "x = 1"
        score, details = score_code_execution(code, [], timeout=5)
        assert score == 0.0
        assert "error" in details

    def test_score_normalized_0_to_1(self) -> None:
        code = "def f(x): return x * 2"
        test_cases = [
            {"call": "f(1)", "expected": 2},
            {"call": "f(5)", "expected": 10},
        ]
        score, _ = score_code_execution(code, test_cases, timeout=5)
        assert 0.0 <= score <= 1.0


class TestResponseScorer:
    """Tests for ResponseScorer class (without PostgreSQL)."""

    def test_init_without_postgres(self) -> None:
        # Should not raise even without PostgreSQL
        scorer = ResponseScorer(config=ScorerConfig(auto_create_table=False))
        assert scorer is not None

    def test_store_result_returns_id_or_none(self) -> None:
        scorer = ResponseScorer(config=ScorerConfig(auto_create_table=False))
        result = scorer.store_result(
            env_name="unit_test_scorer",
            level=1,
            scenario="test scenario",
            response="test response",
            score=0.8,
        )
        # Returns int row id if PostgreSQL is available, None otherwise
        assert result is None or isinstance(result, int)

    def test_get_recent_scores_returns_list(self) -> None:
        scorer = ResponseScorer(config=ScorerConfig(auto_create_table=False))
        scores = scorer.get_recent_scores("unit_test_scorer_nonexistent_env_xyz")
        # Returns list (empty or with results depending on DB)
        assert isinstance(scores, list)
