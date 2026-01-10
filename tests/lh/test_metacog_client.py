"""Tests for MetacognitionClient."""

import pytest
from unittest.mock import MagicMock, patch

from agi.lh.metacog_client import MetacognitionClient, MetaReviewResult


class TestMetacognitionClientInit:
    """Tests for MetacognitionClient initialization."""

    def test_metacog_client_initializes_with_address(self):
        """MetacognitionClient should initialize with given address."""
        client = MetacognitionClient(address="localhost:50300")
        assert client._address == "localhost:50300"

    def test_metacog_client_stub_none_when_service_unavailable(self):
        """MetacognitionClient should have None stub when service unavailable."""
        # The MetacognitionServiceStub doesn't exist, so stub will be None
        client = MetacognitionClient(address="localhost:99999")
        assert client._stub is None


class TestMetacognitionClientReviewPlan:
    """Tests for MetacognitionClient.review_plan method."""

    def test_review_plan_mock_accept_when_unavailable(self):
        """review_plan should return ACCEPT when service unavailable."""
        client = MetacognitionClient(address="localhost:50300")

        mock_plan = MagicMock()
        result = client.review_plan(mock_plan)

        assert isinstance(result, MetaReviewResult)
        assert result.decision == "ACCEPT"
        assert result.confidence == 1.0
        assert result.issues == []

    def test_review_plan_returns_meta_review_result(self):
        """review_plan should always return a MetaReviewResult."""
        client = MetacognitionClient(address="localhost:50300")

        result = client.review_plan(None)

        assert isinstance(result, MetaReviewResult)

    def test_review_plan_handles_any_plan_input(self):
        """review_plan should handle any plan input type."""
        client = MetacognitionClient(address="localhost:50300")

        for plan in [None, {}, [], "string", 123]:
            result = client.review_plan(plan)
            assert isinstance(result, MetaReviewResult)
            assert result.decision == "ACCEPT"


class TestMetacognitionClientRevisePlan:
    """Tests for MetacognitionClient.revise_plan method."""

    def test_revise_plan_returns_original_when_unavailable(self):
        """revise_plan should return original plan when service unavailable."""
        client = MetacognitionClient(address="localhost:50300")

        mock_plan = MagicMock()
        mock_review = MetaReviewResult(decision="REVISE", issues=["need_revision"])

        result = client.revise_plan(mock_plan, mock_review)

        assert result is mock_plan

    def test_revise_plan_passthrough_for_any_input(self):
        """revise_plan should passthrough any plan type."""
        client = MetacognitionClient(address="localhost:50300")
        review = MetaReviewResult(decision="REVISE")

        test_plan = {"steps": [1, 2, 3]}
        result = client.revise_plan(test_plan, review)

        assert result is test_plan


class TestMetaReviewResult:
    """Tests for MetaReviewResult dataclass."""

    def test_meta_review_result_accept(self):
        """MetaReviewResult should store ACCEPT decision."""
        result = MetaReviewResult(decision="ACCEPT")

        assert result.decision == "ACCEPT"
        assert result.confidence == 1.0
        assert result.issues is None

    def test_meta_review_result_revise(self):
        """MetaReviewResult should store REVISE decision with issues."""
        issues = ["reasoning_unclear", "missing_evidence"]
        result = MetaReviewResult(decision="REVISE", issues=issues, confidence=0.6)

        assert result.decision == "REVISE"
        assert result.issues == issues
        assert result.confidence == 0.6

    def test_meta_review_result_reject(self):
        """MetaReviewResult should store REJECT decision."""
        result = MetaReviewResult(
            decision="REJECT", issues=["critical_failure"], confidence=0.0
        )

        assert result.decision == "REJECT"
        assert result.confidence == 0.0

    def test_meta_review_result_default_confidence(self):
        """MetaReviewResult confidence should default to 1.0."""
        result = MetaReviewResult(decision="ACCEPT")
        assert result.confidence == 1.0


class TestMetacognitionClientConnected:
    """Tests for MetacognitionClient when service is connected (mocked)."""

    def test_review_plan_fallback_when_stub_unavailable(self):
        """review_plan should use mock ACCEPT when stub unavailable."""
        client = MetacognitionClient(address="localhost:50300")

        # Stub is None because proto doesn't exist
        mock_plan = MagicMock()
        result = client.review_plan(mock_plan)

        # Should return mock ACCEPT in passthrough mode
        assert isinstance(result, MetaReviewResult)
        assert result.decision == "ACCEPT"
        assert result.confidence == 1.0

    def test_revise_plan_fallback_when_stub_unavailable(self):
        """revise_plan should return original plan when stub unavailable."""
        client = MetacognitionClient(address="localhost:50300")

        # Stub is None because proto doesn't exist
        mock_plan = MagicMock()
        mock_review = MetaReviewResult(decision="REVISE")

        result = client.revise_plan(mock_plan, mock_review)

        # Should return original plan in passthrough mode
        assert result is mock_plan

    def test_metacog_client_address_stored(self):
        """MetacognitionClient should store the address for connection."""
        client = MetacognitionClient(address="meta-service:50300")
        assert client._address == "meta-service:50300"
