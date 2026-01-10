"""Tests for SafetyClient."""

import pytest
from unittest.mock import MagicMock, patch

from agi.lh.safety_client import SafetyClient, SafetyResult


class TestSafetyClientInit:
    """Tests for SafetyClient initialization."""

    def test_safety_client_initializes_with_address(self):
        """SafetyClient should initialize with given address."""
        client = SafetyClient(address="localhost:50200")
        assert client._address == "localhost:50200"

    def test_safety_client_stub_none_when_service_unavailable(self):
        """SafetyClient should have None stub when SafetyService is unavailable."""
        # The SafetyServiceStub doesn't exist, so stub will be None
        client = SafetyClient(address="localhost:99999")
        assert client._stub is None


class TestSafetyClientCheckPlan:
    """Tests for SafetyClient.check_plan method."""

    def test_check_plan_mock_approval_when_unavailable(self):
        """check_plan should return approved=True when service unavailable."""
        client = SafetyClient(address="localhost:50200")

        mock_plan = MagicMock()
        result = client.check_plan(mock_plan)

        assert isinstance(result, SafetyResult)
        assert result.approved is True
        assert "mock_safety" in result.issues

    def test_check_plan_returns_safety_result(self):
        """check_plan should always return a SafetyResult."""
        client = SafetyClient(address="localhost:50200")

        result = client.check_plan(None)

        assert isinstance(result, SafetyResult)

    def test_check_plan_handles_any_plan_input(self):
        """check_plan should handle any plan input type."""
        client = SafetyClient(address="localhost:50200")

        # Test with various inputs
        for plan in [None, {}, [], "string", 123]:
            result = client.check_plan(plan)
            assert isinstance(result, SafetyResult)
            assert result.approved is True


class TestSafetyResult:
    """Tests for SafetyResult dataclass."""

    def test_safety_result_approved_true(self):
        """SafetyResult should store approved=True correctly."""
        result = SafetyResult(approved=True)
        assert result.approved is True
        assert result.issues is None

    def test_safety_result_approved_false_with_issues(self):
        """SafetyResult should store approved=False with issues."""
        issues = ["constraint_violation", "invalid_tool"]
        result = SafetyResult(approved=False, issues=issues)

        assert result.approved is False
        assert result.issues == issues

    def test_safety_result_issues_default_none(self):
        """SafetyResult issues should default to None."""
        result = SafetyResult(approved=True)
        assert result.issues is None


class TestSafetyClientConnected:
    """Tests for SafetyClient when service is connected (mocked)."""

    def test_check_plan_fallback_when_stub_unavailable(self):
        """check_plan should use mock approval when stub unavailable."""
        client = SafetyClient(address="localhost:50200")

        # Stub is None because proto doesn't exist
        mock_plan = MagicMock()
        result = client.check_plan(mock_plan)

        # Should return mock approval in passthrough mode
        assert isinstance(result, SafetyResult)
        assert result.approved is True
        assert "mock_safety" in result.issues

    def test_safety_client_address_stored(self):
        """SafetyClient should store the address for connection."""
        client = SafetyClient(address="safety-service:50200")
        assert client._address == "safety-service:50200"
