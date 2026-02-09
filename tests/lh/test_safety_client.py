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

    def test_safety_client_creates_stub(self):
        """SafetyClient should create stub (gRPC uses lazy connection)."""
        client = SafetyClient(address="localhost:99999")
        # Stub is created even if service isn't running (lazy connect)
        assert client._stub is not None


class TestSafetyClientCheckPlan:
    """Tests for SafetyClient.check_plan method."""

    def test_check_plan_mock_approval_when_not_connected(self):
        """check_plan should return approved=True when not connected."""
        client = SafetyClient(address="localhost:50200")
        # Force not connected state
        client._connected = False

        mock_plan = MagicMock()
        result = client.check_plan(mock_plan)

        assert isinstance(result, SafetyResult)
        assert result.approved is True
        assert any("mock_safety" in issue for issue in result.issues)

    def test_check_plan_returns_safety_result(self):
        """check_plan should always return a SafetyResult."""
        client = SafetyClient(address="localhost:50200")
        # Force not connected to avoid serialization
        client._connected = False

        result = client.check_plan(None)

        assert isinstance(result, SafetyResult)

    def test_check_plan_blocks_on_serialization_error(self):
        """check_plan should block on serialization errors when connected."""
        client = SafetyClient(address="localhost:50200")
        # Client is "connected" so it will try to serialize

        # Invalid plan types cause serialization errors
        for plan in [None, {}, [], "string", 123]:
            result = client.check_plan(plan)
            assert isinstance(result, SafetyResult)
            # Serialization failure returns blocked
            assert result.approved is False


class TestSafetyResult:
    """Tests for SafetyResult dataclass."""

    def test_safety_result_approved_true(self):
        """SafetyResult should store approved=True correctly."""
        result = SafetyResult(approved=True)
        assert result.approved is True
        assert result.issues == []

    def test_safety_result_approved_false_with_issues(self):
        """SafetyResult should store approved=False with issues."""
        issues = ["constraint_violation", "invalid_tool"]
        result = SafetyResult(approved=False, issues=issues)

        assert result.approved is False
        assert result.issues == issues

    def test_safety_result_issues_default_empty(self):
        """SafetyResult issues should default to empty list."""
        result = SafetyResult(approved=True)
        assert result.issues == []


class TestSafetyClientConnected:
    """Tests for SafetyClient when service is connected (mocked)."""

    def test_check_plan_fallback_when_not_connected(self):
        """check_plan should use mock approval when not connected."""
        client = SafetyClient(address="localhost:50200")
        # Force not connected
        client._connected = False

        mock_plan = MagicMock()
        result = client.check_plan(mock_plan)

        # Should return mock approval in passthrough mode
        assert isinstance(result, SafetyResult)
        assert result.approved is True
        assert any("mock_safety" in issue for issue in result.issues)

    def test_safety_client_address_stored(self):
        """SafetyClient should store the address for connection."""
        client = SafetyClient(address="safety-service:50200")
        assert client._address == "safety-service:50200"
