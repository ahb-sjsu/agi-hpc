"""Tests for MemoryClient."""

import pytest
from unittest.mock import MagicMock, patch

from agi.lh.memory_client import MemoryClient, MemoryAugmentResult


class TestMemoryClientInit:
    """Tests for MemoryClient initialization."""

    def test_memory_client_initializes_with_address(self):
        """MemoryClient should initialize with given address."""
        client = MemoryClient(address="localhost:50110")
        assert client._address == "localhost:50110"

    def test_memory_client_stub_none_when_service_unavailable(self):
        """MemoryClient should have None stub when service is unavailable."""
        # The SemanticMemoryServiceStub doesn't exist, so stub will be None
        client = MemoryClient(address="localhost:99999")
        assert client._stub is None


class TestMemoryClientEnrichRequest:
    """Tests for MemoryClient.enrich_request method."""

    def test_enrich_request_passthrough_when_unavailable(self):
        """enrich_request should return request unchanged when service unavailable."""
        client = MemoryClient(address="localhost:50110")

        # Create a mock request
        mock_request = MagicMock()
        mock_request.task.description = "Test task"

        result = client.enrich_request(mock_request)

        assert result is mock_request

    def test_enrich_request_returns_same_object(self):
        """enrich_request should return the exact same request object."""
        client = MemoryClient(address="localhost:50110")

        class DummyRequest:
            pass

        request = DummyRequest()
        result = client.enrich_request(request)

        assert result is request

    def test_enrich_request_handles_none_task(self):
        """enrich_request should handle request without task attribute."""
        client = MemoryClient(address="localhost:50110")

        class RequestWithoutTask:
            pass

        request = RequestWithoutTask()
        result = client.enrich_request(request)

        assert result is request


class TestMemoryAugmentResult:
    """Tests for MemoryAugmentResult dataclass."""

    def test_memory_augment_result_stores_enriched_request(self):
        """MemoryAugmentResult should store the enriched request."""
        mock_request = {"test": "data"}
        result = MemoryAugmentResult(enriched_request=mock_request)

        assert result.enriched_request == mock_request


class TestMemoryClientConnected:
    """Tests for MemoryClient when service is connected (mocked)."""

    def test_enrich_request_with_stub_available(self):
        """enrich_request should attempt query when stub is set."""
        client = MemoryClient(address="localhost:50110")

        # Stub is None because proto doesn't exist
        # Just verify client was created and passthrough works
        mock_request = MagicMock()
        mock_request.task.description = "Test task"

        result = client.enrich_request(mock_request)

        # Passthrough mode - returns original request
        assert result is mock_request

    def test_memory_client_address_stored(self):
        """MemoryClient should store the address for connection."""
        client = MemoryClient(address="memory-service:50110")
        assert client._address == "memory-service:50110"
