"""Skip integration tests unless NATS is available."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip integration tests that need external services."""
    try:
        import nats  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="nats-py not installed")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip)
        return

    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(1)
        s.connect(("localhost", 4222))
        s.close()
    except (ConnectionRefusedError, OSError, socket.timeout):
        skip = pytest.mark.skip(reason="NATS server not running on localhost:4222")
        for item in items:
            if "integration" in str(item.fspath):
                item.add_marker(skip)
