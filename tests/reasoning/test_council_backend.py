# Copyright (c) 2026 Andrew H. Bond. AGI-HPC Responsible AI License v1.0.
"""Tests for the Divine Council transport layer."""

from __future__ import annotations

from typing import Callable, Dict, List
from unittest.mock import patch

import pytest
import requests

from agi.reasoning._council_backend import (
    BackendRequest,
    BackendResponse,
    FallbackBackend,
    LlamaServerBackend,
    new_trace_id,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    """Minimal stand-in for requests.Session that records calls."""

    def __init__(
        self,
        responses: List[_FakeResponse] | None = None,
        exceptions: List[Exception] | None = None,
    ):
        self.responses = list(responses or [])
        self.exceptions = list(exceptions or [])
        self.calls: List[dict] = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.exceptions:
            exc = self.exceptions.pop(0)
            if exc is not None:
                raise exc
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "ok"}}],
            },
        )


def _ok_response(content: str = "ok response"):
    return _FakeResponse(
        200,
        payload={"choices": [{"message": {"content": content}}]},
    )


def _healthy_probe(monkeypatch, backend: LlamaServerBackend, healthy: bool = True):
    """Force the health probe to return a given value for tests."""
    monkeypatch.setattr(
        backend._health, "probe", lambda: healthy
    )


def _make_request(member: str = "judge") -> BackendRequest:
    return BackendRequest(
        system_prompt="sys",
        user_prompt="user",
        max_tokens=100,
        temperature=0.3,
        trace_id=new_trace_id(),
        member_id=member,
    )


# ---------------------------------------------------------------------------
# LlamaServerBackend
# ---------------------------------------------------------------------------


class TestLlamaServerBackend:

    def test_construction_validates_args(self):
        with pytest.raises(ValueError, match="url"):
            LlamaServerBackend("")
        with pytest.raises(ValueError, match="max_retries"):
            LlamaServerBackend("http://x", max_retries=0)
        with pytest.raises(ValueError, match="timeout_s"):
            LlamaServerBackend("http://x", timeout_s=0)

    def test_successful_call(self, monkeypatch):
        session = _FakeSession(responses=[_ok_response("hello")])
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="test",
            session=session,
            sleep=lambda _: None,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        resp = backend.chat(_make_request())

        assert resp.ok is True
        assert resp.content == "hello"
        assert resp.attempts == 1
        assert resp.backend_name == "test"
        assert resp.degraded is False
        assert len(session.calls) == 1

    def test_retries_on_connection_error_then_succeeds(self, monkeypatch):
        session = _FakeSession(
            responses=[_ok_response("second try")],
            exceptions=[requests.ConnectionError("boom")],
        )
        # Only 1 exception, then 1 ok response -> should be ok on attempt 2.
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="retry-test",
            session=session,
            sleep=lambda _: None,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        resp = backend.chat(_make_request())

        assert resp.ok is True
        assert resp.content == "second try"
        assert resp.attempts == 2

    def test_exhausts_retries_and_returns_error(self, monkeypatch):
        session = _FakeSession(
            exceptions=[
                requests.ConnectionError("boom1"),
                requests.ConnectionError("boom2"),
                requests.ConnectionError("boom3"),
            ]
        )
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="exhaust",
            session=session,
            sleep=lambda _: None,
            max_retries=3,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        resp = backend.chat(_make_request())

        assert resp.ok is False
        assert resp.attempts == 3
        assert "transient" in resp.error.lower() or "connection" in resp.error.lower()

    def test_permanent_error_does_not_retry(self, monkeypatch):
        session = _FakeSession(responses=[_FakeResponse(status_code=400, text="bad")])
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="perm",
            session=session,
            sleep=lambda _: None,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        resp = backend.chat(_make_request())

        assert resp.ok is False
        assert resp.attempts == 1
        assert "permanent" in resp.error.lower()

    def test_5xx_is_retryable(self, monkeypatch):
        session = _FakeSession(
            responses=[
                _FakeResponse(status_code=503, text="try later"),
                _ok_response("recovered"),
            ]
        )
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="5xx",
            session=session,
            sleep=lambda _: None,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        resp = backend.chat(_make_request())

        assert resp.ok is True
        assert resp.attempts == 2

    def test_empty_content_is_transient(self, monkeypatch):
        session = _FakeSession(
            responses=[
                _FakeResponse(
                    200, payload={"choices": [{"message": {"content": ""}}]}
                ),
                _ok_response("now has content"),
            ]
        )
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="empty",
            session=session,
            sleep=lambda _: None,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        resp = backend.chat(_make_request())

        assert resp.ok is True
        assert resp.content == "now has content"
        assert resp.attempts == 2

    def test_unhealthy_probe_fails_fast(self, monkeypatch):
        session = _FakeSession()
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="unhealthy",
            session=session,
            sleep=lambda _: None,
        )
        _healthy_probe(monkeypatch, backend, healthy=False)

        resp = backend.chat(_make_request())

        assert resp.ok is False
        assert resp.error == "unhealthy"
        assert resp.attempts == 0
        assert len(session.calls) == 0  # no POSTs attempted

    def test_circuit_opens_after_threshold(self, monkeypatch):
        # All calls fail; circuit should open after 5 consecutive failures.
        session = _FakeSession(
            exceptions=[requests.ConnectionError("x")] * 30
        )
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="circuit",
            session=session,
            sleep=lambda _: None,
            max_retries=1,
            circuit_fail_threshold=3,
            circuit_cooldown_s=60,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        # 3 failures → circuit opens
        for _ in range(3):
            resp = backend.chat(_make_request())
            assert resp.ok is False

        # 4th call: circuit is open, fails immediately without retries
        resp = backend.chat(_make_request())
        assert resp.ok is False
        assert resp.error == "circuit_open"
        assert resp.attempts == 0

    def test_circuit_closes_on_success(self, monkeypatch):
        # 2 failures, then a success resets the counter
        session = _FakeSession(
            responses=[_ok_response("good")],
            exceptions=[requests.ConnectionError("a"), None, requests.ConnectionError("b"), None],
        )
        # Sequence: call 1 = exc (fail), call 2 = no exc + ok response (success)
        # We'll iterate manually rather than rely on interleaving.
        backend = LlamaServerBackend(
            "http://localhost:9999",
            name="close",
            session=session,
            sleep=lambda _: None,
            max_retries=1,
            circuit_fail_threshold=3,
        )
        _healthy_probe(monkeypatch, backend, healthy=True)

        # 1 failure
        resp1 = backend.chat(_make_request())
        assert not resp1.ok
        # 1 success — circuit counter resets
        resp2 = backend.chat(_make_request())
        assert resp2.ok
        # Counter is 0 now, circuit remains closed

    def test_health_snapshot(self, monkeypatch):
        session = _FakeSession()
        backend = LlamaServerBackend(
            "http://localhost:9999", name="snap", session=session
        )
        snap = backend.health_snapshot()
        assert snap["name"] == "snap"
        assert snap["url"] == "http://localhost:9999"
        assert snap["open"] is False
        assert snap["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# FallbackBackend
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal CouncilBackend for testing composition."""

    def __init__(self, name: str, outcomes: List[BackendResponse]):
        self.name = name
        self._outcomes = outcomes
        self.calls = 0

    def chat(self, request: BackendRequest) -> BackendResponse:
        self.calls += 1
        if self._outcomes:
            r = self._outcomes.pop(0)
            r.backend_name = self.name
            return r
        return BackendResponse(
            ok=True, content="default", backend_name=self.name, attempts=1
        )

    def health_snapshot(self) -> Dict:
        return {"name": self.name}


class TestFallbackBackend:

    def test_primary_success_skips_fallback(self):
        primary = _StubBackend("primary", [BackendResponse(ok=True, content="p")])
        fallback = _StubBackend("fb", [])
        comp = FallbackBackend(primary, fallback)

        resp = comp.chat(_make_request())

        assert resp.ok is True
        assert resp.content == "p"
        assert resp.degraded is False
        assert primary.calls == 1
        assert fallback.calls == 0

    def test_primary_failure_routes_to_fallback(self):
        primary = _StubBackend("primary", [BackendResponse(ok=False, error="dead")])
        fallback = _StubBackend("fb", [BackendResponse(ok=True, content="rescued")])
        comp = FallbackBackend(primary, fallback)

        resp = comp.chat(_make_request())

        assert resp.ok is True
        assert resp.content == "rescued"
        assert resp.degraded is True
        assert primary.calls == 1
        assert fallback.calls == 1

    def test_both_fail_returns_error(self):
        primary = _StubBackend("primary", [BackendResponse(ok=False, error="p")])
        fallback = _StubBackend("fb", [BackendResponse(ok=False, error="f")])
        comp = FallbackBackend(primary, fallback)

        resp = comp.chat(_make_request())

        assert resp.ok is False
        assert resp.degraded is True
        assert primary.calls == 1
        assert fallback.calls == 1

    def test_health_snapshot_includes_both(self):
        primary = _StubBackend("p", [])
        fallback = _StubBackend("f", [])
        comp = FallbackBackend(primary, fallback, name="test-fb")
        snap = comp.health_snapshot()
        assert snap["name"] == "test-fb"
        assert snap["primary"] == {"name": "p"}
        assert snap["fallback"] == {"name": "f"}


# ---------------------------------------------------------------------------
# new_trace_id
# ---------------------------------------------------------------------------


class TestTraceId:

    def test_format(self):
        tid = new_trace_id()
        assert len(tid) == 8
        assert all(c in "0123456789abcdef" for c in tid)

    def test_uniqueness(self):
        ids = {new_trace_id() for _ in range(1000)}
        assert len(ids) == 1000  # 8 hex chars → 4.3 B possibilities
