# AGI-HPC Project — Divine Council transport layer
# Copyright (c) 2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""Backend abstraction for the Divine Council.

Decouples the council's deliberation logic from the specifics of
talking to a llama-server. Provides:

- :class:`BackendRequest` / :class:`BackendResponse` — transport DTOs
- :class:`CouncilBackend` — protocol that any backend must satisfy
- :class:`LlamaServerBackend` — the concrete backend for a single
  llama-server endpoint, with health probe, retry with jitter, and
  a per-backend circuit breaker
- :class:`FallbackBackend` — composite that prefers a primary and
  falls back to a secondary when the primary's circuit is open

Rationale (see docs/architecture/COUNCIL_RELIABILITY_PLAN.md):

- Gemma 4 crashes daily-ish in the field. Systemd ``Restart=always``
  will bring it back, but in-flight requests see a failure. Without
  the retry + circuit-breaker + fallback logic here, a crash produces
  a dead council for the duration of the restart window.

- The consensus logic in :mod:`divine_council` needs a three-valued
  outcome (approve / challenge / abstain). Backend errors must map
  to ``abstain`` so they don't silently count as approvals.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------


@dataclass
class BackendRequest:
    """A single request dispatched to a backend.

    ``trace_id`` is a correlation ID that should flow through all
    logs emitted in service of one user query. Set by the council
    at the top of ``deliberate()`` and threaded into every backend
    call so a ``journalctl -u atlas-ego`` search by trace_id yields
    every member's timing and error together.
    """

    system_prompt: str
    user_prompt: str
    max_tokens: int = 1024
    temperature: float = 0.3
    trace_id: str = ""
    member_id: str = ""  # for logging / metrics labels


@dataclass
class BackendResponse:
    """Output of a backend call.

    ``ok=True`` means a full response came back cleanly. ``ok=False``
    means the backend ultimately failed (all retries exhausted, or
    circuit open). The caller (the council) uses ``ok`` to distinguish
    abstain from approve/challenge.
    """

    ok: bool
    content: str = ""
    latency_s: float = 0.0
    attempts: int = 0
    backend_name: str = ""
    degraded: bool = False
    error: str = ""

    def as_error(self, message: str) -> "BackendResponse":
        self.ok = False
        self.error = message
        self.content = ""
        return self


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class _CircuitBreaker:
    """Simple threshold-based circuit breaker.

    Concurrent-safe via a lock. State transitions:

    - CLOSED: requests flow normally.
    - OPEN: requests are rejected immediately (fail-fast).
    - (no half-open state for simplicity; a health probe is used to
      close the circuit early rather than a half-open probe request.)

    The breaker opens after ``fail_threshold`` consecutive failures
    and stays open for ``cooldown_s`` seconds, after which the next
    request attempts the backend again.
    """

    def __init__(
        self,
        fail_threshold: int = 5,
        cooldown_s: float = 30.0,
    ) -> None:
        if fail_threshold < 1:
            raise ValueError("fail_threshold must be >= 1")
        if cooldown_s <= 0:
            raise ValueError("cooldown_s must be > 0")
        self._fail_threshold = fail_threshold
        self._cooldown_s = cooldown_s
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._open_until == 0.0:
                return False
            if time.monotonic() >= self._open_until:
                # Cooldown elapsed — allow the next request through.
                self._open_until = 0.0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._fail_threshold:
                self._open_until = time.monotonic() + self._cooldown_s

    def force_open(self) -> None:
        """Called by health-probe logic when a health check definitively fails."""
        with self._lock:
            self._consecutive_failures = max(
                self._consecutive_failures, self._fail_threshold
            )
            self._open_until = time.monotonic() + self._cooldown_s

    def force_closed(self) -> None:
        """Called by health-probe logic when the backend is confirmed healthy."""
        self.record_success()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "open": self._open_until > 0.0
                and time.monotonic() < self._open_until,
                "consecutive_failures": self._consecutive_failures,
                "seconds_until_retry": max(
                    0.0, self._open_until - time.monotonic()
                ),
            }


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------


class _HealthProbe:
    """Caches the outcome of a llama-server /health check.

    Called before every request; the cache window (``ttl_s``) keeps
    it from firing on every call. A fresh successful probe closes the
    circuit breaker early; a failed probe forces it open.
    """

    def __init__(self, url: str, ttl_s: float = 5.0, timeout_s: float = 2.0) -> None:
        self._url = url.rstrip("/")
        self._ttl_s = ttl_s
        self._timeout_s = timeout_s
        self._last_check = 0.0
        self._last_healthy = False
        self._lock = threading.Lock()

    def probe(self) -> bool:
        """Return True if the backend reports healthy (cached)."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_check < self._ttl_s:
                return self._last_healthy
            healthy = self._do_probe()
            self._last_healthy = healthy
            self._last_check = now
            return healthy

    def _do_probe(self) -> bool:
        try:
            r = requests.get(f"{self._url}/health", timeout=self._timeout_s)
        except requests.RequestException:
            return False
        if r.status_code != 200:
            return False
        try:
            data = r.json()
        except ValueError:
            # /health may return plaintext on some llama-server builds; 200 alone is fine.
            return True
        status = str(data.get("status", "")).lower()
        return status in {"", "ok", "ready"}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class CouncilBackend(Protocol):
    """Transport interface any council backend must satisfy."""

    name: str

    def chat(self, request: BackendRequest) -> BackendResponse:
        """Execute one chat-completion request; never raises."""
        ...

    def health_snapshot(self) -> Dict[str, Any]:
        """Return current health + circuit state for metrics/logs."""
        ...


# ---------------------------------------------------------------------------
# LlamaServerBackend
# ---------------------------------------------------------------------------


class LlamaServerBackend:
    """Concrete backend for a single llama-server endpoint.

    Wraps the OpenAI-compatible ``/v1/chat/completions`` endpoint with:

    - health probe (cached; forces circuit state on failure)
    - retry with exponential backoff + full jitter (only on transient
      failures; 4xx fails immediately)
    - circuit breaker (opens after consecutive failures)
    - single-request timeout

    The backend never raises. Every failure path returns a structured
    :class:`BackendResponse` with ``ok=False``.
    """

    def __init__(
        self,
        url: str,
        name: str = "",
        *,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_base_delay_s: float = 0.5,
        retry_max_delay_s: float = 3.0,
        circuit_fail_threshold: int = 5,
        circuit_cooldown_s: float = 30.0,
        health_ttl_s: float = 5.0,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not url:
            raise ValueError("url is required")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        self.name = name or url
        self._url = url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._retry_base_delay_s = retry_base_delay_s
        self._retry_max_delay_s = retry_max_delay_s
        self._session = session or requests.Session()
        self._sleep = sleep

        self._health = _HealthProbe(self._url, ttl_s=health_ttl_s)
        self._breaker = _CircuitBreaker(
            fail_threshold=circuit_fail_threshold,
            cooldown_s=circuit_cooldown_s,
        )

    # ---- public -------------------------------------------------------

    def chat(self, request: BackendRequest) -> BackendResponse:
        t0 = time.monotonic()

        # Fail fast if circuit is open.
        if self._breaker.is_open():
            elapsed = time.monotonic() - t0
            logger.info(
                "[council-backend] circuit open, rejecting request "
                "(backend=%s, trace_id=%s, member=%s)",
                self.name,
                request.trace_id,
                request.member_id,
            )
            return BackendResponse(
                ok=False,
                backend_name=self.name,
                latency_s=elapsed,
                attempts=0,
                error="circuit_open",
            )

        # Probe health — a cheap no-op most of the time (cached).
        if not self._health.probe():
            self._breaker.force_open()
            elapsed = time.monotonic() - t0
            logger.warning(
                "[council-backend] health check failed, opening circuit "
                "(backend=%s, trace_id=%s)",
                self.name,
                request.trace_id,
            )
            return BackendResponse(
                ok=False,
                backend_name=self.name,
                latency_s=elapsed,
                attempts=0,
                error="unhealthy",
            )

        last_err = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                content = self._do_chat(request)
                elapsed = time.monotonic() - t0
                self._breaker.record_success()
                return BackendResponse(
                    ok=True,
                    content=content,
                    latency_s=elapsed,
                    attempts=attempt,
                    backend_name=self.name,
                )
            except _PermanentError as e:
                # 4xx — don't retry; caller error.
                elapsed = time.monotonic() - t0
                last_err = f"permanent: {e}"
                self._breaker.record_failure()
                logger.warning(
                    "[council-backend] permanent error "
                    "(backend=%s, trace_id=%s, member=%s): %s",
                    self.name,
                    request.trace_id,
                    request.member_id,
                    e,
                )
                return BackendResponse(
                    ok=False,
                    backend_name=self.name,
                    latency_s=elapsed,
                    attempts=attempt,
                    error=last_err,
                )
            except _TransientError as e:
                last_err = f"transient: {e}"
                if attempt >= self._max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.info(
                    "[council-backend] transient error "
                    "(backend=%s, trace_id=%s, member=%s, attempt=%d/%d): "
                    "%s; retrying in %.2fs",
                    self.name,
                    request.trace_id,
                    request.member_id,
                    attempt,
                    self._max_retries,
                    e,
                    delay,
                )
                self._sleep(delay)

        # Out of retries.
        elapsed = time.monotonic() - t0
        self._breaker.record_failure()
        return BackendResponse(
            ok=False,
            backend_name=self.name,
            latency_s=elapsed,
            attempts=self._max_retries,
            error=last_err or "unknown",
        )

    def health_snapshot(self) -> Dict[str, Any]:
        snap = self._breaker.snapshot()
        snap["url"] = self._url
        snap["name"] = self.name
        return snap

    # ---- internals ----------------------------------------------------

    def _do_chat(self, request: BackendRequest) -> str:
        try:
            r = self._session.post(
                f"{self._url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": False,
                },
                timeout=self._timeout_s,
            )
        except requests.Timeout as e:
            raise _TransientError(f"timeout: {e}") from e
        except requests.ConnectionError as e:
            raise _TransientError(f"connection: {e}") from e
        except requests.RequestException as e:
            raise _TransientError(f"request: {e}") from e

        if 500 <= r.status_code < 600:
            raise _TransientError(f"HTTP {r.status_code}: {r.text[:200]}")
        if 400 <= r.status_code < 500:
            raise _PermanentError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise _TransientError(f"HTTP {r.status_code}: {r.text[:200]}")

        try:
            data = r.json()
        except ValueError as e:
            raise _TransientError(f"bad json: {e}") from e

        choices = data.get("choices", [])
        if not choices:
            raise _TransientError("no choices in response")
        msg = choices[0].get("message", {}) or {}
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        if not content.strip():
            raise _TransientError("empty content")
        return content

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter.

        Delay for attempt N is ``random.uniform(0, min(max, base * 2**(N-1)))``.
        """
        target = min(
            self._retry_max_delay_s,
            self._retry_base_delay_s * (2 ** (attempt - 1)),
        )
        return random.uniform(0.0, target)


class _TransientError(Exception):
    """Retryable transport/5xx error."""


class _PermanentError(Exception):
    """Non-retryable 4xx error."""


# ---------------------------------------------------------------------------
# FallbackBackend
# ---------------------------------------------------------------------------


class FallbackBackend:
    """Composite backend: primary first, fallback on open circuit.

    The fallback response is flagged with ``degraded=True`` so callers
    can surface the degradation to users / metrics.

    Use this when you have a secondary model that's already running
    for another purpose (e.g., Spock's Qwen 72B) and can serve as a
    single-model council when the primary is down.
    """

    def __init__(
        self,
        primary: CouncilBackend,
        fallback: CouncilBackend,
        name: str = "fallback",
    ) -> None:
        self.name = name
        self._primary = primary
        self._fallback = fallback

    def chat(self, request: BackendRequest) -> BackendResponse:
        primary_resp = self._primary.chat(request)
        if primary_resp.ok:
            return primary_resp

        # Primary failed. Try fallback.
        logger.info(
            "[council-backend] primary failed (%s), routing to fallback "
            "(trace_id=%s, member=%s)",
            primary_resp.error,
            request.trace_id,
            request.member_id,
        )
        fallback_resp = self._fallback.chat(request)
        fallback_resp.degraded = True
        fallback_resp.backend_name = f"{self.name}/{fallback_resp.backend_name}"
        return fallback_resp

    def health_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "primary": self._primary.health_snapshot(),
            "fallback": self._fallback.health_snapshot(),
        }


# ---------------------------------------------------------------------------
# Utility — produce a fresh trace ID for a deliberation
# ---------------------------------------------------------------------------


def new_trace_id() -> str:
    """Return a short, log-friendly trace ID (8-char UUID4 prefix)."""
    return uuid.uuid4().hex[:8]


__all__ = [
    "BackendRequest",
    "BackendResponse",
    "CouncilBackend",
    "FallbackBackend",
    "LlamaServerBackend",
    "new_trace_id",
]
