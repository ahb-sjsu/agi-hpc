# AGI-HPC Project — Divine Council metrics (optional prometheus)
# Copyright (c) 2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""Prometheus metrics for the Divine Council, gracefully degrading when
``prometheus_client`` is not installed.

This module exposes a small surface area (:class:`CouncilMetrics`) whose
methods are no-ops when prometheus is unavailable. Importing and using
it is safe on any machine — CI, dev laptops, Atlas. On production
(where prometheus_client is installed), the metrics are real and can
be scraped by the existing agi-hpc metrics endpoint.

The no-op pattern means council code does not need its own ``if metrics
else None`` gating; it just calls ``metrics.record_request_outcome(...)``
and trusts the implementation.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:  # pragma: no cover - import-guard
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - import-guard
    _PROM_AVAILABLE = False

    class _NoOp:
        """Stand-in for prometheus_client objects when the lib is missing."""

        def labels(self, **_kwargs: Any) -> "_NoOp":
            return self

        def inc(self, _amount: float = 1.0) -> None:
            pass

        def set(self, _value: float) -> None:
            pass

        def observe(self, _value: float) -> None:
            pass

    Counter = Gauge = Histogram = _NoOp  # type: ignore[misc,assignment]
    CollectorRegistry = None  # type: ignore[misc,assignment]


__all__ = ["CouncilMetrics", "PROMETHEUS_AVAILABLE"]

PROMETHEUS_AVAILABLE = _PROM_AVAILABLE


class CouncilMetrics:
    """Single entry point for council-level metrics.

    Methods are no-ops when prometheus_client is unavailable.

    Usage::

        metrics = CouncilMetrics.default()
        with metrics.track_request(member="judge", backend="gemma4"):
            response = backend.chat(request)
        metrics.record_request_outcome(
            member="judge",
            backend="gemma4",
            outcome="success",
            latency_s=response.latency_s,
        )
    """

    _singleton: Optional["CouncilMetrics"] = None

    def __init__(self, registry: Any = None) -> None:
        # When the library is available, using an isolated registry in
        # tests avoids "Duplicated timeseries in CollectorRegistry".
        # In production, pass None (default global registry).
        kw = (
            {"registry": registry} if (_PROM_AVAILABLE and registry is not None) else {}
        )
        self._request_total = Counter(
            "council_request_total",
            "Divine Council per-member request count",
            ["member", "backend", "outcome"],
            **kw,
        )
        self._request_latency = Histogram(
            "council_request_latency_seconds",
            "Divine Council per-member request latency",
            ["member", "backend", "outcome"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            **kw,
        )
        self._backend_health = Gauge(
            "council_backend_health",
            "1 if backend is healthy, 0 otherwise",
            ["backend"],
            **kw,
        )
        self._circuit_open = Gauge(
            "council_circuit_open",
            "1 if the circuit breaker is open, 0 otherwise",
            ["backend"],
            **kw,
        )
        self._fallback_activations = Counter(
            "council_fallback_active_total",
            "Count of deliberations that routed through the fallback backend",
            **kw,
        )
        self._deliberation_latency = Histogram(
            "council_deliberation_latency_seconds",
            "Divine Council end-to-end deliberation latency",
            ["consensus", "degraded"],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            **kw,
        )
        self._consensus_rate = Gauge(
            "council_consensus_rate",
            "Rolling fraction of deliberations that reached consensus "
            "(simple EWMA, alpha=0.1)",
            **kw,
        )
        self._ewma_consensus = 0.5  # neutral prior

    @classmethod
    def default(cls) -> "CouncilMetrics":
        """Process-wide singleton; use this unless you're testing."""
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    @classmethod
    def reset_singleton_for_tests(cls) -> None:
        """Only for tests — drops the singleton and re-initializes it into
        a fresh (isolated) registry so repeated calls don't collide with
        already-registered series in the global ``prometheus_client`` registry.
        """
        if _PROM_AVAILABLE and CollectorRegistry is not None:
            cls._singleton = cls(registry=CollectorRegistry())
        else:
            cls._singleton = cls()

    # ---- request-level ------------------------------------------------

    @contextmanager
    def track_request(self, *, member: str, backend: str) -> Iterator[None]:
        """Context manager that measures elapsed time for one request.

        On exit, latency is recorded under ``outcome="exception"`` if an
        exception is propagating, else the caller is expected to call
        :meth:`record_request_outcome` to record the success outcome.

        In the current code path, backends never raise (they return
        ``BackendResponse(ok=False)``), so the exception branch is
        defense-in-depth.
        """
        t0 = time.monotonic()
        try:
            yield
        except Exception:
            self.record_request_outcome(
                member=member,
                backend=backend,
                outcome="exception",
                latency_s=time.monotonic() - t0,
            )
            raise

    def record_request_outcome(
        self,
        *,
        member: str,
        backend: str,
        outcome: str,
        latency_s: float,
    ) -> None:
        """Record one member's request result.

        ``outcome`` is one of: ``success``, ``backend_error``, ``timeout``,
        ``circuit_open``, ``unhealthy``, ``exception``.
        """
        labels = {"member": member, "backend": backend, "outcome": outcome}
        self._request_total.labels(**labels).inc()
        self._request_latency.labels(**labels).observe(latency_s)

    # ---- backend-level ------------------------------------------------

    def set_backend_health(self, *, backend: str, healthy: bool) -> None:
        self._backend_health.labels(backend=backend).set(1.0 if healthy else 0.0)

    def set_circuit_state(self, *, backend: str, open_: bool) -> None:
        self._circuit_open.labels(backend=backend).set(1.0 if open_ else 0.0)

    # ---- deliberation-level -------------------------------------------

    def record_deliberation(
        self,
        *,
        consensus: bool,
        degraded: bool,
        latency_s: float,
    ) -> None:
        labels = {
            "consensus": "true" if consensus else "false",
            "degraded": "true" if degraded else "false",
        }
        self._deliberation_latency.labels(**labels).observe(latency_s)
        # EWMA the consensus rate (alpha=0.1 → ~30-sample effective memory)
        alpha = 0.1
        x = 1.0 if consensus else 0.0
        self._ewma_consensus = alpha * x + (1 - alpha) * self._ewma_consensus
        self._consensus_rate.set(self._ewma_consensus)

    def record_fallback_activated(self) -> None:
        self._fallback_activations.inc()
