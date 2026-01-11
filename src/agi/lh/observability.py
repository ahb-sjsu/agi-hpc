# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Observability infrastructure for the Left Hemisphere.

Provides:
- Prometheus metrics for monitoring
- Structured logging with correlation IDs
- Request context propagation
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Generator
import threading

logger = logging.getLogger(__name__)

# Thread-local storage for request context
_context = threading.local()


# ---------------------------------------------------------------------------
# Request Context
# ---------------------------------------------------------------------------


@dataclass
class RequestContext:
    """
    Context for a single request, propagated through the call chain.

    Attributes:
        correlation_id: Unique ID for tracing the request
        request_id: ID of the original request
        start_time: When the request started (monotonic)
        metadata: Additional context metadata
    """

    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    request_id: str = ""
    start_time: float = field(default_factory=time.monotonic)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time since request start in milliseconds."""
        return (time.monotonic() - self.start_time) * 1000


def get_context() -> Optional[RequestContext]:
    """Get the current request context."""
    return getattr(_context, "current", None)


def set_context(ctx: RequestContext) -> None:
    """Set the current request context."""
    _context.current = ctx


def clear_context() -> None:
    """Clear the current request context."""
    if hasattr(_context, "current"):
        delattr(_context, "current")


@contextmanager
def request_context(
    correlation_id: Optional[str] = None,
    request_id: str = "",
    **metadata: Any,
) -> Generator[RequestContext, None, None]:
    """
    Context manager for request tracking.

    Usage:
        with request_context(request_id="plan-001") as ctx:
            # All operations within this block share the context
            process_request()
    """
    ctx = RequestContext(
        correlation_id=correlation_id or uuid.uuid4().hex[:16],
        request_id=request_id,
        metadata=metadata,
    )
    old_ctx = get_context()
    set_context(ctx)
    try:
        yield ctx
    finally:
        if old_ctx:
            set_context(old_ctx)
        else:
            clear_context()


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------


class StructuredLogAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds correlation ID and structured context.

    Usage:
        log = get_structured_logger(__name__)
        log.info("Processing request", plan_id="plan-001", steps=5)
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Add context to log message."""
        ctx = get_context()
        extra = kwargs.get("extra", {})

        # Add correlation ID if available
        if ctx:
            extra["correlation_id"] = ctx.correlation_id
            extra["request_id"] = ctx.request_id
            extra["elapsed_ms"] = f"{ctx.elapsed_ms:.2f}"

        # Add any extra fields passed to the log call
        if self.extra:
            extra.update(self.extra)

        kwargs["extra"] = extra
        return msg, kwargs


def get_structured_logger(name: str) -> StructuredLogAdapter:
    """Get a structured logger for the given module name."""
    return StructuredLogAdapter(logging.getLogger(name), {})


# ---------------------------------------------------------------------------
# Metrics (Prometheus-compatible)
# ---------------------------------------------------------------------------


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, name: str, description: str, labels: Optional[list] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def inc(self, value: float = 1, **label_values: str) -> None:
        """Increment the counter."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def get(self, **label_values: str) -> float:
        """Get current counter value."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        return self._values.get(key, 0)

    def collect(self) -> Dict[tuple, float]:
        """Collect all metric values."""
        with self._lock:
            return dict(self._values)


class Histogram:
    """Thread-safe histogram metric for tracking distributions."""

    DEFAULT_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[list] = None,
        buckets: Optional[tuple] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._counts: Dict[tuple, Dict[float, int]] = {}
        self._sums: Dict[tuple, float] = {}
        self._totals: Dict[tuple, int] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **label_values: str) -> None:
        """Record an observation."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        with self._lock:
            if key not in self._counts:
                self._counts[key] = {b: 0 for b in self.buckets}
                self._sums[key] = 0
                self._totals[key] = 0

            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[key][bucket] += 1

            self._sums[key] += value
            self._totals[key] += 1

    def get_sum(self, **label_values: str) -> float:
        """Get sum of all observations."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        return self._sums.get(key, 0)

    def get_count(self, **label_values: str) -> int:
        """Get count of all observations."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        return self._totals.get(key, 0)

    @contextmanager
    def time(self, **label_values: str) -> Generator[None, None, None]:
        """Context manager to time an operation."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.observe(time.monotonic() - start, **label_values)


class Gauge:
    """Thread-safe gauge metric for current values."""

    def __init__(self, name: str, description: str, labels: Optional[list] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **label_values: str) -> None:
        """Set the gauge value."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1, **label_values: str) -> None:
        """Increment the gauge."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def dec(self, value: float = 1, **label_values: str) -> None:
        """Decrement the gauge."""
        self.inc(-value, **label_values)

    def get(self, **label_values: str) -> float:
        """Get current gauge value."""
        key = tuple(label_values.get(lbl, "") for lbl in self.labels)
        return self._values.get(key, 0)


# ---------------------------------------------------------------------------
# LH Metrics Registry
# ---------------------------------------------------------------------------


class LHMetrics:
    """
    Metrics registry for Left Hemisphere service.

    Provides pre-defined metrics for monitoring LH health and performance.
    """

    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            "lh_requests_total",
            "Total number of planning requests",
            labels=["method", "status"],
        )

        self.request_duration = Histogram(
            "lh_request_duration_seconds",
            "Request duration in seconds",
            labels=["method"],
        )

        # Plan metrics
        self.plan_steps = Histogram(
            "lh_plan_steps",
            "Number of steps in generated plans",
            labels=["planner_type"],
            buckets=(1, 2, 5, 10, 15, 20, 30, 50, float("inf")),
        )

        self.plan_generation_duration = Histogram(
            "lh_plan_generation_seconds",
            "Plan generation duration in seconds",
            labels=["planner_type"],
        )

        # Safety metrics
        self.safety_checks_total = Counter(
            "lh_safety_checks_total",
            "Total safety checks performed",
            labels=["result"],  # approved, rejected
        )

        self.safety_check_duration = Histogram(
            "lh_safety_check_seconds",
            "Safety check duration in seconds",
        )

        # Metacognition metrics
        self.metacog_reviews_total = Counter(
            "lh_metacog_reviews_total",
            "Total metacognition reviews",
            labels=["decision"],  # accept, revise, reject
        )

        self.metacog_revision_count = Histogram(
            "lh_metacog_revisions",
            "Number of revisions per plan",
            buckets=(0, 1, 2, 3, 5, 10, float("inf")),
        )

        # LLM metrics
        self.llm_calls_total = Counter(
            "lh_llm_calls_total",
            "Total LLM API calls",
            labels=["provider", "status"],
        )

        self.llm_tokens_total = Counter(
            "lh_llm_tokens_total",
            "Total LLM tokens used",
            labels=["provider", "type"],  # type: prompt, completion
        )

        self.llm_call_duration = Histogram(
            "lh_llm_call_seconds",
            "LLM API call duration",
            labels=["provider"],
        )

        # Memory metrics
        self.memory_queries_total = Counter(
            "lh_memory_queries_total",
            "Total memory queries",
            labels=["memory_type", "status"],
        )

        self.memory_query_duration = Histogram(
            "lh_memory_query_seconds",
            "Memory query duration",
            labels=["memory_type"],
        )

        # Error metrics
        self.errors_total = Counter(
            "lh_errors_total",
            "Total errors by type",
            labels=["error_type"],
        )

        # Active requests gauge
        self.active_requests = Gauge(
            "lh_active_requests",
            "Number of currently active requests",
        )

    def to_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-compatible metrics string
        """
        lines = []

        def format_metric(metric, metric_type: str) -> None:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} {metric_type}")

            if isinstance(metric, Counter):
                for labels, value in metric.collect().items():
                    label_str = self._format_labels(metric.labels, labels)
                    lines.append(f"{metric.name}{label_str} {value}")

            elif isinstance(metric, Gauge):
                with metric._lock:
                    for labels, value in metric._values.items():
                        label_str = self._format_labels(metric.labels, labels)
                        lines.append(f"{metric.name}{label_str} {value}")

            elif isinstance(metric, Histogram):
                with metric._lock:
                    for labels, buckets in metric._counts.items():
                        label_str = self._format_labels(metric.labels, labels)
                        for bucket, count in buckets.items():
                            le = "+Inf" if bucket == float("inf") else str(bucket)
                            lines.append(
                                f'{metric.name}_bucket{{{label_str[1:-1]},le="{le}"}} {count}'
                                if label_str != "{}"
                                else f'{metric.name}_bucket{{le="{le}"}} {count}'
                            )
                        lines.append(
                            f"{metric.name}_sum{label_str} {metric._sums.get(labels, 0)}"
                        )
                        lines.append(
                            f"{metric.name}_count{label_str} {metric._totals.get(labels, 0)}"
                        )

        # Export all metrics
        format_metric(self.requests_total, "counter")
        format_metric(self.request_duration, "histogram")
        format_metric(self.plan_steps, "histogram")
        format_metric(self.plan_generation_duration, "histogram")
        format_metric(self.safety_checks_total, "counter")
        format_metric(self.safety_check_duration, "histogram")
        format_metric(self.metacog_reviews_total, "counter")
        format_metric(self.llm_calls_total, "counter")
        format_metric(self.llm_tokens_total, "counter")
        format_metric(self.llm_call_duration, "histogram")
        format_metric(self.memory_queries_total, "counter")
        format_metric(self.memory_query_duration, "histogram")
        format_metric(self.errors_total, "counter")
        format_metric(self.active_requests, "gauge")

        return "\n".join(lines)

    def _format_labels(self, label_names: list, label_values: tuple) -> str:
        """Format labels for Prometheus output."""
        if not label_names:
            return "{}"
        pairs = [f'{n}="{v}"' for n, v in zip(label_names, label_values, strict=True)]
        return "{" + ",".join(pairs) + "}"


# Global metrics instance
metrics = LHMetrics()


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def track_request(method: str) -> Callable[[F], F]:
    """
    Decorator to track request metrics.

    Usage:
        @track_request("Plan")
        def Plan(self, request, context):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics.active_requests.inc()
            start_time = time.monotonic()
            status = "success"

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                metrics.errors_total.inc(error_type=type(e).__name__)
                raise
            finally:
                duration = time.monotonic() - start_time
                metrics.requests_total.inc(method=method, status=status)
                metrics.request_duration.observe(duration, method=method)
                metrics.active_requests.dec()

        return wrapper  # type: ignore

    return decorator


def track_llm_call(provider: str) -> Callable[[F], F]:
    """
    Decorator to track LLM API call metrics.

    Usage:
        @track_llm_call("openai")
        def generate(self, prompt):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.monotonic()
            status = "success"

            try:
                result = func(*args, **kwargs)

                # Track token usage if result has usage info
                if hasattr(result, "prompt_tokens"):
                    metrics.llm_tokens_total.inc(
                        result.prompt_tokens, provider=provider, type="prompt"
                    )
                if hasattr(result, "completion_tokens"):
                    metrics.llm_tokens_total.inc(
                        result.completion_tokens, provider=provider, type="completion"
                    )

                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.monotonic() - start_time
                metrics.llm_calls_total.inc(provider=provider, status=status)
                metrics.llm_call_duration.observe(duration, provider=provider)

        return wrapper  # type: ignore

    return decorator
