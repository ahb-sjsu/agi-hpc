# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Resilience patterns for the Left Hemisphere.

Provides:
- Retry logic with exponential backoff
- Circuit breakers for downstream services
- Timeout handling
- Graceful degradation
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry Configuration
# ---------------------------------------------------------------------------


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay: float = 0.1  # seconds
    max_delay: float = 10.0  # seconds
    exponential_base: float = 2.0
    jitter: float = 0.1  # Random jitter factor (0-1)
    retryable_exceptions: tuple = (Exception,)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        delay = min(
            self.initial_delay * (self.exponential_base**attempt),
            self.max_delay,
        )
        # Add jitter
        jitter_amount = delay * self.jitter * random.random()
        return delay + jitter_amount


def retry(
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[F], F]:
    """
    Decorator for retry with exponential backoff.

    Usage:
        @retry(max_attempts=3, retryable_exceptions=(ConnectionError,))
        def call_external_service():
            ...

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Random jitter factor (0-1)
        retryable_exceptions: Tuple of exceptions to retry on
        on_retry: Optional callback called on each retry
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
    )

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_attempts - 1:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            "[Retry] %s failed (attempt %d/%d): %s. Retrying in %.2fs",
                            func.__name__,
                            attempt + 1,
                            config.max_attempts,
                            e,
                            delay,
                        )

                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)

            # All attempts failed
            logger.error(
                "[Retry] %s failed after %d attempts",
                func.__name__,
                config.max_attempts,
            )
            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open to close
    timeout: float = 30.0  # Seconds before trying half-open
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing recovery, limited requests pass through

    Usage:
        breaker = CircuitBreaker("safety-service")

        @breaker
        def call_safety_service():
            ...
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if (
                    self._last_failure_time
                    and time.monotonic() - self._last_failure_time
                    >= self.config.timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(
                        "[CircuitBreaker] %s: OPEN -> HALF_OPEN (timeout elapsed)",
                        self.name,
                    )
            return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(
                        "[CircuitBreaker] %s: HALF_OPEN -> CLOSED (recovered)",
                        self.name,
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this exception should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._state = CircuitState.OPEN
                logger.warning(
                    "[CircuitBreaker] %s: HALF_OPEN -> OPEN (failure in recovery)",
                    self.name,
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "[CircuitBreaker] %s: CLOSED -> OPEN (threshold reached: %d failures)",
                        self.name,
                        self._failure_count,
                    )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info("[CircuitBreaker] %s: Reset to CLOSED", self.name)

    def __call__(self, func: F) -> F:
        """Use circuit breaker as a decorator."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.is_open:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open, rejecting request"
                )

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper  # type: ignore


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    pass


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TimeoutError(Exception):
    """Raised when an operation times out."""

    pass


def with_timeout(
    timeout_seconds: float,
    fallback: Optional[Callable[[], T]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add timeout to a function.

    Note: This uses threading for timeout, which may not interrupt
    blocking I/O. For gRPC calls, use the timeout parameter instead.

    Usage:
        @with_timeout(5.0)
        def slow_operation():
            ...

        @with_timeout(5.0, fallback=lambda: default_value)
        def operation_with_fallback():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result: list = []
            exception: list = []

            def target() -> None:
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    exception.append(e)

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)

            if thread.is_alive():
                logger.warning(
                    "[Timeout] %s exceeded timeout of %.1fs",
                    func.__name__,
                    timeout_seconds,
                )
                if fallback:
                    return fallback()
                raise TimeoutError(
                    f"{func.__name__} exceeded timeout of {timeout_seconds}s"
                )

            if exception:
                raise exception[0]

            return result[0] if result else None  # type: ignore

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def with_fallback(
    fallback_func: Callable[..., T],
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to provide fallback on failure.

    Usage:
        def get_default_plan():
            return default_plan

        @with_fallback(get_default_plan, exceptions=(ConnectionError,))
        def get_plan_from_llm():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(
                    "[Fallback] %s failed with %s, using fallback",
                    func.__name__,
                    type(e).__name__,
                )
                return fallback_func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Bulkhead (Concurrency Limiter)
# ---------------------------------------------------------------------------


class Bulkhead:
    """
    Bulkhead pattern to limit concurrent executions.

    Prevents a single component from consuming all resources.

    Usage:
        bulkhead = Bulkhead("llm-calls", max_concurrent=5)

        @bulkhead
        def call_llm():
            ...
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int = 10,
        max_wait: float = 30.0,
    ):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait = max_wait
        self._semaphore = threading.Semaphore(max_concurrent)
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active_count(self) -> int:
        """Get number of active executions."""
        with self._lock:
            return self._active

    def __call__(self, func: F) -> F:
        """Use bulkhead as a decorator."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            acquired = self._semaphore.acquire(timeout=self.max_wait)
            if not acquired:
                raise BulkheadFullError(
                    f"Bulkhead '{self.name}' is full ({self.max_concurrent} concurrent)"
                )

            with self._lock:
                self._active += 1

            try:
                return func(*args, **kwargs)
            finally:
                with self._lock:
                    self._active -= 1
                self._semaphore.release()

        return wrapper  # type: ignore


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""

    pass


# ---------------------------------------------------------------------------
# Combined Resilience Pattern
# ---------------------------------------------------------------------------


def resilient(
    circuit_breaker: Optional[CircuitBreaker] = None,
    retry_config: Optional[RetryConfig] = None,
    timeout_seconds: Optional[float] = None,
    bulkhead: Optional[Bulkhead] = None,
) -> Callable[[F], F]:
    """
    Combined resilience decorator applying multiple patterns.

    Order of application (outer to inner):
    1. Bulkhead (limit concurrency)
    2. Circuit breaker (fail fast)
    3. Timeout (prevent hanging)
    4. Retry (handle transient failures)

    Usage:
        @resilient(
            circuit_breaker=CircuitBreaker("external-api"),
            retry_config=RetryConfig(max_attempts=3),
            timeout_seconds=10.0,
        )
        def call_external_api():
            ...
    """

    def decorator(func: F) -> F:
        wrapped = func

        # Apply retry (innermost)
        if retry_config:
            wrapped = retry(
                max_attempts=retry_config.max_attempts,
                initial_delay=retry_config.initial_delay,
                max_delay=retry_config.max_delay,
                exponential_base=retry_config.exponential_base,
                jitter=retry_config.jitter,
                retryable_exceptions=retry_config.retryable_exceptions,
            )(wrapped)

        # Apply timeout
        if timeout_seconds:
            wrapped = with_timeout(timeout_seconds)(wrapped)

        # Apply circuit breaker
        if circuit_breaker:
            wrapped = circuit_breaker(wrapped)

        # Apply bulkhead (outermost)
        if bulkhead:
            wrapped = bulkhead(wrapped)

        return wrapped  # type: ignore

    return decorator
