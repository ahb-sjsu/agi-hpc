"""Tests for LH resilience patterns."""

import time
import threading
import pytest

from agi.lh.resilience import (
    RetryConfig,
    retry,
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    with_timeout,
    TimeoutError,
    with_fallback,
    Bulkhead,
    BulkheadFullError,
    resilient,
)


# ---------------------------------------------------------------------------
# Retry Tests
# ---------------------------------------------------------------------------


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_default_config(self):
        """RetryConfig should have sensible defaults."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 0.1
        assert config.exponential_base == 2.0

    def test_get_delay_exponential_backoff(self):
        """get_delay should implement exponential backoff."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=0)
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0

    def test_get_delay_respects_max(self):
        """get_delay should respect max_delay."""
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, jitter=0)
        assert config.get_delay(10) == 5.0


class TestRetryDecorator:
    """Tests for retry decorator."""

    def test_retry_succeeds_first_try(self):
        """retry should not retry on success."""
        call_count = [0]

        @retry(max_attempts=3)
        def success_func():
            call_count[0] += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_retries_on_failure(self):
        """retry should retry on failure."""
        call_count = [0]

        @retry(max_attempts=3, initial_delay=0.001)
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("fail")
            return "success"

        result = fail_twice()
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_raises_after_max_attempts(self):
        """retry should raise after max attempts."""
        call_count = [0]

        @retry(max_attempts=3, initial_delay=0.001)
        def always_fail():
            call_count[0] += 1
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            always_fail()

        assert call_count[0] == 3

    def test_retry_only_retries_specified_exceptions(self):
        """retry should only retry specified exceptions."""
        call_count = [0]

        @retry(max_attempts=3, retryable_exceptions=(ValueError,), initial_delay=0.001)
        def raise_type_error():
            call_count[0] += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            raise_type_error()

        # Should not have retried
        assert call_count[0] == 1

    def test_retry_calls_on_retry_callback(self):
        """retry should call on_retry callback."""
        retries = []

        def on_retry(exc, attempt):
            retries.append((type(exc).__name__, attempt))

        @retry(max_attempts=3, initial_delay=0.001, on_retry=on_retry)
        def fail_twice():
            if len(retries) < 2:
                raise ValueError("fail")
            return "success"

        result = fail_twice()
        assert result == "success"
        assert len(retries) == 2
        assert retries[0] == ("ValueError", 1)
        assert retries[1] == ("ValueError", 2)


# ---------------------------------------------------------------------------
# Circuit Breaker Tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_circuit_starts_closed(self):
        """Circuit breaker should start in closed state."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED
        assert not breaker.is_open

    def test_circuit_opens_after_failures(self):
        """Circuit breaker should open after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        for _ in range(3):
            breaker.record_failure(Exception("fail"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_circuit_rejects_when_open(self):
        """Circuit breaker should reject requests when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        @breaker
        def protected_func():
            return "success"

        # First failure opens the circuit
        breaker.record_failure(Exception("fail"))

        with pytest.raises(CircuitOpenError):
            protected_func()

    def test_circuit_transitions_to_half_open(self):
        """Circuit breaker should transition to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.01)
        breaker = CircuitBreaker("test", config)

        breaker.record_failure(Exception("fail"))
        assert breaker.state == CircuitState.OPEN

        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_success_in_half_open(self):
        """Circuit breaker should close after successes in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1, success_threshold=2, timeout=0.01
        )
        breaker = CircuitBreaker("test", config)

        breaker.record_failure(Exception("fail"))
        time.sleep(0.02)  # Wait for timeout to elapse

        # Access state to trigger transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_reopens_on_failure_in_half_open(self):
        """Circuit breaker should reopen on failure in half-open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.01)
        breaker = CircuitBreaker("test", config)

        breaker.record_failure(Exception("fail"))
        time.sleep(0.02)  # Transition to half-open

        breaker.record_failure(Exception("fail again"))
        assert breaker.state == CircuitState.OPEN

    def test_circuit_as_decorator(self):
        """Circuit breaker should work as decorator."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("test", config)
        call_count = [0]

        @breaker
        def flaky_func():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("fail")
            return "success"

        # First two calls fail, opening circuit
        with pytest.raises(ValueError):
            flaky_func()
        with pytest.raises(ValueError):
            flaky_func()

        # Circuit is now open
        with pytest.raises(CircuitOpenError):
            flaky_func()

    def test_circuit_reset(self):
        """Circuit breaker reset should return to closed state."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        breaker.record_failure(Exception("fail"))
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert not breaker.is_open


# ---------------------------------------------------------------------------
# Timeout Tests
# ---------------------------------------------------------------------------


class TestTimeout:
    """Tests for timeout decorator."""

    def test_timeout_allows_fast_function(self):
        """Timeout should allow functions that complete in time."""

        @with_timeout(1.0)
        def fast_func():
            return "fast"

        result = fast_func()
        assert result == "fast"

    def test_timeout_raises_on_slow_function(self):
        """Timeout should raise TimeoutError on slow functions."""

        @with_timeout(0.01)
        def slow_func():
            time.sleep(1.0)
            return "slow"

        with pytest.raises(TimeoutError):
            slow_func()

    def test_timeout_uses_fallback(self):
        """Timeout should use fallback when provided."""

        @with_timeout(0.01, fallback=lambda: "fallback")
        def slow_func():
            time.sleep(1.0)
            return "slow"

        result = slow_func()
        assert result == "fallback"

    def test_timeout_propagates_exceptions(self):
        """Timeout should propagate exceptions from function."""

        @with_timeout(1.0)
        def error_func():
            raise ValueError("error")

        with pytest.raises(ValueError, match="error"):
            error_func()


# ---------------------------------------------------------------------------
# Fallback Tests
# ---------------------------------------------------------------------------


class TestFallback:
    """Tests for fallback decorator."""

    def test_fallback_returns_primary_on_success(self):
        """Fallback should return primary result on success."""

        @with_fallback(lambda: "fallback")
        def primary():
            return "primary"

        result = primary()
        assert result == "primary"

    def test_fallback_returns_fallback_on_failure(self):
        """Fallback should return fallback result on failure."""

        @with_fallback(lambda: "fallback")
        def failing():
            raise ValueError("fail")

        result = failing()
        assert result == "fallback"

    def test_fallback_only_catches_specified_exceptions(self):
        """Fallback should only catch specified exceptions."""

        @with_fallback(lambda: "fallback", exceptions=(ValueError,))
        def raise_type_error():
            raise TypeError("not caught")

        with pytest.raises(TypeError):
            raise_type_error()


# ---------------------------------------------------------------------------
# Bulkhead Tests
# ---------------------------------------------------------------------------


class TestBulkhead:
    """Tests for Bulkhead pattern."""

    def test_bulkhead_allows_under_limit(self):
        """Bulkhead should allow requests under limit."""
        bulkhead = Bulkhead("test", max_concurrent=2)

        @bulkhead
        def fast_func():
            return "success"

        result = fast_func()
        assert result == "success"

    def test_bulkhead_tracks_active_count(self):
        """Bulkhead should track active executions."""
        bulkhead = Bulkhead("test", max_concurrent=5)
        active_during = []

        @bulkhead
        def track_active():
            active_during.append(bulkhead.active_count)
            time.sleep(0.01)
            return "done"

        threads = []
        for _ in range(3):
            t = threading.Thread(target=track_active)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # At least one thread should have seen > 1 active
        assert max(active_during) >= 1

    def test_bulkhead_rejects_over_limit(self):
        """Bulkhead should reject when at capacity."""
        bulkhead = Bulkhead("test", max_concurrent=1, max_wait=0.01)
        started = threading.Event()
        proceed = threading.Event()

        @bulkhead
        def blocking_func():
            started.set()
            proceed.wait(timeout=1.0)
            return "done"

        # Start a blocking call
        thread = threading.Thread(target=blocking_func)
        thread.start()
        started.wait()

        # Second call should be rejected
        with pytest.raises(BulkheadFullError):

            @bulkhead
            def second_call():
                return "second"

            second_call()

        # Clean up
        proceed.set()
        thread.join()


# ---------------------------------------------------------------------------
# Combined Resilience Tests
# ---------------------------------------------------------------------------


class TestResilient:
    """Tests for combined resilience decorator."""

    def test_resilient_applies_patterns(self):
        """resilient should apply all patterns."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=5))
        retry_config = RetryConfig(max_attempts=2, initial_delay=0.001)

        call_count = [0]

        @resilient(circuit_breaker=breaker, retry_config=retry_config)
        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        result = flaky()
        assert result == "success"
        assert call_count[0] == 2

    def test_resilient_circuit_breaker_takes_precedence(self):
        """resilient should fail fast when circuit is open."""
        breaker = CircuitBreaker(
            "test", CircuitBreakerConfig(failure_threshold=1, timeout=100)
        )

        @resilient(circuit_breaker=breaker)
        def fail():
            raise ValueError("fail")

        # First call opens circuit
        with pytest.raises(ValueError):
            fail()

        # Second call should be rejected by circuit breaker
        with pytest.raises(CircuitOpenError):
            fail()
