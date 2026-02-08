"""Tests for LH observability infrastructure."""

import time
import threading
import pytest

from agi.lh.observability import (
    RequestContext,
    get_context,
    set_context,
    clear_context,
    request_context,
    StructuredLogAdapter,
    get_structured_logger,
    Counter,
    Histogram,
    Gauge,
    LHMetrics,
    metrics,
    track_request,
    track_llm_call,
)

# ---------------------------------------------------------------------------
# RequestContext Tests
# ---------------------------------------------------------------------------


class TestRequestContext:
    """Tests for RequestContext."""

    def test_context_generates_correlation_id(self):
        """Context should generate a correlation ID if not provided."""
        ctx = RequestContext()
        assert len(ctx.correlation_id) == 16

    def test_context_accepts_custom_correlation_id(self):
        """Context should accept a custom correlation ID."""
        ctx = RequestContext(correlation_id="custom-123")
        assert ctx.correlation_id == "custom-123"

    def test_context_tracks_elapsed_time(self):
        """Context should track elapsed time."""
        ctx = RequestContext()
        time.sleep(0.05)  # Sleep 50ms
        assert ctx.elapsed_ms >= 40  # Allow some timing slack

    def test_context_stores_metadata(self):
        """Context should store metadata."""
        ctx = RequestContext(metadata={"plan_id": "plan-001"})
        assert ctx.metadata["plan_id"] == "plan-001"


class TestContextManagement:
    """Tests for context get/set/clear."""

    def teardown_method(self):
        """Clear context after each test."""
        clear_context()

    def test_get_context_returns_none_when_not_set(self):
        """get_context should return None when no context is set."""
        clear_context()
        assert get_context() is None

    def test_set_and_get_context(self):
        """set_context and get_context should work together."""
        ctx = RequestContext(correlation_id="test-123")
        set_context(ctx)
        assert get_context() is ctx

    def test_clear_context(self):
        """clear_context should remove the context."""
        ctx = RequestContext()
        set_context(ctx)
        clear_context()
        assert get_context() is None

    def test_request_context_manager(self):
        """request_context should set and clear context."""
        with request_context(correlation_id="ctx-123", request_id="req-001") as ctx:
            assert get_context() is ctx
            assert ctx.correlation_id == "ctx-123"
            assert ctx.request_id == "req-001"

        assert get_context() is None

    def test_nested_context_managers(self):
        """Nested context managers should restore outer context."""
        with request_context(correlation_id="outer") as outer:
            with request_context(correlation_id="inner") as inner:
                assert get_context() is inner
            assert get_context() is outer

    def test_context_is_thread_local(self):
        """Context should be thread-local."""
        main_ctx = RequestContext(correlation_id="main")
        set_context(main_ctx)

        thread_ctx_seen = []

        def thread_func():
            # Should not see main thread's context
            thread_ctx_seen.append(get_context())
            # Set own context
            thread_ctx = RequestContext(correlation_id="thread")
            set_context(thread_ctx)
            thread_ctx_seen.append(get_context().correlation_id)

        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()

        # Main thread context should be unchanged
        assert get_context() is main_ctx
        # Thread should have seen None initially
        assert thread_ctx_seen[0] is None
        # Thread should have set its own context
        assert thread_ctx_seen[1] == "thread"


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


class TestCounter:
    """Tests for Counter metric."""

    def test_counter_starts_at_zero(self):
        """Counter should start at zero."""
        counter = Counter("test_counter", "Test counter")
        assert counter.get() == 0

    def test_counter_increments(self):
        """Counter should increment."""
        counter = Counter("test_counter", "Test counter")
        counter.inc()
        assert counter.get() == 1
        counter.inc(5)
        assert counter.get() == 6

    def test_counter_with_labels(self):
        """Counter should support labels."""
        counter = Counter("test_counter", "Test counter", labels=["method", "status"])
        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="200")
        counter.inc(method="GET", status="500")

        assert counter.get(method="GET", status="200") == 1
        assert counter.get(method="POST", status="200") == 1
        assert counter.get(method="GET", status="500") == 1
        assert counter.get(method="DELETE", status="200") == 0

    def test_counter_collect(self):
        """Counter should collect all values."""
        counter = Counter("test_counter", "Test counter", labels=["status"])
        counter.inc(status="200")
        counter.inc(status="200")
        counter.inc(status="500")

        collected = counter.collect()
        assert collected[("200",)] == 2
        assert collected[("500",)] == 1


class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_records_observations(self):
        """Histogram should record observations."""
        hist = Histogram("test_hist", "Test histogram")
        hist.observe(0.5)
        hist.observe(1.5)

        assert hist.get_count() == 2
        assert hist.get_sum() == 2.0

    def test_histogram_with_labels(self):
        """Histogram should support labels."""
        hist = Histogram("test_hist", "Test histogram", labels=["method"])
        hist.observe(0.5, method="GET")
        hist.observe(1.5, method="POST")

        assert hist.get_count(method="GET") == 1
        assert hist.get_count(method="POST") == 1

    def test_histogram_time_context_manager(self):
        """Histogram time() should measure duration."""
        hist = Histogram("test_hist", "Test histogram")

        with hist.time():
            time.sleep(0.05)  # Increased from 0.01 for timing reliability

        assert hist.get_count() == 1
        assert hist.get_sum() >= 0.04  # Allow some timing variance


class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_set(self):
        """Gauge should set value."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(42)
        assert gauge.get() == 42

    def test_gauge_inc_dec(self):
        """Gauge should increment and decrement."""
        gauge = Gauge("test_gauge", "Test gauge")
        gauge.set(10)
        gauge.inc(5)
        assert gauge.get() == 15
        gauge.dec(3)
        assert gauge.get() == 12

    def test_gauge_with_labels(self):
        """Gauge should support labels."""
        gauge = Gauge("test_gauge", "Test gauge", labels=["service"])
        gauge.set(100, service="safety")
        gauge.set(200, service="memory")

        assert gauge.get(service="safety") == 100
        assert gauge.get(service="memory") == 200


class TestLHMetrics:
    """Tests for LHMetrics registry."""

    def test_metrics_has_expected_counters(self):
        """LHMetrics should have expected counter metrics."""
        m = LHMetrics()
        assert m.requests_total is not None
        assert m.safety_checks_total is not None
        assert m.llm_calls_total is not None
        assert m.errors_total is not None

    def test_metrics_has_expected_histograms(self):
        """LHMetrics should have expected histogram metrics."""
        m = LHMetrics()
        assert m.request_duration is not None
        assert m.plan_steps is not None
        assert m.llm_call_duration is not None

    def test_metrics_prometheus_format(self):
        """LHMetrics should export Prometheus format."""
        m = LHMetrics()
        m.requests_total.inc(method="Plan", status="success")

        output = m.to_prometheus_format()
        assert "lh_requests_total" in output
        assert "HELP" in output
        assert "TYPE" in output


# ---------------------------------------------------------------------------
# Decorator Tests
# ---------------------------------------------------------------------------


class TestTrackRequestDecorator:
    """Tests for track_request decorator."""

    def test_track_request_records_success(self):
        """track_request should record successful requests."""
        test_metrics = LHMetrics()

        @track_request("TestMethod")
        def successful_func():
            return "success"

        # Patch global metrics
        import agi.lh.observability as obs

        original_metrics = obs.metrics
        obs.metrics = test_metrics

        try:
            result = successful_func()
            assert result == "success"
            assert (
                test_metrics.requests_total.get(method="TestMethod", status="success")
                == 1
            )
        finally:
            obs.metrics = original_metrics

    def test_track_request_records_error(self):
        """track_request should record failed requests."""
        test_metrics = LHMetrics()

        @track_request("TestMethod")
        def failing_func():
            raise ValueError("test error")

        import agi.lh.observability as obs

        original_metrics = obs.metrics
        obs.metrics = test_metrics

        try:
            with pytest.raises(ValueError):
                failing_func()

            assert (
                test_metrics.requests_total.get(method="TestMethod", status="error")
                == 1
            )
            assert test_metrics.errors_total.get(error_type="ValueError") == 1
        finally:
            obs.metrics = original_metrics


class TestTrackLLMCallDecorator:
    """Tests for track_llm_call decorator."""

    def test_track_llm_call_records_metrics(self):
        """track_llm_call should record LLM call metrics."""
        test_metrics = LHMetrics()

        class MockResponse:
            prompt_tokens = 100
            completion_tokens = 50

        @track_llm_call("openai")
        def llm_call():
            return MockResponse()

        import agi.lh.observability as obs

        original_metrics = obs.metrics
        obs.metrics = test_metrics

        try:
            llm_call()  # Result not needed, just checking metrics
            assert (
                test_metrics.llm_calls_total.get(provider="openai", status="success")
                == 1
            )
            assert (
                test_metrics.llm_tokens_total.get(provider="openai", type="prompt")
                == 100
            )
            assert (
                test_metrics.llm_tokens_total.get(provider="openai", type="completion")
                == 50
            )
        finally:
            obs.metrics = original_metrics
