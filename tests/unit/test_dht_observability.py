# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.core.dht.observability module."""

import time
import pytest

from agi.core.dht.observability import (
    SpanContext,
    DHTTracer,
    DHTMetrics,
    track_dht_operation,
)


class TestSpanContext:
    def test_create(self):
        span = SpanContext(
            trace_id="abc123",
            span_id="def456",
            operation="put",
            start_time=time.monotonic(),
        )
        assert span.trace_id == "abc123"
        assert span.operation == "put"
        assert span.status == "ok"
        assert span.end_time is None

    def test_duration_none_before_end(self):
        span = SpanContext(
            trace_id="t",
            span_id="s",
            operation="get",
            start_time=time.monotonic(),
        )
        assert span.duration_ms is None

    def test_duration_after_end(self):
        start = time.monotonic()
        span = SpanContext(
            trace_id="t",
            span_id="s",
            operation="get",
            start_time=start,
        )
        span.end_time = start + 0.1
        assert span.duration_ms == pytest.approx(100.0, abs=1.0)

    def test_attributes(self):
        span = SpanContext(
            trace_id="t",
            span_id="s",
            operation="put",
            start_time=time.monotonic(),
            attributes={"key": "val"},
        )
        assert span.attributes["key"] == "val"


class TestDHTTracer:
    def test_init(self):
        tracer = DHTTracer(service_name="test")
        assert tracer._service_name == "test"

    def test_start_span(self):
        tracer = DHTTracer()
        span = tracer.start_span("put", key="mykey")
        assert span.operation == "put"
        assert "service" in span.attributes
        assert span.attributes.get("key") == "mykey"
        assert span.end_time is None

    def test_end_span(self):
        tracer = DHTTracer()
        span = tracer.start_span("get")
        tracer.end_span(span, status="ok")
        assert span.end_time is not None
        assert span.status == "ok"
        assert span.duration_ms is not None

    def test_end_span_error(self):
        tracer = DHTTracer()
        span = tracer.start_span("delete")
        tracer.end_span(span, status="error")
        assert span.status == "error"

    def test_trace_context_manager(self):
        tracer = DHTTracer()
        with tracer.trace("put", key="k1") as span:
            assert span.operation == "put"
        assert span.end_time is not None
        assert span.status == "ok"

    def test_trace_context_manager_error(self):
        tracer = DHTTracer()
        with pytest.raises(ValueError):
            with tracer.trace("put") as span:
                raise ValueError("test error")
        assert span.status == "error"

    def test_get_traces(self):
        tracer = DHTTracer()
        for i in range(5):
            with tracer.trace(f"op_{i}"):
                pass
        traces = tracer.get_traces(limit=3)
        assert len(traces) == 3


class TestDHTMetrics:
    def test_init(self):
        m = DHTMetrics()
        assert m.dht_operations_total is not None
        assert m.dht_operation_duration_seconds is not None
        assert m.dht_keys_total is not None
        assert m.dht_nodes_total is not None
        assert m.dht_errors_total is not None

    def test_counter_inc(self):
        m = DHTMetrics()
        m.dht_operations_total.inc(operation="put", status="success")
        collected = m.dht_operations_total.collect()
        assert len(collected) > 0

    def test_prometheus_format(self):
        m = DHTMetrics()
        m.dht_operations_total.inc(operation="get", status="success")
        output = m.to_prometheus_format()
        assert "dht_operations_total" in output


class TestTrackDHTOperation:
    def test_sync_decorator(self):
        @track_dht_operation("test_op")
        def my_func():
            return 42

        result = my_func()
        assert result == 42

    def test_sync_decorator_error(self):
        @track_dht_operation("test_err")
        def my_func():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            my_func()
