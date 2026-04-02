# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Integration test for Phase 1 LH NATS service.

Sends a chat request through NATS, verifies the LH service
processes it, and returns a response through the LLM.

Usage (with LH service running):
    python -m pytest tests/integration/test_lh_nats.py -v

Or standalone:
    python tests/integration/test_lh_nats.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import nats
except ImportError:
    nats = None

from agi.common.event import Event


async def test_lh_chat_round_trip() -> bool:
    """Send a chat request to LH and verify we get a response.

    Returns True if the test passes.
    """
    if nats is None:
        logger.error("nats-py not installed, skipping integration test")
        return False

    nc = await nats.connect("nats://localhost:4222")
    logger.info("[test] connected to NATS")

    # Set up response listener
    response_received = asyncio.Event()
    response_data = {}

    async def on_response(msg):
        try:
            event = Event.from_bytes(msg.data)
            response_data.update(event.payload)
            response_data["_trace_id"] = event.trace_id
            response_received.set()
            logger.info("[test] received response: %s", json.dumps(event.payload, indent=2)[:500])
        except Exception as e:
            logger.error("[test] failed to parse response: %s", e)

    # Subscribe to LH response
    sub = await nc.subscribe("agi.lh.response.chat", cb=on_response)

    # Also subscribe to CoT traces
    cot_data = {}

    async def on_cot(msg):
        try:
            event = Event.from_bytes(msg.data)
            cot_data.update(event.payload)
            logger.info("[test] received CoT trace: hemisphere=%s, latency=%.0fms",
                       event.payload.get("hemisphere_decision", "?"),
                       event.payload.get("latency_ms", 0))
        except Exception as e:
            logger.error("[test] failed to parse CoT: %s", e)

    cot_sub = await nc.subscribe("agi.lh.internal.cot", cb=on_cot)

    # Create and publish a chat request
    request_event = Event.create(
        source="test",
        event_type="lh.request.chat",
        payload={
            "prompt": "Explain the concept of distributed consensus in 2-3 sentences.",
        },
    )

    # Publish via JetStream
    js = nc.jetstream()
    ack = await js.publish("agi.lh.request.chat", request_event.to_bytes())
    logger.info("[test] published request, stream seq=%d, trace=%s",
                ack.seq, request_event.trace_id[:8])

    # Wait for response (timeout 120s for LLM generation)
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(response_received.wait(), timeout=120.0)
    except asyncio.TimeoutError:
        logger.error("[test] TIMEOUT waiting for LH response after 120s")
        await nc.close()
        return False

    elapsed = time.perf_counter() - t0
    logger.info("[test] response received in %.1fs", elapsed)

    # Validate response
    passed = True

    if response_data.get("error"):
        logger.error("[test] FAIL: LH returned error: %s", response_data.get("message"))
        passed = False
    elif not response_data.get("text"):
        logger.error("[test] FAIL: empty response text")
        passed = False
    else:
        text = response_data["text"]
        logger.info("[test] PASS: got response (%d chars, %d tokens, %.0fms)",
                    len(text), response_data.get("tokens_used", 0),
                    response_data.get("latency_ms", 0))
        logger.info("[test] response preview: %s", text[:200])

    if cot_data:
        logger.info("[test] PASS: CoT trace received (hemisphere=%s)",
                    cot_data.get("hemisphere_decision", "?"))
    else:
        logger.warning("[test] WARN: no CoT trace received (may arrive later)")

    await sub.unsubscribe()
    await cot_sub.unsubscribe()
    await nc.close()
    return passed


async def test_lh_reason_request() -> bool:
    """Send a reason request and verify response."""
    if nats is None:
        return False

    nc = await nats.connect("nats://localhost:4222")
    response_received = asyncio.Event()
    response_data = {}

    async def on_response(msg):
        try:
            event = Event.from_bytes(msg.data)
            response_data.update(event.payload)
            response_received.set()
        except Exception:
            pass

    sub = await nc.subscribe("agi.lh.response.reason", cb=on_response)

    request_event = Event.create(
        source="test",
        event_type="lh.request.reason",
        payload={
            "prompt": "What are the trade-offs between Raft and Paxos for consensus?",
            "config": {"temperature": 0.2, "max_tokens": 512},
        },
    )

    js = nc.jetstream()
    await js.publish("agi.lh.request.reason", request_event.to_bytes())
    logger.info("[test] published reason request")

    try:
        await asyncio.wait_for(response_received.wait(), timeout=120.0)
    except asyncio.TimeoutError:
        logger.error("[test] TIMEOUT on reason request")
        await nc.close()
        return False

    passed = not response_data.get("error") and bool(response_data.get("text"))
    if passed:
        logger.info("[test] PASS: reason response (%d chars)", len(response_data["text"]))
    else:
        logger.error("[test] FAIL: reason request failed")

    await sub.unsubscribe()
    await nc.close()
    return passed


async def test_telemetry_published() -> bool:
    """Verify that telemetry events are published after a request."""
    if nats is None:
        return False

    nc = await nats.connect("nats://localhost:4222")
    telemetry_received = asyncio.Event()
    telemetry_data = {}

    async def on_telemetry(msg):
        try:
            event = Event.from_bytes(msg.data)
            telemetry_data.update(event.payload)
            telemetry_received.set()
        except Exception:
            pass

    sub = await nc.subscribe("agi.meta.monitor.lh", cb=on_telemetry)

    # Send a request to trigger telemetry
    request_event = Event.create(
        source="test",
        event_type="lh.request.chat",
        payload={"prompt": "Hello, this is a telemetry test."},
    )
    js = nc.jetstream()
    await js.publish("agi.lh.request.chat", request_event.to_bytes())

    try:
        await asyncio.wait_for(telemetry_received.wait(), timeout=120.0)
    except asyncio.TimeoutError:
        logger.error("[test] TIMEOUT waiting for telemetry")
        await nc.close()
        return False

    passed = "requests_processed" in telemetry_data and "avg_latency_ms" in telemetry_data
    if passed:
        logger.info(
            "[test] PASS: telemetry received -- requests=%d avg_latency=%.0fms",
            telemetry_data.get("requests_processed", 0),
            telemetry_data.get("avg_latency_ms", 0),
        )
    else:
        logger.error("[test] FAIL: telemetry missing expected fields")

    await sub.unsubscribe()
    await nc.close()
    return passed


async def run_all_tests() -> int:
    """Run all integration tests and return exit code."""
    results = {}

    logger.info("=" * 60)
    logger.info("AGI-HPC Phase 1 Integration Tests")
    logger.info("=" * 60)

    logger.info("\n--- Test 1: LH Chat Round-Trip ---")
    results["chat"] = await test_lh_chat_round_trip()

    logger.info("\n--- Test 2: LH Reason Request ---")
    results["reason"] = await test_lh_reason_request()

    logger.info("\n--- Test 3: Telemetry Published ---")
    results["telemetry"] = await test_telemetry_published()

    logger.info("\n" + "=" * 60)
    logger.info("Results:")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-20s %s", name, status)
        if not passed:
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("All tests PASSED")
        return 0
    else:
        logger.error("Some tests FAILED")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
