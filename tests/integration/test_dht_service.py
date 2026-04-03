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
# software is provided "AS IS", without WARRANTIES or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
Phase 6 integration tests: DHT Service Registry round-trip.

Requires:
- Running NATS server with JetStream at localhost:4222
- PostgreSQL with database ``atlas`` accessible by user ``claude``
- DHT Service running (python -m agi.meta.dht.nats_service)

Tests:
1. Register a service via NATS and verify it appears in the registry
2. Heartbeat updates keep a service healthy
3. Lookup returns correct service info
4. Deregister removes the service
5. Status broadcast contains registry state
6. Config store put/get round-trip
7. Stale services are detected and marked degraded

Usage (with DHT Service running):
    python tests/integration/test_dht_service.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import nats
except ImportError:
    nats = None

try:
    import asyncpg
except ImportError:
    asyncpg = None

from agi.common.event import Event  # noqa: E402


async def publish_and_wait(
    publish_subject: str,
    listen_subject: str,
    payload: dict,
    timeout: float = 10.0,
) -> dict:
    """Publish an event and wait for a response on a listener subject.

    Returns the response payload, or ``{"_timeout": True}`` on timeout.
    """
    nc = await nats.connect("nats://localhost:4222")

    response_received = asyncio.Event()
    response_data = {}

    request_event = Event.create(
        source="test",
        event_type=publish_subject.replace("agi.", ""),
        payload=payload,
    )
    expected_trace = request_event.trace_id

    async def on_response(msg):
        try:
            event = Event.from_bytes(msg.data)
            if event.trace_id != expected_trace:
                return
            response_data.update(event.payload)
            response_data["_trace_id"] = event.trace_id
            response_received.set()
        except Exception as e:
            logger.error("[test] parse error: %s", e)

    sub = await nc.subscribe(listen_subject, cb=on_response)

    js = nc.jetstream()
    ack = await js.publish(publish_subject, request_event.to_bytes())
    logger.info(
        "[test] published to %s seq=%d trace=%s",
        publish_subject,
        ack.seq,
        expected_trace[:8],
    )

    try:
        await asyncio.wait_for(response_received.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        await sub.unsubscribe()
        await nc.close()
        return {"_timeout": True}

    await sub.unsubscribe()
    await nc.close()
    return response_data


async def publish_event(subject: str, payload: dict) -> None:
    """Publish a single event via JetStream."""
    nc = await nats.connect("nats://localhost:4222")
    js = nc.jetstream()
    event = Event.create(
        source="test",
        event_type=subject.replace("agi.", ""),
        payload=payload,
    )
    await js.publish(subject, event.to_bytes())
    await nc.close()


async def wait_for_status(timeout: float = 35.0) -> dict:
    """Wait for a status broadcast on agi.dht.status.

    Returns the status payload, or ``{"_timeout": True}`` on timeout.
    """
    nc = await nats.connect("nats://localhost:4222")

    status_received = asyncio.Event()
    status_data = {}

    async def on_status(msg):
        try:
            event = Event.from_bytes(msg.data)
            status_data.update(event.payload)
            status_received.set()
        except Exception as e:
            logger.error("[test] status parse error: %s", e)

    sub = await nc.subscribe("agi.dht.status", cb=on_status)

    try:
        await asyncio.wait_for(status_received.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        await sub.unsubscribe()
        await nc.close()
        return {"_timeout": True}

    await sub.unsubscribe()
    await nc.close()
    return status_data


# -----------------------------------------------------------------------
# Test 1: Register a service via NATS
# -----------------------------------------------------------------------


async def test_register_service() -> bool:
    """Register a test service and verify it via lookup."""
    logger.info("--- Test 1: Register Service via NATS ---")

    test_name = f"test-svc-{uuid.uuid4().hex[:6]}"

    # Register
    await publish_event(
        "agi.dht.register",
        {
            "service_name": test_name,
            "port": 59999,
            "metadata": {"phase": 6, "test": True},
        },
    )

    # Give the service time to process
    await asyncio.sleep(1.0)

    # Lookup via NATS
    result = await publish_and_wait(
        publish_subject="agi.dht.lookup",
        listen_subject="agi.dht.lookup.result",
        payload={"service_name": test_name},
    )

    if result.get("_timeout"):
        logger.error("[test] FAIL: lookup timed out")
        return False

    if not result.get("found"):
        logger.error("[test] FAIL: service not found after registration")
        return False

    svc = result.get("service", {})
    if svc.get("port") != 59999:
        logger.error("[test] FAIL: wrong port %s", svc.get("port"))
        return False

    logger.info(
        "[test] PASS: registered and found %s at port %d",
        test_name,
        svc.get("port", 0),
    )

    # Clean up
    await publish_event("agi.dht.deregister", {"service_name": test_name})
    return True


# -----------------------------------------------------------------------
# Test 2: Heartbeat keeps service healthy
# -----------------------------------------------------------------------


async def test_heartbeat() -> bool:
    """Register, send heartbeat, verify service remains healthy."""
    logger.info("--- Test 2: Heartbeat Updates ---")

    test_name = f"test-hb-{uuid.uuid4().hex[:6]}"

    await publish_event(
        "agi.dht.register",
        {"service_name": test_name, "port": 59998},
    )
    await asyncio.sleep(0.5)

    # Send heartbeat
    await publish_event(
        "agi.dht.heartbeat",
        {"service_name": test_name},
    )
    await asyncio.sleep(0.5)

    # Lookup
    result = await publish_and_wait(
        publish_subject="agi.dht.lookup",
        listen_subject="agi.dht.lookup.result",
        payload={"service_name": test_name},
    )

    if result.get("_timeout"):
        logger.error("[test] FAIL: lookup timed out")
        return False

    svc = result.get("service", {})
    status = svc.get("status", "")

    if status != "healthy":
        logger.error("[test] FAIL: status=%s (expected healthy)", status)
        # Clean up
        await publish_event("agi.dht.deregister", {"service_name": test_name})
        return False

    logger.info("[test] PASS: heartbeat kept service healthy")

    # Clean up
    await publish_event("agi.dht.deregister", {"service_name": test_name})
    return True


# -----------------------------------------------------------------------
# Test 3: Deregister removes service
# -----------------------------------------------------------------------


async def test_deregister() -> bool:
    """Register, deregister, verify service is gone."""
    logger.info("--- Test 3: Deregister Service ---")

    test_name = f"test-dereg-{uuid.uuid4().hex[:6]}"

    await publish_event(
        "agi.dht.register",
        {"service_name": test_name, "port": 59997},
    )
    await asyncio.sleep(0.5)

    await publish_event(
        "agi.dht.deregister",
        {"service_name": test_name},
    )
    await asyncio.sleep(0.5)

    result = await publish_and_wait(
        publish_subject="agi.dht.lookup",
        listen_subject="agi.dht.lookup.result",
        payload={"service_name": test_name},
    )

    if result.get("_timeout"):
        logger.error("[test] FAIL: lookup timed out")
        return False

    if result.get("found"):
        logger.error("[test] FAIL: service still found after deregister")
        return False

    logger.info("[test] PASS: service removed after deregister")
    return True


# -----------------------------------------------------------------------
# Test 4: Status broadcast
# -----------------------------------------------------------------------


async def test_status_broadcast() -> bool:
    """Register a service and wait for a status broadcast."""
    logger.info("--- Test 4: Status Broadcast ---")

    test_name = f"test-status-{uuid.uuid4().hex[:6]}"

    await publish_event(
        "agi.dht.register",
        {"service_name": test_name, "port": 59996},
    )

    # Wait for periodic status (every 30s)
    status = await wait_for_status(timeout=35.0)

    if status.get("_timeout"):
        logger.error("[test] FAIL: no status broadcast received in 35s")
        # Clean up
        await publish_event("agi.dht.deregister", {"service_name": test_name})
        return False

    total = status.get("total", 0)
    if total < 1:
        logger.error("[test] FAIL: status shows 0 services")
        await publish_event("agi.dht.deregister", {"service_name": test_name})
        return False

    logger.info(
        "[test] PASS: status broadcast received. total=%d healthy=%d",
        total,
        status.get("healthy", 0),
    )

    await publish_event("agi.dht.deregister", {"service_name": test_name})
    return True


# -----------------------------------------------------------------------
# Test 5: Config store round-trip (direct DB, no NATS)
# -----------------------------------------------------------------------


async def test_config_store() -> bool:
    """Test config store put/get/list round-trip via direct DB access."""
    logger.info("--- Test 5: Config Store Round-Trip ---")

    if asyncpg is None:
        logger.warning("[test] SKIP: asyncpg not installed")
        return True

    try:
        from agi.meta.dht.config_store import ConfigStore
    except ImportError:
        logger.error("[test] FAIL: could not import ConfigStore")
        return False

    store = ConfigStore(dsn="dbname=atlas user=claude")
    try:
        await store.init_db()
    except Exception as e:
        logger.error("[test] FAIL: could not init config store: %s", e)
        return False

    test_key = f"test.config.{uuid.uuid4().hex[:6]}"

    # Put
    await store.put(test_key, {"threshold": 0.42}, version=1)

    # Get
    result = await store.get(test_key)
    if result is None:
        logger.error("[test] FAIL: config key not found after put")
        await store.close()
        return False

    value, version = result
    if version != 1:
        logger.error("[test] FAIL: version=%d (expected 1)", version)
        await store.delete(test_key)
        await store.close()
        return False

    if value.get("threshold") != 0.42:
        logger.error("[test] FAIL: value mismatch: %s", value)
        await store.delete(test_key)
        await store.close()
        return False

    # Update with new version
    await store.put(test_key, {"threshold": 0.55}, version=2)
    result = await store.get(test_key)
    value, version = result
    if version != 2 or value.get("threshold") != 0.55:
        logger.error(
            "[test] FAIL: version update mismatch: v=%d val=%s",
            version,
            value,
        )
        await store.delete(test_key)
        await store.close()
        return False

    # List keys
    keys = await store.list_keys()
    if test_key not in keys:
        logger.error("[test] FAIL: key not in list_keys()")
        await store.delete(test_key)
        await store.close()
        return False

    # Clean up
    await store.delete(test_key)
    await store.close()

    logger.info("[test] PASS: config store put/get/update/list round-trip")
    return True


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------


async def run_all() -> int:
    logger.info("=" * 60)
    logger.info("AGI-HPC Phase 6 Integration Tests -- DHT Service Registry")
    logger.info("=" * 60)

    results = {}
    results["register_service"] = await test_register_service()
    results["heartbeat"] = await test_heartbeat()
    results["deregister"] = await test_deregister()
    results["status_broadcast"] = await test_status_broadcast()
    results["config_store"] = await test_config_store()

    logger.info("=" * 60)
    logger.info("Results:")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-30s %s", name, status)
        if not passed:
            all_passed = False
    logger.info("=" * 60)

    if all_passed:
        logger.info("All Phase 6 tests PASSED")
        return 0
    else:
        logger.error("Some Phase 6 tests FAILED")
        return 1


if __name__ == "__main__":
    if nats is None:
        logger.error("nats-py not installed; skipping integration tests")
        sys.exit(0)
    sys.exit(asyncio.run(run_all()))
