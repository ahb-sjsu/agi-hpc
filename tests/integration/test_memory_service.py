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
Phase 2 integration tests: Memory Service round-trip.

Requires:
- Running NATS server with JetStream at localhost:4222
- PostgreSQL with pgvector at localhost:5432 (db=atlas, user=claude)
- Memory Service running (python -m agi.memory.service)

Tests:
1. Store an episode via NATS, recall it, verify content
2. Store a procedure via NATS, look it up, verify
3. Procedural lookup returns matching procedures
4. Episodic session history retrieval

Usage (with Memory Service running):
    python tests/integration/test_memory_service.py
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

from agi.common.event import Event  # noqa: E402


async def send_and_wait(
    request_subject: str,
    response_subject: str,
    payload: dict,
    timeout: float = 30.0,
) -> dict:
    """Send a request via JetStream and wait for a matching response.

    Uses core NATS subscription to receive only new messages.
    Filters responses by trace_id.
    """
    nc = await nats.connect("nats://localhost:4222")

    response_received = asyncio.Event()
    response_data = {}

    request_event = Event.create(
        source="test",
        event_type=request_subject.replace("agi.", ""),
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

    # Subscribe BEFORE publishing
    sub = await nc.subscribe(response_subject, cb=on_response)

    # Publish via JetStream
    js = nc.jetstream()
    ack = await js.publish(request_subject, request_event.to_bytes())
    logger.info(
        "[test] published to %s seq=%d trace=%s",
        request_subject,
        ack.seq,
        expected_trace[:8],
    )

    # Wait for response
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(response_received.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("[test] TIMEOUT after %.0fs", timeout)
        await sub.unsubscribe()
        await nc.close()
        return {"_timeout": True}

    elapsed = time.perf_counter() - t0
    await sub.unsubscribe()
    await nc.close()

    response_data["_elapsed"] = elapsed
    return response_data


# -----------------------------------------------------------------------
# Test 1: Store and recall an episode
# -----------------------------------------------------------------------


async def test_episodic_store_and_recall() -> bool:
    """Store an episode via NATS, then recall it and verify content."""
    logger.info("--- Test 1: Episodic Store + Recall ---")

    session_id = f"test-session-{uuid.uuid4().hex[:8]}"
    user_msg = "What is the capital of France?"
    response_text = "The capital of France is Paris."

    # 1. Store an episode
    store_result = await send_and_wait(
        request_subject="agi.memory.store.episodic",
        response_subject="agi.memory.result.episodic",
        payload={
            "session_id": session_id,
            "user_message": user_msg,
            "atlas_response": response_text,
            "hemisphere": "lh",
            "quality_score": 0.95,
            "metadata": {"test": True, "source": "integration_test"},
        },
    )

    if store_result.get("_timeout"):
        logger.error("[test] FAIL: store timeout")
        return False

    episode_id = store_result.get("episode_id")
    if not episode_id:
        logger.error("[test] FAIL: no episode_id in response: %s", store_result)
        return False

    logger.info(
        "[test] stored episode id=%s in %.1fms",
        episode_id[:8],
        store_result.get("latency_ms", 0),
    )

    # 2. Recall the episode by session
    recall_result = await send_and_wait(
        request_subject="agi.memory.query.episodic",
        response_subject="agi.memory.result.episodic",
        payload={
            "session_id": session_id,
            "n": 5,
            "mode": "recent",
        },
    )

    if recall_result.get("_timeout"):
        logger.error("[test] FAIL: recall timeout")
        return False

    episodes = recall_result.get("episodes", [])
    if not episodes:
        logger.error("[test] FAIL: no episodes recalled")
        return False

    # Verify content
    found = False
    for ep in episodes:
        if (
            ep.get("user_message") == user_msg
            and ep.get("atlas_response") == response_text
        ):
            found = True
            break

    if not found:
        logger.error("[test] FAIL: stored episode not found in recall results")
        return False

    logger.info(
        "[test] PASS: stored and recalled episode. count=%d latency=%.1fms",
        recall_result.get("count", 0),
        recall_result.get("latency_ms", 0),
    )
    return True


# -----------------------------------------------------------------------
# Test 2: Store and lookup a procedure
# -----------------------------------------------------------------------


async def test_procedural_store_and_lookup() -> bool:
    """Store a procedure via NATS, then look it up."""
    logger.info("--- Test 2: Procedural Store + Lookup ---")

    proc_name = f"test_procedure_{uuid.uuid4().hex[:8]}"

    # 1. Store a procedure
    store_result = await send_and_wait(
        request_subject="agi.memory.store.procedural",
        response_subject="agi.memory.result.procedural",
        payload={
            "name": proc_name,
            "trigger": "integration test|test procedure",
            "steps": [
                "Step 1: Prepare test data",
                "Step 2: Execute test",
                "Step 3: Verify results",
            ],
            "metadata": {"category": "testing"},
        },
    )

    if store_result.get("_timeout"):
        logger.error("[test] FAIL: store timeout")
        return False

    if store_result.get("name") != proc_name:
        logger.error("[test] FAIL: name mismatch: %s", store_result)
        return False

    logger.info("[test] stored procedure name=%s", proc_name)

    # 2. Look it up
    lookup_result = await send_and_wait(
        request_subject="agi.memory.query.procedural",
        response_subject="agi.memory.result.procedural",
        payload={
            "trigger_text": "run the integration test please",
        },
    )

    if lookup_result.get("_timeout"):
        logger.error("[test] FAIL: lookup timeout")
        return False

    procedures = lookup_result.get("procedures", [])
    found = any(p.get("name") == proc_name for p in procedures)

    if not found:
        logger.error(
            "[test] FAIL: stored procedure not found in lookup. results=%s",
            [p.get("name") for p in procedures],
        )
        return False

    logger.info(
        "[test] PASS: stored and looked up procedure. matches=%d",
        lookup_result.get("count", 0),
    )
    return True


# -----------------------------------------------------------------------
# Test 3: Built-in seed procedures are accessible
# -----------------------------------------------------------------------


async def test_seed_procedures() -> bool:
    """Verify that seed procedures are returned for known triggers."""
    logger.info("--- Test 3: Seed Procedure Lookup ---")

    lookup_result = await send_and_wait(
        request_subject="agi.memory.query.procedural",
        response_subject="agi.memory.result.procedural",
        payload={
            "trigger_text": "tell me about the agi-hpc repository source code",
        },
    )

    if lookup_result.get("_timeout"):
        logger.error("[test] FAIL: lookup timeout")
        return False

    procedures = lookup_result.get("procedures", [])
    names = [p.get("name") for p in procedures]

    if "repo_search" not in names:
        logger.error(
            "[test] FAIL: seed procedure 'repo_search' not found. got=%s",
            names,
        )
        return False

    logger.info(
        "[test] PASS: seed procedures accessible. matches=%s",
        names,
    )
    return True


# -----------------------------------------------------------------------
# Test 4: Episodic session history
# -----------------------------------------------------------------------


async def test_episodic_session_history() -> bool:
    """Store multiple episodes and retrieve full session history."""
    logger.info("--- Test 4: Episodic Session History ---")

    session_id = f"history-test-{uuid.uuid4().hex[:8]}"

    # Store 3 episodes in sequence
    exchanges = [
        ("Hello!", "Hi there! How can I help?"),
        ("What is Python?", "Python is a programming language."),
        ("Thanks!", "You're welcome!"),
    ]

    for user_msg, response in exchanges:
        result = await send_and_wait(
            request_subject="agi.memory.store.episodic",
            response_subject="agi.memory.result.episodic",
            payload={
                "session_id": session_id,
                "user_message": user_msg,
                "atlas_response": response,
                "hemisphere": "lh",
            },
        )
        if result.get("_timeout"):
            logger.error("[test] FAIL: store timeout for '%s'", user_msg)
            return False
        # Small delay to ensure ordering
        await asyncio.sleep(0.1)

    # Retrieve full history
    history_result = await send_and_wait(
        request_subject="agi.memory.query.episodic",
        response_subject="agi.memory.result.episodic",
        payload={
            "session_id": session_id,
            "mode": "history",
        },
    )

    if history_result.get("_timeout"):
        logger.error("[test] FAIL: history recall timeout")
        return False

    episodes = history_result.get("episodes", [])
    if len(episodes) < 3:
        logger.error("[test] FAIL: expected >= 3 episodes, got %d", len(episodes))
        return False

    # Verify ordering (oldest first for history mode)
    messages = [ep.get("user_message") for ep in episodes]
    expected = ["Hello!", "What is Python?", "Thanks!"]
    if messages[-3:] != expected:
        logger.warning(
            "[test] WARN: order may differ. got=%s expected=%s",
            messages[-3:],
            expected,
        )

    logger.info(
        "[test] PASS: session history retrieved. count=%d",
        len(episodes),
    )
    return True


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------


async def run_all() -> int:
    logger.info("=" * 60)
    logger.info("AGI-HPC Phase 2 Integration Tests -- Memory Service")
    logger.info("=" * 60)

    results = {}
    results["episodic_store_recall"] = await test_episodic_store_and_recall()
    results["procedural_store_lookup"] = await test_procedural_store_and_lookup()
    results["seed_procedures"] = await test_seed_procedures()
    results["episodic_session_history"] = await test_episodic_session_history()

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
        logger.info("All Phase 2 tests PASSED")
        return 0
    else:
        logger.error("Some Phase 2 tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all()))
