# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Dreaming NATS Service — Scheduled Memory Consolidation.

Runs as a systemd service. During idle periods, triggers dream cycles
that consolidate episodic memories into wiki articles.

Dream-generated wiki articles are Tier 0 in the RAG search cascade —
higher priority than even the hand-written wiki articles (Tier 1),
because they represent validated, consolidated knowledge from actual
interactions.

Search cascade with dreaming:
  Tier 0: Dream-consolidated articles (/wiki/dream-*)  — <1ms, highest priority
  Tier 1: Hand-written wiki articles (/wiki/*)          — <1ms
  Tier 2: PCA-384 IVFFlat vector search                 — 4.4ms
  Tier 3: tsvector FTS fallback                          — 2.9ms

NATS subjects:
    agi.dreaming.start              — Dream cycle initiated
    agi.dreaming.replay             — Episodic replay batch
    agi.dreaming.consolidate        — Article created/updated
    agi.dreaming.dream              — Creative insight generated
    agi.dreaming.complete           — Dream cycle finished
    agi.dreaming.trigger            — Manual trigger (subscribe)

Usage::

    python -m agi.dreaming.nats_service
    python -m agi.dreaming.nats_service --config configs/dreaming_config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agi.dreaming.consolidator import ConsolidatorConfig, MemoryConsolidator

# ---------------------------------------------------------------------------
# NATS integration
# ---------------------------------------------------------------------------


async def _publish(nc, subject: str, data: dict) -> None:
    """Publish a JSON message to NATS."""
    import json

    payload = json.dumps(data, default=str).encode()
    await nc.publish(subject, payload)


async def _connect_nats(servers: list[str]):
    """Connect to NATS with retry."""
    import nats

    for attempt in range(10):
        try:
            nc = await nats.connect(servers=servers)
            logger.info("[dream-svc] Connected to NATS: %s", servers)
            return nc
        except Exception as e:
            logger.warning(
                "[dream-svc] NATS connect attempt %d failed: %s", attempt + 1, e
            )
            await asyncio.sleep(5)
    raise RuntimeError("Failed to connect to NATS after 10 attempts")


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def _should_dream(
    last_dream: float,
    idle_trigger_seconds: int = 3600,
    scheduled_hours: list[int] | None = None,
) -> tuple[bool, str]:
    """Check if it's time to dream."""
    now = datetime.now(timezone.utc)

    # Check scheduled hours (default: 2 AM and 2 PM UTC)
    if scheduled_hours is None:
        scheduled_hours = [2, 14]
    if now.hour in scheduled_hours and now.minute < 5:
        elapsed = time.time() - last_dream
        if elapsed > 3600:  # at least 1 hour since last dream
            return True, f"scheduled (hour={now.hour})"

    # Check idle trigger
    elapsed = time.time() - last_dream
    if elapsed > idle_trigger_seconds:
        return True, f"idle ({elapsed/3600:.1f}h since last)"

    return False, ""


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


async def run_service(config: ConsolidatorConfig, nats_servers: list[str]) -> None:
    """Main service loop."""
    nc = await _connect_nats(nats_servers)
    consolidator = MemoryConsolidator(config)

    last_dream = 0.0
    running = True

    # Handle manual trigger via NATS
    async def on_trigger(msg):
        nonlocal last_dream
        logger.info("[dream-svc] Manual trigger received")
        await _run_dream_cycle(nc, consolidator)
        last_dream = time.time()

    await nc.subscribe("agi.dreaming.trigger", cb=on_trigger)

    # Handle shutdown
    def shutdown_handler():
        nonlocal running
        running = False

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown_handler)
        except NotImplementedError:
            pass  # Windows

    logger.info("[dream-svc] Dreaming service started — waiting for idle periods")
    await _publish(
        nc,
        "agi.dreaming.start",
        {
            "event": "service_started",
            "wiki_dir": config.wiki_dir,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    while running:
        should, reason = _should_dream(last_dream)
        if should:
            logger.info("[dream-svc] Triggering dream cycle: %s", reason)
            await _run_dream_cycle(nc, consolidator)
            last_dream = time.time()

        # Check every 60 seconds
        for _ in range(60):
            if not running:
                break
            await asyncio.sleep(1)

    await nc.close()
    logger.info("[dream-svc] Dreaming service stopped")


async def _run_dream_cycle(nc, consolidator: MemoryConsolidator) -> None:
    """Execute one dream cycle and publish events."""
    await _publish(
        nc,
        "agi.dreaming.start",
        {
            "event": "cycle_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    try:
        result = await consolidator.run_cycle()

        # Publish per-article events
        # (the consolidator logs them, but we also publish to NATS)
        await _publish(
            nc,
            "agi.dreaming.complete",
            {
                "event": "cycle_complete",
                "episodes_processed": result.episodes_processed,
                "clusters_found": result.clusters_found,
                "articles_created": result.articles_created,
                "articles_updated": result.articles_updated,
                "dream_insights": result.dream_insights,
                "duration_seconds": round(result.duration_seconds, 1),
                "errors": result.errors,
                "timestamp": result.timestamp.isoformat(),
            },
        )

    except Exception as e:
        logger.error("[dream-svc] Dream cycle failed: %s", e, exc_info=True)
        await _publish(
            nc,
            "agi.dreaming.complete",
            {
                "event": "cycle_failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Atlas Dreaming Service")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--wiki-dir", default="/home/claude/agi-hpc/wiki")
    parser.add_argument(
        "--llm-url", default="http://localhost:8082"
    )  # Id (fast, 25.9 tok/s)
    parser.add_argument("--nats", default="nats://localhost:4222")
    parser.add_argument("--idle-trigger", type=int, default=3600)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ConsolidatorConfig(
        wiki_dir=args.wiki_dir,
        llm_url=args.llm_url,
    )

    if args.config and yaml:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(config, k):
                setattr(config, k, v)

    nats_servers = [args.nats]
    asyncio.run(run_service(config, nats_servers))


if __name__ == "__main__":
    main()
