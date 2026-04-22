# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS LiveKit agent — worker entry point.

Usage::

    # Atlas systemd unit (preferred) — systemd provides env.
    python -m agi.primer.artemis.livekit_agent

    # Or with an explicit session id:
    ARTEMIS_SESSION_ID=halyard-s01 python -m agi.primer.artemis.livekit_agent

Environment:
    ARTEMIS_SESSION_ID   required — LiveKit room name == session id
    LIVEKIT_URL          required — wss://livekit.example.com
    LIVEKIT_API_KEY      required
    LIVEKIT_API_SECRET   required
    NATS_URL             default: nats://localhost:4222
    ARTEMIS_WHISPER_MODEL default: large-v3
    ARTEMIS_AGENT_IDENTITY default: artemis
    ARTEMIS_DISPLAY_NAME default: ARTEMIS

Signals: SIGTERM/SIGINT cleanly stop the agent (systemd-friendly).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from .agent import AgentConfig, ArtemisLiveKitAgent

logger = logging.getLogger("artemis.livekit.main")


async def _run() -> int:
    config = AgentConfig.from_env()
    agent = ArtemisLiveKitAgent(config)

    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        logger.info("shutdown signal received")
        asyncio.ensure_future(agent.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, RuntimeError):
            pass  # Windows / non-main-thread

    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
