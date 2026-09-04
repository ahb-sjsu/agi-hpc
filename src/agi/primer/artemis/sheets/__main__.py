# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Entry point for ``python -m agi.primer.artemis.sheets``.

Wires the poller built from environment to a NATS publisher, with a
SIGTERM handler for graceful shutdown under systemd.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from .poller import build_poller_from_env

log = logging.getLogger("artemis.sheets.main")


async def _run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    poller = build_poller_from_env()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _stop() -> None:
        log.info("signal received, stopping poller")
        poller.stop()
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            # Windows dev doesn't support add_signal_handler; tests
            # exit via poller.stop() directly.
            pass

    await poller.run_forever()
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
