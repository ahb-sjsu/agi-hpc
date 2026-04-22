# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS service entry point.

Usage::

    python -m agi.primer.artemis
    python -m agi.primer.artemis --nats nats://localhost:4222

Reads configuration from environment (see
:func:`agi.primer.artemis.nats_handler.build_service_from_env`).
CLI flags take precedence where provided.

Run under systemd via ``atlas-artemis.service``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from .nats_handler import build_service_from_env

logger = logging.getLogger("artemis.main")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARTEMIS NATS service")
    p.add_argument(
        "--nats",
        default=None,
        help="NATS URL (default: $NATS_URL or nats://localhost:4222)",
    )
    p.add_argument(
        "--keeper-gate",
        choices=("on", "off"),
        default=None,
        help=(
            "Override keeper approval gate "
            "(default: $ARTEMIS_KEEPER_APPROVAL_REQUIRED)"
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    # Allow CLI overrides to shadow env.
    if args.nats:
        os.environ["NATS_URL"] = args.nats
    if args.keeper_gate is not None:
        os.environ["ARTEMIS_KEEPER_APPROVAL_REQUIRED"] = (
            "true" if args.keeper_gate == "on" else "false"
        )

    service = build_service_from_env()
    await service.start()

    # Signal handling — asyncio loop.add_signal_handler isn't supported
    # on Windows, so we catch NotImplementedError and fall through; on
    # Windows the operator stops the service via Ctrl-C which raises
    # KeyboardInterrupt in asyncio.run.
    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        logger.info("shutdown signal received")
        # Schedule service.stop; can't await here (sync callback).
        asyncio.ensure_future(service.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except (NotImplementedError, RuntimeError):
            pass  # Windows / non-main-thread contexts

    try:
        await service.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        await service.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
