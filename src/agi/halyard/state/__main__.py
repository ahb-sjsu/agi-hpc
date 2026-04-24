# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-state service entrypoint.

Runnable via::

    python -m agi.halyard.state [--host 0.0.0.0] [--port 8090]
                                [--archive-root /archive/halyard]
                                [--nats-url nats://127.0.0.1:4222]

All flags can also be set from env vars:

- ``HALYARD_STATE_HOST``      (default ``0.0.0.0``)
- ``HALYARD_STATE_PORT``      (default ``8090``)
- ``HALYARD_ARCHIVE_ROOT``    (default ``/archive/halyard``)
- ``NATS_URL``                (default unset — bridge not started)

If ``NATS_URL`` is set, the service connects to NATS and wires the
bridge so patches arriving over ``agi.rh.halyard.sheet.*.patch`` get
applied and broadcast. If it's unset, the service runs REST+WS only
and no NATS traffic is consumed. Handy for smoke tests and for
standalone dev.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

from aiohttp import web

from .api.app import WsBroadcaster, build_app
from .bridge import StateBridge
from .store import Store

log = logging.getLogger("halyard.state")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agi.halyard.state",
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HALYARD_STATE_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HALYARD_STATE_PORT", "8090")),
    )
    parser.add_argument(
        "--archive-root",
        default=os.environ.get("HALYARD_ARCHIVE_ROOT", "/archive/halyard"),
    )
    parser.add_argument(
        "--nats-url",
        default=os.environ.get("NATS_URL", ""),
        help="NATS URL (e.g. nats://127.0.0.1:4222). Empty = no bridge.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "info"),
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    store = Store(archive_root=Path(args.archive_root))
    broadcaster = WsBroadcaster()
    app = build_app(store=store, broadcaster=broadcaster)

    nats_client: Any = None
    bridge: StateBridge | None = None
    if args.nats_url:
        try:
            import nats  # type: ignore

            nats_client = await nats.connect(args.nats_url)
            log.info("NATS connected: %s", args.nats_url)

            # ws_broadcast bridges NATS-origin updates into the
            # aiohttp WebSocket fan-out so browser clients see
            # patches that came in through NATS too.
            async def _ws_broadcast(
                session_id: str, payload: dict[str, Any]
            ) -> None:
                await broadcaster.broadcast(session_id, payload)

            bridge = StateBridge(
                store=store,
                nats_client=nats_client,
                ws_broadcast=_ws_broadcast,
            )
            await bridge.run()
        except Exception as e:  # noqa: BLE001
            log.warning(
                "NATS bridge disabled (%s): service continues REST+WS only",
                e,
            )
            nats_client = None
            bridge = None
    else:
        log.info(
            "NATS_URL not set; running REST+WS only (bridge disabled)"
        )

    # aiohttp lifecycle — manual site-per-app loop so we can await
    # a shutdown event alongside the server.
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    log.info(
        "halyard-state serving on http://%s:%d (archive=%s)",
        args.host,
        args.port,
        args.archive_root,
    )

    stop = asyncio.Event()

    def _stop(*_args: Any) -> None:
        log.info("shutdown requested")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            # Windows lacks add_signal_handler; rely on KeyboardInterrupt.
            pass

    await stop.wait()

    log.info("draining aiohttp runner")
    await runner.cleanup()
    if nats_client is not None:
        try:
            await nats_client.drain()
            await nats_client.close()
        except Exception as e:  # noqa: BLE001
            log.warning("NATS shutdown: %s", e)


def main() -> int:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
