# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-keeper-backend entrypoint.

Runnable via::

    python -m agi.halyard.keeper [--host 127.0.0.1] [--port 8091]

Env variables (see each module for detail):

- ``HALYARD_KEEPER_HOST``   — default ``127.0.0.1`` (loopback only —
                              Tailscale is OOB, atlas-caddy fronts.)
- ``HALYARD_KEEPER_PORT``   — default ``8091``
- ``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ``LIVEKIT_API_SECRET`` —
                              LiveKit SFU this backend mints for
- ``KEEPER_USERNAME``, ``KEEPER_PASSWORD`` — HTTP Basic creds for
                              the keeper-only routes (disabled if
                              unset; logged LOUDLY)
- ``KEEPER_IP_ALLOWLIST`` — comma-separated CIDRs; empty = no filter
- ``HALYARD_ARCHIVE_ROOT`` — sessions append-log lives under
                              ``$ARCHIVE/keeper/sessions/``
- ``LOG_LEVEL``             — default ``info``
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

from .app import build_app
from .auth import KeeperAuthConfig
from .livekit import LiveKitConfig
from .sessions import SessionRegistry

log = logging.getLogger("halyard.keeper")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agi.halyard.keeper",
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("HALYARD_KEEPER_HOST", "127.0.0.1"),
        help=(
            "Bind address. DEFAULT LOOPBACK ONLY — the service is "
            "intended to sit behind atlas-caddy. Do NOT bind to "
            "0.0.0.0 unless a trusted reverse-proxy is explicitly "
            "fronting this port."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HALYARD_KEEPER_PORT", "8091")),
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

    lk_cfg = LiveKitConfig.from_env()
    auth_cfg = KeeperAuthConfig.from_env()
    sessions = SessionRegistry(
        archive_root=Path(
            os.environ.get("HALYARD_ARCHIVE_ROOT", "/archive/halyard")
        ),
    )
    app = build_app(livekit=lk_cfg, auth=auth_cfg, sessions=sessions)

    if not auth_cfg.is_enabled():
        log.warning(
            "halyard-keeper: KEEPER_USERNAME/PASSWORD unset — keeper "
            "routes are UNAUTHENTICATED. Set env vars before running "
            "outside a trusted network."
        )
    log.info(
        "halyard-keeper: livekit=%s key=%s…",
        lk_cfg.url,
        lk_cfg.api_key[:4] + "…" if lk_cfg.api_key else "(unset)",
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    log.info(
        "halyard-keeper serving on http://%s:%d", args.host, args.port
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
            pass

    await stop.wait()
    await runner.cleanup()


def main() -> int:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
