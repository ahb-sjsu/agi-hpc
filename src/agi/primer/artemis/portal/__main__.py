# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Entry point for ``python -m agi.primer.artemis.portal``.

Wires:
  - ChatStore from /var/lib/atlas-artemis/chat.sqlite3
  - SheetCache fed by NATS agi.rh.artemis.sheet.*
  - Scene state fed by NATS agi.rh.artemis.scene
  - Wiki search against ATLAS_WIKI_URL (optional)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from typing import Any

from ..chat.store import ChatStore
from .auth import keeper_secret
from .cache import SceneState, SheetCache
from .server import PortalDeps, run_forever

log = logging.getLogger("artemis.portal.main")


async def _run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_path = os.environ.get("ARTEMIS_CHAT_DB", "/var/lib/atlas-artemis/chat.sqlite3")
    port = int(os.environ.get("ARTEMIS_PORTAL_PORT", "8090"))
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    wiki_url = os.environ.get("ATLAS_WIKI_URL", "").rstrip("/")

    secret = keeper_secret()
    chat_store = ChatStore(db_path)
    sheets = SheetCache()
    scene = SceneState()

    nc = await _connect_nats(nats_url)

    async def _publish(subject: str, payload: bytes) -> None:
        await nc.publish(subject, payload)

    async def _wiki_search(q: str, limit: int) -> list[dict[str, Any]]:
        if not wiki_url:
            return []
        import aiohttp

        async with aiohttp.ClientSession() as s:
            url = f"{wiki_url}/search"
            async with s.get(url, params={"q": q, "limit": limit}, timeout=5) as r:
                r.raise_for_status()
                data = await r.json()
        items = data.get("results") or data.get("hits") or []
        return [x for x in items if isinstance(x, dict)]

    deps = PortalDeps(
        chat_store=chat_store,
        sheets=sheets,
        scene=scene,
        publish=_publish,
        wiki_search=_wiki_search,
        secret=secret,
    )

    # NATS → cache hydrations
    async def _on_snapshot(msg: "Any") -> None:  # type: ignore[valid-type]
        try:
            body = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("snapshot parse: %s", e)
            return
        name = str(body.get("name") or "")
        rows = body.get("rows") or []
        if name and rows:
            sheets.replace_all(name, rows)
            log.info("cache snapshot: %s (%d rows)", name, len(rows))

    async def _on_diff(msg: "Any") -> None:  # type: ignore[valid-type]
        try:
            body = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("diff parse: %s", e)
            return
        name = str(body.get("name") or "")
        rows = body.get("rows") or []
        if name and rows:
            sheets.apply(name, rows)

    async def _on_scene(msg: "Any") -> None:  # type: ignore[valid-type]
        try:
            body = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("scene parse: %s", e)
            return
        if "name" in body:
            scene.name = str(body["name"])
        if "flags" in body:
            scene.flags = str(body["flags"])

    await nc.subscribe("agi.rh.artemis.sheet.snapshot.*", cb=_on_snapshot)
    await nc.subscribe("agi.rh.artemis.sheet.*", cb=_on_diff)
    await nc.subscribe("agi.rh.artemis.scene", cb=_on_scene)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _handle_sig() -> None:
        log.info("signal received, stopping portal")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            pass

    runner_task = asyncio.create_task(run_forever(port, deps=deps))
    try:
        await stop.wait()
    finally:
        runner_task.cancel()
        try:
            await runner_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            await nc.drain()
        except Exception:  # noqa: BLE001
            pass
    return 0


async def _connect_nats(url: str):
    import nats  # type: ignore

    return await nats.connect(servers=[url])


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
