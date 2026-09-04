# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Periodic poller for Keeper's published-to-web CSV.

Service loop:
  1. Fetch each configured CSV URL.
  2. Parse into CharacterRow list.
  3. Compute HUD diff vs last tick.
  4. Publish a snapshot (full rows, one-shot) on first cycle; publish
     diffs on subsequent cycles when rows changed.

Transport layer (HTTP + NATS) is injected so unit tests can drive the
loop without a network or broker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from .parser import CharacterRow, diff_rows, parse_csv

log = logging.getLogger("artemis.sheets.poller")

SUBJECT_DIFF_PREFIX = "agi.rh.artemis.sheet"
SUBJECT_SNAPSHOT_PREFIX = "agi.rh.artemis.sheet.snapshot"

Fetcher = Callable[[str], Awaitable[str]]
Publisher = Callable[[str, bytes], Awaitable[None]]


@dataclass
class SheetBinding:
    """One sheet → one NATS subject suffix.

    ``name`` appears in the subject (``agi.rh.artemis.sheet.<name>``)
    and the Keeper can have multiple sheets (characters, npcs, etc.)
    polled concurrently with independent cadences if needed.
    """

    name: str
    url: str


class SheetsPoller:
    """Poll N sheets on a fixed interval; publish diffs + snapshots.

    Transport is dependency-injected:
      fetcher(url)   -> CSV text
      publisher(subj, payload) -> None

    The loop is cooperative — call :meth:`tick_once` for a single
    cycle in tests, :meth:`run_forever` in production.
    """

    def __init__(
        self,
        *,
        bindings: list[SheetBinding],
        fetcher: Fetcher,
        publisher: Publisher,
        poll_interval_s: float = 30.0,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.bindings = bindings
        self.fetcher = fetcher
        self.publisher = publisher
        self.poll_interval_s = poll_interval_s
        self._last: dict[str, list[CharacterRow]] = {b.name: [] for b in bindings}
        self._snapshotted: set[str] = set()
        self._running = False

    async def tick_once(self) -> dict[str, int]:
        """Fetch + publish every binding once. Returns diff count per name.

        Never raises — a single sheet's fetch failure is logged and
        the other sheets still tick.
        """
        counts: dict[str, int] = {}
        for b in self.bindings:
            try:
                text = await self.fetcher(b.url)
            except Exception as e:  # noqa: BLE001
                log.warning("fetch failed for %s: %s", b.name, e)
                counts[b.name] = 0
                continue
            try:
                rows = parse_csv(text)
            except Exception as e:  # noqa: BLE001
                log.warning("parse failed for %s: %s", b.name, e)
                counts[b.name] = 0
                continue
            counts[b.name] = await self._publish(b, rows)
        return counts

    async def run_forever(self) -> None:
        """Tick loop until :meth:`stop` is called or the task is cancelled."""
        self._running = True
        log.info(
            "sheets poller starting: %d sheet(s), interval=%.1fs",
            len(self.bindings),
            self.poll_interval_s,
        )
        try:
            while self._running:
                await self.tick_once()
                # Sleep in slices so stop() triggers within ~1s, not
                # one full interval later.
                remaining = self.poll_interval_s
                while remaining > 0 and self._running:
                    chunk = min(1.0, remaining)
                    await asyncio.sleep(chunk)
                    remaining -= chunk
        finally:
            log.info("sheets poller stopped")

    def stop(self) -> None:
        self._running = False

    # ── internal ────────────────────────────────────────────────

    async def _publish(self, b: SheetBinding, rows: list[CharacterRow]) -> int:
        """Publish snapshot on first tick for this binding; diffs after."""
        if b.name not in self._snapshotted:
            full = [r.to_dict() for r in rows]
            snapshot_subj = f"{SUBJECT_SNAPSHOT_PREFIX}.{b.name}"
            hud_subj = f"{SUBJECT_DIFF_PREFIX}.{b.name}"
            snapshot_payload = json.dumps(
                {"name": b.name, "rows": full}, separators=(",", ":")
            ).encode("utf-8")
            hud_payload = json.dumps(
                {"name": b.name, "rows": [_hud(r) for r in rows]},
                separators=(",", ":"),
            ).encode("utf-8")
            await self.publisher(snapshot_subj, snapshot_payload)
            await self.publisher(hud_subj, hud_payload)
            self._snapshotted.add(b.name)
            self._last[b.name] = rows
            log.info("snapshot: %s (%d rows)", b.name, len(rows))
            return len(rows)

        changed = diff_rows(self._last.get(b.name, []), rows)
        if changed:
            hud_subj = f"{SUBJECT_DIFF_PREFIX}.{b.name}"
            payload = json.dumps(
                {"name": b.name, "rows": changed}, separators=(",", ":")
            ).encode("utf-8")
            await self.publisher(hud_subj, payload)
            log.info("diff: %s (%d rows changed)", b.name, len(changed))
        self._last[b.name] = rows
        return len(changed)


def _hud(r: CharacterRow) -> dict[str, object]:
    from .parser import row_for_hud

    return row_for_hud(r)


# ─────────────────────────────────────────────────────────────────
# Config + factory
# ─────────────────────────────────────────────────────────────────


def _parse_bindings(raw: str) -> list[SheetBinding]:
    """Parse ARTEMIS_SHEET_URLS.

    Format: ``name1=url1;name2=url2`` — semicolons between bindings,
    ``=`` between name and URL. Whitespace around either is stripped.
    """
    out: list[SheetBinding] = []
    for piece in raw.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            log.warning("malformed sheet binding dropped: %r", piece)
            continue
        name, url = piece.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name or not url:
            log.warning("empty name or URL dropped: %r", piece)
            continue
        out.append(SheetBinding(name=name, url=url))
    return out


def build_poller_from_env(
    *,
    fetcher: Fetcher | None = None,
    publisher: Publisher | None = None,
) -> SheetsPoller:
    """Build a :class:`SheetsPoller` from environment.

    Env:
      ARTEMIS_SHEET_URLS   characters=https://...;npcs=https://...
      ARTEMIS_SHEET_POLL_S 30

    ``fetcher`` and ``publisher`` default to real HTTP + NATS. Tests
    inject fakes.
    """
    raw = os.environ.get("ARTEMIS_SHEET_URLS", "").strip()
    if not raw:
        raise RuntimeError(
            "ARTEMIS_SHEET_URLS is empty — set it to 'name=url;...' "
            "to enable the sheets poller."
        )
    bindings = _parse_bindings(raw)
    if not bindings:
        raise RuntimeError("ARTEMIS_SHEET_URLS parsed to zero bindings")
    interval = float(os.environ.get("ARTEMIS_SHEET_POLL_S", "30"))

    fetcher = fetcher or _default_fetcher
    publisher = publisher or _default_publisher()

    return SheetsPoller(
        bindings=bindings,
        fetcher=fetcher,
        publisher=publisher,
        poll_interval_s=interval,
    )


async def _default_fetcher(url: str) -> str:
    """Default HTTP fetcher. Uses ``httpx`` if available, else stdlib.

    Keeper's published CSV is public (`publish to web` → CSV), so
    we intentionally don't send auth.
    """
    try:
        import httpx  # type: ignore

        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.text
    except ImportError:
        # Fallback: urllib in a thread so the loop isn't blocked.
        import urllib.request

        def _get() -> str:
            req = urllib.request.Request(
                url, headers={"User-Agent": "artemis-sheets/1.0"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")

        return await asyncio.get_running_loop().run_in_executor(None, _get)


def _default_publisher() -> Publisher:
    """Lazy-wired NATS publisher.

    Connects on first publish and reuses the connection. Crash-loops
    are expected behaviour when NATS is down — systemd restarts us.
    """
    state: dict[str, object] = {"nc": None}

    async def publish(subject: str, payload: bytes) -> None:
        nc = state["nc"]
        if nc is None:
            import nats  # type: ignore

            url = os.environ.get("NATS_URL", "nats://localhost:4222")
            nc = await nats.connect(servers=[url])
            state["nc"] = nc
            log.info("connected to NATS: %s", url)
        await nc.publish(subject, payload)  # type: ignore[union-attr]

    return publish
