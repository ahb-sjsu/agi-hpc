# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""discord.py wiring for Erebus's Discord faculty.

Thin layer: translate ``discord`` events into :class:`InMessage`, run the
library-free :func:`handler.handle`, and post the outcome. All decisions,
safety, and audit live in the testable core; this file only connects it to
Discord.

Run as ``atlas-erebus-discord.service``. ``discord`` is imported lazily so
the package imports (and its tests) work on a box without the library.

Env: see ``config.py`` (EREBUS_DISCORD_TOKEN, EREBUS_DISCORD_CHANNELS, …).
"""

from __future__ import annotations

import logging
import os

from .audit import AuditLog
from .cognition import http_cognition
from .config import DiscordConfig
from .handler import BotState, Deps, InMessage, handle
from .ratelimit import RateLimiter
from .safety import DemeOutputGate

log = logging.getLogger("discord.bot")


def build_deps(cfg: DiscordConfig) -> Deps:
    """Assemble the real dependency bundle (HTTP cognition + DEME gate)."""
    return Deps(
        cognition=http_cognition(cfg.cognition_url),
        gate=DemeOutputGate.try_build(),
        ratelimit=RateLimiter(
            cfg.per_user_cooldown_s, cfg.per_channel_max, cfg.per_channel_window_s
        ),
        audit=AuditLog(),
        state=BotState(),
        disabled_sentinel=cfg.disabled_sentinel,
    )


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("DISCORD_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = DiscordConfig.from_env()
    if not cfg.token:
        raise SystemExit("EREBUS_DISCORD_TOKEN not set — cannot start Discord faculty")

    import discord  # lazy: only the wiring layer needs the library

    deps = build_deps(cfg)

    intents = discord.Intents.default()
    intents.message_content = True  # requires the MESSAGE CONTENT intent to be enabled

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        log.info("Erebus online as %s (mode=%s, channels=%s)",
                 client.user, cfg.mode, sorted(cfg.channel_ids) or "any-when-mentioned")
        if cfg.post_intro and cfg.channel_ids:
            for cid in cfg.channel_ids:
                ch = client.get_channel(cid)
                if ch is not None:
                    try:
                        await ch.send(cfg.intro_text)
                    except Exception as e:  # noqa: BLE001
                        log.warning("intro post to %s failed: %s", cid, e)

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        mentions_me = client.user in getattr(message, "mentions", [])
        msg = InMessage(
            text=message.content or "",
            author_id=int(message.author.id),
            author_name=str(message.author),
            channel_id=int(message.channel.id),
            is_bot=bool(getattr(message.author, "bot", False)),
            mentions_me=mentions_me,
        )
        # handler is sync + fast (its one network call, cognition, blocks);
        # run it off the event loop so we don't stall the gateway heartbeat.
        import asyncio

        outcome = await asyncio.to_thread(handle, msg, deps, cfg)
        if outcome.action == "reply" and outcome.text:
            try:
                await message.channel.send(outcome.text)
            except Exception as e:  # noqa: BLE001
                log.warning("send failed: %s", e)
        elif outcome.action in ("suppress", "drafted", "ignore"):
            log.info("%s: %s", outcome.action, outcome.reason)

    client.run(cfg.token)


if __name__ == "__main__":
    main()
