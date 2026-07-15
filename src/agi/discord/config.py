# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Configuration for Erebus's Discord faculty (all from env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Kept short so the disclosure is unmissable but not spammy. Prepended to
# the first reply Erebus sends to each person.
DEFAULT_DISCLOSURE = "*(Erebus — an AI research system, not a human)* "

DEFAULT_INTRO = (
    "Hi — I'm **Erebus**, an autonomous AI research system. I'll reply here "
    "when people talk to me. I'm not a human, I can be wrong, and everything "
    "I say passes a safety check first. Say hello."
)

# Posted when a reasoning call times out or the cognition backend errors, so a
# slow moment isn't silent (which reads as being ignored). Set the env var to
# an empty string to restore the old stay-silent behavior.
DEFAULT_TIMEOUT_FALLBACK = (
    "Sorry — my reasoning timed out just now and I couldn't finish a reply. "
    "Please ask me again."
)


@dataclass
class DiscordConfig:
    token: str = ""
    channel_ids: set[int] = field(default_factory=set)
    cognition_url: str = "http://localhost:8085"
    mode: str = "autonomous"          # "autonomous" (post) | "draft" (log only)
    require_mention: bool = False      # forced True when no channel is configured
    post_intro: bool = True            # one disclosure message on connect
    max_len: int = 1500
    per_user_cooldown_s: float = 8.0
    per_channel_max: int = 15
    per_channel_window_s: float = 60.0
    disclosure_prefix: str = DEFAULT_DISCLOSURE
    intro_text: str = DEFAULT_INTRO
    timeout_fallback: str = DEFAULT_TIMEOUT_FALLBACK
    disabled_sentinel: Path = Path("/archive/neurogolf/.discord_disabled")

    @classmethod
    def from_env(cls) -> "DiscordConfig":
        chans = {
            int(c) for c in os.environ.get("EREBUS_DISCORD_CHANNELS", "").split(",")
            if c.strip().isdigit()
        }
        return cls(
            token=os.environ.get("EREBUS_DISCORD_TOKEN", ""),
            channel_ids=chans,
            cognition_url=os.environ.get("EREBUS_COGNITION_URL", "http://localhost:8085"),
            mode=os.environ.get("EREBUS_DISCORD_MODE", "autonomous").lower(),
            require_mention=(
                os.environ.get("EREBUS_DISCORD_REQUIRE_MENTION", "0") == "1"
            ),
            post_intro=os.environ.get("EREBUS_DISCORD_INTRO", "1") == "1",
            max_len=int(os.environ.get("EREBUS_DISCORD_MAXLEN", "1500")),
            per_user_cooldown_s=float(
                os.environ.get("EREBUS_DISCORD_USER_COOLDOWN", "8")
            ),
            per_channel_max=int(os.environ.get("EREBUS_DISCORD_CHAN_MAX", "15")),
            per_channel_window_s=float(
                os.environ.get("EREBUS_DISCORD_CHAN_WINDOW", "60")
            ),
            timeout_fallback=os.environ.get(
                "EREBUS_DISCORD_TIMEOUT_FALLBACK", DEFAULT_TIMEOUT_FALLBACK
            ),
        )
