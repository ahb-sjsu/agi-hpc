# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Rate limiting for Erebus's Discord replies.

Two independent limits keep Erebus from flooding a channel or being pulled
into a tight back-and-forth loop:

- a per-user cooldown (min seconds between replies to the same person), and
- a per-channel sliding-window cap (max replies per channel per window).

In-memory (the bot is a single long-running process). The clock is
injectable so the whole thing is unit-testable without sleeping.
"""

from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    def __init__(
        self,
        per_user_cooldown_s: float = 8.0,
        per_channel_max: int = 15,
        per_channel_window_s: float = 60.0,
        now=time.monotonic,
    ) -> None:
        self.cooldown = per_user_cooldown_s
        self.chan_max = per_channel_max
        self.window = per_channel_window_s
        self._now = now
        self._user_last: dict[int, float] = {}
        self._chan_hits: dict[int, deque] = {}

    def allow(self, user_id: int, channel_id: int) -> bool:
        """True if a reply is permitted now; records the reply if so."""
        t = self._now()
        last = self._user_last.get(user_id)
        if last is not None and (t - last) < self.cooldown:
            return False
        hits = self._chan_hits.setdefault(channel_id, deque())
        while hits and (t - hits[0]) > self.window:
            hits.popleft()
        if len(hits) >= self.chan_max:
            return False
        # permitted → record
        self._user_last[user_id] = t
        hits.append(t)
        return True
