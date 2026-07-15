# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""The Discord message-handling core — deliberately free of the ``discord``
library so it is unit-testable with fakes.

Pipeline for one inbound message::

    filter (bot? channel? mention? empty?) → rate-limit → cognition →
    length-cap → first-contact disclosure → DEME output-gate → audit →
    (post | draft | suppress)

Fail-safes: a cognition error or a gate veto → *suppress* (Erebus stays
silent). Nothing is ever posted that didn't pass the gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("discord.handler")


@dataclass
class InMessage:
    text: str
    author_id: int
    author_name: str
    channel_id: int
    is_bot: bool = False
    mentions_me: bool = False


@dataclass
class Outcome:
    action: str          # "reply" | "drafted" | "suppress" | "ignore"
    text: str = ""
    reason: str = ""


@dataclass
class BotState:
    seen_authors: set = field(default_factory=set)
    channel_conv: dict = field(default_factory=dict)

    def conv_for(self, channel_id: int) -> str:
        return self.channel_conv.setdefault(channel_id, f"discord-{channel_id}")


@dataclass
class Deps:
    cognition: object          # callable(text, conv_id) -> str
    gate: object               # OutputGate
    ratelimit: object          # RateLimiter
    audit: object              # AuditLog
    state: BotState
    disabled_sentinel: Path


def _should_consider(msg: InMessage, cfg) -> str | None:
    """Return a skip-reason if the message shouldn't be answered, else None."""
    if msg.is_bot:
        return "bot-author"
    if cfg.channel_ids and msg.channel_id not in cfg.channel_ids:
        return "other-channel"
    # Safety: if no channel is configured, only ever answer explicit mentions
    # (prevents responding everywhere by accident).
    require_mention = cfg.require_mention or not cfg.channel_ids
    if require_mention and not msg.mentions_me:
        return "not-addressed"
    if not msg.text.strip():
        return "empty"
    return None


def handle(msg: InMessage, deps: Deps, cfg) -> Outcome:
    """Process one inbound message and decide what to do. Never raises."""
    if deps.disabled_sentinel.exists():
        return Outcome("ignore", reason="disabled-sentinel")

    skip = _should_consider(msg, cfg)
    if skip:
        return Outcome("ignore", reason=skip)

    if not deps.ratelimit.allow(msg.author_id, msg.channel_id):
        deps.audit.inbound(msg, note="rate-limited")
        return Outcome("suppress", reason="rate-limited")

    deps.audit.inbound(msg)
    conv = deps.state.conv_for(msg.channel_id)

    # Observe-only input gate: score *what was asked* so the moral diagnostic
    # shows the incoming message's read separately from Erebus's reply. Never
    # blocks — the output gate below stays the enforcing guard. Absent on gates
    # that don't support it (e.g. test fakes), so it's looked up defensively.
    _score_input = getattr(deps.gate, "score_input", None)
    if _score_input is not None:
        try:
            _score_input(msg.text)
        except Exception:  # noqa: BLE001 - scoring must never break a reply
            pass

    try:
        reply = deps.cognition(msg.text, conv)
    except Exception as e:  # noqa: BLE001 - a cognition failure → stay silent
        deps.audit.error(msg, str(e))
        return Outcome("suppress", reason=f"cognition-error: {e}")

    reply = (reply or "").strip()
    if not reply:
        return Outcome("suppress", reason="empty-reply")
    reply = reply[: cfg.max_len]

    # First time talking to this person → prepend the AI disclosure.
    if msg.author_id not in deps.state.seen_authors:
        deps.state.seen_authors.add(msg.author_id)
        reply = cfg.disclosure_prefix + reply

    verdict = deps.gate.check(reply, msg.text)
    posted = verdict.allowed and cfg.mode != "draft"
    deps.audit.outbound(msg, reply, verdict, posted=posted)

    if not verdict.allowed:
        # Fail-safe: gate veto (or no gate) → Erebus does not speak.
        return Outcome("suppress", reason=f"gate: {verdict.reason}")

    deps.audit.conversation(conv, msg, reply)
    if cfg.mode == "draft":
        # Draft mode: reply is gate-approved + logged but not posted, for
        # human review of the first exchanges.
        return Outcome("drafted", text=reply, reason="draft-mode")
    return Outcome("reply", text=reply)
