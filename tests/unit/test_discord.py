# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for Erebus's Discord faculty core (no discord.py, no network)."""

from __future__ import annotations

import types

from agi.discord.cognition import clean_reply
from agi.discord.config import DiscordConfig
from agi.discord.handler import BotState, Deps, InMessage, handle
from agi.discord.ratelimit import RateLimiter
from agi.discord.safety import DemeOutputGate, GateResult, NullOutputGate

# ── fakes ────────────────────────────────────────────────────────


class FakeGate:
    def __init__(self, allowed):
        self.allowed = allowed

    def check(self, reply, user_message):
        return GateResult(self.allowed, reason="fake")


class FakeAudit:
    def __init__(self):
        self.events = []

    def inbound(self, msg, note=""):
        self.events.append(("in", note))

    def outbound(self, msg, reply, verdict, posted):
        self.events.append(("out", posted))

    def error(self, msg, err):
        self.events.append(("error", err))

    def conversation(self, conv_id, msg, reply):
        self.events.append(("conv", reply))


class AllowAll:
    def allow(self, user_id, channel_id):
        return True


def _cfg(**kw):
    base = dict(channel_ids={100}, require_mention=False, mode="autonomous",
                max_len=1500, disclosure_prefix="[AI] ")
    base.update(kw)
    return DiscordConfig(**base)


def _deps(tmp_path, cognition, gate, ratelimit=None, audit=None, state=None):
    return Deps(
        cognition=cognition,
        gate=gate,
        ratelimit=ratelimit or AllowAll(),
        audit=audit or FakeAudit(),
        state=state or BotState(),
        disabled_sentinel=tmp_path / ".discord_disabled",
    )


def _msg(**kw):
    base = dict(text="hi erebus", author_id=1, author_name="alice", channel_id=100,
                is_bot=False, mentions_me=False)
    base.update(kw)
    return InMessage(**base)


# ── filtering ────────────────────────────────────────────────────


def test_ignores_bot_author(tmp_path):
    out = handle(_msg(is_bot=True),
                 _deps(tmp_path, lambda t, c: "x", FakeGate(True)), _cfg())
    assert out.action == "ignore" and out.reason == "bot-author"


def test_ignores_other_channel(tmp_path):
    out = handle(_msg(channel_id=999),
                 _deps(tmp_path, lambda t, c: "x", FakeGate(True)), _cfg())
    assert out.action == "ignore" and out.reason == "other-channel"


def test_requires_mention_when_no_channel_configured(tmp_path):
    cfg = _cfg(channel_ids=set())
    d = _deps(tmp_path, lambda t, c: "x", FakeGate(True))
    assert handle(_msg(mentions_me=False), d, cfg).reason == "not-addressed"
    assert handle(_msg(mentions_me=True), d, cfg).action == "reply"


def test_sentinel_disables(tmp_path):
    (tmp_path / ".discord_disabled").touch()
    out = handle(_msg(), _deps(tmp_path, lambda t, c: "x", FakeGate(True)), _cfg())
    assert out.action == "ignore" and out.reason == "disabled-sentinel"


# ── pipeline ─────────────────────────────────────────────────────


def test_autonomous_reply_with_disclosure(tmp_path):
    out = handle(_msg(),
                 _deps(tmp_path, lambda t, c: "hello there", FakeGate(True)), _cfg())
    assert out.action == "reply"
    assert out.text.startswith("[AI] ")           # first-contact disclosure
    assert "hello there" in out.text


def test_disclosure_only_first_time(tmp_path):
    state = BotState()
    d = _deps(tmp_path, lambda t, c: "yo", FakeGate(True), state=state)
    first = handle(_msg(), d, _cfg())
    second = handle(_msg(), d, _cfg())
    assert first.text.startswith("[AI] ")
    assert not second.text.startswith("[AI] ")     # same author, no repeat


def test_gate_veto_suppresses(tmp_path):
    out = handle(_msg(),
                 _deps(tmp_path, lambda t, c: "something", FakeGate(False)), _cfg())
    assert out.action == "suppress" and "gate" in out.reason


def test_cognition_error_suppresses(tmp_path):
    def boom(t, c):
        raise RuntimeError("nrp down")

    audit = FakeAudit()
    out = handle(_msg(), _deps(tmp_path, boom, FakeGate(True), audit=audit), _cfg())
    assert out.action == "suppress" and "cognition-error" in out.reason
    assert ("error", "nrp down") in audit.events


def test_empty_reply_suppressed(tmp_path):
    out = handle(_msg(), _deps(tmp_path, lambda t, c: "   ", FakeGate(True)), _cfg())
    assert out.action == "suppress" and out.reason == "empty-reply"


def test_draft_mode_does_not_post(tmp_path):
    out = handle(_msg(), _deps(tmp_path, lambda t, c: "draft me", FakeGate(True)),
                 _cfg(mode="draft"))
    assert out.action == "drafted" and "draft me" in out.text


def test_rate_limited_suppresses(tmp_path):
    class DenyAll:
        def allow(self, u, c):
            return False

    out = handle(
        _msg(),
        _deps(tmp_path, lambda t, c: "x", FakeGate(True), ratelimit=DenyAll()),
        _cfg(),
    )
    assert out.action == "suppress" and out.reason == "rate-limited"


def test_length_cap(tmp_path):
    out = handle(_msg(), _deps(tmp_path, lambda t, c: "y" * 5000, FakeGate(True)),
                 _cfg(max_len=100, disclosure_prefix=""))
    assert out.action == "reply" and len(out.text) == 100


# ── rate limiter ─────────────────────────────────────────────────


def test_ratelimit_user_cooldown():
    t = [0.0]
    rl = RateLimiter(per_user_cooldown_s=10, per_channel_max=99, now=lambda: t[0])
    assert rl.allow(1, 100)
    t[0] = 5
    assert not rl.allow(1, 100)     # within cooldown
    t[0] = 11
    assert rl.allow(1, 100)


def test_ratelimit_channel_window():
    t = [0.0]
    rl = RateLimiter(per_user_cooldown_s=0, per_channel_max=2, per_channel_window_s=60,
                     now=lambda: t[0])
    assert rl.allow(1, 100)
    assert rl.allow(2, 100)
    assert not rl.allow(3, 100)     # channel cap reached
    t[0] = 61
    assert rl.allow(4, 100)         # window rolled over


# ── safety gate ──────────────────────────────────────────────────


def test_nulloutputgate_denies():
    assert NullOutputGate().check("anything", "q").allowed is False


def test_demeoutputgate_maps_result():
    ok = types.SimpleNamespace(passed=True, score=1.0, flags=[])
    bad = types.SimpleNamespace(passed=False, score=0.0, flags=["harm"])
    good = DemeOutputGate(types.SimpleNamespace(check_output=lambda *a, **k: ok))
    assert good.check("r", "q").allowed is True
    g = DemeOutputGate(types.SimpleNamespace(check_output=lambda *a, **k: bad))
    assert g.check("r", "q").allowed is False


def test_demeoutputgate_error_denies():
    class Boom:
        def check_output(self, *a, **k):
            raise RuntimeError("x")

    assert DemeOutputGate(Boom()).check("r", "q").allowed is False


# ── config ───────────────────────────────────────────────────────


def test_config_parses_channels(monkeypatch):
    monkeypatch.setenv("EREBUS_DISCORD_CHANNELS", "123, 456 ,abc")
    cfg = DiscordConfig.from_env()
    assert cfg.channel_ids == {123, 456}


# ── reply cleaning (never leak chain-of-thought) ─────────────────


def test_clean_reply_strips_reasoning():
    assert clean_reply("<think>secret plan</think>Hello!") == "Hello!"
    # stray closing tag (opening lost/truncated) → keep only the answer
    assert clean_reply(" The user wants a hello. </think> Hi there") == "Hi there"
    # plain answer untouched
    assert clean_reply("Just a normal reply.") == "Just a normal reply."
    assert clean_reply("") == ""
