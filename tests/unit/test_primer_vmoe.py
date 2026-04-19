"""Unit tests for agi.primer.vmoe.

These are pure-logic tests with a stub async client — no network calls.
Verifies hint routing, cascade fallback ordering, and ensemble /
first_verified semantics (including that first_verified cancels
pending calls once a verified response arrives).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from agi.primer.vmoe import Expert, Response, vMOE

# ── Test fixtures ────────────────────────────────────────────────


def _experts() -> list[Expert]:
    return [
        Expert(
            name="fast-coder",
            model="fast",
            base_url="http://x",
            api_key_env="X",
            role_hints=frozenset({"code", "fast"}),
            priority=10,
        ),
        Expert(
            name="deep-reasoner",
            model="deep",
            base_url="http://x",
            api_key_env="X",
            role_hints=frozenset({"reason"}),
            priority=20,
        ),
        Expert(
            name="slow-coder",
            model="slow",
            base_url="http://x",
            api_key_env="X",
            role_hints=frozenset({"code"}),
            priority=30,
        ),
    ]


@dataclass
class _CallScript:
    """Recipe for one stubbed call: delay, ok, content, error."""

    delay_s: float = 0.0
    ok: bool = True
    content: str = "ok"
    error: str = ""


def _install_call_stub(monkeypatch, script_by_expert: dict[str, _CallScript]):
    """Patch vMOE.call so each named expert returns the scripted result
    after the scripted delay."""

    async def fake_call(self, expert_name, messages, **opts):
        s = script_by_expert[expert_name]
        if s.delay_s:
            await asyncio.sleep(s.delay_s)
        return Response(
            expert=expert_name,
            model=self.experts[expert_name].model,
            content=s.content,
            ok=s.ok,
            latency_s=s.delay_s,
            error=s.error,
        )

    monkeypatch.setattr(vMOE, "call", fake_call)


# ── by_hint / expert registry ────────────────────────────────────


def test_by_hint_returns_matching_experts_sorted_by_priority():
    m = vMOE(_experts())
    pool = m.by_hint("code")
    assert [e.name for e in pool] == ["fast-coder", "slow-coder"]


def test_by_hint_none_returns_all_sorted():
    m = vMOE(_experts())
    pool = m.by_hint(None)
    assert [e.name for e in pool] == ["fast-coder", "deep-reasoner", "slow-coder"]


def test_get_unknown_raises():
    m = vMOE(_experts())
    with pytest.raises(KeyError):
        m.get("nobody")


# ── route ────────────────────────────────────────────────────────


def test_route_uses_highest_priority_for_hint(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {"fast-coder": _CallScript(content="from fast")},
    )
    m = vMOE(_experts())
    r = asyncio.run(m.route([{"role": "user", "content": "hi"}], hint="code"))
    assert r.expert == "fast-coder"
    assert r.content == "from fast"


def test_route_no_match_raises(monkeypatch):
    m = vMOE(_experts())
    with pytest.raises(ValueError):
        asyncio.run(m.route([{"role": "user", "content": "hi"}], hint="vision"))


# ── cascade ──────────────────────────────────────────────────────


def test_cascade_falls_through_on_failure(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {
            "fast-coder": _CallScript(ok=False, error="timeout"),
            "slow-coder": _CallScript(content="slow saves the day"),
        },
    )
    m = vMOE(_experts())
    r = asyncio.run(m.cascade([{"role": "user", "content": "hi"}], hint="code"))
    assert r.expert == "slow-coder"
    assert r.content == "slow saves the day"


def test_cascade_custom_accept_predicate(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {
            "fast-coder": _CallScript(content="short"),  # ok but reject by len
            "slow-coder": _CallScript(content="long enough"),
        },
    )
    m = vMOE(_experts())
    r = asyncio.run(
        m.cascade(
            [{"role": "user", "content": "hi"}],
            hint="code",
            accept=lambda resp: resp.ok and len(resp.content) > 5,
        )
    )
    assert r.expert == "slow-coder"


def test_cascade_returns_last_failure_if_all_fail(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {
            "fast-coder": _CallScript(ok=False, error="a"),
            "slow-coder": _CallScript(ok=False, error="b"),
        },
    )
    m = vMOE(_experts())
    r = asyncio.run(m.cascade([{"role": "user", "content": "hi"}], hint="code"))
    assert not r.ok
    assert r.expert == "slow-coder"  # last tried


# ── ensemble / first_verified ────────────────────────────────────


def test_ensemble_returns_all_successes(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {
            "fast-coder": _CallScript(content="a"),
            "slow-coder": _CallScript(content="b"),
            "deep-reasoner": _CallScript(ok=False, error="nope"),
        },
    )
    m = vMOE(_experts())
    rs = asyncio.run(m.ensemble([{"role": "user", "content": "hi"}], hint=None))
    names = sorted(r.expert for r in rs)
    assert names == ["fast-coder", "slow-coder"]


def test_ensemble_verify_filter(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {
            "fast-coder": _CallScript(content="short"),
            "slow-coder": _CallScript(content="accepted-output"),
            "deep-reasoner": _CallScript(content="also-acceptable"),
        },
    )
    m = vMOE(_experts())
    rs = asyncio.run(
        m.ensemble(
            [{"role": "user", "content": "hi"}],
            hint=None,
            verify=lambda r: "accept" in r.content,
        )
    )
    assert {r.expert for r in rs} == {"slow-coder", "deep-reasoner"}


def test_first_verified_returns_fastest_accepted(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {
            "fast-coder": _CallScript(delay_s=0.02, content="fast-accept"),
            "slow-coder": _CallScript(delay_s=0.20, content="slow-accept"),
            "deep-reasoner": _CallScript(delay_s=0.50, content="deep-reject"),
        },
    )
    m = vMOE(_experts())
    r = asyncio.run(
        m.first_verified(
            [{"role": "user", "content": "hi"}],
            verify=lambda resp: "accept" in resp.content,
            hint=None,
        )
    )
    assert r is not None
    assert r.expert == "fast-coder"


def test_first_verified_returns_none_if_nothing_passes(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {
            "fast-coder": _CallScript(content="no"),
            "slow-coder": _CallScript(content="also-no"),
            "deep-reasoner": _CallScript(ok=False, error="boom"),
        },
    )
    m = vMOE(_experts())
    r = asyncio.run(
        m.first_verified(
            [{"role": "user", "content": "hi"}],
            verify=lambda resp: "yes" in resp.content,
            hint=None,
        )
    )
    assert r is None


def test_first_verified_async_predicate(monkeypatch):
    _install_call_stub(
        monkeypatch,
        {"fast-coder": _CallScript(content="match")},
    )
    m = vMOE(_experts())

    async def ap(resp: Response) -> bool:
        return resp.content == "match"

    r = asyncio.run(
        m.first_verified(
            [{"role": "user", "content": "hi"}],
            verify=ap,
            experts=["fast-coder"],
        )
    )
    assert r is not None and r.expert == "fast-coder"
