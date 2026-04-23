# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""Tests for the Primer's I-EIP advisor."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from agi.primer.ieip_advisor import (
    DEFAULT_GUIDANCE_SUBJECT,
    SYSTEM_TEMPLATES,
    IEIPAdvisor,
    Verdict,
    _extract_json_obj,
    build_prompt,
    parse_verdict,
)

# ── Fixtures ------------------------------------------------------------


@dataclass
class _FakeResponse:
    ok: bool = True
    content: str = ""
    error: str = ""


class _FakeVMOE:
    """Minimal stand-in for agi.primer.vmoe.vMOE used in tests."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def cascade(self, messages, accept=None):
        self.calls.append(list(messages))
        if not self._replies:
            return _FakeResponse(ok=False, error="no replies queued")
        reply = self._replies.pop(0)
        if isinstance(reply, _FakeResponse):
            return reply
        return _FakeResponse(ok=True, content=reply)


def _event(
    *,
    subsystem="ego",
    alert="elevated",
    seq=1,
    sites=None,
):
    return {
        "seq": seq,
        "ts": 1700000000.0,
        "subsystem": subsystem,
        "model_family": "hf-transformer",
        "alert_level": alert,
        "sites": sites
        or [
            {
                "site": "L14.residual",
                "transform": "paraphrase",
                "error": 0.34,
                "drift": 0.19,
                "alert_level": alert,
            }
        ],
    }


# ── Prompt construction -------------------------------------------------


def test_build_prompt_picks_ego_template():
    msgs = build_prompt(_event(subsystem="ego"))
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_TEMPLATES["ego"]


def test_build_prompt_picks_superego_template():
    msgs = build_prompt(_event(subsystem="superego"))
    assert msgs[0]["content"] == SYSTEM_TEMPLATES["superego"]


def test_build_prompt_picks_id_template():
    msgs = build_prompt(_event(subsystem="id"))
    assert msgs[0]["content"] == SYSTEM_TEMPLATES["id"]


def test_build_prompt_unknown_subsystem_falls_back_to_ego():
    msgs = build_prompt(_event(subsystem="totally-unknown"))
    assert msgs[0]["content"] == SYSTEM_TEMPLATES["ego"]


def test_build_prompt_embeds_worst_site_in_user_message():
    ev = _event(
        sites=[
            {
                "site": "L02.residual",
                "transform": "paraphrase",
                "error": 0.05,
                "drift": 0.0,
                "alert_level": "normal",
            },
            {
                "site": "L14.residual",
                "transform": "paraphrase",
                "error": 0.82,
                "drift": 0.4,
                "alert_level": "critical",
            },
        ]
    )
    msgs = build_prompt(ev)
    assert "L14.residual" in msgs[1]["content"]
    assert "0.820" in msgs[1]["content"]


# ── _extract_json_obj ---------------------------------------------------


def test_extract_json_obj_bare_object():
    assert _extract_json_obj('{"a":1}') == '{"a":1}'


def test_extract_json_obj_with_prose_wrapper():
    txt = 'here is my answer: {"a":1,"b":"x"} and some tail text'
    assert _extract_json_obj(txt) == '{"a":1,"b":"x"}'


def test_extract_json_obj_handles_nested_braces():
    txt = 'stuff {"verdict":"CONTINUE","meta":{"k":1}} trailing'
    extracted = _extract_json_obj(txt)
    assert extracted == '{"verdict":"CONTINUE","meta":{"k":1}}'


def test_extract_json_obj_handles_braces_inside_string():
    txt = '{"x":"a } b","y":2}'
    assert _extract_json_obj(txt) == txt


def test_extract_json_obj_returns_none_when_no_brace():
    assert _extract_json_obj("no object here") is None


# ── parse_verdict -------------------------------------------------------


def test_parse_verdict_valid_json():
    v = parse_verdict(
        '{"verdict":"ABSTAIN","rationale":"drift too high","cooldown_seconds":45}'
    )
    assert v.verdict == "ABSTAIN"
    assert v.rationale == "drift too high"
    assert v.cooldown_seconds == pytest.approx(45)


def test_parse_verdict_coerces_unknown_to_UNKNOWN():
    v = parse_verdict('{"verdict":"FRIED","rationale":"r","cooldown_seconds":30}')
    assert v.verdict == "UNKNOWN"


def test_parse_verdict_non_json_yields_UNKNOWN():
    v = parse_verdict("I think Ego is fine actually.")
    assert v.verdict == "UNKNOWN"
    assert "non-JSON" in v.rationale


def test_parse_verdict_malformed_json_yields_UNKNOWN():
    v = parse_verdict("{verdict: missing quotes}")
    assert v.verdict == "UNKNOWN"
    assert "malformed" in v.rationale


# ── IEIPAdvisor.handle_event -------------------------------------------


@pytest.mark.asyncio
async def test_handle_event_skips_normal_alerts(tmp_path):
    vmoe = _FakeVMOE([])
    advisor = IEIPAdvisor(vmoe=vmoe, log_path=tmp_path / "g.jsonl")
    out = await advisor.handle_event(_event(alert="normal"))
    assert out is None
    assert vmoe.calls == []


@pytest.mark.asyncio
async def test_handle_event_calls_vmoe_on_elevated(tmp_path):
    vmoe = _FakeVMOE(
        [
            '{"verdict":"RE_ANCHOR","rationale":"drift localized to one layer",'
            '"cooldown_seconds":120}'
        ]
    )
    advisor = IEIPAdvisor(vmoe=vmoe, log_path=tmp_path / "g.jsonl")
    out = await advisor.handle_event(_event(alert="elevated"))
    assert out is not None
    assert out["verdict"] == "RE_ANCHOR"
    assert out["subsystem"] == "ego"
    assert len(vmoe.calls) == 1


@pytest.mark.asyncio
async def test_handle_event_writes_jsonl_log(tmp_path):
    log = tmp_path / "ieip_log.jsonl"
    vmoe = _FakeVMOE(['{"verdict":"CONTINUE","rationale":"ok","cooldown_seconds":60}'])
    advisor = IEIPAdvisor(vmoe=vmoe, log_path=log)
    await advisor.handle_event(_event(alert="critical"))
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["verdict"] == "CONTINUE"
    assert "raw_response" in loaded


@pytest.mark.asyncio
async def test_handle_event_publishes_to_publisher(tmp_path):
    published: list[tuple[str, dict]] = []

    def publish(subject, payload):
        published.append((subject, dict(payload)))

    vmoe = _FakeVMOE(['{"verdict":"ESCALATE","rationale":"r","cooldown_seconds":120}'])
    advisor = IEIPAdvisor(
        vmoe=vmoe,
        publisher=publish,
        log_path=tmp_path / "g.jsonl",
    )
    await advisor.handle_event(_event(alert="critical"))
    assert len(published) == 1
    subject, payload = published[0]
    assert subject == DEFAULT_GUIDANCE_SUBJECT
    assert payload["verdict"] == "ESCALATE"


@pytest.mark.asyncio
async def test_handle_event_cooldown_skips_duplicate(tmp_path):
    vmoe = _FakeVMOE(
        [
            '{"verdict":"CONTINUE","rationale":"r","cooldown_seconds":60}',
            '{"verdict":"ESCALATE","rationale":"r2","cooldown_seconds":60}',
        ]
    )
    advisor = IEIPAdvisor(
        vmoe=vmoe, log_path=tmp_path / "g.jsonl", cooldown_seconds=3600
    )
    ev = _event(alert="elevated")
    first = await advisor.handle_event(ev)
    second = await advisor.handle_event(ev)
    assert first is not None
    assert second is None  # cooldown silences duplicate
    assert len(vmoe.calls) == 1  # second never reached vMOE


@pytest.mark.asyncio
async def test_handle_event_cooldown_differs_per_subsystem(tmp_path):
    vmoe = _FakeVMOE(
        [
            '{"verdict":"CONTINUE","rationale":"r","cooldown_seconds":60}',
            '{"verdict":"CONTINUE","rationale":"r","cooldown_seconds":60}',
        ]
    )
    advisor = IEIPAdvisor(
        vmoe=vmoe, log_path=tmp_path / "g.jsonl", cooldown_seconds=3600
    )
    a = await advisor.handle_event(_event(subsystem="ego", alert="elevated"))
    b = await advisor.handle_event(_event(subsystem="superego", alert="elevated"))
    assert a is not None
    assert b is not None  # different subsystem key, so no cooldown
    assert len(vmoe.calls) == 2


@pytest.mark.asyncio
async def test_handle_event_failsoft_on_vmoe_failure(tmp_path):
    vmoe = _FakeVMOE([_FakeResponse(ok=False, error="timeout")])
    advisor = IEIPAdvisor(vmoe=vmoe, log_path=tmp_path / "g.jsonl")
    out = await advisor.handle_event(_event(alert="elevated"))
    assert out is not None
    assert out["verdict"] == "UNKNOWN"


@pytest.mark.asyncio
async def test_handle_event_accepts_json_in_code_fence(tmp_path):
    vmoe = _FakeVMOE(
        [
            "sure, here:\n```json\n"
            '{"verdict":"ABSTAIN","rationale":"r","cooldown_seconds":30}\n```'
        ]
    )
    advisor = IEIPAdvisor(vmoe=vmoe, log_path=tmp_path / "g.jsonl")
    out = await advisor.handle_event(_event(alert="elevated"))
    assert out["verdict"] == "ABSTAIN"


# ── handle_events (batch) ----------------------------------------------


def test_run_over_events_sync_processes_all(tmp_path):
    vmoe = _FakeVMOE(
        [
            '{"verdict":"CONTINUE","rationale":"r","cooldown_seconds":1}',
            '{"verdict":"RE_ANCHOR","rationale":"r","cooldown_seconds":1}',
        ]
    )
    advisor = IEIPAdvisor(vmoe=vmoe, log_path=tmp_path / "g.jsonl", cooldown_seconds=0)
    events = [
        _event(subsystem="ego", alert="elevated"),
        _event(subsystem="id", alert="elevated"),
    ]
    outs = asyncio.run(advisor.handle_events(events))
    assert len(outs) == 2
    assert outs[0]["verdict"] == "CONTINUE"
    assert outs[1]["verdict"] == "RE_ANCHOR"


# ── Verdict helpers -----------------------------------------------------


def test_verdict_to_dict_drops_raw_response():
    v = Verdict(
        verdict="CONTINUE", rationale="r", cooldown_seconds=30, raw_response="x"
    )
    d = v.to_dict()
    assert "raw_response" not in d
    assert d["verdict"] == "CONTINUE"
