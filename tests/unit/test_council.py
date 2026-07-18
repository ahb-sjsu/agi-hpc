# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the Director's Divine-Council enrichment hook.

The Council only RE-ORDERS already-gated proposals and must fail open, so these
tests pin: it reorders per the model's ranking; it is inert when disabled, when
no council expert exists, or with <2 proposals; it never raises; and unranked
ids keep a stable fallback order."""

from types import SimpleNamespace

from agi.director.council import build_council_enrich


def _goal(gid, title="t"):
    return SimpleNamespace(id=gid, title=title, provenance={"rationale": "r"})


class FakeMoe:
    """Minimal vMOE stand-in: by_hint + an async cascade returning a fake Response."""

    def __init__(self, content, *, ok=True, has_expert=True, raise_exc=None):
        self._content, self._ok = content, ok
        self._has, self._raise = has_expert, raise_exc

    def by_hint(self, hint, only_healthy=False):
        return ["fable"] if self._has else []

    async def cascade(self, messages, *, hint=None, **kw):
        if self._raise:
            raise self._raise
        return SimpleNamespace(ok=self._ok, content=self._content, expert="fable")


def test_reorders_gated_proposals():
    proposals = [_goal("g1"), _goal("g2"), _goal("g3")]
    enrich = build_council_enrich(FakeMoe('["g3","g1","g2"]'), enabled=True)
    assert [g.id for g in enrich(proposals, {"cycle": 1})] == ["g3", "g1", "g2"]


def test_inert_when_disabled():
    proposals = [_goal("g1"), _goal("g2")]
    enrich = build_council_enrich(FakeMoe('["g2","g1"]'), enabled=False)
    assert [g.id for g in enrich(proposals, {})] == ["g1", "g2"]


def test_fail_open_on_error():
    proposals = [_goal("g1"), _goal("g2")]
    enrich = build_council_enrich(
        FakeMoe("", raise_exc=RuntimeError("boom")), enabled=True)
    assert [g.id for g in enrich(proposals, {})] == ["g1", "g2"]


def test_fail_open_on_not_ok_response():
    proposals = [_goal("g1"), _goal("g2")]
    enrich = build_council_enrich(FakeMoe('["g2","g1"]', ok=False), enabled=True)
    assert [g.id for g in enrich(proposals, {})] == ["g1", "g2"]


def test_noop_without_council_expert():
    proposals = [_goal("g1"), _goal("g2")]
    enrich = build_council_enrich(
        FakeMoe('["g2","g1"]', has_expert=False), enabled=True)
    assert [g.id for g in enrich(proposals, {})] == ["g1", "g2"]


def test_noop_single_proposal():
    proposals = [_goal("g1")]
    enrich = build_council_enrich(FakeMoe('["g1"]'), enabled=True)
    assert [g.id for g in enrich(proposals, {})] == ["g1"]


def test_unranked_ids_kept_stable():
    proposals = [_goal("g1"), _goal("g2"), _goal("g3")]
    enrich = build_council_enrich(FakeMoe('["g2"]'), enabled=True)  # only g2 ranked
    assert [g.id for g in enrich(proposals, {"cycle": 1})] == ["g2", "g1", "g3"]
