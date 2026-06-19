# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).

"""Tests for the deontic maxim gate wiring in the ErisML integration.

Maxim derivation (PlanStep -> action_kind/polarity) and the universalizability
veto (via erisml-lib's deontic gate). Skips cleanly where the generated proto
stubs or erisml-lib are unavailable (e.g. a dev box without grpcio>=1.80).
"""

from __future__ import annotations

import types

import pytest

# These imports pull in the generated stubs; skip the module if they can't load.
facts_builder = pytest.importorskip("agi.safety.erisml.facts_builder")
service = pytest.importorskip("agi.safety.erisml.service")
plan_pb2 = pytest.importorskip("agi.proto_gen.plan_pb2")


def _step(tool_id: str = "noop", safety_tags=None, params=None) -> "plan_pb2.PlanStep":
    step = plan_pb2.PlanStep(step_id="s1", tool_id=tool_id)
    for t in safety_tags or []:
        step.safety_tags.append(t)
    for k, v in (params or {}).items():
        step.params[k] = v
    return step


# ----------------------------------------------------- maxim derivation


def test_derive_maxim_harmful_tool_is_inflict_harm():
    b = facts_builder.PlanStepToEthicalFacts()
    tool = sorted(facts_builder.HARMFUL_TOOLS)[0]
    action_kind, polarity = b._derive_maxim(_step(tool_id=tool))
    assert action_kind == "inflict_harm"
    assert polarity == "affirmed"


def test_derive_maxim_negation_tag():
    b = facts_builder.PlanStepToEthicalFacts()
    _, polarity = b._derive_maxim(
        _step(tool_id="deceive_user", safety_tags=["negated"])
    )
    assert polarity == "negated"


def test_derive_maxim_unknown_is_empty():
    b = facts_builder.PlanStepToEthicalFacts()
    action_kind, _ = b._derive_maxim(_step(tool_id="noop"))
    assert action_kind == ""


# ----------------------------------------------------- deontic veto


def _has_erisml_lib() -> bool:
    try:
        import erisml.ethics.deontic_gate  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_erisml_lib(), reason="erisml-lib not installed")
def test_deontic_veto_fires_for_prohibition():
    facts = types.SimpleNamespace(
        maxim_action_kind="deceive", maxim_polarity="affirmed"
    )
    reason = service.ErisMLServicer._deontic_maxim_veto(facts)
    assert reason is not None
    assert "deontic_universalizability_fail" in reason


@pytest.mark.skipif(not _has_erisml_lib(), reason="erisml-lib not installed")
def test_deontic_veto_silent_for_negated_prohibition():
    facts = types.SimpleNamespace(maxim_action_kind="deceive", maxim_polarity="negated")
    assert service.ErisMLServicer._deontic_maxim_veto(facts) is None


@pytest.mark.skipif(not _has_erisml_lib(), reason="erisml-lib not installed")
def test_deontic_veto_silent_when_no_maxim():
    facts = types.SimpleNamespace(maxim_action_kind="", maxim_polarity="affirmed")
    assert service.ErisMLServicer._deontic_maxim_veto(facts) is None
