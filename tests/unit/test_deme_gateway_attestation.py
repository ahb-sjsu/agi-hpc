"""Unit tests for SafetyGateway.check_hardware_attestation.

Stubbed DCGMAttestor — no real GPU needed. Verifies:
- snapshot-pair mode calls attestor.attest
- trace mode calls attestor.attest_trace
- DCGM-unavailable degrades gracefully with dcgm_unavailable flag
- attestation-fail surfaces via flags + score=0
- NATS publish callback is invoked with the right payload + subject
- Publish failure doesn't break attestation
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agi.safety.deme_gateway import SafetyGateway

# ── stub DCGMAttestor ──────────────────────────────────────────────


class _StubAttestor:
    def __init__(
        self, available: bool = True, attest_result=None, attest_trace_result=None
    ):
        self.available = available
        self._attest = attest_result
        self._attest_trace = attest_trace_result
        self.attest_calls = []
        self.attest_trace_calls = []

    def attest(self, before, after):
        self.attest_calls.append((before, after))
        return self._attest

    def attest_trace(self, samples, thresholds=None):
        self.attest_trace_calls.append(samples)
        return self._attest_trace


def _pass_result():
    # Minimal AttestationResult-like object
    return type(
        "R",
        (),
        {
            "passed": True,
            "computation": True,
            "integrity": True,
            "resource_match": True,
            "reason": "all checks passed",
        },
    )()


def _fail_result(reason="no power delta"):
    return type(
        "R",
        (),
        {
            "passed": False,
            "computation": False,
            "integrity": True,
            "resource_match": False,
            "reason": reason,
        },
    )()


# ── tests ──────────────────────────────────────────────────────────


def test_snapshot_pair_mode_invokes_attest_and_returns_pass():
    stub = _StubAttestor(available=True, attest_result=_pass_result())
    gw = SafetyGateway(dcgm_attestor=stub)
    before = MagicMock()
    after = MagicMock()
    r = gw.check_hardware_attestation(before=before, after=after)
    assert r.passed is True
    assert r.gate == "attestation"
    assert r.score == 1.0
    assert r.flags == []
    assert len(stub.attest_calls) == 1
    assert stub.attest_calls[0] == (before, after)


def test_trace_mode_invokes_attest_trace_and_returns_pass():
    stub = _StubAttestor(available=True, attest_trace_result=_pass_result())
    gw = SafetyGateway(dcgm_attestor=stub)
    r = gw.check_hardware_attestation(trace_samples=[{"power_w": 200, "util_pct": 90}])
    assert r.passed is True
    assert r.decision_proof["mode"] == "trace"
    assert len(stub.attest_trace_calls) == 1


def test_attestation_fail_surfaces_flags_and_zero_score():
    stub = _StubAttestor(
        available=True, attest_result=_fail_result("power delta 2W < 10W")
    )
    gw = SafetyGateway(dcgm_attestor=stub)
    r = gw.check_hardware_attestation(before=MagicMock(), after=MagicMock())
    assert r.passed is False
    assert r.score == 0.0
    assert "attestation_failed" in r.flags
    assert "no_compute_signal" in r.flags
    assert "resource_mismatch" in r.flags
    assert "power delta" in r.decision_proof["reason"]


def test_dcgm_unavailable_degrades_gracefully():
    """DCGM not installed → attestation is skipped and returns passed=True
    with a dcgm_unavailable flag so callers can enforce fail-closed if they
    want to, without the default behaviour blocking every LLM call on a
    host that simply doesn't have DCGM."""
    stub = _StubAttestor(available=False)
    gw = SafetyGateway(dcgm_attestor=stub)
    r = gw.check_hardware_attestation(before=MagicMock(), after=MagicMock())
    assert r.passed is True
    assert "dcgm_unavailable" in r.flags
    assert stub.attest_calls == []  # never invoked


def test_missing_mode_args_raises():
    stub = _StubAttestor(available=True, attest_result=_pass_result())
    gw = SafetyGateway(dcgm_attestor=stub)
    with pytest.raises(ValueError):
        gw.check_hardware_attestation()  # neither snapshot-pair nor trace


def test_nats_publish_invoked_on_attestation_pass():
    stub = _StubAttestor(available=True, attest_result=_pass_result())
    published = []

    def pub(subject, payload):
        published.append((subject, payload))

    gw = SafetyGateway(dcgm_attestor=stub, nats_publish=pub)
    gw.check_hardware_attestation(before=MagicMock(), after=MagicMock())
    assert len(published) == 1
    subject, payload = published[0]
    assert subject == "agi.safety.attestation"
    assert payload["passed"] is True
    assert payload["mode"] == "snapshot_pair"
    assert "ts" in payload
    assert payload["gpu_index"] == 0


def test_nats_publish_invoked_on_attestation_fail():
    stub = _StubAttestor(available=True, attest_result=_fail_result())
    published = []
    gw = SafetyGateway(
        dcgm_attestor=stub,
        nats_publish=lambda s, p: published.append((s, p)),
    )
    gw.check_hardware_attestation(before=MagicMock(), after=MagicMock())
    assert published[0][1]["passed"] is False
    assert "power delta" in published[0][1]["reason"] or published[0][1]["reason"]


def test_nats_publish_invoked_on_skip_when_dcgm_unavailable():
    stub = _StubAttestor(available=False)
    published = []
    gw = SafetyGateway(
        dcgm_attestor=stub,
        nats_publish=lambda s, p: published.append((s, p)),
    )
    gw.check_hardware_attestation(before=MagicMock(), after=MagicMock())
    assert len(published) == 1
    assert published[0][1]["mode"] == "skipped"


def test_nats_publish_failure_does_not_block_attestation():
    """If the publish callback raises, the attestation still returns a
    valid result — we log and move on."""
    stub = _StubAttestor(available=True, attest_result=_pass_result())

    def bad_pub(subject, payload):
        raise RuntimeError("nats server down")

    gw = SafetyGateway(dcgm_attestor=stub, nats_publish=bad_pub)
    r = gw.check_hardware_attestation(before=MagicMock(), after=MagicMock())
    assert r.passed is True
    # No exception escaped to the caller


def test_context_is_passed_through_to_nats_payload():
    stub = _StubAttestor(available=True, attest_result=_pass_result())
    published = []
    gw = SafetyGateway(
        dcgm_attestor=stub,
        nats_publish=lambda s, p: published.append((s, p)),
    )
    gw.check_hardware_attestation(
        before=MagicMock(),
        after=MagicMock(),
        gpu_index=1,
        context={"model": "kimi", "request_id": "abc"},
    )
    payload = published[0][1]
    assert payload["gpu_index"] == 1
    assert payload["context"]["model"] == "kimi"
    assert payload["context"]["request_id"] == "abc"


def test_attestation_with_real_trace_fixture():
    """End-to-end against a real-hardware trace fixture with a real
    DCGMAttestor (whose attest_trace goes through the classifier)."""
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "dcgm_profiles"
    if not (fixtures / "active_cupy.jsonl").exists():
        pytest.skip("dcgm fixtures missing")

    from agi.safety.dcgm_attestation import DCGMAttestor

    # Build an attestor and force it "available" via the classifier path;
    # attest_trace doesn't need dcgmi, only the classifier.
    attestor = DCGMAttestor()
    # Manually flag it available so the gateway doesn't skip
    attestor._available = True
    gw = SafetyGateway(dcgm_attestor=attestor)

    trace = json.loads((fixtures / "active_cupy.jsonl").read_text())["samples"]
    r = gw.check_hardware_attestation(trace_samples=trace)
    assert r.passed is True
    assert r.flags == []

    idle = json.loads((fixtures / "idle_baseline.jsonl").read_text())["samples"]
    r2 = gw.check_hardware_attestation(trace_samples=idle)
    assert r2.passed is False
    assert "attestation_failed" in r2.flags
