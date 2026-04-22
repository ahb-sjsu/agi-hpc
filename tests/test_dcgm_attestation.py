# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Tests for agi.safety.dcgm_attestation — BIP hardware grounding."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agi.safety.dcgm_attestation import (
    AttestationResult,
    DCGMAttestor,
    GPUSnapshot,
    GroundingViolation,
)

# ── Fixtures ──────────────────────────────────────────────────────


def _snap(
    sm: float = 50.0,
    power: float = 200.0,
    ecc_dbe: int = 0,
    ecc_sbe: int = 0,
    mem_used: int = 16000,
    **kw,
) -> GPUSnapshot:
    return GPUSnapshot(
        timestamp=time.time(),
        gpu_index=0,
        sm_utilization=sm,
        memory_utilization=60.0,
        power_draw_w=power,
        temperature_c=72.0,
        ecc_sbe=ecc_sbe,
        ecc_dbe=ecc_dbe,
        memory_used_mib=mem_used,
        **kw,
    )


@pytest.fixture
def attestor():
    att = DCGMAttestor(gpu_index=0, min_sm_utilization=5.0, min_power_delta_w=10.0)
    att._available = True  # skip actual dcgmi check
    return att


# ── GPUSnapshot ───────────────────────────────────────────────────


class TestGPUSnapshot:
    def test_ecc_clean_when_no_errors(self):
        snap = _snap(ecc_dbe=0, ecc_sbe=3)
        assert snap.ecc_clean is True

    def test_ecc_dirty_on_uncorrected_error(self):
        snap = _snap(ecc_dbe=1)
        assert snap.ecc_clean is False

    def test_frozen(self):
        snap = _snap()
        with pytest.raises(AttributeError):
            snap.sm_utilization = 99.0  # type: ignore[misc]


# ── Attestation ───────────────────────────────────────────────────


class TestAttest:
    def test_all_pass_on_healthy_compute(self, attestor):
        before = _snap(sm=0, power=30.0, ecc_dbe=0)
        after = _snap(sm=85, power=250.0, ecc_dbe=0)
        result = attestor.attest(before, after)
        assert result.passed is True
        assert result.computation is True
        assert result.integrity is True
        assert result.resource_match is True
        assert "all checks passed" in result.reason

    def test_fails_computation_when_gpu_idle(self, attestor):
        before = _snap(sm=0, power=30.0)
        after = _snap(sm=0, power=45.0)  # power up but SM idle
        result = attestor.attest(before, after)
        assert result.passed is False
        assert result.computation is False
        assert "SM utilization" in result.reason

    def test_fails_integrity_on_ecc_error(self, attestor):
        before = _snap(sm=0, power=30.0, ecc_dbe=0)
        after = _snap(sm=85, power=250.0, ecc_dbe=2)
        result = attestor.attest(before, after)
        assert result.passed is False
        assert result.integrity is False
        assert "ECC" in result.reason

    def test_fails_resource_when_no_power_delta(self, attestor):
        before = _snap(sm=0, power=30.0)
        after = _snap(sm=50, power=32.0)  # SM active but no power drawn??
        result = attestor.attest(before, after)
        assert result.passed is False
        assert result.resource_match is False
        assert "power delta" in result.reason

    def test_multiple_failures_reported(self, attestor):
        before = _snap(sm=0, power=30.0, ecc_dbe=0)
        after = _snap(sm=0, power=30.0, ecc_dbe=1)
        result = attestor.attest(before, after)
        assert result.passed is False
        assert result.computation is False
        assert result.integrity is False
        assert result.resource_match is False
        # All three reasons present
        assert "SM utilization" in result.reason
        assert "ECC" in result.reason
        assert "power delta" in result.reason


class TestAttestOrRaise:
    def test_returns_on_pass(self, attestor):
        before = _snap(sm=0, power=30.0)
        after = _snap(sm=80, power=250.0)
        result = attestor.attest_or_raise(before, after)
        assert result.passed is True

    def test_raises_on_failure(self, attestor):
        before = _snap(sm=0, power=30.0)
        after = _snap(sm=0, power=30.0)
        with pytest.raises(GroundingViolation, match="SM utilization"):
            attestor.attest_or_raise(before, after)


# ── Snapshot capture ──────────────────────────────────────────────


class TestSnapshot:
    def test_returns_empty_when_dcgm_unavailable(self):
        att = DCGMAttestor(gpu_index=0)
        att._available = False
        snap = att.snapshot()
        assert snap.sm_utilization == 0.0
        assert snap.ecc_dbe == 0

    def test_returns_empty_on_dcgmi_failure(self, attestor):
        with patch(
            "agi.safety.dcgm_attestation.subprocess.check_output",
            side_effect=subprocess.TimeoutExpired("dcgmi", 10),
        ):
            snap = attestor.snapshot()
        assert snap.sm_utilization == 0.0

    def test_parses_dmon_output(self, attestor):
        # Real dcgmi dmon output format (validated on Nautilus GPU node)
        sample_output = (
            "#Entity   GPUTL             MCUTL             POWER             TMPTR        ESVTL        EDVTL        FBUSD             \n"
            "ID                                             W                 C                                                       \n"
            "GPU 0     75                42                185.300           68           0            0            28456             \n"
        )
        snap = attestor._parse_dmon(sample_output)
        assert snap.sm_utilization == 75.0
        assert snap.memory_utilization == 42.0
        assert snap.power_draw_w == 185.3
        assert snap.temperature_c == 68.0
        assert snap.ecc_sbe == 0
        assert snap.ecc_dbe == 0
        assert snap.memory_used_mib == 28456

    def test_parses_real_rtx2080ti_fixture(self, attestor):
        """Parse real dcgmi dmon output captured on RTX 2080 Ti (Turing).

        GPU: NVIDIA GeForce RTX 2080 Ti, driver 590.48.01
        DCGM: 3.3.5, captured on NRP Nautilus Kubernetes cluster.
        Command: dcgmi dmon -e 203,204,155,150,310,311,252 -c 5 -d 1
        """
        fixture = Path(__file__).parent / "fixtures" / "dcgm_dmon_rtx2080ti.txt"
        raw = fixture.read_text()
        snap = attestor._parse_dmon(raw)
        # Last sample wins (parser overwrites on each data line)
        assert snap.sm_utilization == 0.0
        assert snap.memory_utilization == 0.0
        assert snap.power_draw_w == 20.986
        assert snap.temperature_c == 22.0
        assert snap.ecc_sbe == 0
        assert snap.ecc_dbe == 0
        assert snap.memory_used_mib == 0

    def test_handles_na_values(self, attestor):
        sample_output = (
            "#Entity   GPUTL             MCUTL             POWER             TMPTR        ESVTL        EDVTL        FBUSD             \n"
            "ID                                             W                 C                                                       \n"
            "GPU 0     N/A               N/A               N/A               N/A          N/A          N/A          N/A               \n"
        )
        snap = attestor._parse_dmon(sample_output)
        assert snap.sm_utilization == 0.0
        assert snap.ecc_dbe == 0


# ── Availability ──────────────────────────────────────────────────


class TestAvailability:
    def test_available_when_dcgmi_works(self):
        att = DCGMAttestor()
        with patch(
            "agi.safety.dcgm_attestation.subprocess.run",
        ) as mock_run:
            mock_run.return_value.returncode = 0
            assert att.available is True

    def test_unavailable_when_dcgmi_missing(self):
        att = DCGMAttestor(dcgmi_path="/nonexistent/dcgmi")
        with patch(
            "agi.safety.dcgm_attestation.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert att.available is False


import subprocess  # noqa: E402
