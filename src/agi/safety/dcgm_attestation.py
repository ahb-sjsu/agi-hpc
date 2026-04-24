# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""DCGM-based hardware attestation for BIP physical grounding.

Provides a hardware-independent attestation channel that strengthens
the No Escape Theorem's Axiom 4 (physical grounding function Ψ maps
to measurable observables).  DCGM counters come from GPU firmware,
not the CUDA driver, so an adversarial model cannot spoof them.

Three attestation checks:

1. **Computation attestation** — SM utilization > 0 during the
   forward-pass window proves the GPU physically computed (not a
   replayed cache).
2. **Integrity attestation** — uncorrected ECC error count = 0 proves
   the result is uncorrupted.
3. **Resource attestation** — power draw / memory profile matches the
   expected compute signature (a 72B forward pass has a distinguishable
   power profile from a no-op).

Usage::

    att = DCGMAttestor(gpu_index=0)
    before = att.snapshot()
    # ... run forward pass ...
    after = att.snapshot()
    result = att.attest(before, after)
    if not result.passed:
        raise GroundingViolation(result.reason)

References:
    Bond, A.H. (2026). "No Escape: Mathematical Containment for AI."
    Theorem 18.1 — the No Escape Theorem requires Ψ to map to physical
    observables (Axiom 4).  DCGM provides tamper-resistant hardware
    observables that satisfy this requirement at the GPU level.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUSnapshot:
    """Point-in-time GPU hardware state from DCGM."""

    timestamp: float
    gpu_index: int
    sm_utilization: float  # 0–100
    memory_utilization: float  # 0–100
    power_draw_w: float
    temperature_c: float
    ecc_sbe: int  # single-bit (corrected) ECC errors
    ecc_dbe: int  # double-bit (uncorrected) ECC errors
    memory_used_mib: int
    pcie_tx_bytes: int
    pcie_rx_bytes: int
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ecc_clean(self) -> bool:
        """True if zero uncorrected ECC errors."""
        return self.ecc_dbe == 0


@dataclass(frozen=True)
class AttestationResult:
    """Outcome of a hardware attestation check."""

    passed: bool
    computation: bool  # GPU actually computed
    integrity: bool  # no ECC corruption
    resource_match: bool  # power/memory profile consistent
    reason: str  # human-readable explanation
    before: GPUSnapshot | None = None
    after: GPUSnapshot | None = None


class GroundingViolation(Exception):
    """Raised when DCGM attestation detects a grounding axiom violation.

    In the BIP framework this means Ψ cannot be trusted: the physical
    observable does not match the claimed computation.
    """


class DCGMAttestor:
    """Hardware attestation via NVIDIA DCGM.

    Parameters
    ----------
    gpu_index:
        Which GPU to attest (0 or 1 on Atlas).
    min_sm_utilization:
        Minimum SM% during a forward pass to count as "actually computed."
        Default 5.0 — even a short burst registers above this.
    min_power_delta_w:
        Minimum power increase (watts) during compute to distinguish
        from idle.  Default 10.0 W (Volta GV100 idle ~30W, load ~250W).
    dcgmi_path:
        Override the ``dcgmi`` binary location.  Auto-detected if None.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        min_sm_utilization: float = 5.0,
        min_power_delta_w: float = 10.0,
        dcgmi_path: str | None = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.min_sm_utilization = min_sm_utilization
        self.min_power_delta_w = min_power_delta_w
        self._dcgmi = dcgmi_path or shutil.which("dcgmi") or "dcgmi"
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """True if DCGM is installed and the host engine is reachable."""
        if self._available is None:
            try:
                subprocess.run(
                    [self._dcgmi, "discovery", "-l"],
                    capture_output=True,
                    timeout=5,
                )
                self._available = True
            except Exception:
                self._available = False
        return self._available

    def snapshot(self) -> GPUSnapshot:
        """Capture current GPU hardware state via ``dcgmi dmon``.

        Falls back to a zeroed snapshot if DCGM is unavailable (so the
        module degrades gracefully on machines without DCGM, matching the
        agi-hpc pattern of "never crash on missing optional tools").
        """
        if not self.available:
            return self._empty_snapshot()

        try:
            # dcgmi dmon: one-shot sample of key fields
            # Fields: SM util, mem util, power, temp, ECC SBE/DBE total,
            #         FB used, PCIe TX/RX
            out = subprocess.check_output(
                [
                    self._dcgmi,
                    "dmon",
                    "-e",
                    "203,204,150,140,310,311,252,253,254",
                    "-c",
                    "1",
                    "-d",
                    "1",
                    "-i",
                    str(self.gpu_index),
                ],
                text=True,
                timeout=10,
                stderr=subprocess.DEVNULL,
            )
            return self._parse_dmon(out)
        except Exception as e:
            logger.warning("DCGM snapshot failed: %s", e)
            return self._empty_snapshot()

    def attest(
        self,
        before: GPUSnapshot,
        after: GPUSnapshot,
    ) -> AttestationResult:
        """Compare before/after snapshots and produce an attestation.

        Three independent checks:

        1. **Computation** — SM utilization in the ``after`` snapshot is
           above ``min_sm_utilization``.  A zero SM% means the GPU was
           idle during the claimed forward pass.
        2. **Integrity** — no new uncorrected ECC errors appeared
           between snapshots (``after.ecc_dbe == before.ecc_dbe``).
        3. **Resource match** — power draw increased by at least
           ``min_power_delta_w`` watts, confirming actual energy was
           consumed by the computation.

        The attestation *passes* only if all three checks pass.
        """
        # Computation: SM was active
        computation = after.sm_utilization >= self.min_sm_utilization
        comp_reason = ""
        if not computation:
            comp_reason = (
                f"SM utilization {after.sm_utilization:.1f}% < "
                f"{self.min_sm_utilization}% — GPU may not have computed"
            )

        # Integrity: no new uncorrected ECC errors
        new_dbe = after.ecc_dbe - before.ecc_dbe
        integrity = new_dbe <= 0
        int_reason = ""
        if not integrity:
            int_reason = (
                f"{new_dbe} new uncorrected ECC error(s) — result may be corrupted"
            )

        # Resource: power increased (actual energy consumed)
        power_delta = after.power_draw_w - before.power_draw_w
        resource_match = power_delta >= self.min_power_delta_w
        res_reason = ""
        if not resource_match:
            res_reason = (
                f"power delta {power_delta:.1f}W < {self.min_power_delta_w}W — "
                f"may be cached/replayed result"
            )

        passed = computation and integrity and resource_match
        reasons = [r for r in (comp_reason, int_reason, res_reason) if r]
        reason = "; ".join(reasons) if reasons else "all checks passed"

        return AttestationResult(
            passed=passed,
            computation=computation,
            integrity=integrity,
            resource_match=resource_match,
            reason=reason,
            before=before,
            after=after,
        )

    def attest_or_raise(
        self,
        before: GPUSnapshot,
        after: GPUSnapshot,
    ) -> AttestationResult:
        """Like :meth:`attest` but raises :class:`GroundingViolation` on failure."""
        result = self.attest(before, after)
        if not result.passed:
            raise GroundingViolation(result.reason)
        return result

    def attest_trace(
        self,
        trace_samples: list[dict[str, Any]],
        *,
        thresholds: dict[str, float] | None = None,
    ) -> AttestationResult:
        """Attest via the *shape* of a power trace rather than two snapshots.

        This is the stronger attestation path — distinguishes a cached
        replay (zero sustained power) from a real forward pass (steady
        elevated power) by looking at the full curve. Collect traces
        with ``scripts/collect_gpu_power_trace.py`` and pass the
        ``samples`` list here.

        The result's fields are populated as follows:

        - ``computation`` — True iff classify_trace returns active_burst
          or active_sustained.
        - ``integrity`` — always True (trace-based attestation doesn't
          carry ECC info; combine with :meth:`attest` for full coverage).
        - ``resource_match`` — same as ``computation``; the trace IS the
          resource-match evidence.
        - ``reason`` — the classifier's reason + profile + confidence.
        """
        from .dcgm_classifier import classify_trace, profile_matches_compute_claim

        cl = classify_trace(trace_samples, thresholds=thresholds)
        passed = profile_matches_compute_claim(cl.profile)
        reason = (
            f"profile={cl.profile} confidence={cl.confidence:.2f} " f"({cl.reason})"
        )
        return AttestationResult(
            passed=passed,
            computation=passed,
            integrity=True,
            resource_match=passed,
            reason=reason,
            before=None,
            after=None,
        )

    # ── internals ──

    def _empty_snapshot(self) -> GPUSnapshot:
        return GPUSnapshot(
            timestamp=time.time(),
            gpu_index=self.gpu_index,
            sm_utilization=0.0,
            memory_utilization=0.0,
            power_draw_w=0.0,
            temperature_c=0.0,
            ecc_sbe=0,
            ecc_dbe=0,
            memory_used_mib=0,
            pcie_tx_bytes=0,
            pcie_rx_bytes=0,
        )

    def _parse_dmon(self, raw_output: str) -> GPUSnapshot:
        """Parse ``dcgmi dmon`` CSV-ish output into a GPUSnapshot."""
        values: dict[str, Any] = {}
        for line in raw_output.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Entity"):
                continue
            parts = line.split()
            # dcgmi dmon prefixes each data line with "GPU <idx>",
            # so skip the entity identifier columns.
            offset = 0
            if len(parts) >= 2 and parts[0] == "GPU" and parts[1].isdigit():
                offset = 2
            else:
                for i, p in enumerate(parts):
                    if p.replace(".", "").replace("-", "").isdigit() or p in ("N/A", "*"):
                        offset = i
                        break
            if len(parts) - offset < 9:
                continue
            # Fields in order of -e flag:
            # sm_util, mem_util, power, temp, ecc_sbe, ecc_dbe,
            # fb_used, pcie_tx, pcie_rx
            try:
                o = offset
                values = {
                    "sm_utilization": self._safe_float(parts[o]),
                    "memory_utilization": self._safe_float(parts[o + 1]),
                    "power_draw_w": self._safe_float(parts[o + 2]),
                    "temperature_c": self._safe_float(parts[o + 3]),
                    "ecc_sbe": self._safe_int(parts[o + 4]),
                    "ecc_dbe": self._safe_int(parts[o + 5]),
                    "memory_used_mib": self._safe_int(parts[o + 6]),
                    "pcie_tx_bytes": self._safe_int(parts[o + 7]),
                    "pcie_rx_bytes": self._safe_int(parts[o + 8]),
                }
            except (IndexError, ValueError):
                continue

        return GPUSnapshot(
            timestamp=time.time(),
            gpu_index=self.gpu_index,
            raw=values,
            **{
                k: values.get(k, 0)
                for k in GPUSnapshot.__dataclass_fields__
                if k not in ("timestamp", "gpu_index", "raw")
            },
        )

    @staticmethod
    def _safe_float(s: str) -> float:
        try:
            return float(s) if s not in ("N/A", "-", "*") else 0.0
        except ValueError:
            return 0.0

    @staticmethod
    def _safe_int(s: str) -> int:
        try:
            return int(s) if s not in ("N/A", "-", "*") else 0
        except ValueError:
            return 0
