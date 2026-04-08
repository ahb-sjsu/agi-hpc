# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Chaos Testing Framework for AGI-HPC.

Provides fault injection tools for resilience testing:
- Latency injection
- Error injection
- Timeout simulation
- Network partition simulation

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChaosConfig:
    """Configuration for chaos testing."""

    enabled: bool = False
    failure_rate: float = 0.1  # Probability of injecting failure
    latency_ms: float = 100.0  # Added latency
    max_concurrent_faults: int = 1
    fault_types: List[str] = field(
        default_factory=lambda: ["latency", "error", "timeout"]
    )
    seed: Optional[int] = None


@dataclass
class ChaosHandle:
    """Handle to control active chaos injection."""

    fault_id: str
    fault_type: str
    target: str
    start_time: float
    active: bool = True
    _monkey: Optional["ChaosMonkey"] = field(default=None, repr=False)

    def stop(self) -> None:
        """Stop this fault injection."""
        self.active = False
        if self._monkey:
            self._monkey._remove_fault(self.fault_id)
        logger.info("[chaos] stopped fault %s on %s", self.fault_id, self.target)


# ---------------------------------------------------------------------------
# Chaos Monkey
# ---------------------------------------------------------------------------


class ChaosMonkey:
    """Chaos testing tool for resilience testing.

    Injects various types of faults into the system to verify
    that it handles failures gracefully.
    """

    def __init__(self, config: Optional[ChaosConfig] = None) -> None:
        self._config = config or ChaosConfig()
        self._active_faults: Dict[str, ChaosHandle] = {}
        self._rng = random.Random(self._config.seed)
        self._patches: Dict[str, Any] = {}

        logger.info(
            "[chaos] initialized enabled=%s failure_rate=%.2f",
            self._config.enabled,
            self._config.failure_rate,
        )

    def inject_latency(
        self,
        target: str,
        latency_ms: Optional[float] = None,
    ) -> ChaosHandle:
        """Inject latency into a target function/service.

        Args:
            target: Target identifier (e.g., "lh.planner.plan")
            latency_ms: Latency to inject in milliseconds

        Returns:
            ChaosHandle to control the fault
        """
        latency = latency_ms or self._config.latency_ms
        handle = self._create_handle("latency", target)

        logger.info(
            "[chaos] injecting %.0fms latency on %s",
            latency,
            target,
        )

        self._active_faults[handle.fault_id] = handle
        return handle

    def inject_error(
        self,
        target: str,
        error_type: str = "RuntimeError",
        message: str = "Chaos fault injected",
    ) -> ChaosHandle:
        """Inject an error into a target.

        Args:
            target: Target identifier
            error_type: Type of exception to raise
            message: Error message

        Returns:
            ChaosHandle to control the fault
        """
        handle = self._create_handle("error", target)

        logger.info(
            "[chaos] injecting %s on %s: %s",
            error_type,
            target,
            message,
        )

        self._active_faults[handle.fault_id] = handle
        return handle

    def inject_timeout(
        self,
        target: str,
        timeout_sec: float = 0.0,
    ) -> ChaosHandle:
        """Inject a timeout into a target.

        Args:
            target: Target identifier
            timeout_sec: Timeout duration (0 = immediate timeout)

        Returns:
            ChaosHandle to control the fault
        """
        handle = self._create_handle("timeout", target)

        logger.info(
            "[chaos] injecting timeout on %s (%.1fs)",
            target,
            timeout_sec,
        )

        self._active_faults[handle.fault_id] = handle
        return handle

    def inject_partition(
        self,
        target_a: str,
        target_b: str,
    ) -> ChaosHandle:
        """Simulate a network partition between two targets.

        Args:
            target_a: First target
            target_b: Second target

        Returns:
            ChaosHandle to control the fault
        """
        target = f"{target_a}<->{target_b}"
        handle = self._create_handle("partition", target)

        logger.info("[chaos] injecting partition %s <-> %s", target_a, target_b)

        self._active_faults[handle.fault_id] = handle
        return handle

    def random_fault(self, targets: List[str]) -> ChaosHandle:
        """Inject a random fault on a random target.

        Args:
            targets: List of possible targets

        Returns:
            ChaosHandle to control the fault
        """
        if not targets:
            raise ValueError("No targets provided")

        target = self._rng.choice(targets)
        fault_type = self._rng.choice(self._config.fault_types)

        if fault_type == "latency":
            return self.inject_latency(target)
        elif fault_type == "error":
            return self.inject_error(target)
        elif fault_type == "timeout":
            return self.inject_timeout(target)
        else:
            return self.inject_error(target)

    def should_inject(self) -> bool:
        """Check if a fault should be injected based on failure rate."""
        if not self._config.enabled:
            return False
        if len(self._active_faults) >= self._config.max_concurrent_faults:
            return False
        return self._rng.random() < self._config.failure_rate

    def stop_all(self) -> None:
        """Stop all active fault injections."""
        for handle in list(self._active_faults.values()):
            handle.active = False
        self._active_faults.clear()
        logger.info("[chaos] stopped all faults")

    def get_active_faults(self) -> List[ChaosHandle]:
        """Get all active fault handles."""
        return [h for h in self._active_faults.values() if h.active]

    @contextmanager
    def chaos_scope(
        self,
        targets: List[str],
    ) -> Generator[List[ChaosHandle], None, None]:
        """Context manager for scoped chaos injection.

        Injects random faults on enter, stops them on exit.

        Args:
            targets: List of targets for fault injection

        Yields:
            List of active ChaosHandle objects
        """
        handles = []
        try:
            if self._config.enabled and targets:
                handle = self.random_fault(targets)
                handles.append(handle)
            yield handles
        finally:
            for handle in handles:
                handle.stop()

    def _create_handle(self, fault_type: str, target: str) -> ChaosHandle:
        """Create a new chaos handle."""
        return ChaosHandle(
            fault_id=str(uuid.uuid4())[:8],
            fault_type=fault_type,
            target=target,
            start_time=time.monotonic(),
            _monkey=self,
        )

    def _remove_fault(self, fault_id: str) -> None:
        """Remove a fault from active tracking."""
        self._active_faults.pop(fault_id, None)
