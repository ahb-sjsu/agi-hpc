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

"""
Safety NATS Service: Main service running input + output gates.

Connects to the NATS event fabric, subscribes to safety check
subjects, and publishes results, vetoes, and audit logs.

Tracks telemetry: checks/sec, veto count, avg latency, audit log size.
Publishes telemetry to ``agi.meta.monitor.safety``.

Subscribes to:
    agi.safety.check.input   -- pre-LLM input checks
    agi.safety.check.output  -- post-LLM output checks

Publishes to:
    agi.safety.result.input   -- input check results
    agi.safety.result.output  -- output check results
    agi.safety.veto           -- veto notifications
    agi.safety.audit          -- decision proofs
    agi.meta.monitor.safety   -- telemetry

Usage::

    service = SafetyNatsService()
    await service.start()
    # ... runs until stopped ...
    await service.stop()

Phase 3 (Safety Gateway) -- Atlas integration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from agi.common.event import Event
from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig
from agi.safety.deme_gateway import GatewayConfig, SafetyGateway
from agi.safety.input_gate import InputGate
from agi.safety.output_gate import OutputGate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SafetyServiceConfig:
    """Configuration for the Safety NATS Service.

    Attributes:
        nats_servers: NATS server URLs.
        port: Service port (for health/metrics endpoint).
        gateway_config_path: Path to safety_config.yaml.
        telemetry_interval_s: Seconds between telemetry publishes.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    port: int = 50055
    gateway_config_path: Optional[str] = None
    telemetry_interval_s: float = 10.0

    @classmethod
    def from_yaml(cls, path: str) -> SafetyServiceConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        safety = data.get("safety", data)
        nats_cfg = safety.get("nats", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            port=safety.get("port", 50055),
            gateway_config_path=safety.get("gateway_config_path"),
            telemetry_interval_s=safety.get("telemetry_interval_s", 10.0),
        )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass
class SafetyTelemetry:
    """Accumulates safety service metrics."""

    input_checks: int = 0
    output_checks: int = 0
    input_vetoes: int = 0
    output_vetoes: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    audit_log_size: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def total_checks(self) -> int:
        return self.input_checks + self.output_checks

    @property
    def total_vetoes(self) -> int:
        return self.input_vetoes + self.output_vetoes

    @property
    def avg_latency_ms(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.total_latency_ms / self.total_checks

    @property
    def checks_per_sec(self) -> float:
        elapsed = time.monotonic() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self.total_checks / elapsed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_checks": self.input_checks,
            "output_checks": self.output_checks,
            "input_vetoes": self.input_vetoes,
            "output_vetoes": self.output_vetoes,
            "total_checks": self.total_checks,
            "total_vetoes": self.total_vetoes,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "checks_per_sec": round(self.checks_per_sec, 2),
            "errors": self.errors,
            "audit_log_size": self.audit_log_size,
        }


# ---------------------------------------------------------------------------
# Safety NATS Service
# ---------------------------------------------------------------------------


class SafetyNatsService:
    """Main NATS-connected safety service.

    Orchestrates the InputGate and OutputGate, manages telemetry,
    and publishes metrics to the monitoring subject.
    """

    def __init__(self, config: Optional[SafetyServiceConfig] = None) -> None:
        self._config = config or SafetyServiceConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._gateway: Optional[SafetyGateway] = None
        self._input_gate: Optional[InputGate] = None
        self._output_gate: Optional[OutputGate] = None
        self._telemetry = SafetyTelemetry()
        self._running = False
        self._telemetry_task: Optional[asyncio.Task[None]] = None

    @property
    def telemetry(self) -> SafetyTelemetry:
        """Return current telemetry snapshot."""
        return self._telemetry

    @property
    def gateway(self) -> Optional[SafetyGateway]:
        """Return the SafetyGateway instance."""
        return self._gateway

    async def start(self) -> None:
        """Connect to NATS and start processing safety checks."""
        logger.info("[safety-service] starting Phase 3 Safety Service")

        # Load gateway config
        if self._config.gateway_config_path:
            gw_config = GatewayConfig.from_yaml(self._config.gateway_config_path)
        else:
            gw_config = GatewayConfig.default()

        # Initialise gateway and gates
        self._gateway = SafetyGateway(config=gw_config)
        self._input_gate = InputGate(self._gateway)
        self._output_gate = OutputGate(self._gateway)

        logger.info(
            "[safety-service] gateway initialised (DEME=%s)",
            self._gateway.has_deme,
        )

        # Connect to NATS
        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Wire up fabric references
        self._input_gate.set_fabric(self._fabric)
        self._output_gate.set_fabric(self._fabric)

        # Subscribe to check subjects with telemetry wrappers
        await self._fabric.subscribe(
            InputGate.SUBSCRIBE_SUBJECT,
            self._wrap_input_handler,
        )
        await self._fabric.subscribe(
            OutputGate.SUBSCRIBE_SUBJECT,
            self._wrap_output_handler,
        )

        self._running = True

        # Start telemetry publishing loop
        self._telemetry_task = asyncio.create_task(self._telemetry_loop())

        logger.info(
            "[safety-service] ready -- listening on %s, %s",
            InputGate.SUBSCRIBE_SUBJECT,
            OutputGate.SUBSCRIBE_SUBJECT,
        )

    async def stop(self) -> None:
        """Disconnect and clean up."""
        self._running = False

        if self._telemetry_task and not self._telemetry_task.done():
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass

        if self._fabric:
            await self._fabric.disconnect()

        logger.info("[safety-service] stopped")

    # ------------------------------------------------------------------
    # Telemetry-wrapped handlers
    # ------------------------------------------------------------------

    async def _wrap_input_handler(self, event: Event) -> None:
        """Wrap input gate handler with telemetry tracking."""
        t0 = time.perf_counter()
        try:
            await self._input_gate.handle(event)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.input_checks += 1
            self._telemetry.total_latency_ms += latency_ms
            self._telemetry.input_vetoes = self._input_gate.vetoes_total
            self._telemetry.audit_log_size = len(self._gateway.audit_log)
        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[safety-service] error in input handler trace=%s",
                event.trace_id[:8],
            )

    async def _wrap_output_handler(self, event: Event) -> None:
        """Wrap output gate handler with telemetry tracking."""
        t0 = time.perf_counter()
        try:
            await self._output_gate.handle(event)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._telemetry.output_checks += 1
            self._telemetry.total_latency_ms += latency_ms
            self._telemetry.output_vetoes = self._output_gate.vetoes_total
            self._telemetry.audit_log_size = len(self._gateway.audit_log)
        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[safety-service] error in output handler trace=%s",
                event.trace_id[:8],
            )

    # ------------------------------------------------------------------
    # Telemetry publishing
    # ------------------------------------------------------------------

    async def _telemetry_loop(self) -> None:
        """Periodically publish telemetry to agi.meta.monitor.safety."""
        while self._running:
            try:
                await asyncio.sleep(self._config.telemetry_interval_s)
                await self._publish_telemetry()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[safety-service] telemetry publish error")

    async def _publish_telemetry(self) -> None:
        """Publish current telemetry snapshot."""
        if self._fabric and self._fabric.is_connected:
            telemetry_event = Event.create(
                source="safety",
                event_type="meta.monitor.safety",
                payload=self._telemetry.to_dict(),
            )
            await self._fabric.publish("agi.meta.monitor.safety", telemetry_event)
            logger.debug("[safety-service] telemetry: %s", self._telemetry.to_dict())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def run_service(config_path: Optional[str] = None) -> None:
    """Run the Safety Service until interrupted."""
    if config_path:
        config = SafetyServiceConfig.from_yaml(config_path)
    else:
        config = SafetyServiceConfig()

    service = SafetyNatsService(config=config)
    await service.start()

    try:
        while service._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await service.stop()


def main() -> None:
    """CLI entry point for the Safety Service."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AGI-HPC Safety Service (Phase 3)")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to safety_config.yaml",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        asyncio.run(run_service(args.config))
    except KeyboardInterrupt:
        logger.info("[safety-service] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
