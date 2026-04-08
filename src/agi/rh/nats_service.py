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
Right Hemisphere NATS Service for AGI-HPC Phase 4.

Wraps the Qwen 3 32B model into a NATS-connected cognitive service
that handles creative, pattern-seeking, and spatial reasoning tasks.

Subscribes to:
    agi.rh.request.{pattern,spatial,creative}

Uses Qwen 3 32B on localhost:8082 (NOT 8080 which is Gemma 4 / LH).

Publishes responses to:
    agi.rh.response.*

Publishes telemetry to:
    agi.meta.monitor.rh
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

from agi.common.event import Event  # noqa: E402
from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig  # noqa: E402
from agi.meta.llm.client import LLMClient  # noqa: E402
from agi.meta.llm.config import InferenceConfig, RH_PRESET  # noqa: E402
from agi.meta.llm.templates import PromptTemplateRegistry  # noqa: E402

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


@dataclass
class RHNatsServiceConfig:
    """Configuration for the RH NATS service.

    Attributes:
        nats_servers: NATS server URLs.
        llm_base_url: Base URL for the LLM server (Qwen 3 32B).
        llm_timeout: LLM request timeout in seconds.
        llm_model: Model name for API requests.
        enable_cot: Whether to publish chain-of-thought traces.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    llm_base_url: str = "http://localhost:8082"
    llm_timeout: float = 300.0
    llm_model: str = "qwen-3-32b"
    enable_cot: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> RHNatsServiceConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        rh = data.get("rh", data)
        nats_cfg = rh.get("nats", {})
        llm_cfg = rh.get("llm", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            llm_base_url=llm_cfg.get("base_url", "http://localhost:8082"),
            llm_timeout=llm_cfg.get("timeout", 300.0),
            llm_model=llm_cfg.get("model", "qwen-3-32b"),
            enable_cot=rh.get("enable_cot", True),
        )


# -----------------------------------------------------------------
# Telemetry tracker
# -----------------------------------------------------------------


@dataclass
class RHTelemetry:
    """Accumulates RH service metrics."""

    requests_processed: int = 0
    total_latency_ms: float = 0.0
    last_request_type: str = ""
    last_query: str = ""
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_processed == 0:
            return 0.0
        return self.total_latency_ms / self.requests_processed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_processed": self.requests_processed,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_request_type": self.last_request_type,
            "last_query_preview": (self.last_query[:80] if self.last_query else ""),
            "errors": self.errors,
        }


# -----------------------------------------------------------------
# RH NATS Service
# -----------------------------------------------------------------


class RHNatsService:
    """Right Hemisphere as a NATS-connected cognitive service.

    Subscribes to creative/pattern/spatial request subjects, processes
    queries through Qwen 3 32B, and publishes responses back through
    the event fabric.

    Usage::

        service = RHNatsService()
        await service.start()
        # ... service runs until stopped ...
        await service.stop()
    """

    # Subjects this service listens on
    REQUEST_SUBJECTS = [
        "agi.rh.request.pattern",
        "agi.rh.request.spatial",
        "agi.rh.request.creative",
    ]

    def __init__(
        self,
        config: Optional[RHNatsServiceConfig] = None,
    ) -> None:
        self._config = config or RHNatsServiceConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._llm: Optional[LLMClient] = None
        self._templates = PromptTemplateRegistry()
        self._telemetry = RHTelemetry()
        self._running = False

    async def start(self) -> None:
        """Connect to NATS and start processing requests."""
        logger.info("[rh-service] starting Phase 4 RH NATS service")

        # Initialise NATS fabric
        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Initialise LLM client (Qwen 3 on port 8082)
        self._llm = LLMClient(
            base_url=self._config.llm_base_url,
            timeout=self._config.llm_timeout,
            default_model=self._config.llm_model,
        )

        # Subscribe to request subjects
        for subject in self.REQUEST_SUBJECTS:
            await self._fabric.subscribe(subject, self._handle_request)

        self._running = True
        logger.info(
            "[rh-service] ready -- subscribed to %s",
            ", ".join(self.REQUEST_SUBJECTS),
        )

    async def stop(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        if self._llm:
            await self._llm.close()
        if self._fabric:
            await self._fabric.disconnect()
        logger.info("[rh-service] stopped")

    async def _handle_request(self, event: Event) -> None:
        """Process an incoming RH request event.

        Expected payload keys:
            prompt (str): The user query.
            messages (list, optional): Full message list override.
            config (dict, optional): InferenceConfig overrides.
        """
        t0 = time.perf_counter()
        request_type = event.type.split(".")[-1] if event.type else "creative"
        prompt = event.payload.get("prompt", "")
        trace_id = event.trace_id

        logger.info(
            "[rh-service] request type=%s trace=%s prompt=%r",
            request_type,
            trace_id[:8],
            prompt[:60],
        )

        try:
            self._telemetry.last_request_type = request_type
            self._telemetry.last_query = prompt

            # Build system prompt from template
            system_prompt = self._templates.render(
                "rh_creative",
                model_name=self._config.llm_model,
                rag_context="",
            )

            # Build inference config with RH preset (creative, divergent)
            user_config = event.payload.get("config", {})
            inference_config = InferenceConfig(
                temperature=user_config.get("temperature", RH_PRESET.temperature),
                top_p=user_config.get("top_p", RH_PRESET.top_p),
                max_tokens=user_config.get("max_tokens", RH_PRESET.max_tokens),
                system_prompt=system_prompt,
            )

            # Call LLM (Qwen 3 32B)
            messages = event.payload.get("messages")
            result = await self._llm.generate(
                prompt=prompt,
                config=inference_config,
                messages=messages,
            )

            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Update telemetry
            self._telemetry.requests_processed += 1
            self._telemetry.total_latency_ms += latency_ms

            # Publish response
            response_event = Event.create(
                source="rh",
                event_type=f"rh.response.{request_type}",
                payload={
                    "text": result.text,
                    "model": result.model,
                    "tokens_used": result.tokens_used,
                    "latency_ms": round(latency_ms, 1),
                    "request_type": request_type,
                    "finish_reason": result.finish_reason,
                },
                trace_id=trace_id,
            )
            await self._fabric.publish(
                f"agi.rh.response.{request_type}", response_event
            )

            # Publish chain-of-thought trace for metacognition
            if self._config.enable_cot:
                cot_event = Event.create(
                    source="rh",
                    event_type="rh.internal.cot",
                    payload={
                        "request_type": request_type,
                        "prompt_preview": prompt[:200],
                        "response_preview": result.text[:500],
                        "inference_config": {
                            "temperature": inference_config.temperature,
                            "max_tokens": inference_config.max_tokens,
                        },
                        "latency_ms": round(latency_ms, 1),
                        "tokens_used": result.tokens_used,
                    },
                    trace_id=trace_id,
                )
                await self._fabric.publish("agi.rh.internal.cot", cot_event)

            # Publish telemetry
            telemetry_event = Event.create(
                source="rh",
                event_type="meta.monitor.rh",
                payload=self._telemetry.to_dict(),
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.meta.monitor.rh", telemetry_event)

            logger.info(
                "[rh-service] response sent trace=%s latency=%.0fms tokens=%d",
                trace_id[:8],
                latency_ms,
                result.tokens_used,
            )

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[rh-service] error processing request trace=%s",
                trace_id[:8],
            )
            # Publish error response
            if self._fabric and self._fabric.is_connected:
                error_event = Event.create(
                    source="rh",
                    event_type=f"rh.response.{request_type}",
                    payload={
                        "error": True,
                        "message": "Internal RH service error",
                    },
                    trace_id=trace_id,
                )
                await self._fabric.publish(
                    f"agi.rh.response.{request_type}", error_event
                )

    @property
    def telemetry(self) -> RHTelemetry:
        """Return current telemetry snapshot."""
        return self._telemetry


# -----------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------


async def run_service(config_path: Optional[str] = None) -> None:
    """Run the RH NATS service until interrupted."""
    if config_path:
        config = RHNatsServiceConfig.from_yaml(config_path)
    else:
        config = RHNatsServiceConfig()

    service = RHNatsService(config=config)
    await service.start()

    try:
        # Run forever until cancelled
        while service._running:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await service.stop()


def main() -> None:
    """CLI entry point for the RH NATS service."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AGI-HPC RH NATS Service (Phase 4)")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to rh_config.yaml",
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
        logger.info("[rh-service] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
