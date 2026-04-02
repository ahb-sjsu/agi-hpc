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
Left Hemisphere NATS Service for AGI-HPC Phase 1.

Wraps the existing llama-server and RAG into a NATS-connected
cognitive service that:
- Subscribes to agi.lh.request.{chat,plan,reason}
- Retrieves RAG context for each query
- Calls the LLM (Gemma 4 on localhost:8080)
- Publishes responses to agi.lh.response.*
- Publishes chain-of-thought traces for metacognition
- Reports telemetry to agi.meta.monitor.lh
"""

from __future__ import annotations

import asyncio
import json
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
from agi.meta.llm.client import LLMClient, CompletionResult  # noqa: E402
from agi.meta.llm.config import InferenceConfig, LH_PRESET  # noqa: E402
from agi.meta.llm.templates import PromptTemplateRegistry  # noqa: E402
from agi.lh.rag import RAGSearcher, RAGConfig, RAGResult  # noqa: E402


# -----------------------------------------------------------------
# Hemisphere routing (extracted from atlas-rag-server.py)
# -----------------------------------------------------------------

LH_KEYWORDS = {
    "explain", "debug", "error", "fix", "how does", "what is", "define",
    "analyze", "calculate", "prove", "implement", "code", "function",
    "syntax", "compile", "trace", "step by step", "specifically",
    "exact", "precise", "correct", "documentation", "api", "reference",
}

RH_KEYWORDS = {
    "brainstorm", "creative", "imagine", "what if", "pattern", "analogy",
    "design", "vision", "inspire", "explore", "possibilities", "connect",
    "themes", "big picture", "strategy", "reimagine", "innovate",
    "compare across", "similarities", "different angle", "metaphor",
}


def classify_hemisphere(text: str) -> str:
    """Route to lh or rh based on query content."""
    lower = text.lower()
    lh_score = sum(1 for kw in LH_KEYWORDS if kw in lower)
    rh_score = sum(1 for kw in RH_KEYWORDS if kw in lower)
    if rh_score > lh_score and rh_score >= 2:
        return "rh"
    return "lh"


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------

@dataclass
class LHNatsServiceConfig:
    """Configuration for the LH NATS service.

    Attributes:
        nats_servers: NATS server URLs.
        llm_base_url: Base URL for the LLM server (Gemma 4).
        llm_timeout: LLM request timeout in seconds.
        llm_model: Model name for API requests.
        rag_db_dsn: PostgreSQL connection string for RAG.
        rag_embed_model: Sentence-transformer model name.
        rag_top_k: Number of RAG results per query.
        enable_rag: Whether to use RAG context augmentation.
        enable_cot: Whether to publish chain-of-thought traces.
    """

    nats_servers: List[str] = field(
        default_factory=lambda: ["nats://localhost:4222"]
    )
    llm_base_url: str = "http://localhost:8080"
    llm_timeout: float = 300.0
    llm_model: str = "gemma-4-31b"
    rag_db_dsn: str = "dbname=atlas user=claude"
    rag_embed_model: str = "BAAI/bge-m3"
    rag_top_k: int = 6
    enable_rag: bool = True
    enable_cot: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> LHNatsServiceConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        lh = data.get("lh", data)
        nats_cfg = lh.get("nats", {})
        llm_cfg = lh.get("llm", {})
        rag_cfg = lh.get("rag", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            llm_base_url=llm_cfg.get("base_url", "http://localhost:8080"),
            llm_timeout=llm_cfg.get("timeout", 300.0),
            llm_model=llm_cfg.get("model", "gemma-4-31b"),
            rag_db_dsn=rag_cfg.get("db_dsn", "dbname=atlas user=claude"),
            rag_embed_model=rag_cfg.get("embed_model", "BAAI/bge-m3"),
            rag_top_k=rag_cfg.get("top_k", 6),
            enable_rag=rag_cfg.get("enabled", True),
            enable_cot=lh.get("enable_cot", True),
        )


# -----------------------------------------------------------------
# Telemetry tracker
# -----------------------------------------------------------------

@dataclass
class LHTelemetry:
    """Accumulates LH service metrics."""

    requests_processed: int = 0
    total_latency_ms: float = 0.0
    last_hemisphere_decision: str = "lh"
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
            "last_hemisphere_decision": self.last_hemisphere_decision,
            "last_query_preview": self.last_query[:80] if self.last_query else "",
            "errors": self.errors,
        }


# -----------------------------------------------------------------
# LH NATS Service
# -----------------------------------------------------------------

class LHNatsService:
    """Left Hemisphere as a NATS-connected cognitive service.

    Subscribes to request subjects, processes queries through RAG +
    LLM pipeline, and publishes responses back through the event fabric.

    Usage::

        service = LHNatsService()
        await service.start()
        # ... service runs until stopped ...
        await service.stop()
    """

    # Subjects this service listens on
    REQUEST_SUBJECTS = [
        "agi.lh.request.chat",
        "agi.lh.request.plan",
        "agi.lh.request.reason",
    ]

    def __init__(
        self,
        config: Optional[LHNatsServiceConfig] = None,
    ) -> None:
        self._config = config or LHNatsServiceConfig()
        self._fabric: Optional[NatsEventFabric] = None
        self._llm: Optional[LLMClient] = None
        self._rag: Optional[RAGSearcher] = None
        self._templates = PromptTemplateRegistry()
        self._telemetry = LHTelemetry()
        self._running = False

    async def start(self) -> None:
        """Connect to NATS and start processing requests."""
        logger.info("[lh-service] starting Phase 1 LH NATS service")

        # Initialise NATS fabric
        fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
        self._fabric = NatsEventFabric(config=fabric_config)
        await self._fabric.connect()

        # Initialise LLM client
        self._llm = LLMClient(
            base_url=self._config.llm_base_url,
            timeout=self._config.llm_timeout,
            default_model=self._config.llm_model,
        )

        # Initialise RAG searcher
        if self._config.enable_rag:
            rag_config = RAGConfig(
                db_dsn=self._config.rag_db_dsn,
                embed_model_name=self._config.rag_embed_model,
                top_k=self._config.rag_top_k,
            )
            self._rag = RAGSearcher(config=rag_config)
            logger.info("[lh-service] RAG searcher enabled")

        # Subscribe to request subjects
        for subject in self.REQUEST_SUBJECTS:
            await self._fabric.subscribe(subject, self._handle_request)

        self._running = True
        logger.info(
            "[lh-service] ready -- subscribed to %s",
            ", ".join(self.REQUEST_SUBJECTS),
        )

    async def stop(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        if self._llm:
            await self._llm.close()
        if self._fabric:
            await self._fabric.disconnect()
        logger.info("[lh-service] stopped")

    async def _handle_request(self, event: Event) -> None:
        """Process an incoming LH request event.

        Expected payload keys:
            prompt (str): The user query.
            messages (list, optional): Full message list override.
            config (dict, optional): InferenceConfig overrides.
            skip_rag (bool, optional): Skip RAG augmentation.
        """
        t0 = time.perf_counter()
        request_type = event.type.split(".")[-1] if event.type else "chat"
        prompt = event.payload.get("prompt", "")
        trace_id = event.trace_id

        logger.info(
            "[lh-service] request type=%s trace=%s prompt=%r",
            request_type,
            trace_id[:8],
            prompt[:60],
        )

        try:
            # Determine hemisphere routing
            hemisphere = classify_hemisphere(prompt)
            self._telemetry.last_hemisphere_decision = hemisphere
            self._telemetry.last_query = prompt

            # RAG retrieval
            rag_context = ""
            rag_results: List[RAGResult] = []
            if (
                self._rag
                and self._config.enable_rag
                and not event.payload.get("skip_rag", False)
            ):
                rag_results = self._rag.search(prompt)
                rag_context = self._rag.format_context(rag_results)

            # Build system prompt from template
            system_prompt = self._templates.render(
                "lh_analytical",
                model_name=self._config.llm_model,
                rag_context=rag_context,
            )

            # Build inference config
            user_config = event.payload.get("config", {})
            inference_config = InferenceConfig(
                temperature=user_config.get("temperature", LH_PRESET.temperature),
                top_p=user_config.get("top_p", LH_PRESET.top_p),
                max_tokens=user_config.get("max_tokens", LH_PRESET.max_tokens),
                system_prompt=system_prompt,
            )

            # Call LLM
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
                source="lh",
                event_type=f"lh.response.{request_type}",
                payload={
                    "text": result.text,
                    "model": result.model,
                    "tokens_used": result.tokens_used,
                    "latency_ms": round(latency_ms, 1),
                    "hemisphere": hemisphere,
                    "rag_results_count": len(rag_results),
                    "finish_reason": result.finish_reason,
                },
                trace_id=trace_id,
            )
            await self._fabric.publish(
                f"agi.lh.response.{request_type}", response_event
            )

            # Publish chain-of-thought trace for metacognition
            if self._config.enable_cot:
                cot_event = Event.create(
                    source="lh",
                    event_type="lh.internal.cot",
                    payload={
                        "request_type": request_type,
                        "prompt_preview": prompt[:200],
                        "response_preview": result.text[:500],
                        "rag_scores": [
                            {"repo": r.repo, "file": r.file, "score": r.score}
                            for r in rag_results[:3]
                        ],
                        "inference_config": {
                            "temperature": inference_config.temperature,
                            "max_tokens": inference_config.max_tokens,
                        },
                        "hemisphere_decision": hemisphere,
                        "latency_ms": round(latency_ms, 1),
                        "tokens_used": result.tokens_used,
                    },
                    trace_id=trace_id,
                )
                await self._fabric.publish("agi.lh.internal.cot", cot_event)

            # Publish telemetry
            telemetry_event = Event.create(
                source="lh",
                event_type="meta.monitor.lh",
                payload=self._telemetry.to_dict(),
                trace_id=trace_id,
            )
            await self._fabric.publish("agi.meta.monitor.lh", telemetry_event)

            logger.info(
                "[lh-service] response sent trace=%s latency=%.0fms tokens=%d",
                trace_id[:8],
                latency_ms,
                result.tokens_used,
            )

        except Exception:
            self._telemetry.errors += 1
            logger.exception(
                "[lh-service] error processing request trace=%s",
                trace_id[:8],
            )
            # Publish error response
            if self._fabric and self._fabric.is_connected:
                error_event = Event.create(
                    source="lh",
                    event_type=f"lh.response.{request_type}",
                    payload={
                        "error": True,
                        "message": "Internal LH service error",
                    },
                    trace_id=trace_id,
                )
                await self._fabric.publish(
                    f"agi.lh.response.{request_type}", error_event
                )

    @property
    def telemetry(self) -> LHTelemetry:
        """Return current telemetry snapshot."""
        return self._telemetry


# -----------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------

async def run_service(config_path: Optional[str] = None) -> None:
    """Run the LH NATS service until interrupted."""
    if config_path:
        config = LHNatsServiceConfig.from_yaml(config_path)
    else:
        config = LHNatsServiceConfig()

    service = LHNatsService(config=config)
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
    """CLI entry point for the LH NATS service."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="AGI-HPC LH NATS Service (Phase 1)")
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to lh_config.yaml",
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
        logger.info("[lh-service] interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
