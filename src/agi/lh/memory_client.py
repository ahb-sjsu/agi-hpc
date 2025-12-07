"""
MemoryClient for the Left Hemisphere.

Implements the memory queries described in:

- Memory Architecture (VI.A–C)
- Semantic/Episodic/Procedural API (XIV.B.2)

MemoryClient supports:
    • Semantic augmentation of PlanRequest
    • Episodic retrieval for similar past tasks
    • Procedural skill lookup (not yet implemented)

During early development, this client simply returns the input request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import grpc

from agi.proto_gen import (
    memory_pb2,
    memory_pb2_grpc,
)  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


@dataclass
class MemoryAugmentResult:
    """
    Placeholder result container for enriched request information.
    """

    enriched_request: object


class MemoryClient:
    """
    Interfaces with:

        SemanticMemoryService
        EpisodicMemoryService
        ProceduralMemoryService

    Expected RPCs (per whitepaper Section XIV.B.2):

        rpc SemanticQuery(SemanticQuery) returns (SemanticResult)
        rpc EpisodicQuery(EpisodicQuery) returns (EpisodicResult)
        rpc SkillSearch(SkillQuery) returns (SkillResult)

    This stub focuses only on request enrichment: looking up facts,
    tools, relevant episodes, etc.
    """

    def __init__(self, address: str = "semantic:50110") -> None:
        self._address = address
        try:
            self._channel = grpc.insecure_channel(address)
            self._stub = memory_pb2_grpc.SemanticMemoryServiceStub(self._channel)
            logger.info(
                "[LH][MemoryClient] Connected to semantic memory at %s", address
            )
        except Exception:
            self._stub = None
            logger.exception(
                "[LH][MemoryClient] MemoryService unavailable; falling back to passthrough mode"
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enrich_request(self, request) -> object:
        """
        Augment the PlanRequest with semantic/episodic information.

        Eventually this should query:
            • semantic memory (facts, tool schemas)
            • episodic memory (similar past tasks)
            • procedural memory (relevant skills)

        For now, it simply returns the original request.
        """
        if self._stub is None:
            logger.debug(
                "[LH][MemoryClient] Semantic memory unavailable; returning request unchanged"
            )
            return request

        try:
            # TODO: real query based on request.task
            # Example: lookup domain facts
            query = memory_pb2.SemanticQuery(  # type: ignore[attr-defined]
                text=getattr(getattr(request, "task", None), "description", "")
            )
            _ = self._stub.SemanticSearch(query)  # ignore result for now
        except Exception:
            logger.exception("[LH][MemoryClient] semantic lookup failed")

        return request
