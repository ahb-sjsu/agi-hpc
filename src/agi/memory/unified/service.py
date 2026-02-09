# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Unified Memory Service for AGI-HPC.

Implements Sprint 4 requirements:
- EnrichPlanningContext() RPC for combined memory queries
- Parallel queries to Semantic, Episodic, Procedural memory
- Result aggregation and ranking
- Tool schema resolution

This service acts as a facade over the three specialized memory services,
providing a single point of access for the LH planner to gather all
relevant context for plan generation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import grpc

from agi.proto_gen import memory_pb2, memory_pb2_grpc

if TYPE_CHECKING:
    from agi.memory.semantic.client import SemanticMemoryClient
    from agi.memory.episodic.client import EpisodicMemoryClient
    from agi.memory.procedural.client import ProceduralMemoryClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class UnifiedMemoryConfig:
    """
    Configuration for Unified Memory Service.

    Environment variables:
        AGI_SEMANTIC_MEMORY_ADDR: Address of Semantic Memory service
        AGI_EPISODIC_MEMORY_ADDR: Address of Episodic Memory service
        AGI_PROCEDURAL_MEMORY_ADDR: Address of Procedural Memory service
        AGI_MEMORY_TIMEOUT_SEC: Timeout for memory queries
    """

    semantic_addr: str = "localhost:50053"
    episodic_addr: str = "localhost:50052"
    procedural_addr: str = "localhost:50054"
    timeout_sec: float = 5.0
    default_max_facts: int = 10
    default_max_episodes: int = 5
    default_max_skills: int = 10
    enable_caching: bool = True
    cache_ttl_sec: int = 300


# ---------------------------------------------------------------------------
# Planning Context
# ---------------------------------------------------------------------------


@dataclass
class PlanningContext:
    """
    Aggregated planning context from all memory types.

    This is the internal representation used by the LH planner
    to enrich its planning prompts with relevant context.
    """

    facts: List[Dict[str, Any]] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    tool_schemas: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        """Check if any context is available."""
        return bool(self.facts or self.episodes or self.skills)

    def to_prompt_context(self) -> str:
        """
        Format context for LLM prompt injection.

        Returns:
            Formatted string suitable for inclusion in planning prompts.
        """
        parts = []

        if self.facts:
            parts.append("RELEVANT FACTS:")
            for fact in self.facts[:5]:
                parts.append(
                    f"  - {fact.get('content', '')} (confidence: {fact.get('confidence', 0):.2f})"
                )

        if self.episodes:
            parts.append("\nSIMILAR PAST EPISODES:")
            for ep in self.episodes[:3]:
                outcome = "succeeded" if ep.get("success") else "failed"
                parts.append(f"  - {ep.get('task_description', '')} ({outcome})")
                if ep.get("insights"):
                    for insight in ep["insights"][:2]:
                        parts.append(f"    Insight: {insight}")

        if self.skills:
            parts.append("\nAVAILABLE SKILLS:")
            for skill in self.skills[:5]:
                prof = skill.get("proficiency", 0)
                parts.append(
                    f"  - {skill.get('name', '')}: {skill.get('description', '')} (proficiency: {prof:.2f})"
                )

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Unified Memory Service (Client)
# ---------------------------------------------------------------------------


class UnifiedMemoryService:
    """
    Client interface for unified memory queries.

    Provides a simple Python API for querying all memory types
    and aggregating results for planning context.

    Usage:
        service = UnifiedMemoryService(config=config)
        context = await service.enrich_planning_context(
            task_description="Navigate to the red cube",
            task_type="navigation",
        )
        print(context.to_prompt_context())
    """

    def __init__(
        self,
        config: Optional[UnifiedMemoryConfig] = None,
        semantic_client: Optional["SemanticMemoryClient"] = None,
        episodic_client: Optional["EpisodicMemoryClient"] = None,
        procedural_client: Optional["ProceduralMemoryClient"] = None,
    ) -> None:
        self._config = config or UnifiedMemoryConfig()
        self._semantic = semantic_client
        self._episodic = episodic_client
        self._procedural = procedural_client

        # Cache for frequent queries
        self._cache: Dict[str, PlanningContext] = {}

        logger.info("[Memory][Unified] service initialized")

    async def enrich_planning_context(
        self,
        task_description: str,
        task_type: str = "",
        scenario_id: str = "",
        include_semantic: bool = True,
        include_episodic: bool = True,
        include_procedural: bool = True,
        max_facts: int = 0,
        max_episodes: int = 0,
        max_skills: int = 0,
    ) -> PlanningContext:
        """
        Query all memory types for planning context.

        Args:
            task_description: Description of the current task
            task_type: Type of task (navigation, manipulation, etc.)
            scenario_id: Current scenario/environment identifier
            include_semantic: Whether to query semantic memory
            include_episodic: Whether to query episodic memory
            include_procedural: Whether to query procedural memory
            max_facts: Maximum facts to retrieve (0 = use default)
            max_episodes: Maximum episodes to retrieve (0 = use default)
            max_skills: Maximum skills to retrieve (0 = use default)

        Returns:
            PlanningContext with aggregated results
        """
        # Use defaults if not specified
        max_facts = max_facts or self._config.default_max_facts
        max_episodes = max_episodes or self._config.default_max_episodes
        max_skills = max_skills or self._config.default_max_skills

        # Check cache
        cache_key = f"{task_description}:{task_type}:{scenario_id}"
        if self._config.enable_caching and cache_key in self._cache:
            logger.debug("[Memory][Unified] cache hit for %s", cache_key[:50])
            return self._cache[cache_key]

        logger.info(
            "[Memory][Unified] enriching context for task=%s type=%s",
            task_description[:50],
            task_type,
        )

        # Build async tasks for parallel queries
        tasks = []

        if include_semantic:
            tasks.append(self._query_semantic(task_description, max_facts))
        else:
            tasks.append(asyncio.coroutine(lambda: [])())

        if include_episodic:
            tasks.append(
                self._query_episodic(task_description, task_type, max_episodes)
            )
        else:
            tasks.append(asyncio.coroutine(lambda: [])())

        if include_procedural:
            tasks.append(
                self._query_procedural(task_description, task_type, max_skills)
            )
        else:
            tasks.append(asyncio.coroutine(lambda: [])())

        # Execute queries in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.warning("[Memory][Unified] parallel query failed: %s", e)
            results = [[], [], []]

        # Unpack results
        facts = results[0] if not isinstance(results[0], Exception) else []
        episodes = results[1] if not isinstance(results[1], Exception) else []
        skills = results[2] if not isinstance(results[2], Exception) else []

        # Build planning context
        context = PlanningContext(
            facts=facts,
            episodes=episodes,
            skills=skills,
        )

        # Extract tool schemas from skills
        tool_ids = set()
        for skill in skills:
            for action in skill.get("actions", []):
                if action.get("tool_id"):
                    tool_ids.add(action["tool_id"])

        if tool_ids and include_semantic:
            context.tool_schemas = await self._resolve_tool_schemas(list(tool_ids))

        # Cache result
        if self._config.enable_caching:
            self._cache[cache_key] = context

        logger.info(
            "[Memory][Unified] context enriched: %d facts, %d episodes, %d skills",
            len(context.facts),
            len(context.episodes),
            len(context.skills),
        )

        return context

    async def _query_semantic(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Query semantic memory for relevant facts."""
        if self._semantic is None:
            return self._stub_semantic_query(query, max_results)

        try:
            result = await self._semantic.search(
                text=query,
                max_results=max_results,
            )
            return [self._fact_to_dict(f) for f in result.facts]
        except Exception as e:
            logger.warning("[Memory][Unified] semantic query failed: %s", e)
            return []

    async def _query_episodic(
        self,
        query: str,
        task_type: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Query episodic memory for similar episodes."""
        if self._episodic is None:
            return self._stub_episodic_query(query, task_type, max_results)

        try:
            result = await self._episodic.search(
                situation_description=query,
                task_type=task_type,
                max_results=max_results,
            )
            return [self._episode_to_dict(e) for e in result.episodes]
        except Exception as e:
            logger.warning("[Memory][Unified] episodic query failed: %s", e)
            return []

    async def _query_procedural(
        self,
        query: str,
        task_type: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Query procedural memory for relevant skills."""
        if self._procedural is None:
            return self._stub_procedural_query(query, task_type, max_results)

        try:
            result = await self._procedural.search(
                capability_description=query,
                task_type=task_type,
                max_results=max_results,
            )
            return [self._skill_to_dict(s) for s in result.skills]
        except Exception as e:
            logger.warning("[Memory][Unified] procedural query failed: %s", e)
            return []

    async def _resolve_tool_schemas(
        self,
        tool_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Resolve tool schemas from semantic memory."""
        if self._semantic is None:
            return []

        schemas = []
        for tool_id in tool_ids[:10]:  # Limit to prevent too many queries
            try:
                schema = await self._semantic.get_tool_schema(tool_id)
                if schema:
                    schemas.append(self._schema_to_dict(schema))
            except Exception as e:
                logger.debug("[Memory][Unified] tool schema lookup failed: %s", e)

        return schemas

    # ------------------------------------------------------------------ #
    # Stub implementations (used when clients are not available)
    # ------------------------------------------------------------------ #

    def _stub_semantic_query(
        self,
        query: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Return stub facts when semantic memory is not available."""
        return [
            {
                "fact_id": "stub_fact_1",
                "content": f"Placeholder fact related to: {query[:30]}",
                "confidence": 0.8,
                "similarity": 0.7,
                "source": "stub",
                "domains": ["general"],
            }
        ]

    def _stub_episodic_query(
        self,
        query: str,
        task_type: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Return stub episodes when episodic memory is not available."""
        return [
            {
                "episode_id": "stub_episode_1",
                "task_description": f"Past task similar to: {query[:30]}",
                "task_type": task_type or "general",
                "success": True,
                "completion_percentage": 0.95,
                "insights": ["Consider verifying preconditions before execution"],
                "similarity": 0.6,
            }
        ]

    def _stub_procedural_query(
        self,
        query: str,
        task_type: str,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Return stub skills when procedural memory is not available."""
        return [
            {
                "skill_id": "stub_skill_1",
                "name": "generic_skill",
                "description": f"Generic skill for: {query[:30]}",
                "category": task_type or "general",
                "proficiency": 0.7,
                "success_rate": 0.8,
                "execution_count": 10,
                "actions": [],
            }
        ]

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fact_to_dict(fact: Any) -> Dict[str, Any]:
        """Convert protobuf SemanticFact to dict."""
        return {
            "fact_id": fact.fact_id,
            "content": fact.content,
            "confidence": fact.confidence,
            "similarity": fact.similarity,
            "source": fact.source,
            "domains": list(fact.domains),
        }

    @staticmethod
    def _episode_to_dict(episode: Any) -> Dict[str, Any]:
        """Convert protobuf Episode to dict."""
        return {
            "episode_id": episode.episode_id,
            "task_description": episode.task_description,
            "task_type": episode.task_type,
            "scenario_id": episode.scenario_id,
            "success": (
                episode.outcome.success if episode.HasField("outcome") else False
            ),
            "completion_percentage": (
                episode.outcome.completion_percentage
                if episode.HasField("outcome")
                else 0.0
            ),
            "insights": list(episode.insights),
            "similarity": episode.similarity,
        }

    @staticmethod
    def _skill_to_dict(skill: Any) -> Dict[str, Any]:
        """Convert protobuf Skill to dict."""
        return {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "proficiency": skill.proficiency,
            "success_rate": skill.success_rate,
            "execution_count": skill.execution_count,
            "preconditions": list(skill.preconditions),
            "postconditions": list(skill.postconditions),
            "actions": [
                {
                    "sequence": a.sequence,
                    "action_type": a.action_type,
                    "tool_id": a.tool_id,
                    "parameters": dict(a.parameters),
                }
                for a in skill.actions
            ],
            "similarity": skill.similarity,
        }

    @staticmethod
    def _schema_to_dict(schema: Any) -> Dict[str, Any]:
        """Convert protobuf ToolSchema to dict."""
        return {
            "tool_id": schema.tool_id,
            "name": schema.name,
            "description": schema.description,
            "parameters": [
                {
                    "name": p.name,
                    "param_type": p.param_type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in schema.parameters
            ],
            "preconditions": list(schema.preconditions),
            "postconditions": list(schema.postconditions),
        }


# ---------------------------------------------------------------------------
# gRPC Servicer
# ---------------------------------------------------------------------------


class UnifiedMemoryServicer(memory_pb2_grpc.UnifiedMemoryServiceServicer):
    """
    gRPC servicer for UnifiedMemoryService.

    Implements the EnrichPlanningContext RPC by delegating to
    the underlying UnifiedMemoryService.
    """

    def __init__(
        self,
        service: Optional[UnifiedMemoryService] = None,
        config: Optional[UnifiedMemoryConfig] = None,
    ) -> None:
        self._service = service or UnifiedMemoryService(config=config)

    def EnrichPlanningContext(
        self,
        request: memory_pb2.PlanningContextRequest,
        context: grpc.ServicerContext,
    ) -> memory_pb2.PlanningContextResponse:
        """
        Handle EnrichPlanningContext RPC.

        Queries all memory types based on request parameters and
        returns aggregated planning context.
        """
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            planning_context = loop.run_until_complete(
                self._service.enrich_planning_context(
                    task_description=request.task_description,
                    task_type=request.task_type,
                    scenario_id=request.scenario_id,
                    include_semantic=request.include_semantic,
                    include_episodic=request.include_episodic,
                    include_procedural=request.include_procedural,
                    max_facts=request.max_facts,
                    max_episodes=request.max_episodes,
                    max_skills=request.max_skills,
                )
            )
        finally:
            loop.close()

        # Build response
        response = memory_pb2.PlanningContextResponse()

        for fact in planning_context.facts:
            pb_fact = memory_pb2.SemanticFact(
                fact_id=fact.get("fact_id", ""),
                content=fact.get("content", ""),
                confidence=fact.get("confidence", 0.0),
                similarity=fact.get("similarity", 0.0),
                source=fact.get("source", ""),
                domains=fact.get("domains", []),
            )
            response.facts.append(pb_fact)

        for ep in planning_context.episodes:
            pb_episode = memory_pb2.Episode(
                episode_id=ep.get("episode_id", ""),
                task_description=ep.get("task_description", ""),
                task_type=ep.get("task_type", ""),
                scenario_id=ep.get("scenario_id", ""),
                similarity=ep.get("similarity", 0.0),
                insights=ep.get("insights", []),
            )
            if "success" in ep:
                pb_episode.outcome.success = ep["success"]
                pb_episode.outcome.completion_percentage = ep.get(
                    "completion_percentage", 0.0
                )
            response.episodes.append(pb_episode)

        for skill in planning_context.skills:
            pb_skill = memory_pb2.Skill(
                skill_id=skill.get("skill_id", ""),
                name=skill.get("name", ""),
                description=skill.get("description", ""),
                category=skill.get("category", ""),
                proficiency=skill.get("proficiency", 0.0),
                success_rate=skill.get("success_rate", 0.0),
                execution_count=skill.get("execution_count", 0),
                preconditions=skill.get("preconditions", []),
                postconditions=skill.get("postconditions", []),
                similarity=skill.get("similarity", 0.0),
            )
            response.skills.append(pb_skill)

        for schema in planning_context.tool_schemas:
            pb_schema = memory_pb2.ToolSchema(
                tool_id=schema.get("tool_id", ""),
                name=schema.get("name", ""),
                description=schema.get("description", ""),
                preconditions=schema.get("preconditions", []),
                postconditions=schema.get("postconditions", []),
            )
            for param in schema.get("parameters", []):
                pb_param = memory_pb2.ToolParameter(
                    name=param.get("name", ""),
                    param_type=param.get("param_type", ""),
                    description=param.get("description", ""),
                    required=param.get("required", False),
                )
                pb_schema.parameters.append(pb_param)
            response.tool_schemas.append(pb_schema)

        return response
