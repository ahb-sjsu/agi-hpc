# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
MemoryClient for the Left Hemisphere.

Implements memory queries for planning context enrichment:
    - Semantic memory: facts, concepts, tool schemas
    - Episodic memory: similar past tasks, lessons learned
    - Procedural memory: skills, action sequences

When memory services are unavailable, the client operates in passthrough
mode, returning empty context but not blocking planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import grpc

from agi.proto_gen import memory_pb2, memory_pb2_grpc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory Context Result
# ---------------------------------------------------------------------------


@dataclass
class SemanticFact:
    """A fact retrieved from semantic memory."""

    fact_id: str
    content: str
    confidence: float = 1.0
    similarity: float = 0.0
    source: str = ""
    domains: List[str] = field(default_factory=list)


@dataclass
class Episode:
    """A past episode retrieved from episodic memory."""

    episode_id: str
    task_description: str
    task_type: str = ""
    scenario_id: str = ""
    success: bool = False
    similarity: float = 0.0
    insights: List[str] = field(default_factory=list)
    plan_steps: List[str] = field(default_factory=list)


@dataclass
class Skill:
    """A skill retrieved from procedural memory."""

    skill_id: str
    name: str
    description: str
    category: str = ""
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    proficiency: float = 0.0
    success_rate: float = 0.0
    similarity: float = 0.0


@dataclass
class ToolSchema:
    """Schema for a tool/action available in the system."""

    tool_id: str
    name: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class PlanningContext:
    """
    Enriched context for planning, gathered from all memory types.

    This is passed to the planner to inform plan generation with:
    - Relevant domain facts
    - Similar past experiences
    - Available skills
    - Tool schemas
    """

    facts: List[SemanticFact] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    skills: List[Skill] = field(default_factory=list)
    tool_schemas: List[ToolSchema] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        """Check if any context was retrieved."""
        return bool(self.facts or self.episodes or self.skills)

    def to_prompt_context(self) -> str:
        """
        Format the context for inclusion in an LLM prompt.

        Returns a structured text representation of the memory context.
        """
        sections = []

        if self.facts:
            facts_text = "\n".join(
                f"  - {f.content} (confidence: {f.confidence:.2f})" for f in self.facts
            )
            sections.append(f"RELEVANT FACTS:\n{facts_text}")

        if self.episodes:
            episodes_text = "\n".join(
                f"  - {e.task_description} ({'success' if e.success else 'failed'}, "
                f"similarity: {e.similarity:.2f})"
                + (f"\n    Insights: {', '.join(e.insights)}" if e.insights else "")
                for e in self.episodes
            )
            sections.append(f"SIMILAR PAST TASKS:\n{episodes_text}")

        if self.skills:
            skills_text = "\n".join(
                f"  - {s.name}: {s.description} "
                f"(proficiency: {s.proficiency:.2f}, success: {s.success_rate:.2f})"
                for s in self.skills
            )
            sections.append(f"AVAILABLE SKILLS:\n{skills_text}")

        if self.tool_schemas:
            tools_text = "\n".join(
                f"  - {t.tool_id}: {t.description}" for t in self.tool_schemas
            )
            sections.append(f"AVAILABLE TOOLS:\n{tools_text}")

        return "\n\n".join(sections) if sections else ""


# ---------------------------------------------------------------------------
# Memory Client Configuration
# ---------------------------------------------------------------------------


@dataclass
class MemoryClientConfig:
    """Configuration for memory service connections."""

    semantic_address: str = "localhost:50110"
    episodic_address: str = "localhost:50111"
    procedural_address: str = "localhost:50112"
    unified_address: str = "localhost:50113"
    timeout_seconds: float = 5.0
    use_unified: bool = True  # Use unified service if available
    max_facts: int = 10
    max_episodes: int = 5
    max_skills: int = 10
    min_similarity: float = 0.5


# ---------------------------------------------------------------------------
# Memory Client
# ---------------------------------------------------------------------------


class MemoryClient:
    """
    Client for querying memory subsystems.

    Supports three modes:
    1. Unified: Use UnifiedMemoryService for single-call enrichment
    2. Individual: Query each memory service separately
    3. Passthrough: Return empty context when services unavailable
    """

    def __init__(
        self,
        config: Optional[MemoryClientConfig] = None,
        semantic_address: Optional[str] = None,
    ) -> None:
        self._config = config or MemoryClientConfig()

        # Override with legacy parameter
        if semantic_address:
            self._config.semantic_address = semantic_address

        # Initialize stubs
        self._semantic_stub: Optional[memory_pb2_grpc.SemanticMemoryServiceStub] = None
        self._episodic_stub: Optional[memory_pb2_grpc.EpisodicMemoryServiceStub] = None
        self._procedural_stub: Optional[memory_pb2_grpc.ProceduralMemoryServiceStub] = (
            None
        )
        self._unified_stub: Optional[memory_pb2_grpc.UnifiedMemoryServiceStub] = None

        self._channels: List[grpc.Channel] = []
        self._connect()

    def _connect(self) -> None:
        """Establish connections to memory services."""
        # Try unified service first
        if self._config.use_unified:
            try:
                channel = grpc.insecure_channel(self._config.unified_address)
                self._unified_stub = memory_pb2_grpc.UnifiedMemoryServiceStub(channel)
                self._channels.append(channel)
                logger.info(
                    "[LH][MemoryClient] Connected to unified memory at %s",
                    self._config.unified_address,
                )
            except Exception:
                logger.debug(
                    "[LH][MemoryClient] Unified memory unavailable at %s",
                    self._config.unified_address,
                )

        # Connect to individual services
        try:
            channel = grpc.insecure_channel(self._config.semantic_address)
            self._semantic_stub = memory_pb2_grpc.SemanticMemoryServiceStub(channel)
            self._channels.append(channel)
            logger.info(
                "[LH][MemoryClient] Connected to semantic memory at %s",
                self._config.semantic_address,
            )
        except Exception:
            logger.debug(
                "[LH][MemoryClient] Semantic memory unavailable at %s",
                self._config.semantic_address,
            )

        try:
            channel = grpc.insecure_channel(self._config.episodic_address)
            self._episodic_stub = memory_pb2_grpc.EpisodicMemoryServiceStub(channel)
            self._channels.append(channel)
            logger.info(
                "[LH][MemoryClient] Connected to episodic memory at %s",
                self._config.episodic_address,
            )
        except Exception:
            logger.debug(
                "[LH][MemoryClient] Episodic memory unavailable at %s",
                self._config.episodic_address,
            )

        try:
            channel = grpc.insecure_channel(self._config.procedural_address)
            self._procedural_stub = memory_pb2_grpc.ProceduralMemoryServiceStub(channel)
            self._channels.append(channel)
            logger.info(
                "[LH][MemoryClient] Connected to procedural memory at %s",
                self._config.procedural_address,
            )
        except Exception:
            logger.debug(
                "[LH][MemoryClient] Procedural memory unavailable at %s",
                self._config.procedural_address,
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_planning_context(
        self,
        task_description: str,
        task_type: str = "",
        scenario_id: str = "",
    ) -> PlanningContext:
        """
        Get enriched planning context from all memory types.

        Args:
            task_description: Natural language description of the task
            task_type: Type of task (e.g., "navigation", "manipulation")
            scenario_id: Environment/scenario identifier

        Returns:
            PlanningContext with facts, episodes, and skills
        """
        # Try unified service first
        if self._unified_stub:
            try:
                return self._query_unified(task_description, task_type, scenario_id)
            except Exception as e:
                logger.warning(
                    "[LH][MemoryClient] Unified query failed: %s, trying individual",
                    e,
                )

        # Fall back to individual queries
        context = PlanningContext()

        # Query semantic memory
        if self._semantic_stub:
            try:
                context.facts = self._query_semantic(task_description)
            except Exception as e:
                logger.warning("[LH][MemoryClient] Semantic query failed: %s", e)

        # Query episodic memory
        if self._episodic_stub:
            try:
                context.episodes = self._query_episodic(
                    task_description, task_type, scenario_id
                )
            except Exception as e:
                logger.warning("[LH][MemoryClient] Episodic query failed: %s", e)

        # Query procedural memory
        if self._procedural_stub:
            try:
                context.skills = self._query_procedural(task_description, task_type)
            except Exception as e:
                logger.warning("[LH][MemoryClient] Procedural query failed: %s", e)

        return context

    def enrich_request(self, request) -> object:
        """
        Legacy API: Augment the PlanRequest with memory context.

        For backwards compatibility. New code should use get_planning_context().

        Args:
            request: PlanRequest protobuf message

        Returns:
            The original request (context is retrieved separately)
        """
        # Extract task info from request
        task = getattr(request, "task", None)
        env = getattr(request, "environment", None)

        task_description = getattr(task, "description", "") if task else ""
        task_type = getattr(task, "task_type", "") if task else ""
        scenario_id = getattr(env, "scenario_id", "") if env else ""

        if task_description:
            # Trigger context retrieval (side effect: logs and caches)
            context = self.get_planning_context(
                task_description, task_type, scenario_id
            )
            if context.has_context:
                logger.info(
                    "[LH][MemoryClient] Retrieved context: %d facts, %d episodes, %d skills",
                    len(context.facts),
                    len(context.episodes),
                    len(context.skills),
                )

        return request

    def get_tool_schema(self, tool_id: str) -> Optional[ToolSchema]:
        """
        Get schema for a specific tool.

        Args:
            tool_id: Tool identifier (e.g., "lh.task_interpreter")

        Returns:
            ToolSchema if found, None otherwise
        """
        if not self._semantic_stub:
            return None

        try:
            request = memory_pb2.ToolSchemaRequest(tool_id=tool_id)
            response = self._semantic_stub.GetToolSchema(
                request, timeout=self._config.timeout_seconds
            )
            if response.schema:
                return self._convert_tool_schema(response.schema)
        except Exception as e:
            logger.warning(
                "[LH][MemoryClient] Tool schema lookup failed for %s: %s",
                tool_id,
                e,
            )

        return None

    # ------------------------------------------------------------------ #
    # Internal query methods
    # ------------------------------------------------------------------ #

    def _query_unified(
        self,
        task_description: str,
        task_type: str,
        scenario_id: str,
    ) -> PlanningContext:
        """Query unified memory service."""
        request = memory_pb2.PlanningContextRequest(
            task_description=task_description,
            task_type=task_type,
            scenario_id=scenario_id,
            include_semantic=True,
            include_episodic=True,
            include_procedural=True,
            max_facts=self._config.max_facts,
            max_episodes=self._config.max_episodes,
            max_skills=self._config.max_skills,
        )

        response = self._unified_stub.EnrichPlanningContext(
            request, timeout=self._config.timeout_seconds
        )

        return PlanningContext(
            facts=[self._convert_fact(f) for f in response.facts],
            episodes=[self._convert_episode(e) for e in response.episodes],
            skills=[self._convert_skill(s) for s in response.skills],
            tool_schemas=[self._convert_tool_schema(t) for t in response.tool_schemas],
        )

    def _query_semantic(self, text: str) -> List[SemanticFact]:
        """Query semantic memory for relevant facts."""
        request = memory_pb2.SemanticQuery(
            text=text,
            max_results=self._config.max_facts,
            min_similarity=self._config.min_similarity,
        )

        response = self._semantic_stub.SemanticSearch(
            request, timeout=self._config.timeout_seconds
        )

        return [self._convert_fact(f) for f in response.facts]

    def _query_episodic(
        self,
        situation: str,
        task_type: str,
        scenario_id: str,
    ) -> List[Episode]:
        """Query episodic memory for similar past tasks."""
        request = memory_pb2.EpisodicQuery(
            situation_description=situation,
            task_type=task_type,
            scenario_id=scenario_id,
            max_results=self._config.max_episodes,
            min_similarity=self._config.min_similarity,
            success_only=False,  # Learn from failures too
        )

        response = self._episodic_stub.EpisodicSearch(
            request, timeout=self._config.timeout_seconds
        )

        return [self._convert_episode(e) for e in response.episodes]

    def _query_procedural(
        self,
        capability: str,
        task_type: str,
    ) -> List[Skill]:
        """Query procedural memory for relevant skills."""
        request = memory_pb2.SkillQuery(
            capability_description=capability,
            task_type=task_type,
            max_results=self._config.max_skills,
            min_proficiency=0.0,  # Include all skills
        )

        response = self._procedural_stub.SkillSearch(
            request, timeout=self._config.timeout_seconds
        )

        return [self._convert_skill(s) for s in response.skills]

    # ------------------------------------------------------------------ #
    # Conversion helpers
    # ------------------------------------------------------------------ #

    def _convert_fact(self, proto: memory_pb2.SemanticFact) -> SemanticFact:
        """Convert protobuf fact to dataclass."""
        return SemanticFact(
            fact_id=proto.fact_id,
            content=proto.content,
            confidence=proto.confidence,
            similarity=proto.similarity,
            source=proto.source,
            domains=list(proto.domains),
        )

    def _convert_episode(self, proto: memory_pb2.Episode) -> Episode:
        """Convert protobuf episode to dataclass."""
        plan_steps = []
        if proto.plan:
            plan_steps = [s.description for s in proto.plan.steps]

        return Episode(
            episode_id=proto.episode_id,
            task_description=proto.task_description,
            task_type=proto.task_type,
            scenario_id=proto.scenario_id,
            success=proto.outcome.success if proto.outcome else False,
            similarity=proto.similarity,
            insights=list(proto.insights),
            plan_steps=plan_steps,
        )

    def _convert_skill(self, proto: memory_pb2.Skill) -> Skill:
        """Convert protobuf skill to dataclass."""
        return Skill(
            skill_id=proto.skill_id,
            name=proto.name,
            description=proto.description,
            category=proto.category,
            preconditions=list(proto.preconditions),
            postconditions=list(proto.postconditions),
            proficiency=proto.proficiency,
            success_rate=proto.success_rate,
            similarity=proto.similarity,
        )

    def _convert_tool_schema(self, proto: memory_pb2.ToolSchema) -> ToolSchema:
        """Convert protobuf tool schema to dataclass."""
        return ToolSchema(
            tool_id=proto.tool_id,
            name=proto.name,
            description=proto.description,
            parameters=[
                {
                    "name": p.name,
                    "type": p.param_type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in proto.parameters
            ],
            preconditions=list(proto.preconditions),
            postconditions=list(proto.postconditions),
        )

    def close(self) -> None:
        """Close all gRPC channels."""
        for channel in self._channels:
            try:
                channel.close()
            except Exception:
                pass
        self._channels.clear()
