# AGI-HPC Autonomous Research Loop
# Copyright (c) 2026 Andrew H. Bond
# License v1.0 — see LICENSE
"""Autonomous research cycle triggered by knowledge gaps.

When metacognition detects a gap (low memory hit rate, executive
inhibition, consistency contradictions), the research loop:
1. Formulates search queries
2. Searches existing knowledge
3. Identifies what is missing
4. Stores new findings
5. Re-evaluates confidence
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResearchLoopConfig:
    """Configuration for the autonomous research loop."""

    llm_url: str = "http://localhost:8082"
    max_cycles: int = 3
    min_confidence: float = 0.7
    cooldown_seconds: int = 300
    enabled: bool = True


@dataclass
class ResearchGoal:
    """A knowledge gap to investigate."""

    query: str
    source: str  # executive_function, anomaly_detector, user
    priority: int = 3  # 1 (low) to 5 (critical)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResearchResult:
    """Outcome of a research cycle."""

    goal: ResearchGoal
    findings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    cycles_used: int = 0
    elapsed_s: float = 0.0
    entities_added: int = 0
    relationships_added: int = 0


@dataclass
class ResearchTelemetry:
    """Telemetry for the research loop."""

    goals_detected: int = 0
    goals_completed: int = 0
    cycles_total: int = 0
    knowledge_added: int = 0
    avg_confidence: float = 0.0
    last_cycle: Optional[datetime] = None
    _confidence_sum: float = 0.0

    def record(self, result: ResearchResult) -> None:
        self.goals_completed += 1
        self.cycles_total += result.cycles_used
        self.knowledge_added += result.entities_added + result.relationships_added
        self._confidence_sum += result.confidence
        self.avg_confidence = self._confidence_sum / max(self.goals_completed, 1)
        self.last_cycle = datetime.now(timezone.utc)


class ResearchLoop:
    """Autonomous research cycle for filling knowledge gaps.

    Integrates with the knowledge graph and semantic memory
    to detect gaps and fill them through iterative search.
    """

    def __init__(
        self,
        config: Optional[ResearchLoopConfig] = None,
        knowledge_graph=None,
        semantic_memory=None,
        extractor=None,
    ) -> None:
        self._config = config or ResearchLoopConfig()
        self._graph = knowledge_graph
        self._semantic = semantic_memory
        self._extractor = extractor
        self._telemetry = ResearchTelemetry()
        self._last_cycle_time: float = 0
        self._gap_history: list[str] = []

    @property
    def telemetry(self) -> ResearchTelemetry:
        return self._telemetry

    async def research(self, goal: ResearchGoal) -> ResearchResult:
        """Execute a research cycle for one goal."""
        if not self._config.enabled:
            return ResearchResult(goal=goal, confidence=0.0)

        t0 = time.time()
        result = ResearchResult(goal=goal)

        for cycle in range(self._config.max_cycles):
            result.cycles_used = cycle + 1

            # Search existing knowledge
            existing = self._search_existing(goal.query)
            if existing:
                result.findings.extend(existing)

            # Check if we have enough
            confidence = self._assess_confidence(goal.query, result.findings)
            result.confidence = confidence
            if confidence >= self._config.min_confidence:
                break

            # Try to expand knowledge from what we found
            if self._extractor and result.findings:
                combined = "\n".join(result.findings)
                knowledge = self._extractor.extract_from_text(
                    combined, source=f"research:{goal.query}"
                )
                if self._graph and (knowledge.entities or knowledge.relationships):
                    self._graph.store(knowledge)
                    result.entities_added += len(knowledge.entities)
                    result.relationships_added += len(knowledge.relationships)

        result.elapsed_s = time.time() - t0
        self._telemetry.record(result)
        self._last_cycle_time = time.time()

        logger.info(
            "Research complete: %s — confidence=%.2f, %d findings, "
            "%d entities, %d rels, %d cycles, %.1fs",
            goal.query[:50],
            result.confidence,
            len(result.findings),
            result.entities_added,
            result.relationships_added,
            result.cycles_used,
            result.elapsed_s,
        )
        return result

    def _search_existing(self, query: str) -> list[str]:
        """Search semantic memory and knowledge graph for existing info."""
        findings = []

        if self._semantic:
            try:
                results = self._semantic.search(query, top_k=5)
                for r in results:
                    findings.append(r.text if hasattr(r, "text") else str(r))
            except Exception as e:
                logger.debug("Semantic search failed: %s", e)

        if self._graph:
            try:
                words = query.split()
                for word in words[:5]:
                    entity = self._graph.query_entity(word)
                    if entity:
                        findings.append(
                            f"{entity.name} ({entity.entity_type}): "
                            f"{entity.description}"
                        )
                    rels = self._graph.query_relationships(word)
                    for r in rels[:3]:
                        findings.append(f"{r.subject} {r.predicate} {r.object}")
            except Exception as e:
                logger.debug("Knowledge graph search failed: %s", e)

        return findings

    def _assess_confidence(self, query: str, findings: list[str]) -> float:
        """Estimate how well the findings answer the query."""
        if not findings:
            return 0.0
        query_words = set(query.lower().split())
        finding_words = set(" ".join(findings).lower().split())
        overlap = len(query_words & finding_words)
        coverage = overlap / max(len(query_words), 1)
        volume_bonus = min(len(findings) / 10, 0.3)
        return min(coverage + volume_bonus, 1.0)

    def detect_gaps(
        self,
        executive_decisions: Optional[list] = None,
        failed_lookups: Optional[list[str]] = None,
        memory_hit_rate: Optional[float] = None,
    ) -> list[ResearchGoal]:
        """Identify knowledge gaps from metacognition signals."""
        goals = []

        if executive_decisions:
            for decision in executive_decisions:
                if hasattr(decision, "inhibit") and decision.inhibit:
                    reason = getattr(decision, "inhibit_reason", "unknown gap")
                    if reason not in self._gap_history:
                        goals.append(
                            ResearchGoal(
                                query=reason,
                                source="executive_function",
                                priority=4,
                            )
                        )
                        self._gap_history.append(reason)

        if failed_lookups:
            for lookup in failed_lookups:
                if lookup not in self._gap_history:
                    goals.append(
                        ResearchGoal(
                            query=lookup,
                            source="procedural_memory",
                            priority=2,
                        )
                    )
                    self._gap_history.append(lookup)

        if memory_hit_rate is not None and memory_hit_rate < 0.3:
            goal_text = "general knowledge coverage is low"
            if goal_text not in self._gap_history:
                goals.append(
                    ResearchGoal(
                        query=goal_text,
                        source="anomaly_detector",
                        priority=1,
                    )
                )
                self._gap_history.append(goal_text)

        self._telemetry.goals_detected += len(goals)
        return goals

    async def run_idle_cycle(self, **gap_kwargs) -> list[ResearchResult]:
        """Run during idle time. Detect gaps, research top ones."""
        if not self._config.enabled:
            return []

        elapsed = time.time() - self._last_cycle_time
        if elapsed < self._config.cooldown_seconds:
            return []

        goals = self.detect_gaps(**gap_kwargs)
        if not goals:
            return []

        goals.sort(key=lambda g: g.priority, reverse=True)
        results = []
        for goal in goals[:3]:
            result = await self.research(goal)
            results.append(result)

        return results
