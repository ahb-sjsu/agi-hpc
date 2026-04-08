# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
LLM Integration Module for AGI-HPC.

Provides pre-configured LLM integration points for each subsystem:
- LH Planner: Plan generation and refinement
- Metacognition: Plan critique and explanation
- Memory: Embedding generation
- Safety: Fallback reasoning

Sprint 6 Implementation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from agi.core.llm.client import LLMClient
    from agi.core.llm.types import CompletionRequest, Message, MessageRole
except ImportError:
    LLMClient = None  # type: ignore[assignment,misc]
    CompletionRequest = None  # type: ignore[assignment,misc]
    Message = None  # type: ignore[assignment,misc]
    MessageRole = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LLMIntegrationConfig:
    """Configuration for LLM integrations across subsystems."""

    planner_model: str = "claude-3-5-sonnet-20241022"
    planner_temperature: float = 0.3
    planner_max_tokens: int = 4096
    metacog_model: str = "claude-3-5-sonnet-20241022"
    metacog_temperature: float = 0.2
    metacog_max_tokens: int = 2048
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"
    safety_model: str = "claude-3-5-sonnet-20241022"
    safety_temperature: float = 0.0
    safety_max_tokens: int = 1024
    enable_caching: bool = True
    enable_fallback: bool = True
    fallback_provider: str = "ollama"
    fallback_model: str = "llama3"


# ---------------------------------------------------------------------------
# LH Planner Integration
# ---------------------------------------------------------------------------


class LHPlannerIntegration:
    """LLM integration for LH planning subsystem."""

    SYSTEM_PROMPT = (
        "You are a task planner for an AGI system. Given a goal and environment "
        "context, generate a structured plan with ordered steps. Each step must "
        "have: step_id, description, preconditions, postconditions, and action type. "
        "Return valid JSON."
    )

    def __init__(
        self,
        config: Optional[LLMIntegrationConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._config = config or LLMIntegrationConfig()
        self._client = client
        logger.info("[LLM][Planner] initialized model=%s", self._config.planner_model)

    def generate_plan(
        self,
        goal: str,
        context: str = "",
        constraints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a plan for the given goal.

        Args:
            goal: Task goal description
            context: Additional context (environment, memory)
            constraints: Optional constraints on the plan

        Returns:
            Plan dictionary with steps and metadata
        """
        if not self._client:
            logger.warning("[LLM][Planner] no client, returning stub plan")
            return self._stub_plan(goal)

        prompt = f"Goal: {goal}\n"
        if context:
            prompt += f"Context: {context}\n"
        if constraints:
            prompt += f"Constraints: {', '.join(constraints)}\n"
        prompt += "\nGenerate a structured plan as JSON."

        try:
            response = self._client.complete(
                prompt,
                system=self.SYSTEM_PROMPT,
                temperature=self._config.planner_temperature,
                max_tokens=self._config.planner_max_tokens,
            )
            return self._parse_plan(response.content)
        except Exception as e:
            logger.error("[LLM][Planner] generation failed: %s", e)
            return self._stub_plan(goal)

    def refine_plan(
        self,
        plan: Dict[str, Any],
        feedback: str,
    ) -> Dict[str, Any]:
        """Refine an existing plan based on feedback.

        Args:
            plan: Current plan dictionary
            feedback: Feedback for refinement

        Returns:
            Refined plan dictionary
        """
        if not self._client:
            return plan

        prompt = (
            f"Current plan:\n{json.dumps(plan, indent=2)}\n\n"
            f"Feedback: {feedback}\n\n"
            "Refine the plan based on the feedback. Return JSON."
        )

        try:
            response = self._client.complete(
                prompt,
                system=self.SYSTEM_PROMPT,
                temperature=self._config.planner_temperature,
                max_tokens=self._config.planner_max_tokens,
            )
            return self._parse_plan(response.content)
        except Exception as e:
            logger.error("[LLM][Planner] refinement failed: %s", e)
            return plan

    def decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """Decompose a complex task into subtasks.

        Args:
            task: Task description

        Returns:
            List of subtask dictionaries
        """
        if not self._client:
            return [{"subtask_id": "1", "description": task, "type": "generic"}]

        try:
            response = self._client.complete(
                f"Decompose this task into subtasks: {task}\nReturn JSON array.",
                system=self.SYSTEM_PROMPT,
                temperature=self._config.planner_temperature,
            )
            parsed = json.loads(response.content)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception as e:
            logger.error("[LLM][Planner] decomposition failed: %s", e)
            return [{"subtask_id": "1", "description": task, "type": "generic"}]

    def _parse_plan(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into plan dictionary."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_content": content, "steps": [], "parse_error": True}

    @staticmethod
    def _stub_plan(goal: str) -> Dict[str, Any]:
        """Return a stub plan when LLM is unavailable."""
        return {
            "plan_id": "stub-001",
            "goal": goal,
            "steps": [
                {
                    "step_id": "1",
                    "description": goal,
                    "action_type": "generic",
                    "preconditions": [],
                    "postconditions": [],
                }
            ],
            "stub": True,
        }


# ---------------------------------------------------------------------------
# Metacognition Integration
# ---------------------------------------------------------------------------


class MetacognitionIntegration:
    """LLM integration for metacognition subsystem."""

    SYSTEM_PROMPT = (
        "You are a metacognitive reviewer for an AGI system. Analyze plans, "
        "reasoning traces, and decisions for logical consistency, completeness, "
        "and potential issues. Provide structured assessments."
    )

    def __init__(
        self,
        config: Optional[LLMIntegrationConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._config = config or LLMIntegrationConfig()
        self._client = client
        logger.info("[LLM][Meta] initialized model=%s", self._config.metacog_model)

    def critique_plan(
        self,
        plan: Dict[str, Any],
        context: str = "",
    ) -> Dict[str, Any]:
        """Critique a plan for issues and improvements.

        Args:
            plan: Plan dictionary to critique
            context: Additional context

        Returns:
            Critique with issues, suggestions, confidence
        """
        if not self._client:
            return {
                "confidence": 0.5,
                "issues": [],
                "suggestions": [],
                "decision": "ACCEPT",
                "stub": True,
            }

        prompt = f"Plan:\n{json.dumps(plan, indent=2)}\n"
        if context:
            prompt += f"Context: {context}\n"
        prompt += (
            "\nCritique this plan. Return JSON with: confidence (0-1), "
            "issues (list), suggestions (list), decision (ACCEPT/REVISE/REJECT)."
        )

        try:
            response = self._client.complete(
                prompt,
                system=self.SYSTEM_PROMPT,
                temperature=self._config.metacog_temperature,
                max_tokens=self._config.metacog_max_tokens,
            )
            return json.loads(response.content)
        except Exception as e:
            logger.error("[LLM][Meta] critique failed: %s", e)
            return {"confidence": 0.5, "issues": [str(e)], "decision": "REVISE"}

    def generate_explanation(self, decision: str, reasoning: str) -> str:
        """Generate a human-readable explanation for a decision.

        Args:
            decision: The decision made
            reasoning: Raw reasoning trace

        Returns:
            Human-readable explanation string
        """
        if not self._client:
            return f"Decision: {decision}. Based on: {reasoning[:100]}"

        try:
            response = self._client.complete(
                f"Explain this decision in plain language:\n"
                f"Decision: {decision}\nReasoning: {reasoning}",
                system=self.SYSTEM_PROMPT,
                temperature=self._config.metacog_temperature,
            )
            return response.content
        except Exception as e:
            logger.error("[LLM][Meta] explanation failed: %s", e)
            return f"Decision: {decision} (explanation unavailable: {e})"

    def assess_confidence(
        self,
        plan: Dict[str, Any],
        evidence: List[str],
    ) -> float:
        """Assess confidence in a plan given evidence.

        Args:
            plan: Plan dictionary
            evidence: Supporting evidence strings

        Returns:
            Confidence score 0.0 to 1.0
        """
        if not self._client:
            return 0.5

        try:
            response = self._client.complete(
                f"Rate confidence (0.0-1.0) in this plan:\n"
                f"{json.dumps(plan, indent=2)}\n"
                f"Evidence: {json.dumps(evidence)}\n"
                "Return only a number.",
                system=self.SYSTEM_PROMPT,
                temperature=0.0,
            )
            return max(0.0, min(1.0, float(response.content.strip())))
        except Exception:
            return 0.5


# ---------------------------------------------------------------------------
# Memory Embedding Integration
# ---------------------------------------------------------------------------


class MemoryEmbeddingIntegration:
    """LLM integration for memory embedding generation."""

    def __init__(
        self,
        config: Optional[LLMIntegrationConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._config = config or LLMIntegrationConfig()
        self._client = client
        logger.info(
            "[LLM][Embedding] initialized model=%s",
            self._config.embedding_model,
        )

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not self._client:
            # Return stub embeddings (384-dim zeros)
            return [[0.0] * 384 for _ in texts]

        try:
            embeddings = []
            for text in texts:
                emb = self.generate_query_embedding(text)
                embeddings.append(emb)
            return embeddings
        except Exception as e:
            logger.error("[LLM][Embedding] batch generation failed: %s", e)
            return [[0.0] * 384 for _ in texts]

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query.

        Args:
            query: Text to embed

        Returns:
            Embedding vector
        """
        if not self._client:
            return [0.0] * 384

        try:
            # Use client's complete as a fallback for embedding
            # In production, this would use a dedicated embedding endpoint
            self._client.complete(
                f"Generate a semantic representation for: {query}",
                max_tokens=10,
            )
            # In real usage, this would call an embedding API
            logger.debug("[LLM][Embedding] generated embedding for query")
            return [0.0] * 384  # Placeholder
        except Exception as e:
            logger.error("[LLM][Embedding] generation failed: %s", e)
            return [0.0] * 384


# ---------------------------------------------------------------------------
# Safety Fallback Integration
# ---------------------------------------------------------------------------


class SafetyFallbackIntegration:
    """LLM integration for safety fallback reasoning."""

    SAFETY_PROMPT = (
        "You are a safety reviewer for an AGI system. Analyze proposed actions "
        "for potential risks, ethical concerns, and safety violations. Respond "
        "with a structured safety assessment as JSON with: safe (bool), "
        "risk_level (low/medium/high/critical), issues (list), recommendations (list)."
    )

    def __init__(
        self,
        config: Optional[LLMIntegrationConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._config = config or LLMIntegrationConfig()
        self._client = client
        logger.info("[LLM][Safety] initialized model=%s", self._config.safety_model)

    def assess_safety(
        self,
        action: str,
        context: str = "",
    ) -> Dict[str, Any]:
        """Assess the safety of a proposed action.

        Args:
            action: Action description
            context: Additional context

        Returns:
            Safety assessment dictionary
        """
        if not self._client:
            return {
                "safe": True,
                "risk_level": "low",
                "issues": [],
                "recommendations": [],
                "stub": True,
            }

        prompt = f"Action: {action}\n"
        if context:
            prompt += f"Context: {context}\n"
        prompt += "\nAssess the safety of this action. Return JSON."

        try:
            response = self._client.complete(
                prompt,
                system=self.SAFETY_PROMPT,
                temperature=self._config.safety_temperature,
                max_tokens=self._config.safety_max_tokens,
            )
            return json.loads(response.content)
        except Exception as e:
            logger.error("[LLM][Safety] assessment failed: %s", e)
            return {
                "safe": False,
                "risk_level": "high",
                "issues": [f"Assessment failed: {e}"],
                "recommendations": ["Manual review required"],
            }

    def explain_violation(self, violation: str) -> str:
        """Generate explanation for a safety violation.

        Args:
            violation: Violation description

        Returns:
            Human-readable explanation
        """
        if not self._client:
            return f"Safety violation: {violation}"

        try:
            response = self._client.complete(
                f"Explain this safety violation and suggest remediation:\n{violation}",
                system=self.SAFETY_PROMPT,
                temperature=self._config.safety_temperature,
            )
            return response.content
        except Exception as e:
            logger.error("[LLM][Safety] explanation failed: %s", e)
            return f"Safety violation: {violation} (explanation unavailable)"
