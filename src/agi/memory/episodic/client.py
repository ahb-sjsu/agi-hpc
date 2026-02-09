# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Episodic Memory Client for AGI-HPC.

High-level API for episode lifecycle management:
- Starting and ending episodes
- Recording steps and events
- Storing decision proofs
- Querying episode history

Usage:
    from agi.memory.episodic import EpisodicMemoryClient

    client = EpisodicMemoryClient()

    # Start an episode
    episode = await client.start_episode("Navigate to target location")

    # Record steps
    await client.record_step(episode.episode_id, 0, "Plan route")

    # End episode
    await client.end_episode(episode.episode_id, success=True)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agi.memory.episodic.postgres_store import (
    PostgresEpisodicStore,
    PostgresConfig,
    Episode,
    EpisodeStep,
    EpisodeEvent,
    DecisionProof,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode Context Manager
# ---------------------------------------------------------------------------


@dataclass
class EpisodeContext:
    """Context for an active episode."""

    episode_id: str
    task_description: str
    task_type: str = ""
    scenario_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    current_step: int = 0
    _client: Optional["EpisodicMemoryClient"] = None

    async def record_step(
        self,
        description: str,
        tool_id: str = "",
        succeeded: Optional[bool] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a step in this episode."""
        step_index = self.current_step
        self.current_step += 1

        if self._client:
            await self._client.record_step(
                episode_id=self.episode_id,
                step_index=step_index,
                description=description,
                tool_id=tool_id,
                succeeded=succeeded,
                params=params,
            )

        return step_index

    async def record_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an event in this episode."""
        if self._client:
            await self._client.record_event(
                episode_id=self.episode_id,
                step_index=self.current_step,
                event_type=event_type,
                payload=payload,
                tags=tags,
            )

    async def end(
        self,
        success: bool,
        description: str = "",
        insights: Optional[List[str]] = None,
    ) -> None:
        """End this episode."""
        if self._client:
            await self._client.end_episode(
                episode_id=self.episode_id,
                success=success,
                description=description,
                insights=insights,
            )


# ---------------------------------------------------------------------------
# Episodic Memory Client
# ---------------------------------------------------------------------------


class EpisodicMemoryClient:
    """
    High-level client for episodic memory operations.

    Provides episode lifecycle management with:
    - Episode creation and completion
    - Step and event recording
    - Decision proof storage
    - History querying
    """

    def __init__(
        self,
        store: Optional[PostgresEpisodicStore] = None,
        store_config: Optional[PostgresConfig] = None,
        auto_init_schema: bool = True,
    ):
        """
        Initialize episodic memory client.

        Args:
            store: Storage backend (creates default if not provided)
            store_config: Config for storage backend
            auto_init_schema: Whether to initialize DB schema
        """
        if store is None:
            config = store_config or PostgresConfig()
            store = PostgresEpisodicStore(config)

        self._store = store

        if auto_init_schema:
            try:
                store.init_schema()
            except Exception as e:
                logger.warning(
                    "[memory][episodic] schema init failed: %s",
                    e,
                )

        logger.info("[memory][episodic] client initialized")

    async def start_episode(
        self,
        task_description: str,
        task_type: str = "",
        scenario_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EpisodeContext:
        """Start a new episode.

        Args:
            task_description: Description of the task
            task_type: Type of task (e.g., "navigation", "manipulation")
            scenario_id: Scenario identifier
            metadata: Additional metadata

        Returns:
            Episode context for recording steps/events
        """
        episode_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        episode = Episode(
            episode_id=episode_id,
            task_description=task_description,
            task_type=task_type,
            scenario_id=scenario_id,
            start_time=start_time,
            metadata=metadata or {},
        )

        self._store.store_episode(episode)

        context = EpisodeContext(
            episode_id=episode_id,
            task_description=task_description,
            task_type=task_type,
            scenario_id=scenario_id,
            start_time=start_time,
            _client=self,
        )

        logger.info(
            "[memory][episodic] started episode=%s task=%s",
            episode_id[:8],
            task_description[:50],
        )

        return context

    async def end_episode(
        self,
        episode_id: str,
        success: bool,
        description: str = "",
        insights: Optional[List[str]] = None,
    ) -> None:
        """End an episode.

        Args:
            episode_id: Episode ID
            success: Whether the episode was successful
            description: Outcome description
            insights: Learned insights
        """
        episode = self._store.get_episode(episode_id)
        if not episode:
            logger.warning(
                "[memory][episodic] episode not found: %s",
                episode_id,
            )
            return

        end_time = datetime.utcnow()
        duration_ms = 0
        if episode.start_time:
            duration_ms = int((end_time - episode.start_time).total_seconds() * 1000)

        # Update episode
        episode.end_time = end_time
        episode.outcome_success = success
        episode.outcome_description = description
        episode.total_duration_ms = duration_ms
        episode.insights = insights or []
        episode.completion_percentage = 100.0 if success else 0.0

        self._store.store_episode(episode)

        logger.info(
            "[memory][episodic] ended episode=%s success=%s duration=%dms",
            episode_id[:8],
            success,
            duration_ms,
        )

    async def record_step(
        self,
        episode_id: str,
        step_index: int,
        description: str,
        tool_id: str = "",
        step_id: Optional[str] = None,
        succeeded: Optional[bool] = None,
        failure_reason: str = "",
        duration_ms: int = 0,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an episode step.

        Args:
            episode_id: Episode ID
            step_index: Step index (0-based)
            description: Step description
            tool_id: Tool/skill used
            step_id: Optional step ID
            succeeded: Whether step succeeded
            failure_reason: Reason for failure
            duration_ms: Step duration in milliseconds
            params: Step parameters
        """
        step = EpisodeStep(
            episode_id=episode_id,
            step_index=step_index,
            step_id=step_id or str(uuid.uuid4()),
            description=description,
            tool_id=tool_id,
            succeeded=succeeded,
            failure_reason=failure_reason,
            duration_ms=duration_ms,
            params=params or {},
        )

        self._store.store_step(step)

        logger.debug(
            "[memory][episodic] recorded step=%d episode=%s",
            step_index,
            episode_id[:8],
        )

    async def record_event(
        self,
        episode_id: str,
        step_index: int,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an episode event.

        Args:
            episode_id: Episode ID
            step_index: Current step index
            event_type: Type of event
            payload: Event payload
            tags: Event tags
        """
        event = EpisodeEvent(
            episode_id=episode_id,
            step_index=step_index,
            event_type=event_type,
            timestamp_ms=int(time.time() * 1000),
            payload=payload or {},
            tags=tags or {},
        )

        self._store.store_event(event)

    async def record_decision(
        self,
        episode_id: str,
        step_id: str,
        decision: str,
        bond_index: float = 0.0,
        moral_vector: Optional[Dict[str, float]] = None,
    ) -> str:
        """Record a decision proof.

        Args:
            episode_id: Episode ID
            step_id: Step ID
            decision: Decision (ALLOW, BLOCK, REVISE)
            bond_index: Bond index score
            moral_vector: ErisML moral vector

        Returns:
            Proof ID
        """
        # Get previous proof hash
        prev_hash = ""
        # TODO: Fetch last proof hash from chain

        proof = DecisionProof(
            proof_id=str(uuid.uuid4()),
            episode_id=episode_id,
            step_id=step_id,
            timestamp_ms=int(time.time() * 1000),
            decision=decision,
            bond_index=bond_index,
            moral_vector=moral_vector or {},
            previous_proof_hash=prev_hash,
        )
        proof.proof_hash = proof.compute_hash()

        return self._store.store_decision_proof(proof)

    async def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        return self._store.get_episode(episode_id)

    async def query_episodes(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        task_type: Optional[str] = None,
        scenario_id: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Episode]:
        """Query episodes with filters."""
        return self._store.query_episodes(
            start_time=start_time,
            end_time=end_time,
            task_type=task_type,
            scenario_id=scenario_id,
            success=success,
            limit=limit,
        )

    async def get_episode_steps(self, episode_id: str) -> List[EpisodeStep]:
        """Get steps for an episode."""
        return self._store.get_episode_steps(episode_id)

    async def get_episode_events(self, episode_id: str) -> List[EpisodeEvent]:
        """Get events for an episode."""
        return self._store.get_episode_events(episode_id)

    async def verify_proof_chain(self, episode_id: str) -> bool:
        """Verify decision proof chain integrity."""
        return self._store.verify_proof_chain(episode_id)

    def close(self) -> None:
        """Close the client."""
        self._store.close()
        logger.info("[memory][episodic] client closed")
