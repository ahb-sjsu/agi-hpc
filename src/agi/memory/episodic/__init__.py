# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Episodic Memory module for AGI-HPC.

Provides temporal episode storage and retrieval:
- Episode lifecycle management
- Step and event recording
- Decision proof chain for governance
- Temporal range queries

Usage:
    from agi.memory.episodic import EpisodicMemoryClient

    client = EpisodicMemoryClient()

    # Start episode
    episode = await client.start_episode("Navigate to target")

    # Record steps
    await episode.record_step("Plan route")
    await episode.record_step("Execute navigation")

    # End episode
    await episode.end(success=True)
"""

from agi.memory.episodic.client import (
    EpisodicMemoryClient,
    EpisodeContext,
)
from agi.memory.episodic.postgres_store import (
    PostgresEpisodicStore,
    PostgresConfig,
    Episode,
    EpisodeStep,
    EpisodeEvent,
    DecisionProof,
)

__all__ = [
    # Client
    "EpisodicMemoryClient",
    "EpisodeContext",
    # Store
    "PostgresEpisodicStore",
    "PostgresConfig",
    # Data types
    "Episode",
    "EpisodeStep",
    "EpisodeEvent",
    "DecisionProof",
]
