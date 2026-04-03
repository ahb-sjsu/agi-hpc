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
Training Runner for AtlasGym.

Runs N episodes across selected environments, records results in
episodic memory and the training_results table, and publishes
progress to NATS.

Can operate as:
    - Standalone CLI: ``python -m agi.training.runner --env ethics --level 2 --episodes 10``
    - Programmatic API: ``runner = TrainingRunner(); await runner.run("ethics", 10)``
    - NATS service: listens for training requests on ``agi.training.request``
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    from agi.common.event import Event
    from agi.core.events.nats_fabric import NatsEventFabric, NatsFabricConfig
except ImportError:
    Event = None  # type: ignore
    NatsEventFabric = None  # type: ignore
    NatsFabricConfig = None  # type: ignore

try:
    from agi.meta.llm.client import LLMClient
    from agi.meta.llm.config import InferenceConfig
except ImportError:
    LLMClient = None  # type: ignore
    InferenceConfig = None  # type: ignore

from agi.training.curriculum import CurriculumConfig, CurriculumManager
from agi.training.gym_env import AtlasGym
from agi.training.scorer import ResponseScorer, ScorerConfig

# Import environment classes
from agi.training.envs.ethics_env import EthicsEnv, EthicsEnvConfig
from agi.training.envs.reasoning_env import ReasoningEnv
from agi.training.envs.coding_env import CodingEnv
from agi.training.envs.debate_env import DebateEnv
from agi.training.envs.memory_env import MemoryEnv
from agi.training.gym_env import AtlasGymConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingRunnerConfig:
    """Configuration for the TrainingRunner.

    Attributes:
        nats_servers: NATS server URLs.
        enable_nats: Whether to connect to NATS for event publishing.
        llm_base_url: LLM server URL for generating responses.
        llm_timeout: LLM request timeout in seconds.
        llm_model: Model name for API requests.
        db_dsn: PostgreSQL DSN for scoring persistence.
        enable_episodic_storage: Store results in episodic memory.
        session_id: Training session identifier.
    """

    nats_servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    enable_nats: bool = False
    llm_base_url: str = "http://localhost:8080"
    llm_timeout: float = 300.0
    llm_model: str = "gemma-4-31b"
    db_dsn: str = "dbname=atlas user=claude"
    enable_episodic_storage: bool = True
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_yaml(cls, path: str) -> TrainingRunnerConfig:
        """Load configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        training = data.get("training", data)
        nats_cfg = training.get("nats", {})
        llm_cfg = training.get("llm", {})
        pg_cfg = training.get("postgresql", {})
        return cls(
            nats_servers=nats_cfg.get("servers", ["nats://localhost:4222"]),
            enable_nats=nats_cfg.get("enabled", False),
            llm_base_url=llm_cfg.get("base_url", "http://localhost:8080"),
            llm_timeout=llm_cfg.get("timeout", 300.0),
            llm_model=llm_cfg.get("model", "gemma-4-31b"),
            db_dsn=pg_cfg.get("dsn", "dbname=atlas user=claude"),
            enable_episodic_storage=training.get("episodic_storage", True),
        )


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Result of a single training episode."""

    env_name: str = ""
    level: int = 1
    scenario_text: str = ""
    response_text: str = ""
    score: float = 0.0
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    curriculum_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_name": self.env_name,
            "level": self.level,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "latency_ms": round(self.latency_ms, 1),
            "curriculum_action": self.curriculum_action,
        }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

ENV_REGISTRY: Dict[str, type] = {
    "ethics": EthicsEnv,
    "reasoning": ReasoningEnv,
    "coding": CodingEnv,
    "debate": DebateEnv,
    "memory": MemoryEnv,
}


def create_env(
    env_name: str,
    level: int = 1,
    db_dsn: str = "dbname=atlas user=claude",
    enable_nats: bool = False,
    nats_servers: Optional[List[str]] = None,
) -> AtlasGym:
    """Create an environment instance by name.

    Args:
        env_name: One of 'ethics', 'reasoning', 'coding', 'debate', 'memory'.
        level: Initial difficulty level.
        db_dsn: PostgreSQL DSN (used by ethics env).
        enable_nats: Whether to enable NATS events.
        nats_servers: NATS server URLs.

    Returns:
        An AtlasGym subclass instance.

    Raises:
        ValueError: If env_name is not recognised.
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {', '.join(ENV_REGISTRY)}"
        )

    env_cls = ENV_REGISTRY[env_name]

    if env_name == "ethics":
        config = EthicsEnvConfig(
            level=level,
            db_dsn=db_dsn,
            enable_nats=enable_nats,
            nats_servers=nats_servers or ["nats://localhost:4222"],
        )
    else:
        config = AtlasGymConfig(
            env_name=env_name,
            level=level,
            enable_nats=enable_nats,
            nats_servers=nats_servers or ["nats://localhost:4222"],
        )

    return env_cls(config=config)


# ---------------------------------------------------------------------------
# TrainingRunner
# ---------------------------------------------------------------------------


class TrainingRunner:
    """Runs training episodes and manages the full training loop.

    Coordinates between environments, the LLM client (for generating
    Atlas responses), the scorer (for persistence), and the curriculum
    manager (for level progression).

    Usage::

        runner = TrainingRunner()
        results = await runner.run("ethics", episodes=10, level=2)
        print(f"Average score: {sum(r.score for r in results) / len(results)}")
    """

    def __init__(
        self,
        config: Optional[TrainingRunnerConfig] = None,
        curriculum: Optional[CurriculumManager] = None,
        response_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialise the training runner.

        Args:
            config: Runner configuration.
            curriculum: Shared CurriculumManager (created if None).
            response_fn: Optional sync function to generate responses.
                If None, uses the LLM client.
        """
        self._config = config or TrainingRunnerConfig()
        self._curriculum = curriculum or CurriculumManager(
            config=CurriculumConfig(
                db_dsn=self._config.db_dsn,
                enable_nats=self._config.enable_nats,
                nats_servers=self._config.nats_servers,
            )
        )
        self._scorer = ResponseScorer(config=ScorerConfig(db_dsn=self._config.db_dsn))
        self._llm: Optional[Any] = None
        self._fabric: Optional[Any] = None
        self._response_fn = response_fn

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------

    async def _ensure_llm(self) -> None:
        """Lazily create the LLM client."""
        if self._llm is not None or LLMClient is None:
            return
        self._llm = LLMClient(
            base_url=self._config.llm_base_url,
            timeout=self._config.llm_timeout,
            default_model=self._config.llm_model,
        )

    async def _generate_response(self, scenario_text: str) -> str:
        """Generate Atlas's response to a scenario.

        Uses the response_fn if provided, otherwise calls the LLM.
        """
        # Sync callback (for testing or manual override)
        if self._response_fn is not None:
            return self._response_fn(scenario_text)

        # LLM client
        await self._ensure_llm()
        if self._llm is None:
            raise RuntimeError(
                "No LLM client available and no response_fn provided. "
                "Install aiohttp and ensure LLM server is running."
            )

        config = InferenceConfig(
            temperature=0.5,
            max_tokens=2048,
            system_prompt=(
                "You are Atlas, an AGI cognitive architecture being trained. "
                "Respond thoughtfully and precisely to the scenario presented. "
                "Show your reasoning."
            ),
        )
        result = await self._llm.generate(prompt=scenario_text, config=config)
        return result.text

    # ------------------------------------------------------------------
    # NATS integration
    # ------------------------------------------------------------------

    async def _ensure_fabric(self) -> None:
        """Lazily connect to NATS."""
        if not self._config.enable_nats or NatsEventFabric is None or Event is None:
            return
        if self._fabric is not None:
            return
        try:
            fabric_config = NatsFabricConfig(servers=self._config.nats_servers)
            self._fabric = NatsEventFabric(config=fabric_config)
            await self._fabric.connect()
        except Exception:
            logger.warning("[runner] NATS connection failed")
            self._fabric = None

    async def _publish_progress(
        self,
        env_name: str,
        episode_num: int,
        total_episodes: int,
        result: EpisodeResult,
    ) -> None:
        """Publish training progress event."""
        if self._fabric is None or Event is None:
            return
        try:
            event = Event.create(
                source="training",
                event_type="training.progress",
                payload={
                    "env_name": env_name,
                    "episode": episode_num,
                    "total_episodes": total_episodes,
                    "level": result.level,
                    "score": result.score,
                    "curriculum_action": result.curriculum_action,
                    "session_id": self._config.session_id,
                },
            )
            await self._fabric.publish("agi.training.progress", event)
        except Exception:
            logger.debug("[runner] failed to publish progress event")

    # ------------------------------------------------------------------
    # Episodic memory storage
    # ------------------------------------------------------------------

    def _store_episode(self, result: EpisodeResult) -> None:
        """Store training result in episodic memory via the scorer."""
        self._scorer.store_result(
            env_name=result.env_name,
            level=result.level,
            scenario=result.scenario_text,
            response=result.response_text,
            score=result.score,
            metadata=result.score_breakdown,
        )

    # ------------------------------------------------------------------
    # Single episode
    # ------------------------------------------------------------------

    async def run_episode(
        self, env: AtlasGym, level: Optional[int] = None
    ) -> EpisodeResult:
        """Run a single training episode.

        Args:
            env: The environment to run.
            level: Optional level override.

        Returns:
            EpisodeResult with score and details.
        """
        t0 = time.perf_counter()
        options = {"level": level} if level else None
        obs, info = env.reset(options=options)

        # For multi-turn envs (memory), iterate until quiz
        if hasattr(env, "_in_quiz"):
            # Multi-turn: acknowledge facts, then answer quiz
            while True:
                response = await self._generate_response(obs)
                obs, reward, terminated, truncated, info = env.step(response)
                if terminated:
                    break
                if info.get("is_quiz"):
                    # This is the quiz question
                    response = await self._generate_response(obs)
                    obs, reward, terminated, truncated, info = env.step(response)
                    break
        else:
            # Single-turn: generate and submit
            response = await self._generate_response(obs)
            obs, reward, terminated, truncated, info = env.step(response)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Record in curriculum
        curr_action = self._curriculum.record_score(env.env_name, reward)

        result = EpisodeResult(
            env_name=env.env_name,
            level=info.get("level", env.level),
            scenario_text=obs if obs else "(multi-turn)",
            response_text=response if isinstance(response, str) else "",
            score=reward,
            score_breakdown=info.get("score_breakdown", {}),
            latency_ms=latency_ms,
            curriculum_action=curr_action,
        )

        # Persist
        self._store_episode(result)

        return result

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    async def run(
        self,
        env_name: str,
        episodes: int = 10,
        level: Optional[int] = None,
    ) -> List[EpisodeResult]:
        """Run N training episodes in a specific environment.

        If level is None, uses the curriculum manager's current level
        for the environment.

        Args:
            env_name: Environment name.
            episodes: Number of episodes to run.
            level: Optional level override.

        Returns:
            List of EpisodeResult instances.
        """
        await self._ensure_fabric()

        # Use curriculum level if not overridden
        current_level = level or self._curriculum.get_level(env_name)

        env = create_env(
            env_name=env_name,
            level=current_level,
            db_dsn=self._config.db_dsn,
            enable_nats=self._config.enable_nats,
            nats_servers=self._config.nats_servers,
        )

        results: List[EpisodeResult] = []
        logger.info(
            "[runner] starting %d episodes of %s at level %d",
            episodes,
            env_name,
            current_level,
        )

        for i in range(episodes):
            # Update level from curriculum (may have been promoted/demoted)
            current_level = self._curriculum.get_level(env_name)
            env.level = current_level

            result = await self.run_episode(env, level=current_level)
            results.append(result)

            await self._publish_progress(env_name, i + 1, episodes, result)

            logger.info(
                "[runner] episode %d/%d env=%s level=%d score=%.2f%s",
                i + 1,
                episodes,
                env_name,
                result.level,
                result.score,
                f" ({result.curriculum_action})" if result.curriculum_action else "",
            )

        env.close()

        # Summary
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        logger.info(
            "[runner] completed %d episodes of %s -- avg_score=%.3f",
            episodes,
            env_name,
            avg_score,
        )

        return results

    # ------------------------------------------------------------------
    # NATS service mode
    # ------------------------------------------------------------------

    async def start_service(self) -> None:
        """Start the runner as a NATS service.

        Listens for training requests on ``agi.training.request`` and
        runs episodes as directed.
        """
        await self._ensure_fabric()
        if self._fabric is None or Event is None:
            raise RuntimeError(
                "[runner] NATS required for service mode. "
                "Set enable_nats=True and ensure NATS is available."
            )

        async def _handle_request(event: Event) -> None:
            payload = event.payload
            env_name = payload.get("env_name", "reasoning")
            episodes = payload.get("episodes", 10)
            level = payload.get("level")

            logger.info(
                "[runner-service] received training request: "
                "env=%s episodes=%d level=%s",
                env_name,
                episodes,
                level,
            )

            try:
                results = await self.run(env_name, episodes, level)
                avg = sum(r.score for r in results) / len(results) if results else 0.0

                reply = Event.create(
                    source="training",
                    event_type="training.result",
                    payload={
                        "env_name": env_name,
                        "episodes": len(results),
                        "avg_score": round(avg, 3),
                        "results": [r.to_dict() for r in results[-5:]],
                    },
                    trace_id=event.trace_id,
                )
                await self._fabric.publish("agi.training.result", reply)
            except Exception:
                logger.exception("[runner-service] error running training request")

        await self._fabric.subscribe("agi.training.request", _handle_request)
        logger.info("[runner-service] listening on agi.training.request")

    async def close(self) -> None:
        """Clean up all resources."""
        if self._llm is not None:
            await self._llm.close()
            self._llm = None
        if self._fabric is not None:
            await self._fabric.disconnect()
            self._fabric = None
        await self._curriculum.close()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_training_metrics(self) -> Dict[str, Any]:
        """Return training metrics for dashboard consumption.

        Returns:
            Dict with curriculum state, recent scores, totals.
        """
        return self._curriculum.get_training_metrics()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _cli_run(
    env_name: str,
    episodes: int,
    level: Optional[int],
    config_path: Optional[str],
    llm_url: str,
    db_dsn: str,
    nats: bool,
) -> None:
    """CLI async entry point."""
    if config_path:
        config = TrainingRunnerConfig.from_yaml(config_path)
    else:
        config = TrainingRunnerConfig(
            llm_base_url=llm_url,
            db_dsn=db_dsn,
            enable_nats=nats,
        )

    runner = TrainingRunner(config=config)

    try:
        results = await runner.run(env_name, episodes, level)

        # Print summary
        if results:
            avg = sum(r.score for r in results) / len(results)
            print(f"\n{'=' * 60}")
            print(f"Training Complete: {env_name}")
            print(f"{'=' * 60}")
            print(f"Episodes:    {len(results)}")
            print(f"Avg Score:   {avg:.3f}")
            print(f"Min Score:   {min(r.score for r in results):.3f}")
            print(f"Max Score:   {max(r.score for r in results):.3f}")
            print(
                f"Final Level: {runner.get_training_metrics().get('environments', {}).get(env_name, {}).get('current_level', '?')}"
            )
            print(f"{'=' * 60}")
    finally:
        await runner.close()


def main() -> None:
    """CLI entry point for the training runner."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AtlasGym Training Runner",
        prog="python -m agi.training.runner",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=list(ENV_REGISTRY.keys()),
        help="Environment to train in",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Force a specific difficulty level (default: curriculum-managed)",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to training_config.yaml",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8080",
        help="LLM server URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--db-dsn",
        type=str,
        default="dbname=atlas user=claude",
        help="PostgreSQL DSN",
    )
    parser.add_argument(
        "--nats",
        action="store_true",
        help="Enable NATS event publishing",
    )
    parser.add_argument(
        "--service",
        action="store_true",
        help="Run as NATS service (listens for training requests)",
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

    if args.service:
        # NATS service mode
        async def _run_service() -> None:
            config = TrainingRunnerConfig(
                llm_base_url=args.llm_url,
                db_dsn=args.db_dsn,
                enable_nats=True,
            )
            runner = TrainingRunner(config=config)
            await runner.start_service()
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await runner.close()

        try:
            asyncio.run(_run_service())
        except KeyboardInterrupt:
            logger.info("[runner] interrupted by user")
            sys.exit(0)
    else:
        try:
            asyncio.run(
                _cli_run(
                    env_name=args.env,
                    episodes=args.episodes,
                    level=args.level,
                    config_path=args.config,
                    llm_url=args.llm_url,
                    db_dsn=args.db_dsn,
                    nats=args.nats,
                )
            )
        except KeyboardInterrupt:
            logger.info("[runner] interrupted by user")
            sys.exit(0)


if __name__ == "__main__":
    main()
