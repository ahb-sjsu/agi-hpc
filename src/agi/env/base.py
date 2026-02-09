# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Base environment interface for AGI-HPC.

Provides a Gymnasium-compatible API with extensions for robotics:
- Async step execution for real hardware
- Structured observations (sensors, state)
- Structured actions (commands, parameters)
- Environment registry and factory

Usage:
    from agi.env import Environment, create_env

    env = create_env("mujoco:humanoid")
    obs = await env.reset()
    result = await env.step(action)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np

logger = logging.getLogger(__name__)

# Type variables for observation and action spaces
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


# ---------------------------------------------------------------------------
# Space Definitions
# ---------------------------------------------------------------------------


class SpaceType(str, Enum):
    """Types of observation/action spaces."""

    DISCRETE = "discrete"
    BOX = "box"
    DICT = "dict"
    MULTI_DISCRETE = "multi_discrete"
    TUPLE = "tuple"


@dataclass
class Space:
    """
    Space definition compatible with Gymnasium.

    Supports Box, Discrete, MultiDiscrete, and Dict spaces.
    """

    space_type: SpaceType
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    n: Optional[int] = None  # For discrete
    nvec: Optional[List[int]] = None  # For multi-discrete
    spaces: Optional[Dict[str, "Space"]] = None  # For dict

    @classmethod
    def box(
        cls,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
    ) -> "Space":
        """Create a Box space."""
        if shape is not None:
            low_arr = np.full(shape, low, dtype=dtype)
            high_arr = np.full(shape, high, dtype=dtype)
        else:
            low_arr = np.asarray(low, dtype=dtype)
            high_arr = np.asarray(high, dtype=dtype)
            shape = low_arr.shape

        return cls(
            space_type=SpaceType.BOX,
            shape=shape,
            dtype=dtype,
            low=low_arr,
            high=high_arr,
        )

    @classmethod
    def discrete(cls, n: int) -> "Space":
        """Create a Discrete space."""
        return cls(space_type=SpaceType.DISCRETE, n=n)

    @classmethod
    def multi_discrete(cls, nvec: List[int]) -> "Space":
        """Create a MultiDiscrete space."""
        return cls(
            space_type=SpaceType.MULTI_DISCRETE,
            nvec=nvec,
            shape=(len(nvec),),
        )

    @classmethod
    def dict_space(cls, spaces: Dict[str, "Space"]) -> "Space":
        """Create a Dict space."""
        return cls(space_type=SpaceType.DICT, spaces=spaces)

    def sample(self) -> Any:
        """Sample a random element from the space."""
        if self.space_type == SpaceType.BOX:
            return np.random.uniform(self.low, self.high).astype(self.dtype)
        elif self.space_type == SpaceType.DISCRETE:
            return np.random.randint(0, self.n)
        elif self.space_type == SpaceType.MULTI_DISCRETE:
            return np.array([np.random.randint(0, n) for n in self.nvec])
        elif self.space_type == SpaceType.DICT:
            return {k: v.sample() for k, v in self.spaces.items()}
        else:
            raise NotImplementedError(f"Unsupported space type: {self.space_type}")

    def contains(self, x: Any) -> bool:
        """Check if x is in the space."""
        if self.space_type == SpaceType.BOX:
            return (
                isinstance(x, np.ndarray)
                and x.shape == self.shape
                and np.all(x >= self.low)
                and np.all(x <= self.high)
            )
        elif self.space_type == SpaceType.DISCRETE:
            return isinstance(x, (int, np.integer)) and 0 <= x < self.n
        elif self.space_type == SpaceType.MULTI_DISCRETE:
            if not isinstance(x, np.ndarray):
                return False
            return all(0 <= xi < ni for xi, ni in zip(x, self.nvec, strict=False))
        elif self.space_type == SpaceType.DICT:
            if not isinstance(x, dict):
                return False
            return all(k in x and self.spaces[k].contains(x[k]) for k in self.spaces)
        return True


# ---------------------------------------------------------------------------
# Environment Specification
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentSpec:
    """Environment specification and metadata."""

    name: str
    observation_space: Space
    action_space: Space
    max_episode_steps: int = 1000
    reward_threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if "version" not in self.metadata:
            self.metadata["version"] = "1.0.0"


# ---------------------------------------------------------------------------
# Step Results
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result from environment step."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Check if episode is done (terminated or truncated)."""
        return self.terminated or self.truncated


@dataclass
class ResetResult:
    """Result from environment reset."""

    observation: Any
    info: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Protocol defining the environment interface."""

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        ...

    @property
    def observation_space(self) -> Space:
        """Get observation space."""
        ...

    @property
    def action_space(self) -> Space:
        """Get action space."""
        ...

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset the environment."""
        ...

    async def step(self, action: Any) -> StepResult:
        """Execute action and return next state."""
        ...

    async def observe(self) -> Any:
        """Get current observation without stepping."""
        ...

    async def close(self) -> None:
        """Clean up environment resources."""
        ...


# ---------------------------------------------------------------------------
# Base Environment Implementation
# ---------------------------------------------------------------------------


class Environment(ABC, Generic[ObsType, ActType]):
    """
    Abstract base class for environments.

    Follows the Gymnasium API with async support for robotics.
    """

    def __init__(self, name: str = "base"):
        self._name = name
        self._step_count = 0
        self._episode_count = 0
        self._closed = False
        self._rng = np.random.default_rng()
        logger.info("[env] initialized name=%s", name)

    @property
    @abstractmethod
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        pass

    @property
    def observation_space(self) -> Space:
        """Get observation space."""
        return self.spec.observation_space

    @property
    def action_space(self) -> Space:
        """Get action space."""
        return self.spec.action_space

    @property
    def name(self) -> str:
        """Get environment name."""
        return self._name

    @property
    def step_count(self) -> int:
        """Get current step count in episode."""
        return self._step_count

    @property
    def episode_count(self) -> int:
        """Get total episode count."""
        return self._episode_count

    @abstractmethod
    async def _reset_impl(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Implementation of reset logic."""
        pass

    @abstractmethod
    async def _step_impl(
        self, action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """Implementation of step logic."""
        pass

    @abstractmethod
    async def _observe_impl(self) -> ObsType:
        """Implementation of observe logic."""
        pass

    @abstractmethod
    async def _close_impl(self) -> None:
        """Implementation of close logic."""
        pass

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset the environment.

        Args:
            seed: Random seed for reproducibility
            options: Reset options (e.g., initial state)

        Returns:
            Initial observation and info
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._episode_count += 1

        obs, info = await self._reset_impl(seed, options)

        logger.debug(
            "[env][%s] reset episode=%d seed=%s",
            self._name,
            self._episode_count,
            seed,
        )

        return ResetResult(observation=obs, info=info)

    async def step(self, action: ActType) -> StepResult:
        """Execute action and return next state.

        Args:
            action: Action to execute

        Returns:
            Observation, reward, terminated, truncated, info
        """
        if self._closed:
            raise RuntimeError("Environment is closed")

        self._step_count += 1
        obs, reward, terminated, truncated, info = await self._step_impl(action)

        # Check for truncation due to max steps
        if (
            not terminated
            and not truncated
            and self._step_count >= self.spec.max_episode_steps
        ):
            truncated = True
            info["truncated_reason"] = "max_episode_steps"

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    async def observe(self) -> ObsType:
        """Get current observation without stepping."""
        return await self._observe_impl()

    async def close(self) -> None:
        """Clean up environment resources."""
        if not self._closed:
            await self._close_impl()
            self._closed = True
            logger.info("[env][%s] closed", self._name)

    async def render(self) -> Optional[np.ndarray]:
        """Render the environment (optional).

        Returns:
            RGB image array or None
        """
        return None

    def seed(self, seed: int) -> None:
        """Set random seed."""
        self._rng = np.random.default_rng(seed)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# ---------------------------------------------------------------------------
# Async Environment (for real hardware)
# ---------------------------------------------------------------------------


class AsyncEnvironment(Environment[ObsType, ActType]):
    """
    Environment with async step support for real hardware.

    Supports non-blocking action execution with:
    - step_async: Start action without waiting
    - step_wait: Wait for action completion
    - step_poll: Check action status
    """

    def __init__(self, name: str = "async"):
        super().__init__(name)
        self._pending_actions: Dict[str, Any] = {}

    @abstractmethod
    async def step_async(self, action: ActType) -> str:
        """Start async action execution.

        Args:
            action: Action to execute

        Returns:
            Action ID for tracking
        """
        pass

    @abstractmethod
    async def step_wait(
        self,
        action_id: str,
        timeout: Optional[float] = None,
    ) -> StepResult:
        """Wait for async action completion.

        Args:
            action_id: ID from step_async
            timeout: Maximum wait time in seconds

        Returns:
            Step result
        """
        pass

    @abstractmethod
    async def step_poll(self, action_id: str) -> Optional[StepResult]:
        """Poll async action status.

        Args:
            action_id: ID from step_async

        Returns:
            Step result if complete, None if still running
        """
        pass

    async def step(self, action: ActType) -> StepResult:
        """Execute action synchronously (blocking)."""
        action_id = await self.step_async(action)
        return await self.step_wait(action_id)


# ---------------------------------------------------------------------------
# Environment Registry
# ---------------------------------------------------------------------------


class EnvironmentRegistry:
    """Registry for environment types."""

    _instance: Optional["EnvironmentRegistry"] = None
    _envs: Dict[str, type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, env_class: type) -> None:
        """Register an environment class."""
        registry = cls()
        registry._envs[name] = env_class
        logger.info("[env][registry] registered %s", name)

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get an environment class by name."""
        registry = cls()
        return registry._envs.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """List registered environments."""
        registry = cls()
        return list(registry._envs.keys())

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs,
    ) -> Environment:
        """Create an environment instance."""
        env_class = cls.get(name)
        if env_class is None:
            raise ValueError(f"Unknown environment: {name}")
        return env_class(**kwargs)


def register_env(name: str):
    """Decorator to register an environment class."""

    def decorator(cls: type) -> type:
        EnvironmentRegistry.register(name, cls)
        return cls

    return decorator


def create_env(name: str, **kwargs) -> Environment:
    """Create an environment by name.

    Args:
        name: Environment name (e.g., "mujoco:humanoid")
        **kwargs: Environment-specific arguments

    Returns:
        Environment instance
    """
    return EnvironmentRegistry.create(name, **kwargs)
