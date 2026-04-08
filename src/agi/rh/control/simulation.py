# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Simulation Engine Wrappers for AGI-HPC.

Provides a uniform interface over different physics simulation backends:
- MuJoCo wrapper
- NVIDIA Isaac Sim wrapper
- Unity ML-Agents wrapper
- Gazebo wrapper
- Factory for creating wrappers by name

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class SimulationWrapper(Protocol):
    """Protocol defining the interface for simulation engine wrappers."""

    def initialize(self, config: Dict[str, Any]) -> bool: ...
    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> Dict[str, Any]: ...
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]: ...
    def close(self) -> None: ...

    @property
    def observation_space_shape(self) -> tuple: ...

    @property
    def action_space_shape(self) -> tuple: ...


class MuJoCoWrapper:
    """Wrapper for the MuJoCo physics simulator."""

    def __init__(self) -> None:
        self._model: Any = None
        self._data: Any = None
        self._mujoco: Any = None
        self._initialized: bool = False
        self._obs_shape: tuple = (0,)
        self._act_shape: tuple = (0,)
        try:
            import mujoco

            self._mujoco = mujoco
            logger.info("[rh][sim] MuJoCo available: version %s", mujoco.__version__)
        except ImportError:
            logger.warning("[rh][sim] mujoco not available; MuJoCoWrapper in stub mode")

    def initialize(self, config: Dict[str, Any]) -> bool:
        if self._mujoco is None:
            self._obs_shape = tuple(config.get("obs_shape", (12,)))
            self._act_shape = tuple(config.get("act_shape", (6,)))
            self._initialized = True
            return True
        model_path = config.get("model_path", "")
        if not model_path:
            return False
        try:
            self._model = self._mujoco.MjModel.from_xml_path(model_path)
            self._data = self._mujoco.MjData(self._model)
            if "timestep" in config:
                self._model.opt.timestep = config["timestep"]
            self._obs_shape = (self._model.nq + self._model.nv,)
            self._act_shape = (self._model.nu,)
            self._initialized = True
            return True
        except Exception as exc:
            logger.error("[rh][sim] MuJoCo initialization failed: %s", exc)
            return False

    def reset(self) -> np.ndarray:
        if self._mujoco is not None and self._model is not None:
            self._data = self._mujoco.MjData(self._model)
            self._mujoco.mj_forward(self._model, self._data)
            return np.concatenate([self._data.qpos, self._data.qvel])
        return np.zeros(self._obs_shape)

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        if self._mujoco is not None and self._data is not None:
            self._data.ctrl[:] = action[: self._model.nu]
            self._mujoco.mj_step(self._model, self._data)
            obs = np.concatenate([self._data.qpos, self._data.qvel])
        else:
            obs = np.zeros(self._obs_shape)
        return {"observation": obs, "reward": 0.0, "done": False, "info": {}}

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        self._model = None
        self._data = None
        self._initialized = False

    @property
    def observation_space_shape(self) -> tuple:
        return self._obs_shape

    @property
    def action_space_shape(self) -> tuple:
        return self._act_shape


class IsaacSimWrapper:
    """Wrapper for NVIDIA Isaac Sim."""

    def __init__(self) -> None:
        self._simulation_app: Any = None
        self._world: Any = None
        self._isaac_available: bool = False
        self._initialized: bool = False
        self._obs_shape: tuple = (0,)
        self._act_shape: tuple = (0,)
        try:
            from omni.isaac.kit import SimulationApp  # noqa: F401

            self._isaac_available = True
        except ImportError:
            logger.warning(
                "[rh][sim] omni.isaac not available; IsaacSimWrapper in stub mode"
            )

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._obs_shape = tuple(config.get("obs_shape", (24,)))
        self._act_shape = tuple(config.get("act_shape", (6,)))
        if not self._isaac_available:
            self._initialized = True
            return True
        try:
            from omni.isaac.kit import SimulationApp

            self._simulation_app = SimulationApp(
                {"headless": config.get("headless", True)}
            )
            from omni.isaac.core import World

            self._world = World()
            self._world.reset()
            self._initialized = True
            return True
        except Exception as exc:
            logger.error("[rh][sim] Isaac Sim initialization failed: %s", exc)
            return False

    def reset(self) -> np.ndarray:
        if self._world is not None:
            self._world.reset()
        return np.zeros(self._obs_shape)

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        if self._world is not None:
            self._world.step()
        return {
            "observation": np.zeros(self._obs_shape),
            "reward": 0.0,
            "done": False,
            "info": {},
        }

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass
        self._simulation_app = None
        self._world = None
        self._initialized = False

    @property
    def observation_space_shape(self) -> tuple:
        return self._obs_shape

    @property
    def action_space_shape(self) -> tuple:
        return self._act_shape


class UnityWrapper:
    """Wrapper for Unity ML-Agents."""

    def __init__(self) -> None:
        self._env: Any = None
        self._unity_available: bool = False
        self._initialized: bool = False
        self._obs_shape: tuple = (0,)
        self._act_shape: tuple = (0,)
        try:
            from mlagents_envs.environment import UnityEnvironment  # noqa: F401

            self._unity_available = True
        except ImportError:
            logger.warning(
                "[rh][sim] mlagents_envs not available; UnityWrapper in stub mode"
            )

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._obs_shape = tuple(config.get("obs_shape", (36,)))
        self._act_shape = tuple(config.get("act_shape", (4,)))
        if not self._unity_available:
            self._initialized = True
            return True
        try:
            from mlagents_envs.environment import UnityEnvironment

            self._env = UnityEnvironment(
                file_name=config.get("file_name"),
                no_graphics=config.get("no_graphics", True),
            )
            self._env.reset()
            self._initialized = True
            return True
        except Exception as exc:
            logger.error("[rh][sim] Unity initialization failed: %s", exc)
            return False

    def reset(self) -> np.ndarray:
        if self._env is not None:
            self._env.reset()
        return np.zeros(self._obs_shape)

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        return {
            "observation": np.zeros(self._obs_shape),
            "reward": 0.0,
            "done": False,
            "info": {},
        }

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None
        self._initialized = False

    @property
    def observation_space_shape(self) -> tuple:
        return self._obs_shape

    @property
    def action_space_shape(self) -> tuple:
        return self._act_shape


class GazeboWrapper:
    """Wrapper for Gazebo simulation."""

    def __init__(self) -> None:
        self._transport: Any = None
        self._gazebo_available: bool = False
        self._initialized: bool = False
        self._obs_shape: tuple = (0,)
        self._act_shape: tuple = (0,)
        try:
            from gz.transport import Node as GzNode  # noqa: F401

            self._gazebo_available = True
        except ImportError:
            logger.warning(
                "[rh][sim] gz.transport not available; GazeboWrapper in stub mode"
            )

    def initialize(self, config: Dict[str, Any]) -> bool:
        self._obs_shape = tuple(config.get("obs_shape", (18,)))
        self._act_shape = tuple(config.get("act_shape", (6,)))
        if not self._gazebo_available:
            self._initialized = True
            return True
        try:
            from gz.transport import Node as GzNode

            self._transport = GzNode()
            self._initialized = True
            return True
        except Exception as exc:
            logger.error("[rh][sim] Gazebo initialization failed: %s", exc)
            return False

    def reset(self) -> np.ndarray:
        return np.zeros(self._obs_shape)

    def step(self, action: np.ndarray) -> Dict[str, Any]:
        return {
            "observation": np.zeros(self._obs_shape),
            "reward": 0.0,
            "done": False,
            "info": {},
        }

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        self._transport = None
        self._initialized = False

    @property
    def observation_space_shape(self) -> tuple:
        return self._obs_shape

    @property
    def action_space_shape(self) -> tuple:
        return self._act_shape


_WRAPPER_REGISTRY: Dict[str, type] = {
    "mujoco": MuJoCoWrapper,
    "isaac": IsaacSimWrapper,
    "isaac_sim": IsaacSimWrapper,
    "unity": UnityWrapper,
    "gazebo": GazeboWrapper,
}


class SimulationFactory:
    """Factory for creating simulation wrappers by name."""

    @staticmethod
    def create(name: str) -> SimulationWrapper:
        """Create a simulation wrapper by name."""
        key = name.lower().strip()
        wrapper_cls = _WRAPPER_REGISTRY.get(key)
        if wrapper_cls is None:
            available = ", ".join(sorted(_WRAPPER_REGISTRY.keys()))
            raise ValueError(
                f"Unknown simulation backend '{name}'. Available: {available}"
            )
        logger.info("[rh][sim] creating simulation wrapper: %s", key)
        return wrapper_cls()

    @staticmethod
    def available_backends() -> List[str]:
        """Return names of all available simulation backends."""
        return sorted(_WRAPPER_REGISTRY.keys())

    @staticmethod
    def register(name: str, wrapper_cls: type) -> None:
        """Register a custom simulation wrapper."""
        _WRAPPER_REGISTRY[name.lower().strip()] = wrapper_cls
        logger.info("[rh][sim] registered custom backend: %s", name)
