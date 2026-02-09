# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Environment backends for AGI-HPC.

Provides concrete environment implementations:
- MockEnvironment: For testing and development
- MuJoCoEnvironment: For physics simulation

Usage:
    from agi.env import create_env

    # Mock environment for testing
    env = create_env("mock:simple")
    obs = await env.reset()
    result = await env.step(action)

    # MuJoCo environment for simulation
    env = create_env("mujoco:humanoid")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agi.env.base import (
    Environment,
    AsyncEnvironment,
    EnvironmentSpec,
    Space,
    StepResult,
    ResetResult,
    register_env,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock Environment
# ---------------------------------------------------------------------------


@dataclass
class MockConfig:
    """Configuration for mock environment."""

    obs_dim: int = 10
    action_dim: int = 4
    max_episode_steps: int = 100
    reward_per_step: float = 1.0
    terminate_prob: float = 0.01
    step_delay_ms: float = 0.0


@register_env("mock:simple")
@register_env("mock")
class MockEnvironment(Environment[np.ndarray, np.ndarray]):
    """
    Mock environment for testing.

    Provides:
    - Configurable observation/action spaces
    - Deterministic or stochastic rewards
    - Optional step delay for timing tests
    - Termination probability
    """

    def __init__(
        self,
        config: Optional[MockConfig] = None,
        name: str = "mock",
    ):
        super().__init__(name)
        self._config = config or MockConfig()
        self._state: Optional[np.ndarray] = None
        self._spec = self._create_spec()

        logger.info(
            "[env][mock] initialized obs_dim=%d action_dim=%d",
            self._config.obs_dim,
            self._config.action_dim,
        )

    def _create_spec(self) -> EnvironmentSpec:
        """Create environment specification."""
        return EnvironmentSpec(
            name=self._name,
            observation_space=Space.box(
                low=-np.inf,
                high=np.inf,
                shape=(self._config.obs_dim,),
            ),
            action_space=Space.box(
                low=-1.0,
                high=1.0,
                shape=(self._config.action_dim,),
            ),
            max_episode_steps=self._config.max_episode_steps,
            metadata={
                "mock": True,
                "obs_dim": self._config.obs_dim,
                "action_dim": self._config.action_dim,
            },
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._spec

    async def _reset_impl(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._state = self._rng.standard_normal(self._config.obs_dim)

        info = {
            "episode": self._episode_count,
            "seed": seed,
        }

        return self._state.copy(), info

    async def _step_impl(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and update state."""
        # Optional delay for timing tests
        if self._config.step_delay_ms > 0:
            await asyncio.sleep(self._config.step_delay_ms / 1000.0)

        # Update state based on action
        noise = self._rng.standard_normal(self._config.obs_dim) * 0.1
        self._state = self._state + action.mean() * 0.1 + noise

        # Calculate reward
        reward = self._config.reward_per_step

        # Check termination
        terminated = self._rng.random() < self._config.terminate_prob
        truncated = False

        info = {
            "step": self._step_count,
            "action_norm": float(np.linalg.norm(action)),
        }

        return self._state.copy(), reward, terminated, truncated, info

    async def _observe_impl(self) -> np.ndarray:
        """Get current observation."""
        if self._state is None:
            return np.zeros(self._config.obs_dim)
        return self._state.copy()

    async def _close_impl(self) -> None:
        """Clean up resources."""
        self._state = None


# ---------------------------------------------------------------------------
# Configurable Mock (for specific testing scenarios)
# ---------------------------------------------------------------------------


@register_env("mock:navigation")
class NavigationMockEnvironment(MockEnvironment):
    """Mock environment for navigation tasks."""

    def __init__(self, name: str = "mock:navigation"):
        config = MockConfig(
            obs_dim=12,  # position (3) + velocity (3) + goal (3) + sensors (3)
            action_dim=3,  # velocity commands
            max_episode_steps=500,
            reward_per_step=0.0,
        )
        super().__init__(config=config, name=name)
        self._goal: Optional[np.ndarray] = None

    async def _reset_impl(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = await super()._reset_impl(seed, options)

        # Initialize goal position
        self._goal = self._rng.uniform(-5, 5, size=3)
        obs[6:9] = self._goal

        info["goal"] = self._goal.tolist()
        return obs, info

    async def _step_impl(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, _, terminated, truncated, info = await super()._step_impl(action)

        # Update position based on action
        position = obs[:3]
        position += action * 0.1

        # Calculate distance-based reward
        distance = np.linalg.norm(position - self._goal)
        reward = -distance * 0.1

        # Check if goal reached
        if distance < 0.5:
            terminated = True
            reward = 100.0
            info["success"] = True

        obs[:3] = position
        info["distance_to_goal"] = float(distance)

        return obs, reward, terminated, truncated, info


@register_env("mock:manipulation")
class ManipulationMockEnvironment(MockEnvironment):
    """Mock environment for manipulation tasks."""

    def __init__(self, name: str = "mock:manipulation"):
        config = MockConfig(
            obs_dim=24,  # joint positions (7) + velocities (7) + gripper (2) + target (6) + object (2)
            action_dim=8,  # joint velocities (7) + gripper (1)
            max_episode_steps=200,
            reward_per_step=0.0,
        )
        super().__init__(config=config, name=name)
        self._object_pos: Optional[np.ndarray] = None
        self._target_pos: Optional[np.ndarray] = None

    async def _reset_impl(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = await super()._reset_impl(seed, options)

        # Initialize positions
        self._object_pos = self._rng.uniform(-0.3, 0.3, size=3)
        self._target_pos = self._rng.uniform(-0.3, 0.3, size=3)

        info["object_pos"] = self._object_pos.tolist()
        info["target_pos"] = self._target_pos.tolist()

        return obs, info


# ---------------------------------------------------------------------------
# MuJoCo Environment
# ---------------------------------------------------------------------------


@dataclass
class MuJoCoConfig:
    """Configuration for MuJoCo environment."""

    xml_path: str = ""
    frame_skip: int = 5
    render_mode: Optional[str] = None
    camera_name: Optional[str] = None
    width: int = 480
    height: int = 480


@register_env("mujoco:humanoid")
@register_env("mujoco:ant")
@register_env("mujoco:walker2d")
@register_env("mujoco:hopper")
class MuJoCoEnvironment(AsyncEnvironment[np.ndarray, np.ndarray]):
    """
    MuJoCo-based physics simulation environment.

    Provides:
    - High-fidelity physics simulation
    - Async step execution
    - Rendering support
    - Custom model loading

    Requires mujoco package to be installed.
    """

    def __init__(
        self,
        config: Optional[MuJoCoConfig] = None,
        name: str = "mujoco",
    ):
        super().__init__(name)
        self._config = config or MuJoCoConfig()
        self._model = None
        self._data = None
        self._viewer = None
        self._initialized = False
        self._pending_actions: Dict[str, Dict[str, Any]] = {}

        # Try to load MuJoCo
        self._mujoco = self._load_mujoco()

        if self._mujoco and self._config.xml_path:
            self._initialize_model()

        logger.info(
            "[env][mujoco] initialized name=%s model=%s",
            name,
            self._config.xml_path or "none",
        )

    def _load_mujoco(self) -> Optional[Any]:
        """Try to import mujoco."""
        try:
            import mujoco

            return mujoco
        except ImportError:
            logger.warning("[env][mujoco] mujoco not available, using mock fallback")
            return None

    def _initialize_model(self) -> None:
        """Initialize MuJoCo model from XML."""
        if not self._mujoco:
            return

        try:
            self._model = self._mujoco.MjModel.from_xml_path(self._config.xml_path)
            self._data = self._mujoco.MjData(self._model)
            self._initialized = True

            logger.info(
                "[env][mujoco] loaded model nq=%d nv=%d nu=%d",
                self._model.nq,
                self._model.nv,
                self._model.nu,
            )
        except Exception as e:
            logger.error("[env][mujoco] failed to load model: %s", e)

    def _create_spec_from_model(self) -> EnvironmentSpec:
        """Create spec from loaded model."""
        if not self._initialized:
            # Return default spec for mock mode
            return EnvironmentSpec(
                name=self._name,
                observation_space=Space.box(-np.inf, np.inf, shape=(17,)),
                action_space=Space.box(-1.0, 1.0, shape=(6,)),
                max_episode_steps=1000,
            )

        obs_dim = self._model.nq + self._model.nv
        act_dim = self._model.nu

        return EnvironmentSpec(
            name=self._name,
            observation_space=Space.box(-np.inf, np.inf, shape=(obs_dim,)),
            action_space=Space.box(-1.0, 1.0, shape=(act_dim,)),
            max_episode_steps=1000,
            metadata={
                "xml_path": self._config.xml_path,
                "frame_skip": self._config.frame_skip,
            },
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._create_spec_from_model()

    def _get_obs(self) -> np.ndarray:
        """Get current observation from MuJoCo state."""
        if not self._initialized:
            return np.zeros(17)

        return np.concatenate(
            [
                self._data.qpos.flat.copy(),
                self._data.qvel.flat.copy(),
            ]
        )

    async def _reset_impl(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset MuJoCo simulation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._initialized:
            self._mujoco.mj_resetData(self._model, self._data)

            # Add noise to initial state
            noise_scale = options.get("noise_scale", 0.01) if options else 0.01
            self._data.qpos[:] += self._rng.uniform(
                -noise_scale, noise_scale, size=self._model.nq
            )
            self._data.qvel[:] += self._rng.uniform(
                -noise_scale, noise_scale, size=self._model.nv
            )

            self._mujoco.mj_forward(self._model, self._data)

        info = {
            "episode": self._episode_count,
            "initialized": self._initialized,
        }

        return self._get_obs(), info

    async def _step_impl(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action in MuJoCo."""
        if not self._initialized:
            # Mock fallback
            obs = self._rng.standard_normal(17)
            return obs, 1.0, False, False, {"mock": True}

        # Apply action and step simulation
        self._data.ctrl[:] = action

        for _ in range(self._config.frame_skip):
            self._mujoco.mj_step(self._model, self._data)

        obs = self._get_obs()

        # Calculate reward (example: forward velocity)
        reward = self._data.qvel[0]

        # Check termination (example: height check)
        height = self._data.qpos[2] if len(self._data.qpos) > 2 else 1.0
        terminated = height < 0.2

        info = {
            "step": self._step_count,
            "sim_time": self._data.time,
        }

        return obs, reward, terminated, False, info

    async def _observe_impl(self) -> np.ndarray:
        """Get current observation."""
        return self._get_obs()

    async def _close_impl(self) -> None:
        """Clean up MuJoCo resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        self._model = None
        self._data = None
        self._initialized = False

    async def step_async(self, action: np.ndarray) -> str:
        """Start async action execution."""
        action_id = str(uuid.uuid4())

        # For MuJoCo, we can run the simulation in a thread
        async def execute():
            return await self._step_impl(action)

        self._pending_actions[action_id] = {
            "task": asyncio.create_task(execute()),
            "start_time": time.time(),
        }

        return action_id

    async def step_wait(
        self,
        action_id: str,
        timeout: Optional[float] = None,
    ) -> StepResult:
        """Wait for async action completion."""
        if action_id not in self._pending_actions:
            raise ValueError(f"Unknown action ID: {action_id}")

        pending = self._pending_actions[action_id]

        try:
            result = await asyncio.wait_for(
                pending["task"],
                timeout=timeout,
            )
            obs, reward, terminated, truncated, info = result

            self._step_count += 1
            del self._pending_actions[action_id]

            return StepResult(
                observation=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Action {action_id} timed out") from None

    async def step_poll(self, action_id: str) -> Optional[StepResult]:
        """Poll async action status."""
        if action_id not in self._pending_actions:
            return None

        pending = self._pending_actions[action_id]
        if pending["task"].done():
            return await self.step_wait(action_id)

        return None

    async def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if not self._initialized:
            return None

        if self._config.render_mode != "rgb_array":
            return None

        try:
            renderer = self._mujoco.Renderer(
                self._model,
                self._config.height,
                self._config.width,
            )
            renderer.update_scene(self._data)
            return renderer.render()
        except Exception as e:
            logger.warning("[env][mujoco] render failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# PyBullet Environment
# ---------------------------------------------------------------------------


@dataclass
class PyBulletConfig:
    """Configuration for PyBullet environment."""

    urdf_path: str = ""
    use_gui: bool = False
    physics_dt: float = 1 / 240
    control_dt: float = 1 / 60
    gravity: Tuple[float, float, float] = (0, 0, -9.81)
    base_position: Tuple[float, float, float] = (0, 0, 0)
    use_fixed_base: bool = True


@register_env("pybullet")
@register_env("pybullet:robot")
class PyBulletEnvironment(AsyncEnvironment[np.ndarray, np.ndarray]):
    """
    PyBullet-based physics simulation environment.

    Provides:
    - Physics simulation with PyBullet
    - URDF/SDF model loading
    - Multiple robots and objects
    - Camera rendering

    Requires pybullet package to be installed.
    """

    def __init__(
        self,
        config: Optional[PyBulletConfig] = None,
        name: str = "pybullet",
    ):
        super().__init__(name)
        self._config = config or PyBulletConfig()
        self._client_id: int = -1
        self._robot_id: int = -1
        self._joint_ids: List[int] = []
        self._joint_limits: List[Tuple[float, float]] = []
        self._initialized = False
        self._pending_actions: Dict[str, Dict[str, Any]] = {}

        # Try to load PyBullet
        self._pybullet = self._load_pybullet()

        logger.info(
            "[env][pybullet] initialized name=%s urdf=%s gui=%s",
            name,
            self._config.urdf_path or "none",
            self._config.use_gui,
        )

    def _load_pybullet(self) -> Optional[Any]:
        """Try to import pybullet."""
        try:
            import pybullet as p
            import pybullet_data

            return p
        except ImportError:
            logger.warning(
                "[env][pybullet] pybullet not available, using mock fallback"
            )
            return None

    async def initialize(self) -> None:
        """Initialize PyBullet simulation."""
        if not self._pybullet:
            self._initialized = False
            return

        p = self._pybullet

        # Connect to physics server
        if self._config.use_gui:
            self._client_id = p.connect(p.GUI)
        else:
            self._client_id = p.connect(p.DIRECT)

        # Set physics parameters
        p.setGravity(*self._config.gravity, physicsClientId=self._client_id)
        p.setTimeStep(self._config.physics_dt, physicsClientId=self._client_id)

        # Load ground plane
        try:
            import pybullet_data

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        except Exception as e:
            logger.warning("[env][pybullet] failed to load ground plane: %s", e)

        # Load robot if URDF provided
        if self._config.urdf_path:
            try:
                self._robot_id = p.loadURDF(
                    self._config.urdf_path,
                    basePosition=self._config.base_position,
                    useFixedBase=self._config.use_fixed_base,
                    physicsClientId=self._client_id,
                )
                self._discover_joints()
                self._initialized = True

                logger.info(
                    "[env][pybullet] loaded robot id=%d joints=%d",
                    self._robot_id,
                    len(self._joint_ids),
                )
            except Exception as e:
                logger.error("[env][pybullet] failed to load URDF: %s", e)
        else:
            self._initialized = True

    def _discover_joints(self) -> None:
        """Discover movable joints in the robot."""
        if not self._pybullet or self._robot_id < 0:
            return

        p = self._pybullet
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client_id)

        for i in range(n_joints):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._client_id)
            joint_type = info[2]

            # Only include revolute and prismatic joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self._joint_ids.append(i)
                lower_limit = info[8]
                upper_limit = info[9]
                self._joint_limits.append((lower_limit, upper_limit))

    def _create_spec_from_model(self) -> EnvironmentSpec:
        """Create spec from loaded model."""
        n_joints = len(self._joint_ids) if self._joint_ids else 7

        return EnvironmentSpec(
            name=self._name,
            observation_space=Space.box(
                low=-np.inf,
                high=np.inf,
                shape=(n_joints * 2,),  # positions + velocities
            ),
            action_space=Space.box(
                low=-1.0,
                high=1.0,
                shape=(n_joints,),
            ),
            max_episode_steps=1000,
            metadata={
                "urdf_path": self._config.urdf_path,
                "n_joints": n_joints,
            },
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return self._create_spec_from_model()

    def _get_obs(self) -> np.ndarray:
        """Get current observation from PyBullet state."""
        if not self._initialized or not self._pybullet or self._robot_id < 0:
            n_joints = len(self._joint_ids) if self._joint_ids else 7
            return np.zeros(n_joints * 2)

        p = self._pybullet
        positions = []
        velocities = []

        for joint_id in self._joint_ids:
            state = p.getJointState(
                self._robot_id,
                joint_id,
                physicsClientId=self._client_id,
            )
            positions.append(state[0])  # position
            velocities.append(state[1])  # velocity

        return np.concatenate([positions, velocities])

    async def _reset_impl(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset PyBullet simulation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if not self._initialized:
            await self.initialize()

        if self._pybullet and self._robot_id >= 0:
            p = self._pybullet

            # Get initial positions from options or config
            initial_positions = None
            if options and "initial_qpos" in options:
                initial_positions = options["initial_qpos"]

            # Reset joint states
            for i, joint_id in enumerate(self._joint_ids):
                pos = (
                    initial_positions[i]
                    if initial_positions and i < len(initial_positions)
                    else 0.0
                )
                p.resetJointState(
                    self._robot_id,
                    joint_id,
                    targetValue=pos,
                    targetVelocity=0.0,
                    physicsClientId=self._client_id,
                )

        info = {
            "episode": self._episode_count,
            "initialized": self._initialized,
        }

        return self._get_obs(), info

    async def _step_impl(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action in PyBullet."""
        if not self._initialized or not self._pybullet or self._robot_id < 0:
            # Mock fallback
            n_joints = len(self._joint_ids) if self._joint_ids else 7
            obs = self._rng.standard_normal(n_joints * 2)
            return obs, 0.0, False, False, {"mock": True}

        p = self._pybullet

        # Apply action to joints
        for i, joint_id in enumerate(self._joint_ids):
            if i < len(action):
                # Scale action from [-1, 1] to joint limits
                if i < len(self._joint_limits):
                    low, high = self._joint_limits[i]
                    if high > low:
                        target = low + (action[i] + 1) * 0.5 * (high - low)
                    else:
                        target = action[i]
                else:
                    target = action[i]

                p.setJointMotorControl2(
                    self._robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    physicsClientId=self._client_id,
                )

        # Step simulation
        n_substeps = int(self._config.control_dt / self._config.physics_dt)
        for _ in range(n_substeps):
            p.stepSimulation(physicsClientId=self._client_id)

        obs = self._get_obs()

        # Simple reward (can be overridden in subclass)
        reward = 0.0

        # Check termination (can be overridden in subclass)
        terminated = False
        truncated = self._step_count >= self.spec.max_episode_steps

        info = {
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    async def _observe_impl(self) -> np.ndarray:
        """Get current observation."""
        return self._get_obs()

    async def _close_impl(self) -> None:
        """Clean up PyBullet resources."""
        if self._pybullet and self._client_id >= 0:
            self._pybullet.disconnect(self._client_id)
            self._client_id = -1

        self._robot_id = -1
        self._joint_ids = []
        self._joint_limits = []
        self._initialized = False

    async def step_async(self, action: np.ndarray) -> str:
        """Start async action execution."""
        action_id = str(uuid.uuid4())

        async def execute():
            return await self._step_impl(action)

        self._pending_actions[action_id] = {
            "task": asyncio.create_task(execute()),
            "start_time": time.time(),
        }

        return action_id

    async def step_wait(
        self,
        action_id: str,
        timeout: Optional[float] = None,
    ) -> StepResult:
        """Wait for async action completion."""
        if action_id not in self._pending_actions:
            raise ValueError(f"Unknown action ID: {action_id}")

        pending = self._pending_actions[action_id]

        try:
            result = await asyncio.wait_for(
                pending["task"],
                timeout=timeout,
            )
            obs, reward, terminated, truncated, info = result

            self._step_count += 1
            del self._pending_actions[action_id]

            return StepResult(
                observation=obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Action {action_id} timed out") from None

    async def step_poll(self, action_id: str) -> Optional[StepResult]:
        """Poll async action status."""
        if action_id not in self._pending_actions:
            return None

        pending = self._pending_actions[action_id]
        if pending["task"].done():
            return await self.step_wait(action_id)

        return None

    async def render(self) -> Optional[np.ndarray]:
        """Render the environment from camera."""
        if not self._initialized or not self._pybullet:
            return None

        p = self._pybullet

        try:
            # Get camera image
            width = 480
            height = 480

            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0.5],
                distance=2.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=width / height,
                nearVal=0.1,
                farVal=100.0,
            )

            _, _, rgba, _, _ = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                physicsClientId=self._client_id,
            )

            # Convert to RGB
            rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
            return rgb

        except Exception as e:
            logger.warning("[env][pybullet] render failed: %s", e)
            return None

    def add_object(
        self,
        urdf_path: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ) -> int:
        """Add an object to the simulation."""
        if not self._pybullet or self._client_id < 0:
            return -1

        p = self._pybullet

        try:
            obj_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self._client_id,
            )
            return obj_id
        except Exception as e:
            logger.error("[env][pybullet] failed to add object: %s", e)
            return -1

    def get_object_pose(
        self, object_id: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get position and orientation of an object."""
        if not self._pybullet or self._client_id < 0:
            return None

        p = self._pybullet
        pos, orn = p.getBasePositionAndOrientation(
            object_id,
            physicsClientId=self._client_id,
        )
        return np.array(pos), np.array(orn)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def list_backends() -> List[str]:
    """List available environment backends."""
    from agi.env.base import EnvironmentRegistry

    return EnvironmentRegistry.list()


def get_backend_info(name: str) -> Dict[str, Any]:
    """Get information about an environment backend."""
    from agi.env.base import EnvironmentRegistry

    env_class = EnvironmentRegistry.get(name)
    if env_class is None:
        return {"error": f"Unknown backend: {name}"}

    return {
        "name": name,
        "class": env_class.__name__,
        "doc": env_class.__doc__,
        "is_async": issubclass(env_class, AsyncEnvironment),
    }
