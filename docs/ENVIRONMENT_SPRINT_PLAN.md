# Environment Interface Sprint Plan

## Overview

The Environment Interface subsystem provides the bridge between the AGI-HPC cognitive architecture and the physical or simulated world. It abstracts sensors, actuators, and simulation environments behind a unified API, enabling the RH (Right Hemisphere) to perceive and act in diverse environments without knowing implementation details.

## Current State Assessment

### Implemented (Scaffolding)
| Component | Status | Location |
|-----------|--------|----------|
| `env.proto` | **Exists** | `proto/env.proto` |
| RH Environment integration | **Basic** | `src/agi/rh/world_model.py` |
| `env_config.yaml` | **Mentioned** | Configuration reference |

### Key Gaps
1. **No unified sensor API** - Ad-hoc sensor access
2. **No actuator abstraction** - Direct hardware calls
3. **No simulation interface** - No standard sim protocol
4. **No sensor fusion** - Raw sensor data only
5. **No state estimation** - No Kalman/particle filters
6. **No environment registry** - Hardcoded configurations
7. **No time synchronization** - Inconsistent timestamps
8. **No recording/playback** - No data logging

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ENVIRONMENT INTERFACE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     PERCEPTION LAYER                                 │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │    Camera    │  │    LiDAR     │  │    IMU       │             │   │
│   │   │   Sensors    │  │   Sensors    │  │   Sensors    │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │    Force     │  │   Proximity  │  │   Joint      │             │   │
│   │   │   Torque     │  │   Sensors    │  │   Encoders   │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                     SENSOR FUSION                                    │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │   Kalman     │  │   Particle   │  │   Object     │             │   │
│   │   │   Filter     │  │   Filter     │  │   Tracking   │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                     ENVIRONMENT CORE                                 │   │
│   │                                                                      │   │
│   │   ┌──────────────────────────────────────────────────────────────┐  │   │
│   │   │                    Environment Manager                        │  │   │
│   │   │                                                               │  │   │
│   │   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │  │   │
│   │   │   │ Observe  │  │  Step    │  │  Reset   │  │  Close   │    │  │   │
│   │   │   └──────────┘  └──────────┘  └──────────┘  └──────────┘    │  │   │
│   │   └──────────────────────────────────────────────────────────────┘  │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                     ACTION LAYER                                     │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │   Motion     │  │   Gripper    │  │   Base       │             │   │
│   │   │   Control    │  │   Control    │  │   Control    │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                     ENVIRONMENT BACKENDS                             │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │   MuJoCo     │  │   PyBullet   │  │   Isaac      │             │   │
│   │   │    Sim       │  │     Sim      │  │    Sim       │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │   │
│   │   │    ROS2      │  │   Hardware   │  │   Digital    │             │   │
│   │   │   Bridge     │  │   Direct     │  │    Twin      │             │   │
│   │   └──────────────┘  └──────────────┘  └──────────────┘             │   │
│   │                                                                      │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Use Cases in AGI-HPC

| Component | Use Case | Interface |
|-----------|----------|-----------|
| RH World Model | State observation | `observe()` |
| RH Simulation | Action simulation | `step()`, `predict()` |
| LH Planner | Skill execution | `execute_action()` |
| Safety | Emergency stop | `stop()` |
| Memory | Episode recording | `record()`, `playback()` |

---

## Sprint 1: Core Environment Interface

**Goal**: Define and implement the core environment abstraction.

### Tasks

#### 1.1 Environment Protocol

```python
# src/agi/env/base.py
"""Base environment interface following Gymnasium API."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# Type variables for observation and action spaces
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class SpaceType(Enum):
    """Types of observation/action spaces."""
    DISCRETE = "discrete"
    BOX = "box"
    DICT = "dict"
    MULTI_DISCRETE = "multi_discrete"


@dataclass
class Space:
    """Base space definition."""

    space_type: SpaceType
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    n: Optional[int] = None  # For discrete
    nvec: Optional[List[int]] = None  # For multi-discrete

    def sample(self) -> Any:
        """Sample a random element from the space."""
        if self.space_type == SpaceType.BOX:
            return np.random.uniform(self.low, self.high).astype(self.dtype)
        elif self.space_type == SpaceType.DISCRETE:
            return np.random.randint(0, self.n)
        elif self.space_type == SpaceType.MULTI_DISCRETE:
            return np.array([np.random.randint(0, n) for n in self.nvec])
        else:
            raise NotImplementedError()

    def contains(self, x: Any) -> bool:
        """Check if x is in the space."""
        if self.space_type == SpaceType.BOX:
            return (
                isinstance(x, np.ndarray) and
                x.shape == self.shape and
                np.all(x >= self.low) and
                np.all(x <= self.high)
            )
        elif self.space_type == SpaceType.DISCRETE:
            return isinstance(x, (int, np.integer)) and 0 <= x < self.n
        else:
            return True


@dataclass
class EnvironmentSpec:
    """Environment specification."""

    name: str
    observation_space: Space
    action_space: Space
    max_episode_steps: int = 1000
    reward_threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result from environment step."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResetResult:
    """Result from environment reset."""

    observation: Any
    info: Dict[str, Any] = field(default_factory=dict)


class Environment(ABC, Generic[ObsType, ActType]):
    """Abstract base class for environments.

    Follows the Gymnasium API with extensions for robotics.
    """

    @property
    @abstractmethod
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        """Get observation space."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> Space:
        """Get action space."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    async def step(self, action: ActType) -> StepResult:
        """Execute action and return next state.

        Args:
            action: Action to execute

        Returns:
            Observation, reward, terminated, truncated, info
        """
        ...

    @abstractmethod
    async def observe(self) -> ObsType:
        """Get current observation without stepping."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up environment resources."""
        ...

    async def render(self) -> Optional[np.ndarray]:
        """Render the environment (optional)."""
        return None

    def seed(self, seed: int) -> None:
        """Set random seed."""
        np.random.seed(seed)


class AsyncEnvironment(Environment[ObsType, ActType]):
    """Environment with async step support for real hardware."""

    @abstractmethod
    async def step_async(self, action: ActType) -> str:
        """Start async action execution.

        Returns:
            Action ID for tracking
        """
        ...

    @abstractmethod
    async def step_wait(self, action_id: str, timeout: float = None) -> StepResult:
        """Wait for async action completion.

        Args:
            action_id: ID from step_async
            timeout: Maximum wait time

        Returns:
            Step result
        """
        ...

    @abstractmethod
    async def step_poll(self, action_id: str) -> Optional[StepResult]:
        """Poll async action status.

        Returns:
            Step result if complete, None if still running
        """
        ...
```

#### 1.2 Observation Types

```python
# src/agi/env/observations.py
"""Observation data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CameraObservation:
    """Camera sensor observation."""

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None  # (H, W) float32, meters
    semantic: Optional[np.ndarray] = None  # (H, W) int, class IDs
    intrinsics: Optional[np.ndarray] = None  # (3, 3) camera matrix
    extrinsics: Optional[np.ndarray] = None  # (4, 4) pose
    timestamp: float = 0.0
    frame_id: str = ""


@dataclass
class LidarObservation:
    """LiDAR sensor observation."""

    points: np.ndarray  # (N, 3) or (N, 4) with intensity
    intensities: Optional[np.ndarray] = None  # (N,)
    ring_ids: Optional[np.ndarray] = None  # (N,) int
    extrinsics: Optional[np.ndarray] = None  # (4, 4) pose
    timestamp: float = 0.0
    frame_id: str = ""


@dataclass
class IMUObservation:
    """IMU sensor observation."""

    linear_acceleration: np.ndarray  # (3,) m/s^2
    angular_velocity: np.ndarray  # (3,) rad/s
    orientation: Optional[np.ndarray] = None  # (4,) quaternion
    timestamp: float = 0.0


@dataclass
class JointState:
    """Robot joint state."""

    position: np.ndarray  # (n_joints,) rad
    velocity: np.ndarray  # (n_joints,) rad/s
    effort: Optional[np.ndarray] = None  # (n_joints,) Nm
    names: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class GripperState:
    """Gripper state."""

    position: float  # 0 (closed) to 1 (open)
    force: Optional[float] = None  # Gripping force
    is_grasping: bool = False
    timestamp: float = 0.0


@dataclass
class ForceTorqueSensor:
    """Force-torque sensor observation."""

    force: np.ndarray  # (3,) N
    torque: np.ndarray  # (3,) Nm
    frame_id: str = ""
    timestamp: float = 0.0


@dataclass
class RobotState:
    """Complete robot state observation."""

    joint_state: JointState
    gripper_state: Optional[GripperState] = None
    end_effector_pose: Optional[np.ndarray] = None  # (4, 4)
    base_pose: Optional[np.ndarray] = None  # (4, 4) for mobile robots
    force_torque: Optional[ForceTorqueSensor] = None
    timestamp: float = 0.0


@dataclass
class WorldState:
    """Full world state observation."""

    robot: RobotState
    cameras: Dict[str, CameraObservation] = field(default_factory=dict)
    lidars: Dict[str, LidarObservation] = field(default_factory=dict)
    objects: List[Dict] = field(default_factory=list)  # Detected objects
    timestamp: float = 0.0


@dataclass
class ObjectDetection:
    """Detected object in the scene."""

    class_name: str
    class_id: int
    confidence: float
    bbox_2d: Optional[np.ndarray] = None  # (4,) x1, y1, x2, y2
    bbox_3d: Optional[np.ndarray] = None  # (8, 3) corners
    pose: Optional[np.ndarray] = None  # (4, 4)
    mask: Optional[np.ndarray] = None  # (H, W) bool
    point_cloud: Optional[np.ndarray] = None  # (N, 3)
```

#### 1.3 Action Types

```python
# src/agi/env/actions.py
"""Action data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class ControlMode(Enum):
    """Robot control modes."""
    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"
    CARTESIAN = "cartesian"
    IMPEDANCE = "impedance"


@dataclass
class JointCommand:
    """Joint-level control command."""

    positions: Optional[np.ndarray] = None  # Target positions
    velocities: Optional[np.ndarray] = None  # Target velocities
    efforts: Optional[np.ndarray] = None  # Target efforts/torques
    mode: ControlMode = ControlMode.POSITION
    duration: float = 0.0  # Execution time (0 = instant)


@dataclass
class CartesianCommand:
    """Cartesian space control command."""

    target_pose: np.ndarray  # (4, 4) or (7,) position + quaternion
    velocity: Optional[np.ndarray] = None  # (6,) linear + angular
    frame_id: str = "world"
    mode: ControlMode = ControlMode.CARTESIAN
    duration: float = 0.0


@dataclass
class GripperCommand:
    """Gripper control command."""

    position: float  # 0 (closed) to 1 (open)
    force: Optional[float] = None  # Max force
    speed: Optional[float] = None  # Opening/closing speed


@dataclass
class BaseCommand:
    """Mobile base control command."""

    linear_velocity: np.ndarray  # (3,) m/s
    angular_velocity: np.ndarray  # (3,) rad/s
    frame_id: str = "base_link"


@dataclass
class RobotAction:
    """Complete robot action."""

    joint_command: Optional[JointCommand] = None
    cartesian_command: Optional[CartesianCommand] = None
    gripper_command: Optional[GripperCommand] = None
    base_command: Optional[BaseCommand] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """A high-level skill to execute."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    retry_count: int = 0
```

#### 1.4 Configuration

```yaml
# configs/env_config.yaml
environment:
  name: "agi-hpc-env"
  type: "mujoco"  # mujoco, pybullet, isaac, ros2, hardware

  simulation:
    physics_dt: 0.002  # Physics timestep
    control_dt: 0.02   # Control timestep
    render: true
    render_mode: "rgb_array"

  robot:
    name: "franka_panda"
    urdf: "models/panda/panda.urdf"
    initial_joint_positions: [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    control_mode: "position"

  sensors:
    cameras:
      - name: "head_camera"
        type: "rgbd"
        width: 640
        height: 480
        fov: 60
        frame_id: "head_camera_link"
        rate_hz: 30

      - name: "wrist_camera"
        type: "rgb"
        width: 320
        height: 240
        fov: 90
        frame_id: "wrist_camera_link"
        rate_hz: 60

    force_torque:
      - name: "wrist_ft"
        frame_id: "wrist_ft_link"
        rate_hz: 1000

    lidar:
      - name: "base_lidar"
        type: "velodyne"
        channels: 16
        range: 100.0
        rate_hz: 10

  scene:
    table:
      position: [0.5, 0, 0]
      size: [0.8, 1.2, 0.75]

    objects:
      - name: "red_cube"
        type: "box"
        size: [0.05, 0.05, 0.05]
        color: [1, 0, 0, 1]
        position: [0.5, 0.1, 0.8]

  recording:
    enabled: false
    path: "/var/log/agi/recordings"
    format: "hdf5"  # hdf5, rosbag
    topics:
      - "camera/*"
      - "joint_states"
      - "actions"
```

### Acceptance Criteria
```bash
# Test environment interface
python -c "
import asyncio
from agi.env.mujoco_env import MujocoEnvironment

async def test():
    env = MujocoEnvironment(config_path='configs/env_config.yaml')
    await env.initialize()

    result = await env.reset()
    print(f'Initial obs shape: {result.observation.robot.joint_state.position.shape}')

    action = env.action_space.sample()
    step_result = await env.step(action)
    print(f'Reward: {step_result.reward}')

    await env.close()

asyncio.run(test())
"
```

---

## Sprint 2: Simulation Backends

**Goal**: Implement simulation backends for MuJoCo, PyBullet, and Isaac Sim.

### Tasks

#### 2.1 MuJoCo Backend

```python
# src/agi/env/backends/mujoco_env.py
"""MuJoCo simulation environment."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import mujoco
import numpy as np

from agi.env.base import (
    AsyncEnvironment,
    EnvironmentSpec,
    ResetResult,
    Space,
    SpaceType,
    StepResult,
)
from agi.env.observations import (
    CameraObservation,
    JointState,
    GripperState,
    RobotState,
    WorldState,
)
from agi.env.actions import RobotAction, JointCommand

logger = logging.getLogger(__name__)


class MujocoEnvironment(AsyncEnvironment[WorldState, RobotAction]):
    """MuJoCo simulation environment.

    Features:
    - Physics simulation with MuJoCo
    - Multiple camera support
    - Contact/force sensing
    - Domain randomization
    """

    def __init__(
        self,
        model_path: str,
        config: Dict[str, Any] = None,
    ) -> None:
        """Initialize MuJoCo environment."""
        self.model_path = model_path
        self.config = config or {}

        self._model: mujoco.MjModel = None
        self._data: mujoco.MjData = None
        self._renderer: mujoco.Renderer = None

        self._physics_dt = self.config.get("physics_dt", 0.002)
        self._control_dt = self.config.get("control_dt", 0.02)
        self._n_substeps = int(self._control_dt / self._physics_dt)

        self._episode_step = 0
        self._max_episode_steps = self.config.get("max_episode_steps", 1000)

    async def initialize(self) -> None:
        """Initialize the simulation."""
        self._model = mujoco.MjModel.from_xml_path(self.model_path)
        self._data = mujoco.MjData(self._model)

        # Configure timestep
        self._model.opt.timestep = self._physics_dt

        # Initialize renderer if needed
        if self.config.get("render", False):
            self._renderer = mujoco.Renderer(self._model, 480, 640)

        logger.info(
            "MuJoCo environment initialized: %d bodies, %d joints",
            self._model.nbody,
            self._model.njnt,
        )

    @property
    def spec(self) -> EnvironmentSpec:
        """Get environment specification."""
        return EnvironmentSpec(
            name="mujoco_panda",
            observation_space=self.observation_space,
            action_space=self.action_space,
            max_episode_steps=self._max_episode_steps,
        )

    @property
    def observation_space(self) -> Space:
        """Get observation space."""
        # Joint positions + velocities
        n_joints = self._model.nq if self._model else 7
        dim = n_joints * 2

        return Space(
            space_type=SpaceType.BOX,
            shape=(dim,),
            dtype="float32",
            low=np.full(dim, -np.inf, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32),
        )

    @property
    def action_space(self) -> Space:
        """Get action space."""
        n_actuators = self._model.nu if self._model else 7

        return Space(
            space_type=SpaceType.BOX,
            shape=(n_actuators,),
            dtype="float32",
            low=np.full(n_actuators, -1.0, dtype=np.float32),
            high=np.full(n_actuators, 1.0, dtype=np.float32),
        )

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Reset simulation state
        mujoco.mj_resetData(self._model, self._data)

        # Set initial configuration
        if options and "initial_qpos" in options:
            self._data.qpos[:] = options["initial_qpos"]
        elif "initial_joint_positions" in self.config:
            n_joints = len(self.config["initial_joint_positions"])
            self._data.qpos[:n_joints] = self.config["initial_joint_positions"]

        # Forward kinematics
        mujoco.mj_forward(self._model, self._data)

        self._episode_step = 0

        observation = await self.observe()

        return ResetResult(
            observation=observation,
            info={"step": 0},
        )

    async def step(self, action: RobotAction) -> StepResult:
        """Execute action and step simulation."""
        # Apply action
        if action.joint_command is not None:
            self._apply_joint_command(action.joint_command)
        else:
            # Raw action array
            self._data.ctrl[:] = np.array(action)

        # Step physics
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        self._episode_step += 1

        # Get observation
        observation = await self.observe()

        # Compute reward (task-specific)
        reward = self._compute_reward()

        # Check termination
        terminated = self._is_terminated()
        truncated = self._episode_step >= self._max_episode_steps

        info = {
            "step": self._episode_step,
            "sim_time": self._data.time,
        }

        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    async def observe(self) -> WorldState:
        """Get current observation."""
        # Joint state
        n_joints = min(7, self._model.nq)  # Assuming 7-DOF arm
        joint_state = JointState(
            position=self._data.qpos[:n_joints].copy(),
            velocity=self._data.qvel[:n_joints].copy(),
            effort=self._data.qfrc_actuator[:n_joints].copy()
            if hasattr(self._data, "qfrc_actuator") else None,
            timestamp=self._data.time,
        )

        # Gripper state (if available)
        gripper_state = None
        if self._model.nu > n_joints:
            gripper_state = GripperState(
                position=float(self._data.qpos[n_joints]),
                timestamp=self._data.time,
            )

        # End-effector pose
        ee_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        ) if mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        ) >= 0 else None

        ee_pose = None
        if ee_site_id is not None:
            ee_pos = self._data.site_xpos[ee_site_id]
            ee_mat = self._data.site_xmat[ee_site_id].reshape(3, 3)
            ee_pose = np.eye(4)
            ee_pose[:3, :3] = ee_mat
            ee_pose[:3, 3] = ee_pos

        # Robot state
        robot_state = RobotState(
            joint_state=joint_state,
            gripper_state=gripper_state,
            end_effector_pose=ee_pose,
            timestamp=self._data.time,
        )

        # Camera observations
        cameras = {}
        if self._renderer and self.config.get("cameras"):
            for cam_config in self.config["cameras"]:
                cam_name = cam_config["name"]
                cam_id = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
                )
                if cam_id >= 0:
                    self._renderer.update_scene(self._data, cam_id)
                    rgb = self._renderer.render()
                    cameras[cam_name] = CameraObservation(
                        rgb=rgb,
                        timestamp=self._data.time,
                        frame_id=cam_name,
                    )

        return WorldState(
            robot=robot_state,
            cameras=cameras,
            timestamp=self._data.time,
        )

    async def close(self) -> None:
        """Clean up resources."""
        if self._renderer:
            del self._renderer
            self._renderer = None

        logger.info("MuJoCo environment closed")

    async def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self._renderer:
            self._renderer.update_scene(self._data)
            return self._renderer.render()
        return None

    def _apply_joint_command(self, cmd: JointCommand) -> None:
        """Apply joint command to actuators."""
        if cmd.positions is not None:
            # Position control
            n = len(cmd.positions)
            self._data.ctrl[:n] = cmd.positions
        elif cmd.velocities is not None:
            # Velocity control (if supported)
            n = len(cmd.velocities)
            self._data.ctrl[:n] = cmd.velocities
        elif cmd.efforts is not None:
            # Effort/torque control
            n = len(cmd.efforts)
            self._data.ctrl[:n] = cmd.efforts

    def _compute_reward(self) -> float:
        """Compute task reward (override in subclass)."""
        return 0.0

    def _is_terminated(self) -> bool:
        """Check if episode should terminate (override in subclass)."""
        return False

    async def step_async(self, action: RobotAction) -> str:
        """Start async action (for compatibility)."""
        result = await self.step(action)
        return "sync_action"

    async def step_wait(self, action_id: str, timeout: float = None) -> StepResult:
        """Wait for async action (returns immediately for sync sim)."""
        return await self.observe()

    async def step_poll(self, action_id: str) -> Optional[StepResult]:
        """Poll async action (always complete for sync sim)."""
        return await self.step_wait(action_id)
```

#### 2.2 PyBullet Backend

```python
# src/agi/env/backends/pybullet_env.py
"""PyBullet simulation environment."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pybullet as p
import pybullet_data

from agi.env.base import (
    Environment,
    EnvironmentSpec,
    ResetResult,
    Space,
    SpaceType,
    StepResult,
)
from agi.env.observations import JointState, RobotState, WorldState
from agi.env.actions import RobotAction, JointCommand

logger = logging.getLogger(__name__)


class PyBulletEnvironment(Environment[WorldState, RobotAction]):
    """PyBullet simulation environment."""

    def __init__(
        self,
        urdf_path: str,
        config: Dict[str, Any] = None,
        gui: bool = False,
    ) -> None:
        """Initialize PyBullet environment."""
        self.urdf_path = urdf_path
        self.config = config or {}
        self.gui = gui

        self._client_id: int = -1
        self._robot_id: int = -1
        self._joint_ids: list = []
        self._physics_dt = self.config.get("physics_dt", 1/240)
        self._control_dt = self.config.get("control_dt", 1/60)

        self._episode_step = 0

    async def initialize(self) -> None:
        """Initialize PyBullet."""
        if self.gui:
            self._client_id = p.connect(p.GUI)
        else:
            self._client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        p.setTimeStep(self._physics_dt, physicsClientId=self._client_id)

        # Load ground plane
        p.loadURDF("plane.urdf", physicsClientId=self._client_id)

        # Load robot
        self._robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client_id,
        )

        # Get movable joints
        n_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client_id)
        for i in range(n_joints):
            info = p.getJointInfo(self._robot_id, i, physicsClientId=self._client_id)
            if info[2] != p.JOINT_FIXED:
                self._joint_ids.append(i)

        logger.info(
            "PyBullet environment initialized: %d joints",
            len(self._joint_ids),
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="pybullet_robot",
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @property
    def observation_space(self) -> Space:
        n_joints = len(self._joint_ids) if self._joint_ids else 7
        dim = n_joints * 2
        return Space(
            space_type=SpaceType.BOX,
            shape=(dim,),
            dtype="float32",
            low=np.full(dim, -np.inf, dtype=np.float32),
            high=np.full(dim, np.inf, dtype=np.float32),
        )

    @property
    def action_space(self) -> Space:
        n_joints = len(self._joint_ids) if self._joint_ids else 7
        return Space(
            space_type=SpaceType.BOX,
            shape=(n_joints,),
            dtype="float32",
            low=np.full(n_joints, -1.0, dtype=np.float32),
            high=np.full(n_joints, 1.0, dtype=np.float32),
        )

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Reset joint positions
        initial_positions = (
            options.get("initial_qpos")
            if options else self.config.get("initial_joint_positions", [0] * len(self._joint_ids))
        )

        for i, joint_id in enumerate(self._joint_ids):
            if i < len(initial_positions):
                p.resetJointState(
                    self._robot_id,
                    joint_id,
                    initial_positions[i],
                    physicsClientId=self._client_id,
                )

        self._episode_step = 0
        observation = await self.observe()

        return ResetResult(observation=observation, info={"step": 0})

    async def step(self, action: RobotAction) -> StepResult:
        """Execute action."""
        # Apply action
        if isinstance(action, RobotAction) and action.joint_command:
            positions = action.joint_command.positions
        else:
            positions = np.array(action)

        for i, joint_id in enumerate(self._joint_ids):
            if i < len(positions):
                p.setJointMotorControl2(
                    self._robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=positions[i],
                    physicsClientId=self._client_id,
                )

        # Step simulation
        n_substeps = int(self._control_dt / self._physics_dt)
        for _ in range(n_substeps):
            p.stepSimulation(physicsClientId=self._client_id)

        self._episode_step += 1
        observation = await self.observe()

        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=self._episode_step >= 1000,
            info={"step": self._episode_step},
        )

    async def observe(self) -> WorldState:
        """Get current observation."""
        positions = []
        velocities = []

        for joint_id in self._joint_ids:
            state = p.getJointState(
                self._robot_id,
                joint_id,
                physicsClientId=self._client_id,
            )
            positions.append(state[0])
            velocities.append(state[1])

        joint_state = JointState(
            position=np.array(positions),
            velocity=np.array(velocities),
        )

        robot_state = RobotState(joint_state=joint_state)
        return WorldState(robot=robot_state)

    async def close(self) -> None:
        """Disconnect from PyBullet."""
        if self._client_id >= 0:
            p.disconnect(self._client_id)
            self._client_id = -1
```

### Deliverables
- [ ] MuJoCo backend implementation
- [ ] PyBullet backend implementation
- [ ] Isaac Sim backend (basic)
- [ ] Backend factory pattern
- [ ] Configuration loading

---

## Sprint 3: Sensor Fusion

**Goal**: Implement sensor fusion for state estimation.

### Tasks

#### 3.1 Kalman Filter for State Estimation

```python
# src/agi/env/fusion/kalman.py
"""Extended Kalman Filter for state estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Kalman filter state."""

    mean: np.ndarray  # State estimate
    covariance: np.ndarray  # Estimation uncertainty
    timestamp: float = 0.0


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear state estimation.

    Fuses multiple sensor observations into a coherent state estimate.
    """

    def __init__(
        self,
        state_dim: int,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        state_transition: Optional[Callable] = None,
        measurement_model: Optional[Callable] = None,
    ) -> None:
        """Initialize EKF."""
        self.state_dim = state_dim
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance

        # Default: linear state transition (identity)
        self._f = state_transition or (lambda x, dt: x)

        # Jacobian of state transition
        self._F = None

        # Measurement model
        self._h = measurement_model or (lambda x: x)

        # Current state
        self._state: Optional[KalmanState] = None

    def initialize(
        self,
        initial_state: np.ndarray,
        initial_covariance: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize filter with initial state."""
        if initial_covariance is None:
            initial_covariance = np.eye(self.state_dim)

        self._state = KalmanState(
            mean=initial_state.copy(),
            covariance=initial_covariance.copy(),
            timestamp=timestamp,
        )

    def predict(self, dt: float, control: Optional[np.ndarray] = None) -> KalmanState:
        """Predict step: propagate state forward in time."""
        if self._state is None:
            raise RuntimeError("Filter not initialized")

        # State transition
        predicted_mean = self._f(self._state.mean, dt)

        # Compute Jacobian (numerical differentiation)
        F = self._compute_jacobian_f(self._state.mean, dt)

        # Propagate covariance
        predicted_cov = F @ self._state.covariance @ F.T + self.Q * dt

        self._state = KalmanState(
            mean=predicted_mean,
            covariance=predicted_cov,
            timestamp=self._state.timestamp + dt,
        )

        return self._state

    def update(
        self,
        measurement: np.ndarray,
        measurement_model: Optional[Callable] = None,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> KalmanState:
        """Update step: incorporate new measurement."""
        if self._state is None:
            raise RuntimeError("Filter not initialized")

        h = measurement_model or self._h
        R = measurement_noise if measurement_noise is not None else self.R

        # Expected measurement
        expected = h(self._state.mean)

        # Innovation (measurement residual)
        y = measurement - expected

        # Measurement Jacobian
        H = self._compute_jacobian_h(self._state.mean, h)

        # Innovation covariance
        S = H @ self._state.covariance @ H.T + R

        # Kalman gain
        K = self._state.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        updated_mean = self._state.mean + K @ y

        # Update covariance (Joseph form for stability)
        I = np.eye(self.state_dim)
        IKH = I - K @ H
        updated_cov = (
            IKH @ self._state.covariance @ IKH.T +
            K @ R @ K.T
        )

        self._state = KalmanState(
            mean=updated_mean,
            covariance=updated_cov,
            timestamp=self._state.timestamp,
        )

        return self._state

    def _compute_jacobian_f(
        self,
        x: np.ndarray,
        dt: float,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute Jacobian of state transition numerically."""
        n = len(x)
        F = np.zeros((n, n))

        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            F[:, i] = (self._f(x_plus, dt) - self._f(x_minus, dt)) / (2 * eps)

        return F

    def _compute_jacobian_h(
        self,
        x: np.ndarray,
        h: Callable,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Compute Jacobian of measurement model numerically."""
        n = len(x)
        y0 = h(x)
        m = len(y0)
        H = np.zeros((m, n))

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            H[:, i] = (h(x_plus) - y0) / eps

        return H

    @property
    def state(self) -> KalmanState:
        """Get current state estimate."""
        if self._state is None:
            raise RuntimeError("Filter not initialized")
        return self._state


class IMUFusion(ExtendedKalmanFilter):
    """IMU + other sensor fusion filter.

    State: [position, velocity, orientation (quaternion)]
    """

    def __init__(
        self,
        process_noise_accel: float = 0.1,
        process_noise_gyro: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        """Initialize IMU fusion filter."""
        # State: [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        state_dim = 10

        Q = np.diag([
            0.01, 0.01, 0.01,  # Position
            process_noise_accel, process_noise_accel, process_noise_accel,  # Velocity
            process_noise_gyro, process_noise_gyro, process_noise_gyro, process_noise_gyro,  # Orientation
        ])

        R = np.eye(6) * measurement_noise  # 6-DOF pose measurement

        super().__init__(
            state_dim=state_dim,
            process_noise=Q,
            measurement_noise=R,
            state_transition=self._imu_state_transition,
        )

    def _imu_state_transition(
        self,
        state: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """IMU state transition model."""
        new_state = state.copy()

        # Position update: p = p + v * dt
        new_state[:3] += state[3:6] * dt

        # Orientation stays same (updated by gyro in predict_imu)

        return new_state

    def predict_imu(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        dt: float,
    ) -> KalmanState:
        """Predict with IMU measurements."""
        if self._state is None:
            raise RuntimeError("Filter not initialized")

        state = self._state.mean.copy()

        # Update velocity with accelerometer
        # Need to remove gravity and rotate to world frame
        # Simplified: assuming world-aligned IMU
        state[3:6] += accel * dt

        # Update orientation with gyroscope
        quat = state[6:10]
        dq = self._gyro_to_quat_delta(gyro, dt)
        state[6:10] = self._quat_multiply(quat, dq)
        state[6:10] /= np.linalg.norm(state[6:10])  # Normalize

        # Update position
        state[:3] += state[3:6] * dt

        # Propagate covariance
        F = self._compute_jacobian_f(self._state.mean, dt)
        predicted_cov = F @ self._state.covariance @ F.T + self.Q * dt

        self._state = KalmanState(
            mean=state,
            covariance=predicted_cov,
            timestamp=self._state.timestamp + dt,
        )

        return self._state

    def _gyro_to_quat_delta(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        """Convert angular velocity to quaternion delta."""
        angle = np.linalg.norm(gyro) * dt
        if angle < 1e-8:
            return np.array([1, 0, 0, 0])

        axis = gyro / np.linalg.norm(gyro)
        return np.concatenate([
            [np.cos(angle / 2)],
            np.sin(angle / 2) * axis,
        ])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
```

#### 3.2 Object Tracker

```python
# src/agi/env/fusion/tracking.py
"""Multi-object tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object."""

    track_id: int
    class_name: str
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    covariance: np.ndarray  # (6, 6)
    confidence: float
    age: int = 0  # Frames since creation
    hits: int = 0  # Detection matches
    misses: int = 0  # Consecutive missed detections
    last_detection: Optional[Dict] = None


class MultiObjectTracker:
    """Hungarian algorithm-based multi-object tracker.

    Features:
    - Association via IoU or distance
    - Track management (creation, deletion)
    - Kalman prediction between detections
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        distance_threshold: float = 1.0,
    ) -> None:
        """Initialize tracker."""
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold

        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 0

    def update(
        self,
        detections: List[Dict],
        dt: float,
    ) -> List[TrackedObject]:
        """Update tracks with new detections.

        Args:
            detections: List of detection dicts with 'position', 'class_name', etc.
            dt: Time since last update

        Returns:
            List of confirmed tracks
        """
        # Predict existing tracks
        for track in self._tracks.values():
            track.position += track.velocity * dt
            track.age += 1

        if not detections:
            # No detections, increment miss count
            for track in list(self._tracks.values()):
                track.misses += 1
                if track.misses > self.max_age:
                    del self._tracks[track.track_id]
            return self._get_confirmed_tracks()

        # Compute cost matrix (distance-based)
        track_ids = list(self._tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            track = self._tracks[tid]
            for j, det in enumerate(detections):
                det_pos = np.array(det["position"])
                cost_matrix[i, j] = np.linalg.norm(track.position - det_pos)

        # Hungarian matching
        matched, unmatched_tracks, unmatched_dets = self._match(
            cost_matrix,
            track_ids,
            detections,
        )

        # Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            track = self._tracks[track_id]

            # Update position with detection
            det_pos = np.array(det["position"])
            alpha = 0.7  # Simple exponential smoothing
            track.velocity = alpha * (det_pos - track.position) / dt + (1 - alpha) * track.velocity
            track.position = det_pos
            track.confidence = det.get("confidence", 1.0)
            track.hits += 1
            track.misses = 0
            track.last_detection = det

        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            self._tracks[track_id].misses += 1
            if self._tracks[track_id].misses > self.max_age:
                del self._tracks[track_id]

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track(det)

        return self._get_confirmed_tracks()

    def _match(
        self,
        cost_matrix: np.ndarray,
        track_ids: List[int],
        detections: List[Dict],
    ) -> tuple:
        """Match tracks to detections using Hungarian algorithm."""
        from scipy.optimize import linear_sum_assignment

        if cost_matrix.size == 0:
            return [], track_ids, list(range(len(detections)))

        # Apply threshold
        cost_matrix[cost_matrix > self.distance_threshold] = 1e6

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1e6:
                matched.append((track_ids[row], col))
                unmatched_tracks.discard(track_ids[row])
                unmatched_dets.discard(col)

        return matched, list(unmatched_tracks), list(unmatched_dets)

    def _create_track(self, detection: Dict) -> TrackedObject:
        """Create a new track from detection."""
        track = TrackedObject(
            track_id=self._next_id,
            class_name=detection.get("class_name", "unknown"),
            position=np.array(detection["position"]),
            velocity=np.zeros(3),
            covariance=np.eye(6),
            confidence=detection.get("confidence", 1.0),
        )
        self._tracks[track.track_id] = track
        self._next_id += 1
        return track

    def _get_confirmed_tracks(self) -> List[TrackedObject]:
        """Get tracks that meet confirmation criteria."""
        return [
            track for track in self._tracks.values()
            if track.hits >= self.min_hits
        ]
```

### Deliverables
- [ ] Extended Kalman Filter
- [ ] IMU fusion
- [ ] Multi-object tracker
- [ ] Particle filter (optional)

---

## Sprint 4: ROS2 Bridge

**Goal**: Integrate with ROS2 for real robot communication.

### Tasks

#### 4.1 ROS2 Environment Wrapper

```python
# src/agi/env/backends/ros2_env.py
"""ROS2 environment bridge."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from agi.env.base import (
    AsyncEnvironment,
    EnvironmentSpec,
    ResetResult,
    Space,
    SpaceType,
    StepResult,
)
from agi.env.observations import (
    CameraObservation,
    JointState,
    RobotState,
    WorldState,
)
from agi.env.actions import RobotAction, JointCommand

logger = logging.getLogger(__name__)


class ROS2Environment(AsyncEnvironment[WorldState, RobotAction]):
    """ROS2 bridge environment for real robots.

    Subscribes to sensor topics and publishes control commands.
    """

    def __init__(
        self,
        node_name: str = "agi_env",
        config: Dict[str, Any] = None,
    ) -> None:
        """Initialize ROS2 environment."""
        self.node_name = node_name
        self.config = config or {}

        self._node = None
        self._subscribers = {}
        self._publishers = {}
        self._latest_obs: Dict[str, Any] = {}
        self._pending_actions: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize ROS2 node."""
        import rclpy
        from rclpy.node import Node

        rclpy.init()
        self._node = Node(self.node_name)

        # Setup subscribers
        await self._setup_subscribers()

        # Setup publishers
        await self._setup_publishers()

        logger.info("ROS2 environment initialized: %s", self.node_name)

    async def _setup_subscribers(self) -> None:
        """Setup ROS2 subscribers."""
        from sensor_msgs.msg import JointState as JointStateMsg
        from sensor_msgs.msg import Image

        # Joint state subscriber
        self._subscribers["joint_states"] = self._node.create_subscription(
            JointStateMsg,
            self.config.get("joint_states_topic", "/joint_states"),
            self._on_joint_state,
            10,
        )

        # Camera subscribers
        for cam_config in self.config.get("cameras", []):
            topic = cam_config["topic"]
            self._subscribers[topic] = self._node.create_subscription(
                Image,
                topic,
                lambda msg, t=topic: self._on_image(t, msg),
                10,
            )

    async def _setup_publishers(self) -> None:
        """Setup ROS2 publishers."""
        from trajectory_msgs.msg import JointTrajectory

        self._publishers["command"] = self._node.create_publisher(
            JointTrajectory,
            self.config.get("command_topic", "/arm_controller/command"),
            10,
        )

    def _on_joint_state(self, msg) -> None:
        """Handle joint state message."""
        self._latest_obs["joint_state"] = JointState(
            position=np.array(msg.position),
            velocity=np.array(msg.velocity),
            effort=np.array(msg.effort) if msg.effort else None,
            names=list(msg.name),
        )

    def _on_image(self, topic: str, msg) -> None:
        """Handle image message."""
        import cv_bridge
        bridge = cv_bridge.CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        self._latest_obs[f"camera_{topic}"] = CameraObservation(
            rgb=image,
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            frame_id=msg.header.frame_id,
        )

    @property
    def spec(self) -> EnvironmentSpec:
        return EnvironmentSpec(
            name="ros2_robot",
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @property
    def observation_space(self) -> Space:
        n_joints = self.config.get("n_joints", 7)
        return Space(
            space_type=SpaceType.BOX,
            shape=(n_joints * 2,),
            dtype="float32",
            low=np.full(n_joints * 2, -np.inf),
            high=np.full(n_joints * 2, np.inf),
        )

    @property
    def action_space(self) -> Space:
        n_joints = self.config.get("n_joints", 7)
        return Space(
            space_type=SpaceType.BOX,
            shape=(n_joints,),
            dtype="float32",
            low=np.full(n_joints, -np.pi),
            high=np.full(n_joints, np.pi),
        )

    async def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ResetResult:
        """Reset robot to home position."""
        # Send home position command
        home_position = self.config.get(
            "home_position",
            [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
        )

        action = RobotAction(
            joint_command=JointCommand(
                positions=np.array(home_position),
                duration=5.0,
            )
        )

        await self.step(action)

        # Wait for robot to settle
        import asyncio
        await asyncio.sleep(2.0)

        observation = await self.observe()
        return ResetResult(observation=observation, info={})

    async def step(self, action: RobotAction) -> StepResult:
        """Send action to robot."""
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration

        msg = JointTrajectory()
        msg.joint_names = self.config.get("joint_names", [])

        point = JointTrajectoryPoint()

        if action.joint_command:
            point.positions = action.joint_command.positions.tolist()
            duration = action.joint_command.duration or 0.1
        else:
            point.positions = np.array(action).tolist()
            duration = 0.1

        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
        msg.points = [point]

        self._publishers["command"].publish(msg)

        # Spin to receive feedback
        import rclpy
        rclpy.spin_once(self._node, timeout_sec=0.1)

        observation = await self.observe()

        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )

    async def observe(self) -> WorldState:
        """Get current observation."""
        import rclpy

        # Spin to get latest messages
        rclpy.spin_once(self._node, timeout_sec=0.1)

        joint_state = self._latest_obs.get("joint_state", JointState(
            position=np.zeros(7),
            velocity=np.zeros(7),
        ))

        cameras = {
            k.replace("camera_", ""): v
            for k, v in self._latest_obs.items()
            if k.startswith("camera_")
        }

        return WorldState(
            robot=RobotState(joint_state=joint_state),
            cameras=cameras,
        )

    async def close(self) -> None:
        """Shutdown ROS2."""
        import rclpy

        if self._node:
            self._node.destroy_node()
        rclpy.shutdown()

    async def step_async(self, action: RobotAction) -> str:
        """Start async action."""
        import uuid
        action_id = str(uuid.uuid4())
        self._pending_actions[action_id] = {"action": action, "status": "pending"}

        # Send command
        await self.step(action)
        self._pending_actions[action_id]["status"] = "executing"

        return action_id

    async def step_wait(
        self,
        action_id: str,
        timeout: float = None,
    ) -> StepResult:
        """Wait for action completion."""
        import asyncio

        start_time = asyncio.get_event_loop().time()

        while True:
            result = await self.step_poll(action_id)
            if result is not None:
                return result

            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise TimeoutError(f"Action {action_id} timed out")

            await asyncio.sleep(0.1)

    async def step_poll(self, action_id: str) -> Optional[StepResult]:
        """Poll action status."""
        if action_id not in self._pending_actions:
            return None

        action_info = self._pending_actions[action_id]

        # For now, assume actions complete immediately
        # In practice, would check joint error thresholds
        observation = await self.observe()
        del self._pending_actions[action_id]

        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )
```

### Deliverables
- [ ] ROS2 environment wrapper
- [ ] Sensor message subscribers
- [ ] Command publishers
- [ ] Action servers integration
- [ ] tf2 transforms

---

## Sprint 5: Recording and Playback

**Goal**: Implement data recording for offline training and debugging.

### Tasks

#### 5.1 HDF5 Recorder

```python
# src/agi/env/recording/hdf5_recorder.py
"""HDF5 data recorder for environment episodes."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from agi.env.observations import WorldState
from agi.env.actions import RobotAction

logger = logging.getLogger(__name__)


class HDF5Recorder:
    """Record environment episodes to HDF5 files.

    Structure:
    - /metadata (attrs)
    - /episodes/
      - /episode_0/
        - /observations/
          - /joint_positions (N, J)
          - /joint_velocities (N, J)
          - /cameras/
            - /head_camera/ (N, H, W, 3)
        - /actions/ (N, A)
        - /rewards/ (N,)
        - /timestamps/ (N,)
    """

    def __init__(
        self,
        path: Path,
        compress: bool = True,
        chunk_size: int = 100,
    ) -> None:
        """Initialize recorder."""
        self.path = Path(path)
        self.compress = compress
        self.chunk_size = chunk_size

        self._file: Optional[h5py.File] = None
        self._episode_group: Optional[h5py.Group] = None
        self._episode_count = 0
        self._step_count = 0

        # Buffers for batched writing
        self._buffers: Dict[str, List] = {}

    async def open(self) -> None:
        """Open HDF5 file for writing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        self._file.attrs["version"] = "1.0"
        self._file.create_group("episodes")

        logger.info("Opened HDF5 recorder: %s", self.path)

    async def close(self) -> None:
        """Close HDF5 file."""
        if self._episode_group:
            await self._flush_buffers()

        if self._file:
            self._file.close()
            self._file = None

        logger.info("Closed HDF5 recorder: %d episodes", self._episode_count)

    async def start_episode(self, metadata: Dict[str, Any] = None) -> None:
        """Start recording a new episode."""
        if self._episode_group:
            await self._flush_buffers()

        episode_name = f"episode_{self._episode_count}"
        self._episode_group = self._file["episodes"].create_group(episode_name)

        if metadata:
            for k, v in metadata.items():
                self._episode_group.attrs[k] = v

        self._episode_group.create_group("observations")
        self._step_count = 0
        self._buffers.clear()

        logger.debug("Started episode %d", self._episode_count)

    async def end_episode(self) -> None:
        """End current episode."""
        await self._flush_buffers()
        self._episode_group.attrs["length"] = self._step_count
        self._episode_count += 1
        self._episode_group = None

        logger.debug("Ended episode, %d steps", self._step_count)

    async def record_step(
        self,
        observation: WorldState,
        action: RobotAction,
        reward: float,
        info: Dict[str, Any] = None,
    ) -> None:
        """Record a single step."""
        # Buffer observation data
        self._buffer("joint_positions", observation.robot.joint_state.position)
        self._buffer("joint_velocities", observation.robot.joint_state.velocity)
        self._buffer("timestamps", observation.timestamp)
        self._buffer("rewards", reward)

        # Buffer action
        if action.joint_command and action.joint_command.positions is not None:
            self._buffer("actions", action.joint_command.positions)

        # Buffer camera data
        for cam_name, cam_obs in observation.cameras.items():
            self._buffer(f"camera_{cam_name}", cam_obs.rgb)

        self._step_count += 1

        # Flush if buffer is full
        if len(self._buffers.get("joint_positions", [])) >= self.chunk_size:
            await self._flush_buffers()

    def _buffer(self, key: str, value: Any) -> None:
        """Add value to buffer."""
        if key not in self._buffers:
            self._buffers[key] = []
        self._buffers[key].append(value)

    async def _flush_buffers(self) -> None:
        """Write buffers to HDF5."""
        if not self._buffers or not self._episode_group:
            return

        obs_group = self._episode_group["observations"]
        compression = "gzip" if self.compress else None

        for key, values in self._buffers.items():
            if not values:
                continue

            data = np.array(values)

            if key.startswith("camera_"):
                # Store cameras separately
                cam_name = key.replace("camera_", "")
                if "cameras" not in obs_group:
                    obs_group.create_group("cameras")
                if cam_name not in obs_group["cameras"]:
                    obs_group["cameras"].create_dataset(
                        cam_name,
                        data=data,
                        maxshape=(None, *data.shape[1:]),
                        compression=compression,
                        chunks=(min(10, len(data)), *data.shape[1:]),
                    )
                else:
                    ds = obs_group["cameras"][cam_name]
                    ds.resize(ds.shape[0] + len(data), axis=0)
                    ds[-len(data):] = data
            else:
                # Store other data
                target = obs_group if key not in ["actions", "rewards"] else self._episode_group

                if key not in target:
                    target.create_dataset(
                        key,
                        data=data,
                        maxshape=(None, *data.shape[1:]) if len(data.shape) > 1 else (None,),
                        compression=compression,
                    )
                else:
                    ds = target[key]
                    ds.resize(ds.shape[0] + len(data), axis=0)
                    ds[-len(data):] = data

        self._buffers.clear()


class HDF5Player:
    """Playback episodes from HDF5 files."""

    def __init__(self, path: Path) -> None:
        """Initialize player."""
        self.path = Path(path)
        self._file: Optional[h5py.File] = None

    async def open(self) -> None:
        """Open HDF5 file for reading."""
        self._file = h5py.File(self.path, "r")

    async def close(self) -> None:
        """Close file."""
        if self._file:
            self._file.close()
            self._file = None

    @property
    def episode_count(self) -> int:
        """Get number of episodes."""
        return len(self._file["episodes"])

    async def get_episode(self, index: int) -> Dict[str, np.ndarray]:
        """Get all data for an episode."""
        ep = self._file["episodes"][f"episode_{index}"]

        data = {
            "joint_positions": ep["observations"]["joint_positions"][:],
            "joint_velocities": ep["observations"]["joint_velocities"][:],
            "timestamps": ep["observations"]["timestamps"][:],
        }

        if "actions" in ep:
            data["actions"] = ep["actions"][:]

        if "rewards" in ep:
            data["rewards"] = ep["rewards"][:]

        if "cameras" in ep["observations"]:
            for cam_name in ep["observations"]["cameras"]:
                data[f"camera_{cam_name}"] = ep["observations"]["cameras"][cam_name][:]

        return data

    async def iterate_steps(self, episode_index: int):
        """Iterate through episode steps."""
        data = await self.get_episode(episode_index)
        length = len(data["joint_positions"])

        for i in range(length):
            yield {
                "joint_positions": data["joint_positions"][i],
                "joint_velocities": data["joint_velocities"][i],
                "timestamp": data["timestamps"][i],
                "action": data.get("actions", [None] * length)[i],
                "reward": data.get("rewards", [0.0] * length)[i],
            }
```

### Deliverables
- [ ] HDF5 recorder
- [ ] HDF5 player
- [ ] ROSBag integration (optional)
- [ ] Streaming recording

---

## Sprint 6: Unit Tests

**Goal**: Achieve 80%+ test coverage.

### Tasks

#### Test Categories
- [ ] `test_environment_protocol`
- [ ] `test_mujoco_backend`
- [ ] `test_pybullet_backend`
- [ ] `test_observation_types`
- [ ] `test_action_types`
- [ ] `test_kalman_filter`
- [ ] `test_object_tracker`
- [ ] `test_hdf5_recorder`

---

## File Structure After Completion

```
src/agi/env/
├── __init__.py
├── base.py                 # Environment protocol
├── observations.py         # Observation types
├── actions.py              # Action types
├── config.py               # Configuration
├── registry.py             # Environment registry
├── backends/
│   ├── __init__.py
│   ├── mujoco_env.py       # MuJoCo backend
│   ├── pybullet_env.py     # PyBullet backend
│   ├── isaac_env.py        # Isaac Sim backend
│   └── ros2_env.py         # ROS2 bridge
├── fusion/
│   ├── __init__.py
│   ├── kalman.py           # Kalman filter
│   ├── particle.py         # Particle filter
│   └── tracking.py         # Object tracking
├── recording/
│   ├── __init__.py
│   ├── hdf5_recorder.py    # HDF5 recording
│   └── rosbag_recorder.py  # ROSBag recording
└── wrappers/
    ├── __init__.py
    ├── observation.py      # Observation wrappers
    └── action.py           # Action wrappers

proto/
└── env.proto               # gRPC definitions

configs/
└── env_config.yaml

tests/env/
├── __init__.py
├── conftest.py
├── test_base.py
├── test_mujoco.py
├── test_pybullet.py
├── test_fusion.py
└── test_recording.py
```

---

## Priority Order

1. **Sprint 1** - Critical: Core interface needed by RH
2. **Sprint 2** - High: Simulation backends for development
3. **Sprint 4** - High: ROS2 for real hardware
4. **Sprint 3** - Medium: Sensor fusion for accuracy
5. **Sprint 5** - Medium: Recording for training
6. **Sprint 6** - High: Tests for correctness

---

## Dependencies

```toml
# pyproject.toml additions for environment
[project.optional-dependencies]
env = [
    "numpy>=1.24",
    "scipy>=1.11",
    "h5py>=3.10",
]

env-mujoco = [
    "agi-hpc[env]",
    "mujoco>=3.1",
]

env-pybullet = [
    "agi-hpc[env]",
    "pybullet>=3.2",
]

env-isaac = [
    "agi-hpc[env]",
    # Isaac Sim from Omniverse
]

env-ros2 = [
    "agi-hpc[env]",
    # ROS2 packages
]
```
