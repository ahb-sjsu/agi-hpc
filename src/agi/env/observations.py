# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Structured observation types for environment interface.

Provides standardized data structures for different sensor types:
- Camera (RGB, depth, semantic)
- LiDAR (point cloud)
- IMU (acceleration, angular velocity)
- Proprioception (joint positions, velocities)
- Force/Torque sensors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SensorType(str, Enum):
    """Types of sensors."""
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    IMU = "imu"
    JOINT_STATE = "joint_state"
    FORCE_TORQUE = "force_torque"
    CONTACT = "contact"
    PROXIMITY = "proximity"
    GPS = "gps"
    ODOMETRY = "odometry"


# ---------------------------------------------------------------------------
# Base Observation
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """Base observation with timestamp and metadata."""

    timestamp: float  # Seconds since epoch
    frame_id: str = ""  # Reference frame
    sensor_id: str = ""  # Sensor identifier
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Camera Observations
# ---------------------------------------------------------------------------


@dataclass
class CameraObservation(Observation):
    """Camera sensor observation."""

    rgb: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8))
    depth: Optional[np.ndarray] = None  # (H, W) float32, meters
    semantic: Optional[np.ndarray] = None  # (H, W) int, class IDs
    instance: Optional[np.ndarray] = None  # (H, W) int, instance IDs
    intrinsics: Optional[np.ndarray] = None  # (3, 3) camera matrix
    extrinsics: Optional[np.ndarray] = None  # (4, 4) pose matrix

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_semantic(self) -> bool:
        return self.semantic is not None

    def project_to_3d(
        self,
        u: int,
        v: int,
    ) -> Optional[np.ndarray]:
        """Project pixel to 3D point using depth."""
        if self.depth is None or self.intrinsics is None:
            return None

        z = self.depth[v, u]
        if z <= 0:
            return None

        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.array([x, y, z])


# ---------------------------------------------------------------------------
# LiDAR Observations
# ---------------------------------------------------------------------------


@dataclass
class PointCloud:
    """Point cloud data structure."""

    points: np.ndarray  # (N, 3) XYZ
    intensities: Optional[np.ndarray] = None  # (N,) intensity values
    colors: Optional[np.ndarray] = None  # (N, 3) RGB
    normals: Optional[np.ndarray] = None  # (N, 3) surface normals
    ring: Optional[np.ndarray] = None  # (N,) laser ring indices

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    def crop_box(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
    ) -> "PointCloud":
        """Crop to bounding box."""
        mask = np.all(
            (self.points >= min_bound) & (self.points <= max_bound),
            axis=1,
        )
        return PointCloud(
            points=self.points[mask],
            intensities=self.intensities[mask] if self.intensities is not None else None,
            colors=self.colors[mask] if self.colors is not None else None,
            normals=self.normals[mask] if self.normals is not None else None,
            ring=self.ring[mask] if self.ring is not None else None,
        )


@dataclass
class LidarObservation(Observation):
    """LiDAR sensor observation."""

    point_cloud: PointCloud = field(
        default_factory=lambda: PointCloud(points=np.zeros((0, 3)))
    )
    horizontal_fov: float = 360.0  # degrees
    vertical_fov: float = 30.0  # degrees
    range_min: float = 0.1  # meters
    range_max: float = 100.0  # meters

    @property
    def num_points(self) -> int:
        return self.point_cloud.num_points


# ---------------------------------------------------------------------------
# IMU Observations
# ---------------------------------------------------------------------------


@dataclass
class ImuObservation(Observation):
    """Inertial Measurement Unit observation."""

    linear_acceleration: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # m/s^2
    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )  # rad/s
    orientation: Optional[np.ndarray] = None  # (4,) quaternion [x,y,z,w]
    orientation_covariance: Optional[np.ndarray] = None  # (3, 3)
    linear_acceleration_covariance: Optional[np.ndarray] = None  # (3, 3)
    angular_velocity_covariance: Optional[np.ndarray] = None  # (3, 3)

    def get_euler(self) -> Optional[np.ndarray]:
        """Convert orientation to Euler angles (roll, pitch, yaw)."""
        if self.orientation is None:
            return None

        x, y, z, w = self.orientation

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Proprioceptive Observations
# ---------------------------------------------------------------------------


@dataclass
class JointState:
    """State of a single joint."""

    name: str
    position: float  # radians or meters
    velocity: float  # rad/s or m/s
    effort: float = 0.0  # Nm or N


@dataclass
class JointStateObservation(Observation):
    """Robot joint state observation."""

    joint_states: List[JointState] = field(default_factory=list)

    @property
    def positions(self) -> np.ndarray:
        return np.array([j.position for j in self.joint_states])

    @property
    def velocities(self) -> np.ndarray:
        return np.array([j.velocity for j in self.joint_states])

    @property
    def efforts(self) -> np.ndarray:
        return np.array([j.effort for j in self.joint_states])

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.joint_states]

    def get_joint(self, name: str) -> Optional[JointState]:
        """Get joint by name."""
        for j in self.joint_states:
            if j.name == name:
                return j
        return None


# ---------------------------------------------------------------------------
# Force/Torque Observations
# ---------------------------------------------------------------------------


@dataclass
class Wrench:
    """Force and torque wrench."""

    force: np.ndarray  # (3,) N
    torque: np.ndarray  # (3,) Nm

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.force))


@dataclass
class ForceTorqueObservation(Observation):
    """Force/torque sensor observation."""

    wrench: Wrench = field(
        default_factory=lambda: Wrench(
            force=np.zeros(3),
            torque=np.zeros(3),
        )
    )
    wrench_covariance: Optional[np.ndarray] = None  # (6, 6)


# ---------------------------------------------------------------------------
# Contact Observations
# ---------------------------------------------------------------------------


@dataclass
class ContactPoint:
    """Single contact point."""

    position: np.ndarray  # (3,) world coordinates
    normal: np.ndarray  # (3,) contact normal
    force: float  # N
    body_a: str = ""  # First body name
    body_b: str = ""  # Second body name


@dataclass
class ContactObservation(Observation):
    """Contact sensor observation."""

    contacts: List[ContactPoint] = field(default_factory=list)

    @property
    def num_contacts(self) -> int:
        return len(self.contacts)

    @property
    def total_force(self) -> float:
        return sum(c.force for c in self.contacts)

    @property
    def in_contact(self) -> bool:
        return len(self.contacts) > 0


# ---------------------------------------------------------------------------
# Pose/Odometry Observations
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """3D pose representation."""

    position: np.ndarray  # (3,) xyz
    orientation: np.ndarray  # (4,) quaternion [x,y,z,w]

    @classmethod
    def identity(cls) -> "Pose":
        return cls(
            position=np.zeros(3),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        x, y, z, w = self.orientation

        matrix = np.eye(4)
        matrix[0, 0] = 1 - 2*y*y - 2*z*z
        matrix[0, 1] = 2*x*y - 2*z*w
        matrix[0, 2] = 2*x*z + 2*y*w
        matrix[1, 0] = 2*x*y + 2*z*w
        matrix[1, 1] = 1 - 2*x*x - 2*z*z
        matrix[1, 2] = 2*y*z - 2*x*w
        matrix[2, 0] = 2*x*z - 2*y*w
        matrix[2, 1] = 2*y*z + 2*x*w
        matrix[2, 2] = 1 - 2*x*x - 2*y*y
        matrix[:3, 3] = self.position

        return matrix


@dataclass
class Twist:
    """Velocity representation."""

    linear: np.ndarray  # (3,) m/s
    angular: np.ndarray  # (3,) rad/s


@dataclass
class OdometryObservation(Observation):
    """Odometry observation."""

    pose: Pose = field(default_factory=Pose.identity)
    twist: Twist = field(
        default_factory=lambda: Twist(
            linear=np.zeros(3),
            angular=np.zeros(3),
        )
    )
    pose_covariance: Optional[np.ndarray] = None  # (6, 6)
    twist_covariance: Optional[np.ndarray] = None  # (6, 6)


# ---------------------------------------------------------------------------
# Composite Observation
# ---------------------------------------------------------------------------


@dataclass
class RobotObservation:
    """Complete robot observation combining all sensors."""

    timestamp: float
    cameras: Dict[str, CameraObservation] = field(default_factory=dict)
    lidars: Dict[str, LidarObservation] = field(default_factory=dict)
    imu: Optional[ImuObservation] = None
    joint_states: Optional[JointStateObservation] = None
    force_torque: Dict[str, ForceTorqueObservation] = field(default_factory=dict)
    contacts: Optional[ContactObservation] = None
    odometry: Optional[OdometryObservation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_primary_camera(self) -> Optional[CameraObservation]:
        """Get the primary camera observation."""
        if "primary" in self.cameras:
            return self.cameras["primary"]
        if self.cameras:
            return next(iter(self.cameras.values()))
        return None
