# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
PyBullet Physics Engine Adapter for AGI-HPC RH World Model.

PyBullet provides general-purpose physics simulation with:
- Rigid body dynamics
- Collision detection
- Constraint solving
- OpenGL rendering

This adapter wraps the pybullet Python package.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agi.rh.world_model.physics.base import (
    BasePhysicsEngine,
    PhysicsConfig,
    CollisionInfo,
)

logger = logging.getLogger(__name__)


class PyBulletEngine(BasePhysicsEngine):
    """
    PyBullet-based physics engine for general simulation.

    Provides physics simulation for:
    - Rigid body dynamics
    - Robot simulation (URDF loading)
    - Collision detection
    - Constraint solving

    Configuration:
        engine_type: "pybullet"
        timestep: Simulation timestep (default 1/240)
        render_mode: "headless", "gui", or "rgb_array"
    """

    def __init__(self, config: Optional[PhysicsConfig] = None) -> None:
        cfg = config or PhysicsConfig(engine_type="pybullet", timestep=1.0 / 240.0)
        super().__init__(cfg)

        self._client_id: Optional[int] = None
        self._body_ids: Dict[str, int] = {}
        self._pybullet = None

    def _init_engine(self) -> None:
        """Initialize PyBullet engine."""
        try:
            import pybullet as p
            import pybullet_data

            self._pybullet = p

            # Connect to physics server
            if self._config.render_mode == "gui":
                self._client_id = p.connect(p.GUI)
            else:
                self._client_id = p.connect(p.DIRECT)

            # Set additional search path for URDF files
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Set gravity
            p.setGravity(*self._config.gravity, physicsClientId=self._client_id)

            # Set timestep
            p.setTimeStep(self._config.timestep, physicsClientId=self._client_id)

            # Load ground plane
            p.loadURDF("plane.urdf", physicsClientId=self._client_id)

            logger.info(
                "[RH][PyBullet] Engine initialized mode=%s timestep=%.4f",
                self._config.render_mode,
                self._config.timestep,
            )

        except ImportError:
            logger.warning(
                "[RH][PyBullet] pybullet package not installed, using stub mode"
            )
            self._pybullet = None
            self._client_id = None
        except Exception as e:
            logger.warning("[RH][PyBullet] Initialization failed: %s, using stub", e)
            self._pybullet = None
            self._client_id = None

    def _step_engine(
        self,
        actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Step PyBullet simulation forward."""
        if self._pybullet is None or self._client_id is None:
            return

        p = self._pybullet

        try:
            # Apply actions if provided
            if actions:
                for obj_id, action in actions.items():
                    if obj_id in self._body_ids:
                        body_id = self._body_ids[obj_id]
                        # Apply force at center of mass
                        if len(action) >= 3:
                            p.applyExternalForce(
                                body_id,
                                -1,  # Link index (-1 for base)
                                action[:3].tolist(),
                                [0, 0, 0],  # Position (local)
                                p.LINK_FRAME,
                                physicsClientId=self._client_id,
                            )
                        # Apply torque
                        if len(action) >= 6:
                            p.applyExternalTorque(
                                body_id,
                                -1,
                                action[3:6].tolist(),
                                p.LINK_FRAME,
                                physicsClientId=self._client_id,
                            )

            # Step simulation
            p.stepSimulation(physicsClientId=self._client_id)

        except Exception as e:
            logger.warning("[RH][PyBullet] Step failed: %s", e)

    def _get_object_state(self, object_id: str) -> Dict[str, np.ndarray]:
        """Get state of an object from PyBullet."""
        if self._pybullet is None or object_id not in self._body_ids:
            return {
                "position": np.zeros(3),
                "orientation": np.array([0, 0, 0, 1]),
                "linear_velocity": np.zeros(3),
                "angular_velocity": np.zeros(3),
            }

        p = self._pybullet

        try:
            body_id = self._body_ids[object_id]

            pos, orn = p.getBasePositionAndOrientation(
                body_id, physicsClientId=self._client_id
            )
            lin_vel, ang_vel = p.getBaseVelocity(
                body_id, physicsClientId=self._client_id
            )

            return {
                "position": np.array(pos),
                "orientation": np.array(orn),
                "linear_velocity": np.array(lin_vel),
                "angular_velocity": np.array(ang_vel),
            }

        except Exception as e:
            logger.warning("[RH][PyBullet] Get state failed: %s", e)
            return {
                "position": np.zeros(3),
                "orientation": np.array([0, 0, 0, 1]),
                "linear_velocity": np.zeros(3),
                "angular_velocity": np.zeros(3),
            }

    def _add_object_to_engine(
        self,
        object_id: str,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float],
        **kwargs,
    ) -> None:
        """Add object to PyBullet simulation."""
        if self._pybullet is None or self._client_id is None:
            # Stub mode: just track the object
            self._body_ids[object_id] = len(self._body_ids)
            return

        p = self._pybullet

        try:
            # Get object properties
            shape = kwargs.get("shape", "box")
            mass = kwargs.get("mass", 1.0)
            size = kwargs.get("size", [0.1, 0.1, 0.1])
            color = kwargs.get("color", [1, 0, 0, 1])

            # Create collision shape
            if shape == "box":
                half_extents = [s / 2 for s in size]
                col_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    physicsClientId=self._client_id,
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=color,
                    physicsClientId=self._client_id,
                )
            elif shape == "sphere":
                radius = size[0] / 2 if isinstance(size, (list, tuple)) else size / 2
                col_shape = p.createCollisionShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    physicsClientId=self._client_id,
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_SPHERE,
                    radius=radius,
                    rgbaColor=color,
                    physicsClientId=self._client_id,
                )
            elif shape == "cylinder":
                radius = size[0] / 2
                height = size[2] if len(size) > 2 else size[0]
                col_shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    height=height,
                    physicsClientId=self._client_id,
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=height,
                    rgbaColor=color,
                    physicsClientId=self._client_id,
                )
            else:
                # Default to box
                half_extents = [0.05, 0.05, 0.05]
                col_shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    physicsClientId=self._client_id,
                )
                vis_shape = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=color,
                    physicsClientId=self._client_id,
                )

            # Create multi-body
            body_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self._client_id,
            )

            self._body_ids[object_id] = body_id

            logger.debug(
                "[RH][PyBullet] Added %s at %s (body_id=%d)",
                object_id,
                position,
                body_id,
            )

        except Exception as e:
            logger.warning("[RH][PyBullet] Failed to add object: %s", e)
            self._body_ids[object_id] = -1

    def _remove_object_from_engine(self, object_id: str) -> None:
        """Remove object from PyBullet simulation."""
        if self._pybullet is None or self._client_id is None:
            if object_id in self._body_ids:
                del self._body_ids[object_id]
            return

        p = self._pybullet

        if object_id in self._body_ids:
            try:
                body_id = self._body_ids[object_id]
                if body_id >= 0:
                    p.removeBody(body_id, physicsClientId=self._client_id)
                del self._body_ids[object_id]
            except Exception as e:
                logger.warning("[RH][PyBullet] Failed to remove object: %s", e)

    def _check_collision_engine(
        self,
        object_a: str,
        object_b: Optional[str] = None,
    ) -> List[CollisionInfo]:
        """Check collisions in PyBullet."""
        collisions = []

        if self._pybullet is None or self._client_id is None:
            return collisions

        if object_a not in self._body_ids:
            return collisions

        p = self._pybullet
        body_a = self._body_ids[object_a]

        try:
            if object_b is not None:
                # Check specific pair
                if object_b not in self._body_ids:
                    return collisions

                body_b = self._body_ids[object_b]
                contacts = p.getContactPoints(
                    body_a, body_b, physicsClientId=self._client_id
                )
            else:
                # Check all contacts for body_a
                contacts = p.getContactPoints(body_a, physicsClientId=self._client_id)

            for contact in contacts:
                # contact: (contactFlag, bodyA, bodyB, linkIndexA, linkIndexB,
                #           positionOnA, positionOnB, contactNormalOnB,
                #           contactDistance, normalForce, ...)

                obj_b_name = self._get_object_name(contact[2])

                collisions.append(
                    CollisionInfo(
                        object_a=object_a,
                        object_b=obj_b_name,
                        contact_point=contact[5],  # positionOnA
                        contact_normal=contact[7],  # contactNormalOnB
                        penetration_depth=-contact[8],  # contactDistance (negative)
                        impulse=contact[9] if len(contact) > 9 else 0.0,  # normalForce
                    )
                )

        except Exception as e:
            logger.warning("[RH][PyBullet] Collision check failed: %s", e)

        return collisions

    def _get_object_name(self, body_id: int) -> str:
        """Get object name from body ID."""
        for name, bid in self._body_ids.items():
            if bid == body_id:
                return name
        return f"body_{body_id}"

    def load_urdf(
        self,
        urdf_path: str,
        position: Tuple[float, float, float] = (0, 0, 0),
        orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        object_id: Optional[str] = None,
    ) -> str:
        """
        Load a URDF model into the simulation.

        Args:
            urdf_path: Path to URDF file
            position: Initial position
            orientation: Initial orientation (quaternion)
            object_id: Optional ID for the loaded model

        Returns:
            Object ID of the loaded model
        """
        if self._pybullet is None or self._client_id is None:
            obj_id = object_id or f"urdf_{len(self._body_ids)}"
            self._body_ids[obj_id] = -1
            return obj_id

        p = self._pybullet

        try:
            body_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                baseOrientation=orientation,
                physicsClientId=self._client_id,
            )

            obj_id = object_id or f"urdf_{body_id}"
            self._body_ids[obj_id] = body_id

            logger.info(
                "[RH][PyBullet] Loaded URDF %s as %s (body_id=%d)",
                urdf_path,
                obj_id,
                body_id,
            )

            return obj_id

        except Exception as e:
            logger.error("[RH][PyBullet] Failed to load URDF: %s", e)
            raise

    def close(self) -> None:
        """Clean up PyBullet resources."""
        if self._pybullet is not None and self._client_id is not None:
            try:
                self._pybullet.disconnect(self._client_id)
            except Exception as e:
                logger.warning("[RH][PyBullet] Disconnect failed: %s", e)

        self._pybullet = None
        self._client_id = None
        self._body_ids.clear()
        super().close()
