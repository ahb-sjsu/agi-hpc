# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
MuJoCo Physics Engine Adapter for AGI-HPC RH World Model.

MuJoCo (Multi-Joint dynamics with Contact) provides high-fidelity
physics simulation for robotics applications, featuring:
- Accurate contact dynamics
- Efficient constraint solving
- Differentiable physics
- GPU acceleration

This adapter wraps DeepMind's mujoco Python bindings.
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


class MuJoCoEngine(BasePhysicsEngine):
    """
    MuJoCo-based physics engine for high-fidelity simulation.

    Provides accurate physics simulation for:
    - Robot dynamics and control
    - Contact-rich manipulation
    - Multi-body systems
    - Soft body simulation

    Configuration:
        engine_type: "mujoco"
        timestep: Simulation timestep (default 0.002 for MuJoCo)
    """

    def __init__(self, config: Optional[PhysicsConfig] = None) -> None:
        cfg = config or PhysicsConfig(engine_type="mujoco", timestep=0.002)
        super().__init__(cfg)

        self._model = None
        self._data = None
        self._body_ids: Dict[str, int] = {}

    def _init_engine(self) -> None:
        """Initialize MuJoCo engine."""
        try:
            import mujoco

            # Create a simple empty world model
            xml_string = """
            <mujoco>
                <worldbody>
                    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                    <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>
                </worldbody>
            </mujoco>
            """

            self._model = mujoco.MjModel.from_xml_string(xml_string)
            self._data = mujoco.MjData(self._model)

            # Set timestep
            self._model.opt.timestep = self._config.timestep

            # Set gravity
            self._model.opt.gravity[:] = self._config.gravity

            logger.info(
                "[RH][MuJoCo] Engine initialized timestep=%.4f",
                self._model.opt.timestep,
            )

        except ImportError:
            logger.warning("[RH][MuJoCo] mujoco package not installed, using stub mode")
            self._model = None
            self._data = None
        except Exception as e:
            logger.warning("[RH][MuJoCo] Initialization failed: %s, using stub", e)
            self._model = None
            self._data = None

    def _step_engine(
        self,
        actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Step MuJoCo simulation forward."""
        if self._model is None:
            return

        try:
            import mujoco

            # Apply actions if provided
            if actions:
                for obj_id, action in actions.items():
                    if obj_id in self._body_ids:
                        # Apply force/torque to body
                        body_id = self._body_ids[obj_id]
                        if len(action) >= 3:
                            self._data.xfrc_applied[body_id, :3] = action[:3]
                        if len(action) >= 6:
                            self._data.xfrc_applied[body_id, 3:6] = action[3:6]

            # Step simulation
            mujoco.mj_step(self._model, self._data)

        except Exception as e:
            logger.warning("[RH][MuJoCo] Step failed: %s", e)

    def _get_object_state(self, object_id: str) -> Dict[str, np.ndarray]:
        """Get state of an object from MuJoCo."""
        if self._model is None or object_id not in self._body_ids:
            return {
                "position": np.zeros(3),
                "orientation": np.array([0, 0, 0, 1]),
                "linear_velocity": np.zeros(3),
                "angular_velocity": np.zeros(3),
            }

        try:
            body_id = self._body_ids[object_id]

            return {
                "position": self._data.xpos[body_id].copy(),
                "orientation": self._data.xquat[body_id].copy(),
                "linear_velocity": self._data.cvel[body_id, 3:6].copy(),
                "angular_velocity": self._data.cvel[body_id, :3].copy(),
            }

        except Exception as e:
            logger.warning("[RH][MuJoCo] Get state failed: %s", e)
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
        """Add object to MuJoCo simulation."""
        if self._model is None:
            # Stub mode: just track the object
            self._body_ids[object_id] = len(self._body_ids)
            return

        # Note: MuJoCo requires model recompilation to add bodies dynamically
        # For now, we track objects in stub mode
        logger.warning(
            "[RH][MuJoCo] Dynamic object addition not fully supported, "
            "using stub tracking for %s",
            object_id,
        )
        self._body_ids[object_id] = len(self._body_ids)

    def _remove_object_from_engine(self, object_id: str) -> None:
        """Remove object from MuJoCo simulation."""
        if object_id in self._body_ids:
            del self._body_ids[object_id]

    def _check_collision_engine(
        self,
        object_a: str,
        object_b: Optional[str] = None,
    ) -> List[CollisionInfo]:
        """Check collisions in MuJoCo."""
        collisions = []

        if self._model is None or self._data is None:
            return collisions

        try:
            # Check contact array
            for i in range(self._data.ncon):
                contact = self._data.contact[i]

                # Get body IDs from geom IDs
                geom1_body = self._model.geom_bodyid[contact.geom1]
                geom2_body = self._model.geom_bodyid[contact.geom2]

                # Check if object_a is involved
                obj_a_id = self._body_ids.get(object_a, -1)
                if geom1_body != obj_a_id and geom2_body != obj_a_id:
                    continue

                # Check if object_b is specified and involved
                if object_b is not None:
                    obj_b_id = self._body_ids.get(object_b, -1)
                    if geom1_body != obj_b_id and geom2_body != obj_b_id:
                        continue

                # Find object names from body IDs
                obj1_name = self._get_object_name(geom1_body)
                obj2_name = self._get_object_name(geom2_body)

                collisions.append(
                    CollisionInfo(
                        object_a=obj1_name,
                        object_b=obj2_name,
                        contact_point=tuple(contact.pos),
                        contact_normal=tuple(contact.frame[:3]),
                        penetration_depth=contact.dist,
                        impulse=0.0,  # Would need to compute from contact forces
                    )
                )

        except Exception as e:
            logger.warning("[RH][MuJoCo] Collision check failed: %s", e)

        return collisions

    def _get_object_name(self, body_id: int) -> str:
        """Get object name from body ID."""
        for name, bid in self._body_ids.items():
            if bid == body_id:
                return name
        return f"body_{body_id}"

    def load_model(self, model_path: str) -> None:
        """
        Load a MuJoCo model from XML file.

        Args:
            model_path: Path to MuJoCo XML model file
        """
        try:
            import mujoco

            self._model = mujoco.MjModel.from_xml_path(model_path)
            self._data = mujoco.MjData(self._model)

            # Update body ID mapping
            self._body_ids.clear()
            for i in range(self._model.nbody):
                name = self._model.body(i).name
                if name:
                    self._body_ids[name] = i

            self._initialized = True
            logger.info(
                "[RH][MuJoCo] Loaded model from %s with %d bodies",
                model_path,
                self._model.nbody,
            )

        except Exception as e:
            logger.error("[RH][MuJoCo] Failed to load model: %s", e)
            raise

    def close(self) -> None:
        """Clean up MuJoCo resources."""
        self._model = None
        self._data = None
        self._body_ids.clear()
        super().close()
