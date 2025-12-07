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
Right Hemisphere – Event Loop

Coordinates RH behavior using the EventFabric:

    • Subscribes to:
        - plan.step_ready      (from LH PlanService)
        - perception.state_update  (from external env / sensors) [optional]

    • Dispatches:
        - Steps to ControlService for execution
        - Observations to Perception for state updates
        - (SimulationService is gRPC-based and separate)

Architecture references:
    - Section IV.B  – RH Node (Perception + World Model + Control)
    - Section V     – Event Fabric
    - Section XI    – Sensorimotor Loop
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from agi.core.events.fabric import EventFabric
from agi.proto_gen import plan_pb2

from agi.rh.perception import Perception
from agi.rh.world_model import WorldModel
from agi.rh.control_service import ControlService

logger = logging.getLogger(__name__)


class RHEventLoop:
    """
    Right Hemisphere event loop.

    This is deliberately lightweight: it wires EventFabric callbacks to
    Perception and ControlService. SimulationService uses gRPC and the
    same underlying components but is separate.
    """

    def __init__(
        self,
        fabric: EventFabric,
        perception: Perception,
        world_model: WorldModel,
        control: ControlService,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self._fabric = fabric
        self._perception = perception
        self._world_model = world_model
        self._control = control
        self._config = config or {}

        self._shutdown_event = asyncio.Event()

        logger.info("[RH][EventLoop] initialized")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Register fabric subscriptions and wait forever.

        This is meant to be run as an asyncio task from rh_service.py.
        """
        logger.info("[RH][EventLoop] starting subscriptions")

        # Subscribe to plan.step_ready (LH → RH)
        self._fabric.subscribe("plan.step_ready", self._on_plan_step_ready)

        # Optional: perception.state_update from external env / sensors
        self._fabric.subscribe(
            "perception.state_update",
            self._on_perception_state_update,
        )

        logger.info("[RH][EventLoop] subscriptions active; entering idle loop")
        await self._shutdown_event.wait()
        logger.info("[RH][EventLoop] shutdown requested, exiting")

    def stop(self) -> None:
        """
        Signal the event loop to stop (if needed).
        """
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()

    # ------------------------------------------------------------------ #
    # EventFabric handlers (called from fabric backend threads)
    # ------------------------------------------------------------------ #

    def _on_plan_step_ready(self, message: Dict[str, Any]) -> None:
        """
        Fabric callback for plan.step_ready.

        Expected payload (from LH PlanService):

            {
                "node_id": "LH",
                "index": int,
                "step": { ... }  # serialized PlanStep-like dict
            }
        """
        logger.debug("[RH][EventLoop] received plan.step_ready: %s", message)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("[RH][EventLoop] no running event loop; dropping step")
            return

        loop.create_task(self._handle_plan_step_async(message))

    def _on_perception_state_update(self, message: Dict[str, Any]) -> None:
        """
        Fabric callback for perception.state_update.

        Payload can either be:
            - full perceptual state dict, or
            - { "frame": raw_frame, ... }
        """
        logger.debug("[RH][EventLoop] received perception.state_update")

        frame = message.get("frame", message)
        try:
            self._perception.update_observation(frame)
        except Exception:
            logger.exception("[RH][EventLoop] failed to update perception state")

    # ------------------------------------------------------------------ #
    # Async handlers
    # ------------------------------------------------------------------ #

    async def _handle_plan_step_async(self, message: Dict[str, Any]) -> None:
        """
        Asynchronous handler for a single plan step event.
        """
        step_data = message.get("step") or {}
        index = message.get("index", -1)

        step_proto = self._to_plan_step_proto(step_data, index=index)

        logger.info(
            "[RH][EventLoop] executing step index=%d id=%s kind=%s",
            step_proto.index,
            step_proto.step_id,
            step_proto.kind,
        )

        try:
            actions = self._control.translate_step(step_proto)
            await self._control.execute_actions(actions)
        except Exception:
            logger.exception("[RH][EventLoop] error executing plan step")

    # ------------------------------------------------------------------ #
    # Utility: convert serialized step dict → PlanStep proto
    # ------------------------------------------------------------------ #

    def _to_plan_step_proto(
        self,
        data: Dict[str, Any],
        index: int,
    ) -> plan_pb2.PlanStep:
        """
        Reconstruct a minimal PlanStep protobuf from LH's serialized dict.

        The LH PlanService serialized steps with fields like:
            - step_id
            - description
            - kind
            - parent_id
            - requires_simulation
            - safety_tags
            - tool_id
            - params
        """
        step = plan_pb2.PlanStep()
        step.step_id = str(data.get("step_id", f"step_{index}"))
        step.index = index
        step.level = int(data.get("level", 0))
        step.kind = str(data.get("kind", "action"))
        step.description = str(
            data.get("description", data.get("repr", "unnamed-step"))
        )
        step.parent_id = str(data.get("parent_id", ""))

        step.requires_simulation = bool(data.get("requires_simulation", False))

        safety_tags = data.get("safety_tags") or []
        for tag in safety_tags:
            step.safety_tags.append(str(tag))

        tool_id = data.get("tool_id")
        if tool_id is not None:
            step.tool_id = str(tool_id)

        params = data.get("params") or {}
        for k, v in params.items():
            step.params[str(k)] = str(v)

        return step
