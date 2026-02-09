# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Right Hemisphere (RH) Service - Main Entry Point.

Implements the RH service as described in:
    - Section IV.B  – RH Node (Perception + World Model + Control)
    - Section XI    – Sensorimotor Loop
    - Section XIV   – Cognitive APIs (SimulationService)

This service:
    - Starts gRPC server with SimulationService
    - Initializes Perception, WorldModel, ControlService
    - Starts RHEventLoop for EventFabric subscriptions
    - Provides health check endpoints
    - Handles graceful shutdown

Usage:
    python -m agi.rh.service --port 50057 --config configs/rh.yaml

Environment Variables:
    AGI_RH_PORT              gRPC port (default: 50057)
    AGI_RH_CONFIG            Config file path
    AGI_FABRIC_MODE          EventFabric mode (local/zmq/redis)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from typing import Optional

from agi.core.api.grpc_server import GRPCServer, GRPCServerConfig
from agi.core.events.fabric import EventFabric
from agi.proto_gen import plan_pb2_grpc

from agi.rh.config import RHConfig, load_rh_config
from agi.rh.perception import Perception
from agi.rh.world_model import WorldModel
from agi.rh.control_service import ControlService, ControlConfig
from agi.rh.simulation_service import SimulationService
from agi.rh.rh_event_loop import RHEventLoop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RH Service
# ---------------------------------------------------------------------------


class RHService:
    """
    Right Hemisphere Service.

    Coordinates all RH components:
        - Perception: sensory processing
        - WorldModel: short-horizon prediction
        - ControlService: action execution
        - SimulationService: gRPC API for LH
        - RHEventLoop: EventFabric subscriptions

    Lifecycle:
        1. Initialize components from config
        2. Start gRPC server
        3. Start EventLoop (async)
        4. Wait for shutdown signal
        5. Graceful cleanup
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[RHConfig] = None):
        """
        Initialize RH Service.

        Args:
            config: RH configuration (uses defaults if not provided)
        """
        self._config = config or RHConfig()
        self._start_time = time.time()
        self._shutdown_event = asyncio.Event()

        # Initialize components
        self._fabric = self._create_fabric()
        self._perception = self._create_perception()
        self._world_model = self._create_world_model()
        self._control = self._create_control()

        # Initialize services
        self._simulation_service = SimulationService(
            world_model=self._world_model,
            perception=self._perception,
            control=self._control,
            fabric=self._fabric,
        )

        # Event loop (optional)
        self._event_loop: Optional[RHEventLoop] = None
        if self._config.enable_event_loop:
            self._event_loop = RHEventLoop(
                fabric=self._fabric,
                perception=self._perception,
                world_model=self._world_model,
                control=self._control,
            )

        # gRPC server
        self._grpc_server = self._create_grpc_server()

        logger.info(
            "[RH][Service] initialized version=%s port=%d",
            self.VERSION,
            self._config.grpc.port,
        )

    # ------------------------------------------------------------------ #
    # Component Factory Methods
    # ------------------------------------------------------------------ #

    def _create_fabric(self) -> EventFabric:
        """Create EventFabric instance."""
        return EventFabric(mode=self._config.fabric.mode)

    def _create_perception(self) -> Perception:
        """Create Perception instance."""
        return Perception(
            model_name=self._config.perception.model_name,
            device=self._config.perception.device,
        )

    def _create_world_model(self) -> WorldModel:
        """Create WorldModel instance."""
        return WorldModel(
            model_name=self._config.world_model.model_name,
            horizon=self._config.world_model.horizon,
        )

    def _create_control(self) -> ControlService:
        """Create ControlService instance."""
        control_config = ControlConfig(
            controller_type=self._config.control.controller_type,
            default_env=self._config.control.default_env,
            action_timeout_sec=self._config.control.action_timeout_sec,
            max_retries=self._config.control.max_retries,
            safety_enabled=self._config.control.safety_enabled,
        )
        return ControlService(config=control_config)

    def _create_grpc_server(self) -> GRPCServer:
        """Create gRPC server with SimulationService."""
        grpc_config = GRPCServerConfig(
            port=self._config.grpc.port,
            max_workers=self._config.grpc.max_workers,
            max_message_mb=self._config.grpc.max_message_mb,
            enable_reflection=self._config.grpc.enable_reflection,
            reflection_service_names=["agi.plan.v1.SimulationService"],
        )

        server = GRPCServer(grpc_config)

        # Register SimulationService
        server.add_servicer(
            self._simulation_service,
            plan_pb2_grpc.add_SimulationServiceServicer_to_server,
        )

        return server

    # ------------------------------------------------------------------ #
    # Service Lifecycle
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Run the RH service (blocking).

        Starts:
            1. gRPC server
            2. EventLoop (in background)
            3. Signal handlers

        Blocks until shutdown signal received.
        """
        logger.info("[RH][Service] starting...")

        # Install signal handlers
        self._install_signal_handlers()

        # Start gRPC server
        self._grpc_server.start()
        logger.info(
            "[RH][Service] gRPC server running on port %d",
            self._config.grpc.port,
        )

        # Start event loop in background
        if self._event_loop:
            asyncio.ensure_future(self._run_event_loop())

        print(f"[RH] Right Hemisphere service running on port {self._config.grpc.port}")
        print("[RH] Press Ctrl+C to stop")

        # Wait for termination
        self._grpc_server.wait()

        logger.info("[RH][Service] shutdown complete")

    async def run_async(self) -> None:
        """
        Run the RH service asynchronously.

        For use when integrating with an existing asyncio event loop.
        """
        logger.info("[RH][Service] starting async...")

        # Start gRPC server
        self._grpc_server.start()
        logger.info(
            "[RH][Service] gRPC server running on port %d",
            self._config.grpc.port,
        )

        # Start event loop
        if self._event_loop:
            event_loop_task = asyncio.create_task(self._event_loop.start())

        print(f"[RH] Right Hemisphere service running on port {self._config.grpc.port}")

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cleanup
        if self._event_loop:
            self._event_loop.stop()
            await event_loop_task

        self._grpc_server.stop(grace=5.0)
        logger.info("[RH][Service] async shutdown complete")

    async def _run_event_loop(self) -> None:
        """Run the RH event loop in the background."""
        try:
            await self._event_loop.start()
        except Exception:
            logger.exception("[RH][Service] event loop error")

    def stop(self) -> None:
        """Stop the RH service."""
        logger.info("[RH][Service] stopping...")

        if self._event_loop:
            self._event_loop.stop()

        self._shutdown_event.set()
        self._grpc_server.stop(grace=5.0)

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""

        def handler(signum, frame):
            logger.info("[RH][Service] received signal %s", signum)
            self.stop()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        logger.debug("[RH][Service] signal handlers installed")

    # ------------------------------------------------------------------ #
    # Health Check
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Get service health status."""
        uptime_ms = int((time.time() - self._start_time) * 1000)

        return {
            "healthy": True,
            "status": "ready",
            "version": self.VERSION,
            "uptime_ms": uptime_ms,
            "components": {
                "perception": {
                    "healthy": True,
                    "model": self._config.perception.model_name,
                },
                "world_model": {
                    "healthy": True,
                    "model": self._config.world_model.model_name,
                    "horizon": self._config.world_model.horizon,
                },
                "control": {
                    "healthy": True,
                    "type": self._config.control.controller_type,
                },
                "fabric": {
                    "healthy": True,
                    "mode": self._config.fabric.mode,
                },
            },
        }

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def perception(self) -> Perception:
        """Get Perception instance."""
        return self._perception

    @property
    def world_model(self) -> WorldModel:
        """Get WorldModel instance."""
        return self._world_model

    @property
    def control(self) -> ControlService:
        """Get ControlService instance."""
        return self._control

    @property
    def simulation_service(self) -> SimulationService:
        """Get SimulationService instance."""
        return self._simulation_service


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AGI-HPC Right Hemisphere (RH) Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="gRPC server port (overrides config and env)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/rh.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )

    parser.add_argument(
        "--no-event-loop",
        action="store_true",
        help="Disable EventFabric event loop",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_rh_config(args.config)

    # Apply CLI overrides
    if args.port is not None:
        config.grpc.port = args.port
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.no_event_loop:
        config.enable_event_loop = False

    # Setup logging
    setup_logging(config.log_level)

    logger.info("[RH] Starting Right Hemisphere service")
    logger.info(
        "[RH] Config: port=%d, perception=%s, world_model=%s",
        config.grpc.port,
        config.perception.model_name,
        config.world_model.model_name,
    )

    # Create and run service
    try:
        service = RHService(config)
        service.run()
        return 0
    except KeyboardInterrupt:
        logger.info("[RH] Interrupted by user")
        return 0
    except Exception:
        logger.exception("[RH] Service failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
