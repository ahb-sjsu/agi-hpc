# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from agi.core.api.grpc_server import GRPCServer, GRPCServerConfig
from agi.core.events.fabric import EventFabric
from agi.proto_gen import plan_pb2_grpc

from agi.lh.planner import Planner
from agi.lh.plan_service import PlanService, LHPlanServiceConfig
from agi.lh.memory_client import MemoryClient
from agi.lh.safety_client import SafetyClient
from agi.lh.metacog_client import MetacognitionClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class LHServiceConfig:
    port: int = field(default_factory=lambda: int(os.getenv("AGI_LH_PORT", "50100")))
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("AGI_LH_MAX_WORKERS", "16"))
    )
    memory_addr: str = field(
        default_factory=lambda: os.getenv("AGI_LH_MEMORY_ADDR", "localhost:50110")
    )
    safety_addr: str = field(
        default_factory=lambda: os.getenv("AGI_LH_SAFETY_ADDR", "localhost:50200")
    )
    meta_addr: str = field(
        default_factory=lambda: os.getenv("AGI_LH_META_ADDR", "localhost:50300")
    )
    fabric_mode: str = field(
        default_factory=lambda: os.getenv("AGI_FABRIC_MODE", "local")
    )
    fabric_identity: str = field(
        default_factory=lambda: os.getenv("AGI_FABRIC_IDENTITY", "LH")
    )
    enable_safety: bool = field(
        default_factory=lambda: os.getenv("AGI_LH_ENABLE_SAFETY", "true").lower()
        == "true"
    )
    enable_metacognition: bool = field(
        default_factory=lambda: os.getenv("AGI_LH_ENABLE_META", "true").lower()
        == "true"
    )
    enable_reflection: bool = field(
        default_factory=lambda: os.getenv("AGI_LH_ENABLE_REFLECTION", "true").lower()
        == "true"
    )


class LHService:
    def __init__(self, config: Optional[LHServiceConfig] = None) -> None:
        self._cfg = config or LHServiceConfig()
        self._server: Optional[GRPCServer] = None
        self._fabric: Optional[EventFabric] = None
        logger.info(
            "[LH] Initializing port=%d fabric=%s", self._cfg.port, self._cfg.fabric_mode
        )

    def start(self) -> None:
        self._fabric = EventFabric(
            mode=self._cfg.fabric_mode, identity=self._cfg.fabric_identity
        )
        memory_client = MemoryClient(address=self._cfg.memory_addr)
        safety_client = SafetyClient(address=self._cfg.safety_addr)
        metacog_client = MetacognitionClient(address=self._cfg.meta_addr)
        planner = Planner()

        plan_service_config = LHPlanServiceConfig(
            enable_safety=self._cfg.enable_safety,
            enable_metacognition=self._cfg.enable_metacognition,
            node_id=self._cfg.fabric_identity,
        )
        plan_service = PlanService(
            planner=planner,
            memory=memory_client,
            safety=safety_client,
            metacog=metacog_client,
            fabric=self._fabric,
            config=plan_service_config,
        )

        grpc_config = GRPCServerConfig(
            port=self._cfg.port,
            max_workers=self._cfg.max_workers,
            enable_reflection=self._cfg.enable_reflection,
            reflection_service_names=["agi.plan.v1.PlanService"],
        )
        self._server = GRPCServer(config=grpc_config)
        self._server.add_servicer(
            plan_service, plan_pb2_grpc.add_PlanServiceServicer_to_server
        )
        self._server.add_signal_handlers()
        self._server.start()
        logger.info("[LH] Started on port %d", self._cfg.port)

    def wait(self) -> None:
        if self._server:
            self._server.wait()

    def stop(self) -> None:
        logger.info("[LH] Stopping...")
        if self._server:
            self._server.stop(grace=5.0)
        if self._fabric:
            self._fabric.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGI-HPC LH Service")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--fabric-mode", choices=["local", "zmq", "ucx"], default=None)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--no-meta", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config = LHServiceConfig()
    if args.port:
        config.port = args.port
    if args.fabric_mode:
        config.fabric_mode = args.fabric_mode
    if args.no_safety:
        config.enable_safety = False
    if args.no_meta:
        config.enable_metacognition = False

    service = LHService(config=config)
    try:
        service.start()
        service.wait()
    except KeyboardInterrupt:
        logger.info("[LH] Interrupted")
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
