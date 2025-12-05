#!/usr/bin/env python3
r"""
generate_services.py (improved v2)

Creates AGI-HPC service skeletons:

- LH service (planning, meta, event handling, gRPC stub)
- RH service (perception, world-model stub, gRPC stub)
- Event fabric abstraction
- Safe YAML config loader
- Common gRPC server infrastructure

Every file creation is logged.

Usage:
    python generate_services.py
    python generate_services.py --overwrite
"""

import argparse
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------

def log(msg: str):
    print(f"[generate] {msg}")

def write(path: Path, content: str, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        log(f"SKIP    (exists): {path}")
        return False
    path.write_text(content, encoding="utf-8")
    log(f"WRITE   {path}")
    return True

# ---------------------------------------------------------------------
# File content templates
# ---------------------------------------------------------------------

CONFIG_LOADER = dedent(r'''\
"""
Safe YAML config loader for AGI-HPC services.
"""

from dataclasses import dataclass
from typing import Any, Dict
import yaml
import os

class ConfigError(Exception):
    pass

@dataclass
class ServiceConfig:
    name: str
    rpc_port: int
    event_uri: str
    metrics_port: int
    extra: Dict[str, Any]

def load_config(path: str) -> ServiceConfig:
    if not os.path.exists(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if "service" not in raw:
        raise ConfigError("Missing 'service' section in config.")

    svc = raw["service"]

    required = ["name", "rpc_port", "event_uri", "metrics_port"]
    for key in required:
        if key not in svc:
            raise ConfigError(f"Missing required config key: {key}")

    return ServiceConfig(
        name=svc["name"],
        rpc_port=int(svc["rpc_port"]),
        event_uri=str(svc["event_uri"]),
        metrics_port=int(svc["metrics_port"]),
        extra=raw,
    )
''')

EVENT_FABRIC = dedent(r'''\
"""
Event fabric stub.

This is a simple in-process pub/sub dispatcher.
Later replace with UCX/ZeroMQ-based transport for inter-node communication.
"""

from typing import Callable, Dict, List
import threading

class EventFabric:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, message: dict):
        with self._lock:
            handlers = list(self._subscribers.get(topic, []))
        for fn in handlers:
            fn(message)

    def subscribe(self, topic: str, handler: Callable):
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        print(f"[event-fabric] Subscribed to {topic} -> {handler.__name__}")
''')

GRPC_SERVER = dedent(r'''\
"""
Reusable gRPC server utilities for AGI-HPC services.
"""

import grpc
from concurrent import futures

class GRPCServer:
    def __init__(self, port: int):
        self.port = port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))

    def add_servicer(self, servicer, add_fn):
        add_fn(servicer, self.server)

    def start(self):
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start()
        print(f"[grpc] Running on port {self.port}")
        return self.server

    def wait(self):
        self.server.wait_for_termination()
''')

LH_SERVICE = dedent(r'''\
"""
Left Hemisphere (LH) service skeleton.
"""

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

class LHService:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.grpc = GRPCServer(self.config.rpc_port)

    def on_state_update(self, msg: dict):
        print(f"[LH] Received state update: {msg}")
        # TODO: planning
        # TODO: memory queries
        # TODO: metacognition
        # TODO: publish plan.step_ready
        # self.fabric.publish("plan.step_ready", {...})

    def run(self):
        self.fabric.subscribe("perception.state_update", self.on_state_update)
        print("[LH] Service ready.")
        self.grpc.start()
        self.grpc.wait()

def main():
    svc = LHService("configs/lh_config.yaml")
    svc.run()

if __name__ == "__main__":
    main()
''')

RH_SERVICE = dedent(r'''\
"""
Right Hemisphere (RH) service skeleton.
"""

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

class RHService:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.grpc = GRPCServer(self.config.rpc_port)

    def on_plan_ready(self, msg: dict):
        print(f"[RH] Received plan-step: {msg}")
        # TODO: read env state
        # TODO: world-model simulation
        # TODO: generate simulation.result
        # TODO: actuator control

    def run(self):
        self.fabric.subscribe("plan.step_ready", self.on_plan_ready)
        print("[RH] Service ready.")
        self.grpc.start()
        self.grpc.wait()

def main():
    svc = RHService("configs/rh_config.yaml")
    svc.run()

if __name__ == "__main__":
    main()
''')

# ---------------------------------------------------------------------
# Drive file generation
# ---------------------------------------------------------------------

def generate(overwrite: bool):
    log("Starting service skeleton generation...")

    base = Path("src/agi")

    targets = {
        "common/config_loader.py": CONFIG_LOADER,
        "core/events/fabric.py": EVENT_FABRIC,
        "core/api/grpc_server.py": GRPC_SERVER,
        "lh/service.py": LH_SERVICE,
        "rh/service.py": RH_SERVICE,
    }

    created = 0
    skipped = 0

    for rel, content in targets.items():
        path = base / rel
        ok = write(path, content, overwrite)
        if ok:
            created += 1
        else:
            skipped += 1

    log("")
    log(f"Summary:")
    log(f"  Created files : {created}")
    log(f"  Skipped files : {skipped}")
    log("Done.")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    generate(args.overwrite)
