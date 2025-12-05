\
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
