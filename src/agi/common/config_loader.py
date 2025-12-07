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
