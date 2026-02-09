# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Right Hemisphere (RH) Configuration.

Provides configuration management for RH subsystem:
- Service settings (port, workers)
- Perception model settings
- World model parameters
- Control parameters
- Environment variable overrides

Usage:
    from agi.rh.config import RHConfig, load_rh_config

    # Load from YAML file
    config = load_rh_config("configs/rh.yaml")

    # Use defaults with env overrides
    config = RHConfig()

Environment Variables:
    AGI_RH_PORT              gRPC port (default: 50057)
    AGI_RH_GRPC_WORKERS      Max gRPC workers (default: 16)
    AGI_RH_PERCEPTION_MODEL  Perception model name
    AGI_RH_PERCEPTION_DEVICE Device for perception (cpu/cuda)
    AGI_RH_WORLD_MODEL       World model name
    AGI_RH_WORLD_HORIZON     World model rollout horizon
    AGI_RH_CONTROLLER_TYPE   Controller type
    AGI_RH_ENV_NAME          Default environment name
    AGI_FABRIC_MODE          EventFabric mode (local/zmq/redis)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment Variable Helpers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid int for %s: %s, using default %d", name, val, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid float for %s: %s, using default %f", name, val, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _env_str(name: str, default: str) -> str:
    """Get string from environment variable."""
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PerceptionConfig:
    """Perception subsystem configuration."""

    model_name: str = field(
        default_factory=lambda: _env_str("AGI_RH_PERCEPTION_MODEL", "dummy_encoder")
    )
    device: str = field(
        default_factory=lambda: _env_str("AGI_RH_PERCEPTION_DEVICE", "cpu")
    )
    batch_size: int = 1
    max_objects: int = 100
    confidence_threshold: float = 0.5
    enable_tracking: bool = False


@dataclass
class WorldModelConfig:
    """World model subsystem configuration."""

    model_name: str = field(
        default_factory=lambda: _env_str("AGI_RH_WORLD_MODEL", "dummy_world_model")
    )
    horizon: int = field(default_factory=lambda: _env_int("AGI_RH_WORLD_HORIZON", 5))
    physics_dt: float = 0.01  # Physics timestep in seconds
    risk_threshold: float = 0.5
    enable_gpu: bool = False


@dataclass
class ControlConfig:
    """Control subsystem configuration."""

    controller_type: str = field(
        default_factory=lambda: _env_str("AGI_RH_CONTROLLER_TYPE", "rule_based")
    )
    default_env: str = field(
        default_factory=lambda: _env_str("AGI_RH_ENV_NAME", "mock:simple")
    )
    action_timeout_sec: float = 30.0
    max_retries: int = 3
    safety_enabled: bool = True


@dataclass
class EventFabricConfig:
    """EventFabric configuration."""

    mode: str = field(default_factory=lambda: _env_str("AGI_FABRIC_MODE", "local"))
    zmq_pub_address: str = field(
        default_factory=lambda: _env_str("AGI_FABRIC_ZMQ_PUB", "tcp://localhost:5555")
    )
    zmq_sub_address: str = field(
        default_factory=lambda: _env_str("AGI_FABRIC_ZMQ_SUB", "tcp://localhost:5556")
    )
    redis_url: str = field(
        default_factory=lambda: _env_str(
            "AGI_FABRIC_REDIS_URL", "redis://localhost:6379"
        )
    )


@dataclass
class GRPCConfig:
    """gRPC server configuration."""

    port: int = field(default_factory=lambda: _env_int("AGI_RH_PORT", 50057))
    max_workers: int = field(
        default_factory=lambda: _env_int("AGI_RH_GRPC_WORKERS", 16)
    )
    max_message_mb: int = 64
    enable_reflection: bool = field(
        default_factory=lambda: _env_bool("AGI_RH_GRPC_REFLECTION", True)
    )


@dataclass
class RHConfig:
    """
    Complete RH service configuration.

    Combines all subsystem configurations with service-level settings.
    """

    # Service identification
    service_name: str = "rh-service"
    version: str = "1.0.0"

    # Subsystem configurations
    grpc: GRPCConfig = field(default_factory=GRPCConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    fabric: EventFabricConfig = field(default_factory=EventFabricConfig)

    # Logging
    log_level: str = field(default_factory=lambda: _env_str("AGI_RH_LOG_LEVEL", "INFO"))

    # Feature flags
    enable_event_loop: bool = True
    enable_health_check: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.grpc.port < 1 or self.grpc.port > 65535:
            raise ValueError(f"Invalid port: {self.grpc.port}")
        if self.world_model.horizon < 1:
            raise ValueError(f"Invalid horizon: {self.world_model.horizon}")


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def load_rh_config(config_path: Optional[str] = None) -> RHConfig:
    """
    Load RH configuration from YAML file with environment overrides.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        RHConfig instance
    """
    config = RHConfig()

    if config_path:
        path = Path(config_path)
        if path.exists():
            config = _load_from_yaml(path, config)
        else:
            logger.warning("Config file not found: %s, using defaults", config_path)

    logger.info(
        "[RH][Config] loaded port=%d perception=%s world_model=%s",
        config.grpc.port,
        config.perception.model_name,
        config.world_model.model_name,
    )

    return config


def _load_from_yaml(path: Path, default: RHConfig) -> RHConfig:
    """Load configuration from YAML file."""
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        if not data:
            return default

        return _merge_config(default, data)

    except ImportError:
        logger.warning("PyYAML not installed, using defaults")
        return default
    except Exception as e:
        logger.warning("Failed to load config from %s: %s", path, e)
        return default


def _merge_config(config: RHConfig, data: Dict[str, Any]) -> RHConfig:
    """Merge YAML data into config, preserving env overrides."""
    # gRPC settings
    if "grpc" in data:
        grpc_data = data["grpc"]
        if "port" in grpc_data and not os.getenv("AGI_RH_PORT"):
            config.grpc.port = grpc_data["port"]
        if "max_workers" in grpc_data and not os.getenv("AGI_RH_GRPC_WORKERS"):
            config.grpc.max_workers = grpc_data["max_workers"]
        if "max_message_mb" in grpc_data:
            config.grpc.max_message_mb = grpc_data["max_message_mb"]
        if "enable_reflection" in grpc_data:
            config.grpc.enable_reflection = grpc_data["enable_reflection"]

    # Perception settings
    if "perception" in data:
        perc_data = data["perception"]
        if "model_name" in perc_data and not os.getenv("AGI_RH_PERCEPTION_MODEL"):
            config.perception.model_name = perc_data["model_name"]
        if "device" in perc_data and not os.getenv("AGI_RH_PERCEPTION_DEVICE"):
            config.perception.device = perc_data["device"]
        if "batch_size" in perc_data:
            config.perception.batch_size = perc_data["batch_size"]
        if "max_objects" in perc_data:
            config.perception.max_objects = perc_data["max_objects"]
        if "confidence_threshold" in perc_data:
            config.perception.confidence_threshold = perc_data["confidence_threshold"]

    # World model settings
    if "world_model" in data:
        wm_data = data["world_model"]
        if "model_name" in wm_data and not os.getenv("AGI_RH_WORLD_MODEL"):
            config.world_model.model_name = wm_data["model_name"]
        if "horizon" in wm_data and not os.getenv("AGI_RH_WORLD_HORIZON"):
            config.world_model.horizon = wm_data["horizon"]
        if "physics_dt" in wm_data:
            config.world_model.physics_dt = wm_data["physics_dt"]
        if "risk_threshold" in wm_data:
            config.world_model.risk_threshold = wm_data["risk_threshold"]

    # Control settings
    if "control" in data:
        ctrl_data = data["control"]
        if "controller_type" in ctrl_data and not os.getenv("AGI_RH_CONTROLLER_TYPE"):
            config.control.controller_type = ctrl_data["controller_type"]
        if "default_env" in ctrl_data and not os.getenv("AGI_RH_ENV_NAME"):
            config.control.default_env = ctrl_data["default_env"]
        if "action_timeout_sec" in ctrl_data:
            config.control.action_timeout_sec = ctrl_data["action_timeout_sec"]
        if "safety_enabled" in ctrl_data:
            config.control.safety_enabled = ctrl_data["safety_enabled"]

    # Fabric settings
    if "fabric" in data:
        fab_data = data["fabric"]
        if "mode" in fab_data and not os.getenv("AGI_FABRIC_MODE"):
            config.fabric.mode = fab_data["mode"]
        if "zmq_pub_address" in fab_data:
            config.fabric.zmq_pub_address = fab_data["zmq_pub_address"]
        if "zmq_sub_address" in fab_data:
            config.fabric.zmq_sub_address = fab_data["zmq_sub_address"]
        if "redis_url" in fab_data:
            config.fabric.redis_url = fab_data["redis_url"]

    # Service settings
    if "service_name" in data:
        config.service_name = data["service_name"]
    if "log_level" in data and not os.getenv("AGI_RH_LOG_LEVEL"):
        config.log_level = data["log_level"]
    if "enable_event_loop" in data:
        config.enable_event_loop = data["enable_event_loop"]
    if "enable_health_check" in data:
        config.enable_health_check = data["enable_health_check"]

    return config
