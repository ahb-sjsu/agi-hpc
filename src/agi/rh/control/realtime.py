# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Real-Time Controllers for AGI-HPC.

Provides low-level feedback controllers for the RH control subsystem:
- PID (Proportional-Integral-Derivative) control
- MPC (Model Predictive Control) stub
- Impedance control for compliant manipulation

Sprint 6 Implementation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PID Controller
# ---------------------------------------------------------------------------


@dataclass
class PIDConfig:
    """Configuration for a PID controller."""

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    output_min: float = -float("inf")
    output_max: float = float("inf")
    integral_max: float = 100.0
    dt: float = 0.01


class PIDController:
    """Classic PID feedback controller with anti-windup."""

    def __init__(self, config: Optional[PIDConfig] = None) -> None:
        self._config = config or PIDConfig()
        self._integral: float = 0.0
        self._prev_error: Optional[float] = None
        self._last_time: Optional[float] = None
        logger.info(
            "[rh][realtime] PIDController initialized kp=%.3f ki=%.3f kd=%.3f",
            self._config.kp,
            self._config.ki,
            self._config.kd,
        )

    def update(
        self, setpoint: float, measured: float, dt: Optional[float] = None
    ) -> float:
        """Compute the PID control output for the current timestep."""
        now = time.monotonic()
        if dt is None:
            if self._last_time is not None:
                dt = now - self._last_time
            else:
                dt = self._config.dt
        self._last_time = now
        dt = max(dt, 1e-9)
        error = setpoint - measured
        p_term = self._config.kp * error
        self._integral += error * dt
        self._integral = max(
            -self._config.integral_max, min(self._config.integral_max, self._integral)
        )
        i_term = self._config.ki * self._integral
        if self._prev_error is not None:
            d_term = self._config.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0
        self._prev_error = error
        output = p_term + i_term + d_term
        output = max(self._config.output_min, min(self._config.output_max, output))
        return output

    def reset(self) -> None:
        """Reset the controller state."""
        self._integral = 0.0
        self._prev_error = None
        self._last_time = None

    def update_vector(
        self, setpoint: np.ndarray, measured: np.ndarray, dt: Optional[float] = None
    ) -> np.ndarray:
        """Compute PID output for a multi-dimensional setpoint."""
        now = time.monotonic()
        if dt is None:
            dt = (
                (now - self._last_time)
                if self._last_time is not None
                else self._config.dt
            )
        self._last_time = now
        dt = max(dt, 1e-9)
        error = setpoint - measured
        p_term = self._config.kp * error
        if (
            not isinstance(self._integral, np.ndarray)
            or self._integral.shape != error.shape
        ):
            self._integral = np.zeros_like(error, dtype=np.float64)
        self._integral += error * dt
        self._integral = np.clip(
            self._integral, -self._config.integral_max, self._config.integral_max
        )
        i_term = self._config.ki * self._integral
        if self._prev_error is not None and isinstance(self._prev_error, np.ndarray):
            d_term = self._config.kd * (error - self._prev_error) / dt
        else:
            d_term = np.zeros_like(error)
        self._prev_error = error.copy()
        output = p_term + i_term + d_term
        return np.clip(output, self._config.output_min, self._config.output_max)


# ---------------------------------------------------------------------------
# MPC Controller (Stub)
# ---------------------------------------------------------------------------


@dataclass
class MPCConfig:
    """Configuration for Model Predictive Control."""

    horizon: int = 20
    dt: float = 0.01
    state_dim: int = 6
    control_dim: int = 3
    q_weight: float = 1.0
    r_weight: float = 0.01
    control_min: float = -10.0
    control_max: float = 10.0


class MPCController:
    """Model Predictive Controller (stub). Full QP solver deferred."""

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        dynamics_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._config = config or MPCConfig()
        self._dynamics_fn = dynamics_fn
        logger.info(
            "[rh][realtime] MPCController initialized (stub) horizon=%d",
            self._config.horizon,
        )

    def compute(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute MPC control action. Currently proportional placeholder."""
        error = (
            reference[: self._config.control_dim] - state[: self._config.control_dim]
        )
        action = np.clip(
            self._config.q_weight * error,
            self._config.control_min,
            self._config.control_max,
        )
        return action

    def reset(self) -> None:
        """Reset MPC internal state."""
        pass


# ---------------------------------------------------------------------------
# Impedance Controller
# ---------------------------------------------------------------------------


@dataclass
class ImpedanceConfig:
    """Configuration for impedance control: F = K*(xd-x) + D*(vd-v)."""

    stiffness: float = 100.0
    damping: float = 20.0
    inertia: float = 1.0
    dim: int = 3
    force_limit: float = 50.0


class ImpedanceController:
    """Impedance controller for compliant robot manipulation."""

    def __init__(self, config: Optional[ImpedanceConfig] = None) -> None:
        self._config = config or ImpedanceConfig()
        self._stiffness_matrix = np.eye(self._config.dim) * self._config.stiffness
        self._damping_matrix = np.eye(self._config.dim) * self._config.damping
        logger.info(
            "[rh][realtime] ImpedanceController initialized K=%.1f D=%.1f dim=%d",
            self._config.stiffness,
            self._config.damping,
            self._config.dim,
        )

    def compute(
        self,
        desired_pos: np.ndarray,
        desired_vel: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
    ) -> np.ndarray:
        """Compute impedance control force/torque."""
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel
        force = self._stiffness_matrix @ pos_error + self._damping_matrix @ vel_error
        force_magnitude = float(np.linalg.norm(force))
        if force_magnitude > self._config.force_limit:
            force = force * (self._config.force_limit / force_magnitude)
        return force

    def set_stiffness(self, stiffness: float) -> None:
        """Update the virtual stiffness."""
        self._config.stiffness = stiffness
        self._stiffness_matrix = np.eye(self._config.dim) * stiffness

    def set_damping(self, damping: float) -> None:
        """Update the virtual damping."""
        self._config.damping = damping
        self._damping_matrix = np.eye(self._config.dim) * damping

    def set_matrices(
        self, stiffness_matrix: np.ndarray, damping_matrix: np.ndarray
    ) -> None:
        """Set full stiffness and damping matrices for anisotropic control."""
        self._stiffness_matrix = np.array(stiffness_matrix, dtype=np.float64)
        self._damping_matrix = np.array(damping_matrix, dtype=np.float64)
