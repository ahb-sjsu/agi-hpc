# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Anomaly Detection for AGI-HPC Metacognition Subsystem.

Implements Sprint 5 requirements:
- Confidence drift detection
- Latency spike detection
- Decision pattern anomalies
- Outcome mismatch detection

Detects unusual patterns in metacognitive behavior that might
indicate problems with reasoning, calibration, or system health.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Types
# ---------------------------------------------------------------------------


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    CONFIDENCE_DRIFT = "confidence_drift"
    CONFIDENCE_DROP = "confidence_drop"
    CONFIDENCE_SPIKE = "confidence_spike"
    LATENCY_SPIKE = "latency_spike"
    DECISION_PATTERN = "decision_pattern"
    OUTCOME_MISMATCH = "outcome_mismatch"
    ERROR_RATE_SPIKE = "error_rate_spike"
    CALIBRATION_DRIFT = "calibration_drift"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class Anomaly:
    """
    Detected anomaly in metacognitive behavior.

    Represents an unusual pattern that may indicate problems
    with reasoning, calibration, or system health.
    """

    anomaly_type: AnomalyType
    description: str
    severity: AnomalySeverity
    confidence: float  # Confidence in the anomaly detection (0-1)
    detected_at: int  # Timestamp in milliseconds
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity_score(self) -> float:
        """Return numeric severity score."""
        return {
            AnomalySeverity.INFO: 0.3,
            AnomalySeverity.WARNING: 0.6,
            AnomalySeverity.CRITICAL: 0.9,
        }.get(self.severity, 0.5)


@dataclass
class AnomalyDetectorConfig:
    """Configuration for the anomaly detector."""

    # Confidence drift detection
    confidence_drift_threshold: float = 0.15
    confidence_drop_threshold: float = 0.2
    confidence_spike_threshold: float = 0.25

    # Latency detection
    latency_spike_multiplier: float = 3.0
    latency_critical_multiplier: float = 5.0

    # Outcome mismatch
    outcome_mismatch_threshold: float = 0.3

    # Decision patterns
    decision_streak_threshold: int = 10
    error_rate_threshold: float = 0.3

    # Window sizes
    window_size: int = 50
    min_samples_for_detection: int = 10

    # Alert settings
    alert_cooldown_ms: int = 5000


@dataclass
class DecisionRecord:
    """Record of a metacognitive decision for pattern analysis."""

    timestamp: int
    decision: str  # APPROVE, REVISE, ESCALATE
    confidence: float
    outcome: Optional[bool] = None  # True=success, False=failure, None=unknown
    latency_ms: int = 0


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """
    Detects anomalies in metacognitive behavior.

    Monitors patterns in confidence, latency, decisions, and outcomes
    to detect unusual behavior that may indicate problems.

    Usage:
        detector = AnomalyDetector()

        # After each metacognitive review
        anomalies = detector.check_all(
            confidence=0.85,
            latency_ms=150,
            decision="APPROVE",
        )

        for anomaly in anomalies:
            if anomaly.severity == AnomalySeverity.CRITICAL:
                notify_operators(anomaly)
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None) -> None:
        self._config = config or AnomalyDetectorConfig()

        # History tracking
        self._confidence_history: Deque[float] = deque(maxlen=self._config.window_size)
        self._latency_history: Deque[int] = deque(maxlen=self._config.window_size)
        self._decision_history: Deque[DecisionRecord] = deque(
            maxlen=self._config.window_size
        )

        # State tracking
        self._last_alert_times: Dict[AnomalyType, int] = {}
        self._total_decisions = 0
        self._total_errors = 0

        logger.info("[Meta][Anomaly] Detector initialized")

    def check_all(
        self,
        confidence: float,
        latency_ms: int,
        decision: str,
        outcome: Optional[bool] = None,
    ) -> List[Anomaly]:
        """
        Run all anomaly checks and return detected anomalies.

        Args:
            confidence: Confidence score from metacognitive review
            latency_ms: Time taken for the review in milliseconds
            decision: Decision made (APPROVE, REVISE, ESCALATE)
            outcome: Optional outcome of the decision

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Record the decision
        record = DecisionRecord(
            timestamp=int(time.time() * 1000),
            decision=decision,
            confidence=confidence,
            outcome=outcome,
            latency_ms=latency_ms,
        )
        self._decision_history.append(record)
        self._total_decisions += 1

        if outcome is False:
            self._total_errors += 1

        # Run all checks
        confidence_anomaly = self.check_confidence_drift(confidence)
        if confidence_anomaly:
            anomalies.append(confidence_anomaly)

        latency_anomaly = self.check_latency_spike(latency_ms)
        if latency_anomaly:
            anomalies.append(latency_anomaly)

        pattern_anomaly = self.check_decision_pattern(decision)
        if pattern_anomaly:
            anomalies.append(pattern_anomaly)

        if outcome is not None:
            mismatch_anomaly = self.check_outcome_mismatch(confidence, outcome)
            if mismatch_anomaly:
                anomalies.append(mismatch_anomaly)

        error_anomaly = self.check_error_rate()
        if error_anomaly:
            anomalies.append(error_anomaly)

        # Log anomalies
        for anomaly in anomalies:
            logger.warning(
                "[Meta][Anomaly] Detected %s: %s (severity=%s)",
                anomaly.anomaly_type.value,
                anomaly.description,
                anomaly.severity.value,
            )

        return anomalies

    def check_confidence_drift(self, current_confidence: float) -> Optional[Anomaly]:
        """
        Check for unusual confidence drift.

        Detects:
        - Gradual drift over time
        - Sudden drops in confidence
        - Sudden spikes (overconfidence)
        """
        self._confidence_history.append(current_confidence)

        if len(self._confidence_history) < self._config.min_samples_for_detection:
            return None

        recent = list(self._confidence_history)[-10:]
        older = (
            list(self._confidence_history)[-20:-10]
            if len(self._confidence_history) >= 20
            else recent
        )

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        drift = recent_mean - older_mean

        # Check for confidence drop
        if drift < -self._config.confidence_drop_threshold:
            if not self._should_alert(AnomalyType.CONFIDENCE_DROP):
                return None

            return Anomaly(
                anomaly_type=AnomalyType.CONFIDENCE_DROP,
                description=f"Confidence dropped by {abs(drift):.2f} "
                f"(from {older_mean:.2f} to {recent_mean:.2f})",
                severity=(
                    AnomalySeverity.WARNING
                    if abs(drift) < 0.3
                    else AnomalySeverity.CRITICAL
                ),
                confidence=min(
                    1.0, abs(drift) / self._config.confidence_drop_threshold
                ),
                detected_at=int(time.time() * 1000),
                context={
                    "recent_mean": recent_mean,
                    "older_mean": older_mean,
                    "drift": drift,
                },
            )

        # Check for confidence spike (overconfidence)
        if drift > self._config.confidence_spike_threshold:
            if not self._should_alert(AnomalyType.CONFIDENCE_SPIKE):
                return None

            return Anomaly(
                anomaly_type=AnomalyType.CONFIDENCE_SPIKE,
                description=f"Confidence spiked by {drift:.2f} "
                f"(from {older_mean:.2f} to {recent_mean:.2f})",
                severity=AnomalySeverity.WARNING,
                confidence=min(1.0, drift / self._config.confidence_spike_threshold),
                detected_at=int(time.time() * 1000),
                context={
                    "recent_mean": recent_mean,
                    "older_mean": older_mean,
                    "drift": drift,
                },
            )

        # Check for general drift
        if abs(drift) > self._config.confidence_drift_threshold:
            if not self._should_alert(AnomalyType.CONFIDENCE_DRIFT):
                return None

            direction = "increased" if drift > 0 else "decreased"
            return Anomaly(
                anomaly_type=AnomalyType.CONFIDENCE_DRIFT,
                description=f"Confidence {direction} by {abs(drift):.2f}",
                severity=AnomalySeverity.INFO,
                confidence=min(
                    1.0, abs(drift) / self._config.confidence_drift_threshold
                ),
                detected_at=int(time.time() * 1000),
                context={
                    "recent_mean": recent_mean,
                    "older_mean": older_mean,
                    "drift": drift,
                },
            )

        return None

    def check_latency_spike(self, review_time_ms: int) -> Optional[Anomaly]:
        """
        Check for unusual latency spikes.

        Detects reviews that take significantly longer than normal.
        """
        self._latency_history.append(review_time_ms)

        if len(self._latency_history) < self._config.min_samples_for_detection:
            return None

        # Exclude current value from baseline
        history = list(self._latency_history)[:-1]
        mean_latency = np.mean(history)
        std_latency = np.std(history) or 1.0

        z_score = (review_time_ms - mean_latency) / std_latency

        if z_score > self._config.latency_critical_multiplier:
            if not self._should_alert(AnomalyType.LATENCY_SPIKE):
                return None

            return Anomaly(
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                description=f"Review took {review_time_ms}ms "
                f"(expected ~{mean_latency:.0f}ms, {z_score:.1f}x std)",
                severity=AnomalySeverity.CRITICAL,
                confidence=min(1.0, z_score / 10.0),
                detected_at=int(time.time() * 1000),
                context={
                    "review_time_ms": review_time_ms,
                    "expected_ms": mean_latency,
                    "z_score": z_score,
                },
            )

        if z_score > self._config.latency_spike_multiplier:
            if not self._should_alert(AnomalyType.LATENCY_SPIKE):
                return None

            return Anomaly(
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                description=f"Review took {review_time_ms}ms "
                f"(expected ~{mean_latency:.0f}ms)",
                severity=AnomalySeverity.WARNING,
                confidence=min(1.0, z_score / 5.0),
                detected_at=int(time.time() * 1000),
                context={
                    "review_time_ms": review_time_ms,
                    "expected_ms": mean_latency,
                    "z_score": z_score,
                },
            )

        return None

    def check_decision_pattern(self, decision: str) -> Optional[Anomaly]:
        """
        Check for unusual decision patterns.

        Detects:
        - Long streaks of the same decision
        - Unusual decision distributions
        """
        if len(self._decision_history) < self._config.min_samples_for_detection:
            return None

        # Check for decision streaks
        recent_decisions = [d.decision for d in list(self._decision_history)[-20:]]

        # Count consecutive same decisions
        streak = 1
        for i in range(len(recent_decisions) - 1, 0, -1):
            if recent_decisions[i] == recent_decisions[i - 1]:
                streak += 1
            else:
                break

        if streak >= self._config.decision_streak_threshold:
            if not self._should_alert(AnomalyType.DECISION_PATTERN):
                return None

            return Anomaly(
                anomaly_type=AnomalyType.DECISION_PATTERN,
                description=f"Decision streak: {streak} consecutive '{decision}' decisions",
                severity=AnomalySeverity.WARNING,
                confidence=min(
                    1.0, streak / (self._config.decision_streak_threshold * 2)
                ),
                detected_at=int(time.time() * 1000),
                context={
                    "streak_length": streak,
                    "decision": decision,
                    "recent_decisions": recent_decisions[-10:],
                },
            )

        return None

    def check_outcome_mismatch(
        self,
        confidence: float,
        outcome: bool,
    ) -> Optional[Anomaly]:
        """
        Check for mismatch between confidence and outcome.

        Detects cases where high confidence leads to failure
        or low confidence leads to success.
        """
        # High confidence failure
        if confidence > 0.8 and outcome is False:
            if not self._should_alert(AnomalyType.OUTCOME_MISMATCH):
                return None

            return Anomaly(
                anomaly_type=AnomalyType.OUTCOME_MISMATCH,
                description=f"High confidence ({confidence:.2f}) but outcome was failure",
                severity=AnomalySeverity.WARNING,
                confidence=confidence,
                detected_at=int(time.time() * 1000),
                context={
                    "confidence": confidence,
                    "outcome": outcome,
                    "mismatch_type": "overconfident_failure",
                },
            )

        # Track calibration over time
        recent_records = [d for d in self._decision_history if d.outcome is not None][
            -20:
        ]

        if len(recent_records) >= 10:
            # Calculate expected vs actual success rate
            high_conf_records = [r for r in recent_records if r.confidence > 0.7]
            if high_conf_records:
                expected_success_rate = np.mean(
                    [r.confidence for r in high_conf_records]
                )
                actual_success_rate = np.mean(
                    [1.0 if r.outcome else 0.0 for r in high_conf_records]
                )

                calibration_error = abs(expected_success_rate - actual_success_rate)

                if calibration_error > self._config.outcome_mismatch_threshold:
                    if not self._should_alert(AnomalyType.CALIBRATION_DRIFT):
                        return None

                    return Anomaly(
                        anomaly_type=AnomalyType.CALIBRATION_DRIFT,
                        description=f"Calibration error: expected {expected_success_rate:.2f} "
                        f"success rate, actual {actual_success_rate:.2f}",
                        severity=AnomalySeverity.WARNING,
                        confidence=min(1.0, calibration_error / 0.5),
                        detected_at=int(time.time() * 1000),
                        context={
                            "expected_success_rate": expected_success_rate,
                            "actual_success_rate": actual_success_rate,
                            "calibration_error": calibration_error,
                            "sample_size": len(high_conf_records),
                        },
                    )

        return None

    def check_error_rate(self) -> Optional[Anomaly]:
        """
        Check for unusual error rate spikes.
        """
        if self._total_decisions < self._config.min_samples_for_detection:
            return None

        # Check recent error rate
        recent_records = [d for d in self._decision_history if d.outcome is not None][
            -20:
        ]

        if len(recent_records) < 10:
            return None

        recent_error_rate = sum(1 for r in recent_records if r.outcome is False) / len(
            recent_records
        )

        if recent_error_rate > self._config.error_rate_threshold:
            if not self._should_alert(AnomalyType.ERROR_RATE_SPIKE):
                return None

            return Anomaly(
                anomaly_type=AnomalyType.ERROR_RATE_SPIKE,
                description=f"Error rate spike: {recent_error_rate:.1%} "
                f"(threshold: {self._config.error_rate_threshold:.1%})",
                severity=(
                    AnomalySeverity.CRITICAL
                    if recent_error_rate > 0.5
                    else AnomalySeverity.WARNING
                ),
                confidence=min(
                    1.0, recent_error_rate / self._config.error_rate_threshold
                ),
                detected_at=int(time.time() * 1000),
                context={
                    "recent_error_rate": recent_error_rate,
                    "sample_size": len(recent_records),
                    "threshold": self._config.error_rate_threshold,
                },
            )

        return None

    def _should_alert(self, anomaly_type: AnomalyType) -> bool:
        """Check if we should alert for this anomaly type (cooldown check)."""
        now = int(time.time() * 1000)
        last_alert = self._last_alert_times.get(anomaly_type, 0)

        if now - last_alert < self._config.alert_cooldown_ms:
            return False

        self._last_alert_times[anomaly_type] = now
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        confidence_list = list(self._confidence_history)
        latency_list = list(self._latency_history)

        return {
            "total_decisions": self._total_decisions,
            "total_errors": self._total_errors,
            "error_rate": (
                self._total_errors / self._total_decisions
                if self._total_decisions > 0
                else 0.0
            ),
            "confidence_mean": np.mean(confidence_list) if confidence_list else 0.0,
            "confidence_std": np.std(confidence_list) if confidence_list else 0.0,
            "latency_mean_ms": np.mean(latency_list) if latency_list else 0.0,
            "latency_std_ms": np.std(latency_list) if latency_list else 0.0,
            "window_size": self._config.window_size,
            "samples_collected": len(self._decision_history),
        }

    def reset(self) -> None:
        """Reset detector state."""
        self._confidence_history.clear()
        self._latency_history.clear()
        self._decision_history.clear()
        self._last_alert_times.clear()
        self._total_decisions = 0
        self._total_errors = 0
        logger.info("[Meta][Anomaly] Detector reset")
