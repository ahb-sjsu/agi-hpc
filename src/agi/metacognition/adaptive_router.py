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
Adaptive Temperature Router for AGI-HPC Phase 7.2a.

Tracks per-topic confidence averages and adjusts suggested temperature:
    - Topics with consistently low confidence -> higher temperature (exploration)
    - Topics with consistently high confidence -> lower temperature (precision)

Stores state in PostgreSQL topic_confidence table and publishes
temperature adjustments to NATS agi.meta.adjust.temperature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore

try:
    import nats as nats_mod
except ImportError:
    nats_mod = None  # type: ignore


# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------


@dataclass
class AdaptiveTemperatureConfig:
    """Configuration for adaptive temperature routing.

    Attributes:
        db_dsn: PostgreSQL connection string.
        nats_url: NATS server URL.
        base_temperature: Default temperature when no data.
        min_temperature: Floor for temperature adjustment.
        max_temperature: Ceiling for temperature adjustment.
        high_confidence_threshold: Above this, decrease temperature.
        low_confidence_threshold: Below this, increase temperature.
        ema_alpha: Exponential moving average decay for confidence updates.
    """

    db_dsn: str = "dbname=atlas user=claude"
    nats_url: str = "nats://localhost:4222"
    base_temperature: float = 0.7
    min_temperature: float = 0.3
    max_temperature: float = 1.2
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.5
    ema_alpha: float = 0.3


# -----------------------------------------------------------------
# Adaptive Temperature Router
# -----------------------------------------------------------------


class AdaptiveTemperatureRouter:
    """Tracks per-topic confidence and suggests temperature adjustments.

    Maintains an exponential moving average of confidence per topic.
    When confidence is consistently high, suggests lower temperature
    for precision. When low, suggests higher temperature for exploration.

    Args:
        config: AdaptiveTemperatureConfig instance.
    """

    def __init__(
        self,
        config: Optional[AdaptiveTemperatureConfig] = None,
    ) -> None:
        self._config = config or AdaptiveTemperatureConfig()
        self._nats_client = None
        # In-memory cache of topic -> (avg_confidence, sample_count, temperature)
        self._cache: Dict[str, dict] = {}
        self._load_from_db()
        logger.info(
            "[adaptive-temp] initialised (base_temp=%.2f)",
            self._config.base_temperature,
        )

    # ------ Load from DB ------

    def _load_from_db(self) -> None:
        """Load existing topic confidence data from PostgreSQL."""
        if psycopg2 is None:
            return
        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute("""SELECT topic, avg_confidence, sample_count,
                              suggested_temperature
                       FROM topic_confidence""")
                for row in cur.fetchall():
                    self._cache[row[0]] = {
                        "avg_confidence": float(row[1]),
                        "sample_count": int(row[2]),
                        "temperature": float(row[3]),
                    }
            conn.close()
            logger.info("[adaptive-temp] loaded %d topics from DB", len(self._cache))
        except Exception as e:
            logger.warning("[adaptive-temp] DB load failed: %s", e)

    # ------ Temperature computation ------

    def _compute_temperature(self, avg_confidence: float) -> float:
        """Compute suggested temperature from average confidence.

        High confidence -> lower temperature (more deterministic).
        Low confidence -> higher temperature (more exploratory).
        """
        cfg = self._config
        if avg_confidence >= cfg.high_confidence_threshold:
            # High confidence: scale down from base
            # At confidence=1.0, temp = min_temperature
            t = (avg_confidence - cfg.high_confidence_threshold) / (
                1.0 - cfg.high_confidence_threshold
            )
            temp = cfg.base_temperature - t * (
                cfg.base_temperature - cfg.min_temperature
            )
        elif avg_confidence <= cfg.low_confidence_threshold:
            # Low confidence: scale up from base
            # At confidence=0.0, temp = max_temperature
            t = (cfg.low_confidence_threshold - avg_confidence) / (
                cfg.low_confidence_threshold
            )
            temp = cfg.base_temperature + t * (
                cfg.max_temperature - cfg.base_temperature
            )
        else:
            # Middle range: use base temperature
            temp = cfg.base_temperature

        return max(cfg.min_temperature, min(cfg.max_temperature, temp))

    # ------ Update confidence for a topic ------

    def update(self, topic: str, confidence: float) -> dict:
        """Update confidence tracking for a topic.

        Uses exponential moving average to smooth confidence updates.

        Args:
            topic: Topic string.
            confidence: New confidence observation.

        Returns:
            Dict with updated avg_confidence, sample_count, and temperature.
        """
        alpha = self._config.ema_alpha

        if topic in self._cache:
            entry = self._cache[topic]
            old_avg = entry["avg_confidence"]
            new_avg = alpha * confidence + (1 - alpha) * old_avg
            entry["avg_confidence"] = new_avg
            entry["sample_count"] += 1
            entry["temperature"] = self._compute_temperature(new_avg)
        else:
            new_avg = confidence
            self._cache[topic] = {
                "avg_confidence": new_avg,
                "sample_count": 1,
                "temperature": self._compute_temperature(new_avg),
            }

        entry = self._cache[topic]
        logger.info(
            "[adaptive-temp] topic=%s conf=%.2f->%.2f temp=%.2f (n=%d)",
            topic,
            confidence,
            entry["avg_confidence"],
            entry["temperature"],
            entry["sample_count"],
        )

        # Persist to DB (best-effort)
        self._persist_topic(topic, entry)

        return dict(entry)

    # ------ Persist to DB ------

    def _persist_topic(self, topic: str, entry: dict) -> None:
        """Upsert topic confidence data to PostgreSQL."""
        if psycopg2 is None:
            return
        try:
            conn = psycopg2.connect(self._config.db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO topic_confidence
                           (topic, avg_confidence, sample_count,
                            suggested_temperature, last_updated)
                       VALUES (%s, %s, %s, %s, NOW())
                       ON CONFLICT (topic) DO UPDATE SET
                           avg_confidence = EXCLUDED.avg_confidence,
                           sample_count = EXCLUDED.sample_count,
                           suggested_temperature = EXCLUDED.suggested_temperature,
                           last_updated = NOW()""",
                    (
                        topic,
                        entry["avg_confidence"],
                        entry["sample_count"],
                        entry["temperature"],
                    ),
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("[adaptive-temp] DB persist failed: %s", e)

    # ------ Query methods ------

    def get_temperature(self, topic: str) -> float:
        """Get the current suggested temperature for a topic.

        Args:
            topic: Topic string.

        Returns:
            Suggested temperature. Falls back to base_temperature if unknown.
        """
        if topic in self._cache:
            return self._cache[topic]["temperature"]
        return self._config.base_temperature

    def get_all_topics(self) -> Dict[str, dict]:
        """Return all tracked topics with their confidence and temperature."""
        return dict(self._cache)

    def get_summary(self) -> dict:
        """Return a summary of adaptive temperature state.

        Returns:
            Dict with topic count, average temperature, and per-topic details.
        """
        if not self._cache:
            return {
                "topic_count": 0,
                "avg_temperature": self._config.base_temperature,
                "topics": {},
            }

        temps = [e["temperature"] for e in self._cache.values()]
        return {
            "topic_count": len(self._cache),
            "avg_temperature": sum(temps) / len(temps),
            "base_temperature": self._config.base_temperature,
            "topics": {
                topic: {
                    "avg_confidence": round(e["avg_confidence"], 3),
                    "sample_count": e["sample_count"],
                    "temperature": round(e["temperature"], 3),
                }
                for topic, e in sorted(
                    self._cache.items(),
                    key=lambda x: x[1]["sample_count"],
                    reverse=True,
                )
            },
        }

    # ------ NATS publishing ------

    async def publish_temperature(self, topic: str) -> None:
        """Publish temperature adjustment event to NATS.

        Publishes to agi.meta.adjust.temperature with the current
        topic temperature data.
        """
        if nats_mod is None:
            return

        try:
            import json

            if self._nats_client is None:
                self._nats_client = await nats_mod.connect(
                    self._config.nats_url,
                )

            entry = self._cache.get(topic, {})
            payload = json.dumps(
                {
                    "topic": topic,
                    "temperature": entry.get(
                        "temperature", self._config.base_temperature
                    ),
                    "avg_confidence": entry.get("avg_confidence", 0.5),
                    "sample_count": entry.get("sample_count", 0),
                }
            ).encode()

            await self._nats_client.publish(
                "agi.meta.adjust.temperature",
                payload,
            )
            logger.debug("[adaptive-temp] published temperature for topic=%s", topic)
        except Exception as e:
            logger.warning("[adaptive-temp] NATS publish failed: %s", e)
