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
Hemisphere Disagreement Metric for AGI-HPC Phase 7.1.

Measures semantic distance between Superego and Id responses
after each debate round, producing a calibrated confidence score.

Architecture:
    - Embed both hemisphere responses with BGE-M3
    - Compute cosine similarity
    - Map to calibrated confidence via rolling calibration curve
    - Publish to agi.meta.monitor.confidence via NATS
    - Store history in PostgreSQL confidence_log table
    - Track Expected Calibration Error (ECE) over last 100 interactions
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    import nats as nats_mod
except ImportError:
    nats_mod = None  # type: ignore


# -----------------------------------------------------------------
# Data types
# -----------------------------------------------------------------


@dataclass
class DisagreementResult:
    """Result of a hemisphere disagreement measurement.

    Attributes:
        spock_text: LH response text.
        kirk_text: RH response text.
        similarity: Cosine similarity between embeddings (0-1).
        confidence: Mapped calibrated confidence (0-1).
        hemisphere_that_led: Which hemisphere was more relevant ("lh" or "rh").
        compute_time_ms: Time to compute the metric in milliseconds.
    """

    spock_text: str
    kirk_text: str
    similarity: float
    confidence: float
    hemisphere_that_led: str
    compute_time_ms: float = 0.0


@dataclass
class CalibrationPair:
    """A single calibration observation.

    Attributes:
        predicted_confidence: The confidence score we predicted.
        user_accepted: Whether the user accepted the response.
    """

    predicted_confidence: float
    user_accepted: bool


# -----------------------------------------------------------------
# Disagreement Metric
# -----------------------------------------------------------------


class DisagreementMetric:
    """Computes hemisphere disagreement and calibrated confidence.

    Uses BGE-M3 embeddings to measure semantic distance between LH and
    RH responses. Maintains a rolling calibration curve for ECE tracking.

    Args:
        embed_model: Pre-loaded SentenceTransformer model (BGE-M3).
        db_dsn: PostgreSQL connection string.
        calibration_window: Number of recent interactions to track for ECE.
        nats_url: NATS server URL for publishing confidence events.
    """

    def __init__(
        self,
        embed_model: Optional[object] = None,
        db_dsn: str = "dbname=atlas user=claude",
        calibration_window: int = 100,
        nats_url: str = "nats://localhost:4222",
    ) -> None:
        self._embed_model = embed_model
        self._db_dsn = db_dsn
        self._calibration_window = calibration_window
        self._nats_url = nats_url
        self._calibration_history: Deque[CalibrationPair] = deque(
            maxlen=calibration_window,
        )
        self._nats_client = None
        logger.info(
            "[disagreement] initialised (calibration_window=%d)",
            calibration_window,
        )

    # ------ Embedding & similarity ------

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using BGE-M3 with L2 normalisation."""
        if self._embed_model is None:
            raise RuntimeError("No embedding model provided")
        return self._embed_model.encode(texts, normalize_embeddings=True)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------ Confidence mapping ------

    @staticmethod
    def map_confidence(similarity: float) -> float:
        """Map cosine similarity to calibrated confidence.

        Thresholds:
            > 0.85 -> high confidence (0.9)
            0.5-0.85 -> medium confidence (linear 0.4-0.85)
            < 0.5 -> low confidence (0.3)
        """
        if similarity > 0.85:
            return 0.9
        elif similarity >= 0.5:
            # Linear interpolation: 0.5 -> 0.4, 0.85 -> 0.85
            t = (similarity - 0.5) / 0.35
            return 0.4 + t * 0.45
        else:
            return 0.3

    # ------ Determine which hemisphere led ------

    def _determine_leader(
        self,
        spock_emb: np.ndarray,
        kirk_emb: np.ndarray,
        query_emb: np.ndarray,
    ) -> str:
        """Determine which hemisphere was more relevant to the query."""
        spock_sim = self._cosine_similarity(spock_emb, query_emb)
        kirk_sim = self._cosine_similarity(kirk_emb, query_emb)
        return "lh" if spock_sim >= kirk_sim else "rh"

    # ------ Main compute method ------

    def compute(
        self,
        spock_text: str,
        kirk_text: str,
        query: str = "",
        topic: str = "",
    ) -> DisagreementResult:
        """Compute disagreement between Spock and Kirk responses.

        Embeds both responses, computes cosine similarity, maps to
        confidence, stores in DB, and returns the result.

        Args:
            spock_text: LH (Spock) response text.
            kirk_text: RH (Kirk) response text.
            query: Original user query (for relevance comparison).
            topic: Topic classification for per-topic tracking.

        Returns:
            DisagreementResult with similarity, confidence, and leader.
        """
        t0 = time.monotonic()

        # Embed all texts in one batch for efficiency
        texts = [spock_text, kirk_text]
        if query:
            texts.append(query)

        embeddings = self._embed(texts)
        spock_emb = embeddings[0]
        kirk_emb = embeddings[1]
        query_emb = embeddings[2] if query else None

        # Cosine similarity
        similarity = self._cosine_similarity(spock_emb, kirk_emb)

        # Map to confidence
        confidence = self.map_confidence(similarity)

        # Determine leader
        if query_emb is not None:
            leader = self._determine_leader(spock_emb, kirk_emb, query_emb)
        else:
            leader = "lh"

        elapsed_ms = (time.monotonic() - t0) * 1000

        result = DisagreementResult(
            spock_text=spock_text,
            kirk_text=kirk_text,
            similarity=similarity,
            confidence=confidence,
            hemisphere_that_led=leader,
            compute_time_ms=elapsed_ms,
        )

        logger.info(
            "[disagreement] sim=%.3f conf=%.2f leader=%s (%.0fms)",
            similarity,
            confidence,
            leader,
            elapsed_ms,
        )

        # Store in DB (best-effort)
        self._store_result(result, query, topic)

        return result

    # ------ PostgreSQL storage ------

    def _store_result(
        self,
        result: DisagreementResult,
        query: str,
        topic: str,
    ) -> None:
        """Store disagreement result in confidence_log table."""
        if psycopg2 is None:
            return
        try:
            conn = psycopg2.connect(self._db_dsn)
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO confidence_log
                       (query, spock_response, kirk_response,
                        similarity, confidence, topic)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (
                        query,
                        result.spock_text[:10000],
                        result.kirk_text[:10000],
                        result.similarity,
                        result.confidence,
                        topic or None,
                    ),
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("[disagreement] DB store failed: %s", e)

    # ------ Calibration & ECE ------

    def record_feedback(self, confidence: float, user_accepted: bool) -> None:
        """Record user feedback for calibration tracking.

        Args:
            confidence: The confidence score that was predicted.
            user_accepted: Whether the user accepted/liked the response.
        """
        self._calibration_history.append(
            CalibrationPair(
                predicted_confidence=confidence,
                user_accepted=user_accepted,
            )
        )

    def compute_ece(self, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error over recent interactions.

        ECE measures how well-calibrated the confidence scores are.
        A perfectly calibrated system has ECE = 0.

        Args:
            n_bins: Number of bins for the calibration histogram.

        Returns:
            ECE value (0-1). Lower is better.
        """
        if len(self._calibration_history) < 5:
            return 0.0  # Not enough data

        pairs = list(self._calibration_history)
        confs = np.array([p.predicted_confidence for p in pairs])
        accs = np.array([1.0 if p.user_accepted else 0.0 for p in pairs])

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total = len(pairs)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (confs >= lo) & (confs <= hi)
            else:
                mask = (confs >= lo) & (confs < hi)
            count = int(mask.sum())
            if count == 0:
                continue
            avg_conf = float(confs[mask].mean())
            avg_acc = float(accs[mask].mean())
            ece += (count / total) * abs(avg_acc - avg_conf)

        return ece

    # ------ Aggregation queries ------

    def get_recent_stats(self, n: int = 20) -> dict:
        """Get aggregate stats from recent confidence_log entries.

        Args:
            n: Number of recent entries to aggregate.

        Returns:
            Dict with avg_confidence, agreement_rate, ece, and per-topic breakdown.
        """
        stats: dict = {
            "avg_confidence": 0.0,
            "agreement_rate": 0.0,
            "ece": self.compute_ece(),
            "total_logged": 0,
            "topics": {},
        }
        if psycopg2 is None:
            return stats

        try:
            conn = psycopg2.connect(self._db_dsn)
            with conn.cursor() as cur:
                # Overall averages from last N
                cur.execute(
                    """SELECT AVG(confidence), AVG(similarity),
                              COUNT(*)
                       FROM (
                           SELECT confidence, similarity
                           FROM confidence_log
                           ORDER BY timestamp DESC
                           LIMIT %s
                       ) sub""",
                    (n,),
                )
                row = cur.fetchone()
                if row and row[2] > 0:
                    stats["avg_confidence"] = float(row[0] or 0)
                    stats["agreement_rate"] = float(row[1] or 0)
                    stats["total_logged"] = int(row[2])

                # Per-topic breakdown
                cur.execute(
                    """SELECT topic, AVG(confidence), COUNT(*)
                       FROM confidence_log
                       WHERE topic IS NOT NULL
                       GROUP BY topic
                       ORDER BY COUNT(*) DESC
                       LIMIT 20"""
                )
                for row in cur.fetchall():
                    stats["topics"][row[0]] = {
                        "avg_confidence": float(row[1]),
                        "count": int(row[2]),
                    }
            conn.close()
        except Exception as e:
            logger.warning("[disagreement] stats query failed: %s", e)

        return stats

    # ------ NATS publishing (async, best-effort) ------

    async def publish_confidence(self, result: DisagreementResult) -> None:
        """Publish confidence event to NATS.

        Publishes to agi.meta.monitor.confidence with the disagreement
        result serialised as JSON.
        """
        if nats_mod is None:
            return

        try:
            import json

            if self._nats_client is None:
                self._nats_client = await nats_mod.connect(self._nats_url)

            payload = json.dumps({
                "similarity": result.similarity,
                "confidence": result.confidence,
                "hemisphere_that_led": result.hemisphere_that_led,
                "compute_time_ms": result.compute_time_ms,
            }).encode()

            await self._nats_client.publish(
                "agi.meta.monitor.confidence",
                payload,
            )
            logger.debug("[disagreement] published confidence to NATS")
        except Exception as e:
            logger.warning("[disagreement] NATS publish failed: %s", e)
