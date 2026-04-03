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
Safety Adapter: Converts chat interactions into EthicalFacts for the DEME pipeline.

Maps user messages, LLM responses, and action descriptions to the
structured EthicalFacts schema used by the ErisML DEME pipeline.
When ErisML is not available, provides a lightweight standalone
facts dataclass that mirrors the essential fields.

Phase 3 (Safety Gateway) -- Atlas integration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing ErisML's EthicalFacts; fall back to a standalone stub.
# ---------------------------------------------------------------------------

try:
    from erisml.ethics.facts import (
        AutonomyAndAgency,
        Consequences,
        Context,
        EthicalFacts,
        JusticeAndFairness,
        PrivacyAndDataGovernance,
        RightsAndDuties,
        VirtueAndCare,
    )

    ERISML_AVAILABLE = True
    logger.info("[safety-adapter] ErisML EthicalFacts loaded")
except ImportError:
    ERISML_AVAILABLE = False
    logger.info("[safety-adapter] ErisML not available; using standalone facts stub")

    # ------------------------------------------------------------------
    # Standalone stubs mirroring the ErisML dataclasses we need.
    # These are intentionally minimal: just enough for the adapter and
    # gateway to function without the full ErisML library.
    # ------------------------------------------------------------------

    @dataclass
    class Consequences:  # type: ignore[no-redef]
        expected_benefit: float = 0.0
        expected_harm: float = 0.0
        urgency: float = 0.0
        affected_count: int = 0
        short_term: Dict[str, Any] = field(default_factory=dict)
        long_term: Dict[str, Any] = field(default_factory=dict)
        probabilities: Dict[str, float] = field(default_factory=dict)

    @dataclass
    class JusticeAndFairness:  # type: ignore[no-redef]
        discriminates_on_protected_attr: bool = False
        prioritizes_most_disadvantaged: bool = False
        distributive_pattern: Optional[str] = None
        exploits_vulnerable_population: bool = False
        exacerbates_power_imbalance: bool = False
        affected_groups: Dict[str, Any] = field(default_factory=dict)
        equity_metrics: Dict[str, float] = field(default_factory=dict)

    @dataclass
    class RightsAndDuties:  # type: ignore[no-redef]
        violates_rights: bool = False
        has_valid_consent: bool = True
        violates_explicit_rule: bool = False
        role_duty_conflict: bool = False
        rights_infringed: Dict[str, Any] = field(default_factory=dict)
        duties_upheld: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class VirtueAndCare:  # type: ignore[no-redef]
        expresses_compassion: bool = True
        betrays_trust: bool = False
        respects_person_as_end: bool = True
        virtues_promoted: List[str] = field(default_factory=list)
        care_considerations: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class AutonomyAndAgency:  # type: ignore[no-redef]
        has_meaningful_choice: bool = True
        supports_self_determination: bool = True
        manipulative_intent_detected: bool = False
        manipulative_design_present: bool = False
        coercion_or_undue_influence: bool = False
        can_withdraw_without_penalty: bool = True
        freedom_metrics: Dict[str, float] = field(default_factory=dict)
        informed_consent: bool = False

    @dataclass
    class PrivacyAndDataGovernance:  # type: ignore[no-redef]
        privacy_invasion_level: float = 0.0
        data_minimization_respected: bool = True
        secondary_use_without_consent: bool = False
        data_retention_excessive: bool = False
        reidentification_risk: float = 0.0
        collection_is_minimal: bool = True
        data_usage: str = "consensual"
        retention_policy: str = "standard"

    @dataclass
    class Context:  # type: ignore[no-redef]
        domain: str = "general"
        constraints: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class EthicalFacts:  # type: ignore[no-redef]
        option_id: str
        scenario_id: str = "default"
        metadata: Dict[str, Any] = field(default_factory=dict)
        consequences: Optional[Consequences] = None
        justice_and_fairness: Optional[JusticeAndFairness] = None
        rights_and_duties: Optional[RightsAndDuties] = None
        virtue_and_care: Optional[VirtueAndCare] = None
        autonomy_and_agency: Optional[AutonomyAndAgency] = None
        privacy_and_data: Optional[PrivacyAndDataGovernance] = None
        context: Optional[Context] = None
        tags: List[str] = field(default_factory=list)
        extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PII / sensitive content detection patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: Dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}

# Patterns that suggest prompt injection / jailbreak attempts
_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+(instructions|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?DAN", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+(have\s+)?no\s+(restrictions|rules)", re.IGNORECASE),
    re.compile(r"bypass\s+(your\s+)?(safety|content)\s+(filter|policy)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+guardrails", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now|developer\s+mode|sudo\s+mode", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Safety Adapter
# ---------------------------------------------------------------------------


class SafetyAdapter:
    """Converts chat interactions into EthicalFacts for the DEME pipeline.

    The adapter performs lightweight NLP heuristics to populate the
    structured EthicalFacts dimensions:

    * **Consequences**: estimated harm/benefit from dangerous content signals.
    * **Rights & Duties**: PII detection, consent inference.
    * **Privacy**: PII counts, data-governance flags.
    * **Justice & Fairness**: discriminatory-language detection.
    * **Autonomy & Agency**: prompt-injection / manipulation detection.
    * **Virtue & Care**: deceptive or trust-violating content.

    These heuristics intentionally err on the side of caution; the
    full DEME pipeline (if available) provides the authoritative score.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}
        self._extra_blocked: List[re.Pattern[str]] = []

        # Allow additional blocked patterns from config
        for pat_str in self._config.get("extra_blocked_patterns", []):
            try:
                self._extra_blocked.append(re.compile(pat_str, re.IGNORECASE))
            except re.error:
                logger.warning("[safety-adapter] invalid regex in config: %s", pat_str)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_facts(
        self,
        user_msg: str,
        response: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> EthicalFacts:
        """Build an EthicalFacts instance from a chat interaction.

        Args:
            user_msg: The user's input message.
            response: The LLM's response (empty for input-gate checks).
            context: Optional additional context (session_id, hemisphere, etc.).

        Returns:
            An EthicalFacts instance populated from heuristic analysis.
        """
        ctx = context or {}
        combined_text = f"{user_msg} {response}".strip()

        # Detect PII
        pii_flags = self._detect_pii(combined_text)
        pii_count = sum(len(matches) for matches in pii_flags.values())

        # Detect injection attempts
        injection_hits = self._detect_injection(user_msg)

        # Estimate harm/benefit
        harm_score = self._estimate_harm(combined_text, pii_count, injection_hits)

        # Build sub-structures
        consequences = Consequences(
            expected_harm=harm_score,
            expected_benefit=max(0.0, 1.0 - harm_score),
            urgency=0.8 if harm_score > 0.5 else 0.2,
            affected_count=ctx.get("affected_count", 1),
        )

        rights = RightsAndDuties(
            violates_rights=pii_count > 0 and not ctx.get("has_consent", False),
            has_valid_consent=ctx.get("has_consent", True),
            violates_explicit_rule=bool(injection_hits),
        )

        privacy = PrivacyAndDataGovernance(
            privacy_invasion_level=min(1.0, pii_count * 0.25),
            data_minimization_respected=pii_count == 0,
            secondary_use_without_consent=pii_count > 0
            and not ctx.get("has_consent", False),
            reidentification_risk=min(1.0, pii_count * 0.2),
        )

        justice = JusticeAndFairness(
            discriminates_on_protected_attr=self._detect_discrimination(combined_text),
        )

        autonomy = AutonomyAndAgency(
            has_meaningful_choice=True,
            manipulative_intent_detected=bool(injection_hits),
            manipulative_design_present=bool(injection_hits),
            coercion_or_undue_influence=bool(injection_hits),
        )

        virtue = VirtueAndCare(
            expresses_compassion=harm_score < 0.3,
            betrays_trust=bool(injection_hits),
            respects_person_as_end=harm_score < 0.5,
        )

        context_obj = Context(
            domain=ctx.get("domain", "chat"),
            constraints={"hemisphere": ctx.get("hemisphere", "lh")},
        )

        # Build tags
        tags: List[str] = []
        if pii_flags:
            tags.extend(f"pii:{k}" for k in pii_flags if pii_flags[k])
        if injection_hits:
            tags.append("injection_attempt")
        if harm_score > 0.7:
            tags.append("high_harm")

        return EthicalFacts(
            option_id=ctx.get("option_id", "chat_interaction"),
            scenario_id=ctx.get("session_id", "default"),
            metadata={
                "user_msg_length": len(user_msg),
                "response_length": len(response),
                "pii_flags": {k: len(v) for k, v in pii_flags.items()},
                "injection_hits": len(injection_hits),
                "source": "safety_adapter",
            },
            consequences=consequences,
            rights_and_duties=rights,
            privacy_and_data=privacy,
            justice_and_fairness=justice,
            autonomy_and_agency=autonomy,
            virtue_and_care=virtue,
            context=context_obj,
            tags=tags,
        )

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Return dict of PII type -> list of matches found in *text*."""
        results: Dict[str, List[str]] = {}
        for name, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                results[name] = matches
        return results

    def _detect_injection(self, text: str) -> List[str]:
        """Return list of matched injection pattern descriptions."""
        hits: List[str] = []
        for i, pattern in enumerate(_INJECTION_PATTERNS):
            if pattern.search(text):
                hits.append(f"injection_pattern_{i}")
        for pattern in self._extra_blocked:
            if pattern.search(text):
                hits.append("config_blocked_pattern")
        return hits

    def _estimate_harm(
        self, text: str, pii_count: int, injection_hits: List[str]
    ) -> float:
        """Estimate a [0, 1] harm score from heuristic signals."""
        score = 0.0

        # PII leakage is harmful
        score += min(0.4, pii_count * 0.1)

        # Injection attempts are high-harm
        if injection_hits:
            score += 0.5

        # Dangerous content keywords (weapons, self-harm, etc.)
        dangerous_keywords = [
            "how to make a bomb",
            "how to hack",
            "how to kill",
            "synthesize drugs",
            "create malware",
            "exploit vulnerability",
        ]
        text_lower = text.lower()
        for kw in dangerous_keywords:
            if kw in text_lower:
                score += 0.3
                break

        return min(1.0, score)

    def _detect_discrimination(self, text: str) -> bool:
        """Detect obviously discriminatory language patterns."""
        # Intentionally conservative -- only the most blatant patterns.
        # The full DEME pipeline handles nuanced fairness assessment.
        discrimination_patterns = [
            re.compile(
                r"\b(all|every)\s+(men|women|blacks|whites|asians|jews|muslims"
                r"|christians|hindus|gays|lesbians)\s+(are|should)\b",
                re.IGNORECASE,
            ),
        ]
        for pat in discrimination_patterns:
            if pat.search(text):
                return True
        return False
