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
Safety Gateway: Three-layer DEME-integrated gate for LLM input/output.

Wraps the ErisML DEME pipeline (when available) and provides a
standalone reflex layer that works independently of ErisML.

Layers:
    1. **Reflex**: Fast keyword/pattern checks (<1ms) -- always runs.
    2. **Tactical**: Full MoralVector assessment via ErisML DEME (if available).
    3. **Strategic**: Decision-proof logging for audit trail.

Phase 3 (Safety Gateway) -- Atlas integration.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from agi.safety.adapter import SafetyAdapter

# ---------------------------------------------------------------------------
# Try importing ErisML DEME pipeline
# ---------------------------------------------------------------------------

try:
    from erisml.ethics.layers.pipeline import DEMEPipeline, PipelineConfig
    from erisml.ethics.moral_vector import MoralVector

    ERISML_AVAILABLE = True
    logger.info("[safety-gateway] ErisML DEME pipeline loaded")
except ImportError:
    ERISML_AVAILABLE = False
    DEMEPipeline = None  # type: ignore[assignment,misc]
    PipelineConfig = None  # type: ignore[assignment,misc]
    MoralVector = None  # type: ignore[assignment,misc]
    logger.info("[safety-gateway] ErisML not available; reflex-only mode")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SafetyResult:
    """Result of a safety check.

    Attributes:
        passed: Whether the content passed the safety check.
        score: Overall safety score [0.0 = unsafe, 1.0 = safe].
        flags: List of flag strings identifying specific concerns.
        decision_proof: Audit-trail dictionary with layer outputs.
        gate: Which gate produced this result (input/output/action).
        latency_ms: Total check latency in milliseconds.
    """

    passed: bool
    score: float
    flags: List[str] = field(default_factory=list)
    decision_proof: Dict[str, Any] = field(default_factory=dict)
    gate: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return asdict(self)


@dataclass
class ReflexPattern:
    """A single reflex-layer pattern check.

    Attributes:
        name: Human-readable name for the pattern.
        pattern: Compiled regex pattern.
        category: Category of concern (pii, injection, dangerous, blocked).
        severity: Severity weight [0.0, 1.0].  >= veto_threshold triggers veto.
        veto: Whether a match should cause an immediate veto.
    """

    name: str
    pattern: re.Pattern[str]
    category: str
    severity: float = 0.5
    veto: bool = False


@dataclass
class GatewayConfig:
    """Configuration for the Safety Gateway.

    Attributes:
        veto_threshold: Score below which content is vetoed [0.0, 1.0].
        reflex_patterns: List of reflex-layer pattern checks.
        enable_tactical: Whether to run the tactical (DEME) layer.
        enable_strategic: Whether to log strategic decision proofs.
        deme_profile: Name of the DEME governance profile.
        blocked_terms: List of terms that trigger immediate veto.
        pii_patterns: Named PII regex patterns.
        injection_patterns: Prompt injection regex patterns.
    """

    veto_threshold: float = 0.3
    reflex_patterns: List[ReflexPattern] = field(default_factory=list)
    enable_tactical: bool = True
    enable_strategic: bool = True
    deme_profile: str = "default"
    blocked_terms: List[str] = field(default_factory=list)
    pii_patterns: Dict[str, str] = field(default_factory=dict)
    injection_patterns: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> GatewayConfig:
        """Load gateway configuration from a YAML file."""
        if yaml is None:
            raise RuntimeError("pyyaml is required: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        safety = data.get("safety", data)
        reflex_cfg = safety.get("reflex", {})
        deme_cfg = safety.get("deme", {})

        # Build reflex patterns from config
        patterns: List[ReflexPattern] = []

        # PII patterns
        for name, pat_str in reflex_cfg.get("pii_patterns", {}).items():
            try:
                patterns.append(
                    ReflexPattern(
                        name=f"pii_{name}",
                        pattern=re.compile(pat_str),
                        category="pii",
                        severity=0.4,
                        veto=False,
                    )
                )
            except re.error:
                logger.warning("[gateway-config] invalid PII regex: %s", pat_str)

        # Injection patterns
        for i, pat_str in enumerate(reflex_cfg.get("injection_patterns", [])):
            try:
                patterns.append(
                    ReflexPattern(
                        name=f"injection_{i}",
                        pattern=re.compile(pat_str, re.IGNORECASE),
                        category="injection",
                        severity=0.8,
                        veto=True,
                    )
                )
            except re.error:
                logger.warning("[gateway-config] invalid injection regex: %s", pat_str)

        # Blocked terms -> simple word-boundary patterns
        blocked = reflex_cfg.get("blocked_terms", [])
        for term in blocked:
            patterns.append(
                ReflexPattern(
                    name=f"blocked_{term}",
                    pattern=re.compile(re.escape(term), re.IGNORECASE),
                    category="blocked",
                    severity=0.9,
                    veto=True,
                )
            )

        return cls(
            veto_threshold=safety.get("veto_threshold", 0.3),
            reflex_patterns=patterns,
            enable_tactical=deme_cfg.get("enable_tactical", True),
            enable_strategic=deme_cfg.get("enable_strategic", True),
            deme_profile=deme_cfg.get("profile", "default"),
            blocked_terms=blocked,
            pii_patterns=reflex_cfg.get("pii_patterns", {}),
            injection_patterns=reflex_cfg.get("injection_patterns", []),
        )

    @classmethod
    def default(cls) -> GatewayConfig:
        """Create a gateway config with built-in default patterns."""
        patterns: List[ReflexPattern] = []

        # Default PII patterns
        default_pii = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
            "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone_us": (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        }
        for name, pat_str in default_pii.items():
            patterns.append(
                ReflexPattern(
                    name=f"pii_{name}",
                    pattern=re.compile(pat_str),
                    category="pii",
                    severity=0.4,
                    veto=False,
                )
            )

        # Default injection patterns
        default_injection = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"disregard\s+(all\s+)?prior\s+(instructions|rules)",
            r"you\s+are\s+now\s+(a\s+)?DAN",
            r"jailbreak",
            r"pretend\s+you\s+(have\s+)?no\s+(restrictions|rules)",
            r"bypass\s+(your\s+)?(safety|content)\s+(filter|policy)",
            r"act\s+as\s+if\s+you\s+have\s+no\s+guardrails",
            r"do\s+anything\s+now|developer\s+mode|sudo\s+mode",
        ]
        for i, pat_str in enumerate(default_injection):
            patterns.append(
                ReflexPattern(
                    name=f"injection_{i}",
                    pattern=re.compile(pat_str, re.IGNORECASE),
                    category="injection",
                    severity=0.8,
                    veto=True,
                )
            )

        # Default dangerous content patterns
        default_dangerous = [
            (r"how\s+to\s+make\s+a\s+bomb", "dangerous_weapons"),
            (r"how\s+to\s+synthesize\s+(drugs|meth|fentanyl)", "dangerous_drugs"),
            (r"how\s+to\s+create\s+malware", "dangerous_malware"),
        ]
        for pat_str, name in default_dangerous:
            patterns.append(
                ReflexPattern(
                    name=name,
                    pattern=re.compile(pat_str, re.IGNORECASE),
                    category="dangerous",
                    severity=0.9,
                    veto=True,
                )
            )

        return cls(
            veto_threshold=0.3,
            reflex_patterns=patterns,
            enable_tactical=True,
            enable_strategic=True,
        )


# ---------------------------------------------------------------------------
# Safety Gateway
# ---------------------------------------------------------------------------


class SafetyGateway:
    """Three-layer safety gateway wrapping the DEME pipeline.

    Provides three check methods:

    * :meth:`check_input` -- pre-LLM input gate.
    * :meth:`check_output` -- post-LLM output gate.
    * :meth:`check_action` -- pre-actuator gate for physical/system actions.

    The reflex layer always runs (pure Python regex, <1ms).
    The tactical layer runs the full ErisML DEME pipeline when available.
    The strategic layer logs decision proofs for audit compliance.
    """

    def __init__(
        self,
        config: Optional[GatewayConfig] = None,
        adapter: Optional[SafetyAdapter] = None,
    ) -> None:
        self._config = config or GatewayConfig.default()
        self._adapter = adapter or SafetyAdapter()
        self._deme: Optional[Any] = None
        self._audit_log: List[Dict[str, Any]] = []

        # Initialise DEME pipeline if available and enabled
        if (
            ERISML_AVAILABLE
            and self._config.enable_tactical
            and DEMEPipeline is not None
        ):
            try:
                # Disable ErisML's internal proof generation; we build
                # our own decision proofs in the strategic layer.
                deme_cfg = PipelineConfig(generate_proofs=False)
                self._deme = DEMEPipeline(config=deme_cfg)
                logger.info("[safety-gateway] DEME pipeline initialised")
            except Exception:
                logger.exception("[safety-gateway] failed to initialise DEME pipeline")
                self._deme = None

    @property
    def has_deme(self) -> bool:
        """Return True if the DEME tactical layer is available."""
        return self._deme is not None

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Return the accumulated audit log entries."""
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Public check methods
    # ------------------------------------------------------------------

    def check_input(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyResult:
        """Pre-LLM input gate.

        Runs reflex + tactical checks on user input before it reaches
        the LLM.

        Args:
            user_message: Raw user input text.
            context: Optional context (session_id, hemisphere, etc.).

        Returns:
            SafetyResult indicating whether the input is safe.
        """
        t0 = time.perf_counter()
        ctx = context or {}
        proof_layers: List[Dict[str, Any]] = []

        # Layer 1: Reflex (always runs)
        reflex_result = self._run_reflex(user_message, "input")
        proof_layers.append(reflex_result)

        if reflex_result["vetoed"]:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            result = SafetyResult(
                passed=False,
                score=0.0,
                flags=reflex_result["flags"],
                decision_proof=self._build_proof("input", proof_layers, user_message),
                gate="input",
                latency_ms=latency_ms,
            )
            self._log_decision(result, user_message)
            return result

        # Layer 2: Tactical (DEME pipeline, if available)
        tactical_score = 1.0
        tactical_flags: List[str] = []
        tactical_vetoed = False

        if self._deme is not None and self._config.enable_tactical:
            tactical_result = self._run_tactical(user_message, "", ctx)
            proof_layers.append(tactical_result)
            tactical_score = tactical_result.get("score", 1.0)
            tactical_flags = tactical_result.get("flags", [])
            tactical_vetoed = tactical_result.get("vetoed", False)

        # Combine scores: reflex score weighted 0.6, tactical 0.4
        reflex_score = reflex_result.get("score", 1.0)
        if self._deme is not None:
            combined_score = 0.6 * reflex_score + 0.4 * tactical_score
        else:
            combined_score = reflex_score

        all_flags = reflex_result["flags"] + tactical_flags
        passed = combined_score >= self._config.veto_threshold and not tactical_vetoed

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = SafetyResult(
            passed=passed,
            score=combined_score,
            flags=all_flags,
            decision_proof=self._build_proof("input", proof_layers, user_message),
            gate="input",
            latency_ms=latency_ms,
        )

        # Layer 3: Strategic (audit logging)
        self._log_decision(result, user_message)

        return result

    def check_output(
        self,
        response: str,
        user_message: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyResult:
        """Post-LLM output gate.

        Runs full DEME pipeline on LLM output before it reaches the user.

        Args:
            response: LLM-generated response text.
            user_message: Original user input (for context).
            context: Optional context dict.

        Returns:
            SafetyResult indicating whether the output is safe.
        """
        t0 = time.perf_counter()
        ctx = context or {}
        proof_layers: List[Dict[str, Any]] = []

        # Layer 1: Reflex on response
        reflex_result = self._run_reflex(response, "output")
        proof_layers.append(reflex_result)

        if reflex_result["vetoed"]:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            result = SafetyResult(
                passed=False,
                score=0.0,
                flags=reflex_result["flags"],
                decision_proof=self._build_proof("output", proof_layers, response),
                gate="output",
                latency_ms=latency_ms,
            )
            self._log_decision(result, response)
            return result

        # Layer 2: Tactical (full DEME) on response
        tactical_score = 1.0
        tactical_flags: List[str] = []
        tactical_vetoed = False

        if self._deme is not None and self._config.enable_tactical:
            tactical_result = self._run_tactical(user_message, response, ctx)
            proof_layers.append(tactical_result)
            tactical_score = tactical_result.get("score", 1.0)
            tactical_flags = tactical_result.get("flags", [])
            tactical_vetoed = tactical_result.get("vetoed", False)

        # Combine scores
        reflex_score = reflex_result.get("score", 1.0)
        if self._deme is not None:
            combined_score = 0.5 * reflex_score + 0.5 * tactical_score
        else:
            combined_score = reflex_score

        all_flags = reflex_result["flags"] + tactical_flags
        passed = combined_score >= self._config.veto_threshold and not tactical_vetoed

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = SafetyResult(
            passed=passed,
            score=combined_score,
            flags=all_flags,
            decision_proof=self._build_proof("output", proof_layers, response),
            gate="output",
            latency_ms=latency_ms,
        )

        self._log_decision(result, response)
        return result

    def check_action(
        self,
        action_desc: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SafetyResult:
        """Pre-actuator gate for system/physical actions.

        Args:
            action_desc: Description of the action to be taken.
            context: Optional context dict.

        Returns:
            SafetyResult indicating whether the action is safe.
        """
        t0 = time.perf_counter()
        ctx = context or {}
        proof_layers: List[Dict[str, Any]] = []

        # Layer 1: Reflex
        reflex_result = self._run_reflex(action_desc, "action")
        proof_layers.append(reflex_result)

        if reflex_result["vetoed"]:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            result = SafetyResult(
                passed=False,
                score=0.0,
                flags=reflex_result["flags"],
                decision_proof=self._build_proof("action", proof_layers, action_desc),
                gate="action",
                latency_ms=latency_ms,
            )
            self._log_decision(result, action_desc)
            return result

        # Layer 2: Tactical
        tactical_score = 1.0
        tactical_flags: List[str] = []
        tactical_vetoed = False

        if self._deme is not None and self._config.enable_tactical:
            tactical_result = self._run_tactical(action_desc, "", ctx)
            proof_layers.append(tactical_result)
            tactical_score = tactical_result.get("score", 1.0)
            tactical_flags = tactical_result.get("flags", [])
            tactical_vetoed = tactical_result.get("vetoed", False)

        reflex_score = reflex_result.get("score", 1.0)
        if self._deme is not None:
            combined_score = 0.5 * reflex_score + 0.5 * tactical_score
        else:
            combined_score = reflex_score

        all_flags = reflex_result["flags"] + tactical_flags
        passed = combined_score >= self._config.veto_threshold and not tactical_vetoed

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = SafetyResult(
            passed=passed,
            score=combined_score,
            flags=all_flags,
            decision_proof=self._build_proof("action", proof_layers, action_desc),
            gate="action",
            latency_ms=latency_ms,
        )

        self._log_decision(result, action_desc)
        return result

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _run_reflex(self, text: str, gate: str) -> Dict[str, Any]:
        """Run reflex-layer pattern matching.

        This layer is pure Python regex and completes in <1ms.
        It does NOT depend on ErisML.

        Returns a dict with keys: layer, vetoed, flags, score, latency_us.
        """
        t0 = time.perf_counter()
        flags: List[str] = []
        max_severity = 0.0
        vetoed = False

        for rp in self._config.reflex_patterns:
            if rp.pattern.search(text):
                flags.append(f"{rp.category}:{rp.name}")
                max_severity = max(max_severity, rp.severity)
                if rp.veto:
                    vetoed = True

        # Score is inverse of max severity
        score = max(0.0, 1.0 - max_severity) if flags else 1.0

        latency_us = int((time.perf_counter() - t0) * 1_000_000)
        return {
            "layer": "reflex",
            "gate": gate,
            "vetoed": vetoed,
            "flags": flags,
            "score": score,
            "latency_us": latency_us,
            "pattern_count": len(self._config.reflex_patterns),
            "matches": len(flags),
        }

    def _run_tactical(
        self,
        user_msg: str,
        response: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run tactical layer using the ErisML DEME pipeline.

        Returns a dict with keys: layer, vetoed, flags, score, latency_ms.
        """
        t0 = time.perf_counter()

        try:
            # Build EthicalFacts via the adapter
            facts = self._adapter.build_facts(user_msg, response, context)

            # Run the DEME pipeline
            decision = self._deme.decide([facts])

            # Extract results
            vetoed = facts.option_id in decision.forbidden_options
            moral_landscape = decision.moral_landscape

            # Compute aggregate score from moral landscape
            score = 1.0
            flags: List[str] = []

            if hasattr(moral_landscape, "vectors") and moral_landscape.vectors:
                vec = list(moral_landscape.vectors.values())[0]
                # Average of the 8+1 moral vector dimensions
                dims = [
                    vec.physical_harm,
                    vec.rights_respect,
                    vec.fairness_equity,
                    vec.autonomy_respect,
                    vec.privacy_protection,
                    vec.societal_environmental,
                    vec.virtue_care,
                    vec.legitimacy_trust,
                    vec.epistemic_quality,
                ]
                # Invert physical_harm (lower harm = higher score)
                dims[0] = 1.0 - dims[0]
                score = sum(dims) / len(dims)

                if hasattr(vec, "veto_flags") and vec.veto_flags:
                    flags.extend(f"deme:{f}" for f in vec.veto_flags)
                    vetoed = True

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "layer": "tactical",
                "vetoed": vetoed,
                "flags": flags,
                "score": score,
                "latency_ms": round(latency_ms, 2),
                "rationale": getattr(decision, "rationale", ""),
            }

        except Exception:
            logger.exception("[safety-gateway] tactical layer error")
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "layer": "tactical",
                "vetoed": False,
                "flags": ["tactical_error"],
                "score": 0.5,
                "latency_ms": round(latency_ms, 2),
                "error": True,
            }

    # ------------------------------------------------------------------
    # Strategic layer (audit)
    # ------------------------------------------------------------------

    def _build_proof(
        self,
        gate: str,
        layers: List[Dict[str, Any]],
        content_text: str,
    ) -> Dict[str, Any]:
        """Build a decision proof for the audit trail."""
        content_hash = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
        return {
            "proof_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gate": gate,
            "content_hash": content_hash,
            "layers": layers,
            "profile": self._config.deme_profile,
            "erisml_available": ERISML_AVAILABLE,
        }

    def _log_decision(self, result: SafetyResult, content: str) -> None:
        """Log decision to the strategic audit trail."""
        if not self._config.enable_strategic:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gate": result.gate,
            "passed": result.passed,
            "score": result.score,
            "flags": result.flags,
            "latency_ms": result.latency_ms,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "proof_id": result.decision_proof.get("proof_id", ""),
        }
        self._audit_log.append(entry)

        if not result.passed:
            logger.warning(
                "[safety-gateway] VETO gate=%s score=%.2f flags=%s",
                result.gate,
                result.score,
                result.flags,
            )
        else:
            logger.debug(
                "[safety-gateway] PASS gate=%s score=%.2f latency=%.1fms",
                result.gate,
                result.score,
                result.latency_ms,
            )
