# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Divine Council — Multi-agent Ego deliberation.

Replaces the single Ego mediator with a council of specialized
sub-agents that deliberate in parallel and reach consensus. All
council members share a **single llama-server** process running
Gemma 4 26B-A4B MoE (26B total params, 4B active per token) with
``--parallel 8``. The model loads once (~15 GB RAM); each parallel
slot adds ~300 MB of KV cache — total footprint ~18 GB.

Council Members (7):
    Judge      — Impartial evaluator. Scores correctness, logic.
    Advocate   — Devil's advocate. Challenges consensus, finds flaws.
    Synthesizer— Integration expert. Merges perspectives, resolves tension.
    Ethicist   — Moral compass. Checks alignment, fairness, safety.
    Historian  — Precedent tracker. References prior decisions, patterns.
    Futurist   — Consequence mapper. Second-order effects, long-term impact.
    Pragmatist — Feasibility assessor. Resources, constraints, viability.

Cognitive science grounding:
    - Minsky (1986): Society of Mind — intelligence as many agents
    - Mercier & Sperber (2011): Argumentative Theory of Reasoning
    - Surowiecki (2004): Wisdom of Crowds
    - Schank (1982): Dynamic Memory (case-based reasoning → Historian)
    - Gilbert & Wilson (2007): Prospection (Futurist)
    - Simon (1955): Bounded Rationality (Pragmatist)

Reliability & correctness (see docs/architecture/COUNCIL_RELIABILITY_PLAN.md):
    - Transport layer is abstracted via :mod:`_council_backend`:
      health probes, retries with jitter, circuit breaker, optional
      fallback backend (Spock's Qwen 72B) when Gemma 4 is down.
    - Three-valued votes (approve / challenge / abstain) — backend
      errors are ``abstain``, not silent approvals.
    - Ethicist veto: moderate or serious flags block consensus
      regardless of approval count.
    - Per-member ``max_tokens`` tuned so Historian / Futurist /
      Synthesizer have room; display truncation lives in
      :meth:`CouncilVerdict.format_log`, not the data model.
    - Correlation IDs (``trace_id``) thread through all logs.
    - Optional Prometheus metrics via :mod:`_council_metrics`.

Usage:
    council = DivineCouncil.default()  # Gemma 4 primary + Spock fallback
    verdict = council.deliberate(query, superego_response, id_response)
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from agi.reasoning._council_backend import (
    BackendRequest,
    CouncilBackend,
    FallbackBackend,
    LlamaServerBackend,
    new_trace_id,
)
from agi.reasoning._council_metrics import CouncilMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Council member definitions
# ---------------------------------------------------------------------------

MODEL_NAME = "Gemma 4 26B-A4B"

# Per-member token budget overrides. Roles that need to cite precedent
# or trace consequences get more headroom than roles that evaluate briefly.
MEMBER_MAX_TOKENS: Dict[str, int] = {
    "judge": 768,
    "advocate": 768,
    "synthesizer": 2048,
    "ethicist": 768,
    "historian": 2048,
    "futurist": 2048,
    "pragmatist": 1024,
}

DEFAULT_MAX_TOKENS = 1024


COUNCIL_MEMBERS = {
    "judge": {
        "name": "Judge",
        "system_prompt": (
            "You are the Judge on the Divine Council.\n\n"
            "MISSION: Impartial evaluation of reasoning quality. "
            "You are the calibration standard for the council.\n\n"
            "RULES:\n"
            "1. Assess accuracy, logical coherence, and completeness.\n"
            "2. Score each response on a scale of 1-10.\n"
            "3. Provide a brief justification for each score.\n"
            "4. Note any factual errors or logical fallacies.\n"
            "5. Do not advocate — only evaluate.\n\n"
            "SUCCESS METRICS: Your scores correlate with ground "
            "truth. Higher-scored responses are objectively better."
        ),
        "color": "#4a9eff",
    },
    "advocate": {
        "name": "Advocate",
        "system_prompt": (
            "You are the Devil's Advocate on the Divine Council.\n\n"
            "MISSION: Prevent groupthink through rigorous challenge. "
            "You are the immune system of the council.\n\n"
            "RULES:\n"
            "1. Find flaws, unstated assumptions, and edge cases.\n"
            "2. Challenge even when consensus seems strong.\n"
            "3. Propose specific counterarguments, not vague doubt.\n"
            "4. If everyone agrees, you must disagree and say why.\n"
            "5. Be rigorous but constructive — strengthen, "
            "don't obstruct.\n\n"
            "SUCCESS METRICS: Your challenges expose real "
            "weaknesses. The final answer is stronger because "
            "of your objections."
        ),
        "color": "#f87171",
    },
    "synthesizer": {
        "name": "Synthesizer",
        "system_prompt": (
            "You are the Synthesizer on the Divine Council.\n\n"
            "MISSION: Integration and resolution. Produce an answer "
            "that is better than any individual perspective.\n\n"
            "RULES:\n"
            "1. Take the strongest elements from each perspective.\n"
            "2. Resolve tensions rather than picking sides.\n"
            "3. Address the Advocate's challenges directly.\n"
            "4. Your synthesis must answer the original question.\n"
            "5. Be concise and authoritative — no hedging.\n\n"
            "SUCCESS METRICS: Your synthesis is preferred over "
            "any individual response. Users find it clear, "
            "complete, and balanced."
        ),
        "color": "#4ade80",
    },
    "ethicist": {
        "name": "Ethicist",
        "system_prompt": (
            "You are the Ethicist on the Divine Council.\n\n"
            "MISSION: Moral evaluation grounded in specific "
            "frameworks. You are the conscience of the council.\n\n"
            "RULES:\n"
            "1. Check for fairness, potential harm, and bias.\n"
            "2. Reference specific moral frameworks when flagging.\n"
            "3. If ethically sound, confirm concisely.\n"
            "4. Flag concerns with severity: minor, moderate, "
            "serious.\n"
            "5. Consider impact on vulnerable populations.\n\n"
            "SUCCESS METRICS: No harmful content passes your "
            "review. Your flags cite specific principles, not "
            "vague unease."
        ),
        "color": "#f59e0b",
    },
    "historian": {
        "name": "Historian",
        "system_prompt": (
            "You are the Historian on the Divine Council.\n\n"
            "MISSION: Case-based reasoning from prior experience. "
            "You are the institutional memory of the council.\n\n"
            "RULES:\n"
            "1. Identify precedents — similar past decisions and "
            "their outcomes.\n"
            "2. Note patterns: what worked before and what failed.\n"
            "3. Flag if the current proposal repeats a known mistake.\n"
            "4. Cite the specific prior case when making a claim.\n"
            "5. If no precedent exists, say so — novelty is useful "
            "information.\n\n"
            "SUCCESS METRICS: Your precedent analysis prevents "
            "repeated mistakes and surfaces proven approaches."
        ),
        "color": "#a78bfa",
    },
    "futurist": {
        "name": "Futurist",
        "system_prompt": (
            "You are the Futurist on the Divine Council.\n\n"
            "MISSION: Prospective reasoning — second-order effects "
            "and long-term consequences. You are the early warning "
            "system of the council.\n\n"
            "RULES:\n"
            "1. Trace consequences forward: if we do X, then Y, "
            "then Z.\n"
            "2. Identify unintended side effects and cascading "
            "impacts.\n"
            "3. Consider how the decision ages — is it still good "
            "in a week? A month?\n"
            "4. Flag irreversible commitments that deserve extra "
            "scrutiny.\n"
            "5. Distinguish likely consequences from speculative "
            "ones.\n\n"
            "SUCCESS METRICS: Your foresight catches downstream "
            "problems before they materialize."
        ),
        "color": "#06b6d4",
    },
    "pragmatist": {
        "name": "Pragmatist",
        "system_prompt": (
            "You are the Pragmatist on the Divine Council.\n\n"
            "MISSION: Feasibility assessment under real-world "
            "constraints. You are the reality check of the council.\n\n"
            "RULES:\n"
            "1. Evaluate resource requirements: time, compute, "
            "data, human effort.\n"
            "2. Identify the simplest path that achieves the goal.\n"
            "3. Flag over-engineered proposals — good enough now "
            "beats perfect later.\n"
            "4. Consider operational burden: who maintains this?\n"
            "5. If infeasible, propose a viable alternative.\n\n"
            "SUCCESS METRICS: Your assessments are calibrated — "
            "feasible plans succeed, flagged plans would have failed."
        ),
        "color": "#ec4899",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class VoteOutcome(str, Enum):
    """Three-valued vote outcome.

    Historically a backend error silently scored 5.0 which then counted
    as approval (``score >= 4, no flags``). The fix: use an explicit
    ``ABSTAIN`` outcome for error cases so they cannot be tallied as
    approvals.
    """

    APPROVE = "approve"
    CHALLENGE = "challenge"
    ABSTAIN = "abstain"


# Ethical-flag severities understood by the consensus rule.
ETHICAL_SEVERITIES = ("minor", "moderate", "serious")
VETO_SEVERITIES = {"moderate", "serious"}


@dataclass
class CouncilVote:
    """One council member's deliberation output."""

    member: str
    response: str
    score: float = 5.0  # 1-10 for judge, 0-1 for others (free-form)
    flags: List[str] = field(default_factory=list)
    ethical_severity: str = ""  # "", "minor", "moderate", "serious"
    outcome: VoteOutcome = VoteOutcome.APPROVE
    latency_s: float = 0.0
    attempts: int = 0
    backend: str = ""
    degraded: bool = False
    error: str = ""

    @property
    def is_error(self) -> bool:
        return self.outcome == VoteOutcome.ABSTAIN and bool(self.error)


@dataclass
class CouncilVerdict:
    """The council's collective verdict."""

    consensus: bool
    synthesis: str
    trace_id: str = ""
    votes: Dict[str, CouncilVote] = field(default_factory=dict)
    approval_count: int = 0
    challenge_count: int = 0
    abstain_count: int = 0
    ethical_flags: List[str] = field(default_factory=list)
    ethical_veto: bool = False
    degraded: bool = False
    total_latency_s: float = 0.0
    method: str = "divine_council"

    def format_log(self, preview_chars: int = 400) -> str:
        """Format for UI display. Truncates per-member for the UI only;
        the underlying data model keeps full responses."""
        lines = [f"### Divine Council Deliberation (trace_id={self.trace_id})\n"]
        for member_id, vote in self.votes.items():
            info = COUNCIL_MEMBERS.get(member_id, {})
            name = info.get("name", member_id)
            tag = ""
            if vote.is_error:
                tag = f" [ABSTAINED: {vote.error}]"
            elif vote.degraded:
                tag = " [DEGRADED]"
            preview = vote.response[:preview_chars]
            if len(vote.response) > preview_chars:
                preview += "…"
            lines.append(f"**{name}** ({vote.latency_s:.1f}s){tag}:\n{preview}\n")
        lines.append(
            f"\n**Consensus:** "
            f"{'Yes' if self.consensus else 'No'} "
            f"({self.approval_count} approve, "
            f"{self.challenge_count} challenge, "
            f"{self.abstain_count} abstain)"
        )
        if self.ethical_veto:
            lines.append("**Ethical veto active** (moderate+ severity flag).")
        if self.ethical_flags:
            lines.append(f"**Ethical flags:** {', '.join(self.ethical_flags)}")
        if self.degraded:
            lines.append("*Verdict produced in degraded mode (fallback backend).*")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Divine Council
# ---------------------------------------------------------------------------


class DivineCouncil:
    """Multi-agent Ego deliberation engine.

    Runs all council members in parallel and aggregates their
    judgments into a consensus verdict with synthesized response.

    The transport layer is pluggable. In production, the ``backend``
    is typically a :class:`FallbackBackend` composing the Gemma 4
    primary and Spock (Qwen 72B) fallback. For tests, pass a stub.
    """

    DEFAULT_URL = "http://localhost:8084"
    FALLBACK_URL = "http://localhost:8080"  # Spock / Qwen 72B

    def __init__(
        self,
        backend: CouncilBackend,
        *,
        metrics: Optional[CouncilMetrics] = None,
    ) -> None:
        self._backend = backend
        self._metrics = metrics or CouncilMetrics.default()

    # ---- construction helpers ----------------------------------------

    @classmethod
    def default(
        cls,
        primary_url: str = DEFAULT_URL,
        fallback_url: Optional[str] = None,
    ) -> "DivineCouncil":
        """Build a council with Gemma 4 primary and Spock fallback.

        If ``fallback_url`` is None, uses :attr:`FALLBACK_URL`.
        If fallback_url is ``""``, disables the fallback (primary only).
        """
        primary = LlamaServerBackend(primary_url, name="gemma4")
        if fallback_url == "":
            return cls(primary)
        fb = LlamaServerBackend(
            fallback_url or cls.FALLBACK_URL,
            name="spock",
        )
        return cls(FallbackBackend(primary, fb, name="gemma4+spock"))

    @classmethod
    def from_backend(cls, backend: CouncilBackend) -> "DivineCouncil":
        """For tests / custom backends."""
        return cls(backend)

    # ---- internals ----------------------------------------------------

    def _call_member(
        self,
        member_id: str,
        brief: str,
        trace_id: str,
    ) -> CouncilVote:
        """Dispatch one member to the backend and convert the response."""
        info = COUNCIL_MEMBERS.get(member_id, {})
        system = info.get("system_prompt", "You are a council member.")
        max_tokens = MEMBER_MAX_TOKENS.get(member_id, DEFAULT_MAX_TOKENS)

        request = BackendRequest(
            system_prompt=system,
            user_prompt=brief,
            max_tokens=max_tokens,
            temperature=0.3,
            trace_id=trace_id,
            member_id=member_id,
        )

        with self._metrics.track_request(
            member=member_id, backend=self._backend_label()
        ):
            response = self._backend.chat(request)

        outcome_label = "success" if response.ok else (response.error or "error")
        self._metrics.record_request_outcome(
            member=member_id,
            backend=response.backend_name or self._backend_label(),
            outcome=outcome_label,
            latency_s=response.latency_s,
        )

        if not response.ok:
            logger.warning(
                "[council] member=%s trace_id=%s FAILED: %s "
                "(backend=%s, latency=%.2fs, attempts=%d)",
                member_id,
                trace_id,
                response.error,
                response.backend_name,
                response.latency_s,
                response.attempts,
            )
            return CouncilVote(
                member=member_id,
                response="",
                outcome=VoteOutcome.ABSTAIN,
                latency_s=response.latency_s,
                attempts=response.attempts,
                backend=response.backend_name,
                degraded=response.degraded,
                error=response.error,
            )

        content = response.content
        score = _extract_score(content)
        flags = _extract_flags(member_id, content)
        severity = _extract_ethical_severity(member_id, content)

        if member_id == "advocate":
            outcome = VoteOutcome.CHALLENGE
        elif flags or score < 4 or severity in VETO_SEVERITIES:
            outcome = VoteOutcome.CHALLENGE
        else:
            outcome = VoteOutcome.APPROVE

        logger.info(
            "[council] member=%s trace_id=%s outcome=%s "
            "(score=%.1f, flags=%s, backend=%s, latency=%.2fs, attempts=%d)",
            member_id,
            trace_id,
            outcome.value,
            score,
            flags,
            response.backend_name,
            response.latency_s,
            response.attempts,
        )

        return CouncilVote(
            member=member_id,
            response=content,
            score=score,
            flags=flags,
            ethical_severity=severity,
            outcome=outcome,
            latency_s=response.latency_s,
            attempts=response.attempts,
            backend=response.backend_name,
            degraded=response.degraded,
        )

    def _backend_label(self) -> str:
        return getattr(self._backend, "name", "unknown")

    # ---- public API ---------------------------------------------------

    def deliberate(
        self,
        query: str,
        superego_response: str = "",
        id_response: str = "",
        context: str = "",
        trace_id: Optional[str] = None,
    ) -> CouncilVerdict:
        """Run full council deliberation.

        All members receive the same brief and deliberate in parallel.
        The Synthesizer produces the final answer.

        Parameters
        ----------
        query: the user's original question
        superego_response: Superego's analytical response
        id_response: Id's creative response
        context: optional additional context (ignored today; reserved)
        trace_id: correlation ID; if None, a fresh one is generated

        Returns
        -------
        :class:`CouncilVerdict` with full member responses, per-vote
        outcomes, consensus flag, and degradation status.
        """
        tid = trace_id or new_trace_id()
        t0 = time.monotonic()

        logger.info(
            "[council] deliberate start trace_id=%s "
            "(query_len=%d, superego=%d, id=%d)",
            tid,
            len(query),
            len(superego_response),
            len(id_response),
        )

        brief = _build_brief(query, superego_response, id_response)

        # All members deliberate in parallel. The single llama-server
        # with --parallel 8 handles concurrency; ThreadPoolExecutor here
        # just fires the HTTP requests concurrently.
        votes: Dict[str, CouncilVote] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(COUNCIL_MEMBERS),
            thread_name_prefix="council",
        ) as ex:
            futures = {
                ex.submit(self._call_member, mid, brief, tid): mid
                for mid in COUNCIL_MEMBERS
            }
            for future in concurrent.futures.as_completed(futures):
                mid = futures[future]
                votes[mid] = future.result()

        verdict = _tally(votes, trace_id=tid)
        verdict.total_latency_s = time.monotonic() - t0

        # Fallback tracking
        if any(v.degraded for v in votes.values()):
            self._metrics.record_fallback_activated()

        self._metrics.record_deliberation(
            consensus=verdict.consensus,
            degraded=verdict.degraded,
            latency_s=verdict.total_latency_s,
        )

        logger.info(
            "[council] deliberate done trace_id=%s "
            "(approve=%d challenge=%d abstain=%d consensus=%s "
            "degraded=%s veto=%s latency=%.2fs)",
            tid,
            verdict.approval_count,
            verdict.challenge_count,
            verdict.abstain_count,
            verdict.consensus,
            verdict.degraded,
            verdict.ethical_veto,
            verdict.total_latency_s,
        )

        return verdict


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _build_brief(query: str, superego: str, id_: str) -> str:
    brief = f"The user asked: {query}\n\n"
    if superego:
        brief += f"Superego (analytical) responded:\n{superego[:1200]}\n\n"
    if id_:
        brief += f"Id (creative) responded:\n{id_[:1200]}\n\n"
    brief += (
        "Based on these responses, provide your assessment according "
        "to your role on the council."
    )
    return brief


_SCORE_RX = re.compile(r"(\d+)\s*/\s*10")


def _extract_score(content: str) -> float:
    m = _SCORE_RX.search(content)
    if not m:
        return 5.0
    try:
        return min(10.0, max(0.0, float(m.group(1))))
    except ValueError:
        return 5.0


_FLAG_KEYWORDS = ("bias", "harm", "unfair", "unsafe")


def _extract_flags(member_id: str, content: str) -> List[str]:
    if member_id != "ethicist":
        return []
    lc = content.lower()
    flags: List[str] = []
    for fw in _FLAG_KEYWORDS:
        if fw in lc:
            flags.append(fw)
    if "concern" in lc and "no concern" not in lc and "no concerns" not in lc:
        flags.append("concern")
    return flags


_SEVERITY_RX = re.compile(r"\b(minor|moderate|serious)\b")


def _extract_ethical_severity(member_id: str, content: str) -> str:
    if member_id != "ethicist":
        return ""
    lc = content.lower()
    # Only count severity when adjacent to 'concern', 'flag', 'risk', or 'severity'
    m = re.search(
        r"(?:severity|concern|flag|risk)[^\n]{0,40}?\b(minor|moderate|serious)\b",
        lc,
    )
    if m:
        return m.group(1)
    m = re.search(
        r"\b(minor|moderate|serious)\b[^\n]{0,40}?(?:severity|concern|flag|risk)",
        lc,
    )
    if m:
        return m.group(1)
    return ""


def _tally(votes: Dict[str, CouncilVote], *, trace_id: str) -> CouncilVerdict:
    """Aggregate per-member votes into a verdict.

    Key fixes from the historical implementation:

    1. ``ABSTAIN`` (backend errors) does not count as approval.
    2. Ethicist flag with severity moderate/serious is a hard veto.
    3. Synthesis falls back to the highest-scoring *non-abstaining*
       member when the Synthesizer itself abstained.
    """
    approve = 0
    challenge = 0
    abstain = 0
    ethical_flags: List[str] = []
    ethical_veto = False
    degraded_any = False

    for vote in votes.values():
        if vote.outcome == VoteOutcome.APPROVE:
            approve += 1
        elif vote.outcome == VoteOutcome.CHALLENGE:
            challenge += 1
        else:
            abstain += 1
        if vote.degraded:
            degraded_any = True
        if vote.member == "ethicist":
            ethical_flags = list(vote.flags)
            if vote.ethical_severity in VETO_SEVERITIES:
                ethical_veto = True

    # Synthesis selection
    synth_vote = votes.get("synthesizer")
    if synth_vote and not synth_vote.is_error and synth_vote.response.strip():
        synthesis = synth_vote.response
    else:
        # Fall back to the highest-scoring non-abstaining member
        candidates = [v for v in votes.values() if v.outcome != VoteOutcome.ABSTAIN]
        if candidates:
            best = max(candidates, key=lambda v: v.score)
            synthesis = (
                f"[Synthesizer abstained; best available response from "
                f"{best.member}]\n\n{best.response}"
            )
        else:
            synthesis = "(All council members abstained — no verdict available.)"

    # Consensus rule
    # - Require a clear majority of non-advocate members to approve.
    # - Ethical veto (moderate/serious severity) overrides approval count.
    # - Abstain counts against consensus: with 5+ abstains we call it a failure.
    non_advocate_members = len(COUNCIL_MEMBERS) - 1  # Advocate always challenges
    consensus_threshold = (non_advocate_members // 2) + 1  # majority of 6 = 4
    consensus = (
        approve >= consensus_threshold
        and not ethical_veto
        and abstain < (len(COUNCIL_MEMBERS) // 2)
    )

    return CouncilVerdict(
        consensus=consensus,
        synthesis=synthesis,
        trace_id=trace_id,
        votes=votes,
        approval_count=approve,
        challenge_count=challenge,
        abstain_count=abstain,
        ethical_flags=ethical_flags,
        ethical_veto=ethical_veto,
        degraded=degraded_any,
    )


__all__ = [
    "COUNCIL_MEMBERS",
    "CouncilVerdict",
    "CouncilVote",
    "DivineCouncil",
    "MODEL_NAME",
    "VoteOutcome",
]
