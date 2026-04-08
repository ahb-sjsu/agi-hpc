# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Divine Council — Multi-agent Ego deliberation.

Replaces the single Ego mediator with a council of specialized
sub-agents that deliberate in parallel and reach consensus. Each
council member is a Gemma 4 E4B instance with a distinct persona
and evaluation lens.

Council Members:
    Judge      — Impartial evaluator. Scores correctness, logic.
    Advocate   — Devil's advocate. Challenges consensus, finds flaws.
    Synthesizer— Integration expert. Merges perspectives, resolves tension.
    Ethicist   — Moral compass. Checks alignment, fairness, safety.

Cognitive science grounding:
    - Minsky (1986): Society of Mind — intelligence as many agents
    - Mercier & Sperber (2011): Argumentative Theory of Reasoning —
      reasoning evolved for social deliberation, not solo thinking
    - Surowiecki (2004): Wisdom of Crowds — diverse independent
      judgments aggregate better than individual expert judgment

The council improves on a single Ego because:
    1. Diversity: Different lenses catch different problems
    2. Adversarial: The Advocate prevents groupthink
    3. Parallel: All members deliberate simultaneously (fast)
    4. Robust: No single point of failure in judgment

Usage:
    council = DivineCouncil()
    verdict = council.deliberate(query, superego_response, id_response)
    # verdict.consensus = True
    # verdict.synthesis = "The balanced answer is..."
    # verdict.votes = {"judge": "approve", "advocate": "challenge", ...}
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Council member definitions
# ---------------------------------------------------------------------------

COUNCIL_MEMBERS = {
    "judge": {
        "name": "Judge",
        "system_prompt": (
            "You are the Judge on the Divine Council. Your role is "
            "impartial evaluation. Assess the accuracy, logical "
            "coherence, and completeness of the responses presented. "
            "Score each on a scale of 1-10. Be precise and fair. "
            "Do not advocate — only evaluate."
        ),
        "color": "#4a9eff",
    },
    "advocate": {
        "name": "Advocate",
        "system_prompt": (
            "You are the Devil's Advocate on the Divine Council. "
            "Your role is to challenge the consensus. Find flaws, "
            "unstated assumptions, edge cases, and counterarguments. "
            "If everyone agrees, you must disagree and explain why. "
            "Be rigorous but constructive — your goal is to prevent "
            "groupthink, not to obstruct."
        ),
        "color": "#f87171",
    },
    "synthesizer": {
        "name": "Synthesizer",
        "system_prompt": (
            "You are the Synthesizer on the Divine Council. Your "
            "role is integration. Take the strongest elements from "
            "each perspective and weave them into a coherent, "
            "balanced answer. Resolve tensions rather than picking "
            "sides. Your synthesis should be better than any "
            "individual response. Be concise and authoritative."
        ),
        "color": "#4ade80",
    },
    "ethicist": {
        "name": "Ethicist",
        "system_prompt": (
            "You are the Ethicist on the Divine Council. Your role "
            "is moral evaluation. Check the responses for fairness, "
            "potential harm, bias, and alignment with ethical "
            "principles. Flag any concerns. If the responses are "
            "ethically sound, confirm. Your assessment should be "
            "grounded in specific moral frameworks."
        ),
        "color": "#f59e0b",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CouncilVote:
    """One council member's deliberation output."""

    member: str
    response: str
    score: float  # 1-10 for judge, 0-1 for others
    flags: List[str] = field(default_factory=list)
    latency_s: float = 0.0


@dataclass
class CouncilVerdict:
    """The council's collective verdict."""

    consensus: bool
    synthesis: str
    votes: Dict[str, CouncilVote] = field(default_factory=dict)
    approval_count: int = 0
    challenge_count: int = 0
    ethical_flags: List[str] = field(default_factory=list)
    total_latency_s: float = 0.0
    method: str = "divine_council"

    def format_log(self) -> str:
        """Format for the UI collapsible."""
        lines = ["### Divine Council Deliberation\n"]
        for member_id, vote in self.votes.items():
            info = COUNCIL_MEMBERS.get(member_id, {})
            name = info.get("name", member_id)
            lines.append(
                f"**{name}** " f"({vote.latency_s:.1f}s):\n" f"{vote.response[:300]}\n"
            )
        lines.append(
            f"\n**Consensus:** "
            f"{'Yes' if self.consensus else 'No'} "
            f"({self.approval_count} approve, "
            f"{self.challenge_count} challenge)"
        )
        if self.ethical_flags:
            lines.append(f"**Ethical flags:** {', '.join(self.ethical_flags)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Divine Council
# ---------------------------------------------------------------------------


class DivineCouncil:
    """Multi-agent Ego deliberation engine.

    Runs all council members in parallel and aggregates their
    judgments into a consensus verdict with synthesized response.
    """

    def __init__(
        self,
        member_urls: Optional[Dict[str, str]] = None,
        timeout: int = 120,
    ) -> None:
        self._urls = member_urls or {
            "judge": "http://localhost:8084",
            "advocate": "http://localhost:8085",
            "synthesizer": "http://localhost:8086",
            "ethicist": "http://localhost:8087",
        }
        self._timeout = timeout

    def _call_member(
        self,
        member_id: str,
        prompt: str,
    ) -> CouncilVote:
        """Call one council member."""
        url = self._urls.get(member_id, "http://localhost:8084")
        info = COUNCIL_MEMBERS.get(member_id, {})
        system = info.get("system_prompt", "You are a council member.")

        t0 = time.monotonic()
        try:
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 256,
                    "stream": False,
                },
                timeout=self._timeout,
            )
            data = resp.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "") or msg.get("reasoning_content", "")

            # Extract score if present
            score = 5.0
            score_match = re.search(r"(\d+)/10", content)
            if score_match:
                score = min(10.0, float(score_match.group(1)))

            # Extract flags
            flags = []
            if member_id == "ethicist":
                lc = content.lower()
                for fw in ["bias", "harm", "unfair", "unsafe"]:
                    if fw in lc:
                        flags.append(fw)
                if "concern" in lc and "no concern" not in lc:
                    flags.append("concern")

            elapsed = time.monotonic() - t0
            return CouncilVote(
                member=member_id,
                response=content[:500],
                score=score,
                flags=flags,
                latency_s=elapsed,
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning("[council] %s failed: %s", member_id, e)
            return CouncilVote(
                member=member_id,
                response=f"(error: {e})",
                score=5.0,
                latency_s=elapsed,
            )

    def deliberate(
        self,
        query: str,
        superego_response: str = "",
        id_response: str = "",
        context: str = "",
    ) -> CouncilVerdict:
        """Run full council deliberation.

        All members receive the same brief and deliberate in
        parallel. The Synthesizer produces the final answer,
        informed by Judge scores and Advocate challenges.

        Args:
            query: The user's original question.
            superego_response: Superego's analytical response.
            id_response: Id's creative response.
            context: Optional additional context.

        Returns:
            CouncilVerdict with synthesis and vote details.
        """
        t0 = time.monotonic()

        # Build the brief for all council members
        brief = f"The user asked: {query}\n\n"
        if superego_response:
            brief += (
                f"Superego (analytical) responded:\n" f"{superego_response[:400]}\n\n"
            )
        if id_response:
            brief += f"Id (creative) responded:\n" f"{id_response[:400]}\n\n"
        brief += (
            "Based on these responses, provide your assessment "
            "according to your role on the council."
        )

        # All members deliberate in parallel
        votes: Dict[str, CouncilVote] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(self._call_member, mid, brief): mid for mid in COUNCIL_MEMBERS
            }
            for future in concurrent.futures.as_completed(futures):
                mid = futures[future]
                votes[mid] = future.result()

        # Count approvals vs challenges
        # Advocate always challenges. Others approve unless
        # they give an explicitly low score (<4) or have flags.
        approval = 0
        challenge = 0
        for mid, vote in votes.items():
            if mid == "advocate":
                challenge += 1
            elif vote.flags:
                challenge += 1
            elif vote.score < 4:
                challenge += 1
            else:
                approval += 1

        # Ethical flags from ethicist
        ethical_flags = votes.get(
            "ethicist", CouncilVote(member="ethicist", response="", score=5)
        ).flags

        # Use Synthesizer's response as the final answer
        synthesis = votes.get(
            "synthesizer", CouncilVote(member="synthesizer", response="", score=5)
        ).response

        # If Synthesizer failed, fall back to highest-scored response
        if not synthesis or "(error" in synthesis:
            best = max(
                votes.values(),
                key=lambda v: v.score,
            )
            synthesis = best.response

        elapsed = time.monotonic() - t0
        consensus = approval >= 2 and len(ethical_flags) == 0

        verdict = CouncilVerdict(
            consensus=consensus,
            synthesis=synthesis,
            votes=votes,
            approval_count=approval,
            challenge_count=challenge,
            ethical_flags=ethical_flags,
            total_latency_s=elapsed,
        )

        logger.info(
            "[council] Deliberation: %d approve, %d challenge, " "consensus=%s, %.1fs",
            approval,
            challenge,
            consensus,
            elapsed,
        )

        return verdict
