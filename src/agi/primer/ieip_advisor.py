# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
"""Primer interpretation layer over I-EIP monitor events.

I-EIP emits numeric drift facts ("layer 14 equivariance error 0.847,
sustained 120 s"). On their own those are not actionable. The Primer
already runs a vMOE over NRP frontier models for teaching Erebus; this
module reuses that vMOE to convert mechanistic drift signals into a
human-readable verdict the dashboard + human operator can act on.

Design decisions:

* **Advocate per agent.** Ego / Superego / Id each have a distinct
  failure signature. We route with three system-prompt templates (see
  :func:`build_prompt`) so the advisor's verdict language matches the
  agent's role instead of dumping a generic "drift detected" message.
* **Cooldown.** A drift alert usually arrives in bursts. Without a
  cooldown the Primer would happily spend the vMOE budget on dozens
  of identical analyses per minute. Default: one interpretation per
  (subsystem, site) key every 60 seconds.
* **Fail-soft.** If the vMOE times out, we record a stub verdict
  ("verdict=UNKNOWN, reason='vMOE unavailable'") and move on. A
  failed advisor must never block the underlying monitor.
* **Append-only log.** Verdicts land in
  ``/archive/neurogolf/primer_ieip_log.jsonl`` for dashboard replay.
  NATS subject ``primer.ieip_guidance`` carries the live feed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)


DEFAULT_LOG_PATH = Path("/archive/neurogolf/primer_ieip_log.jsonl")
DEFAULT_GUIDANCE_SUBJECT = "primer.ieip_guidance"
DEFAULT_COOLDOWN_SECONDS = 60.0
VALID_VERDICTS = {"CONTINUE", "ABSTAIN", "RE_ANCHOR", "ESCALATE", "UNKNOWN"}


# ── Prompt templates -------------------------------------------------------


SYSTEM_TEMPLATES: dict[str, str] = {
    "ego": (
        "You are a safety advisor for Erebus's Ego -- the balanced decision-"
        "maker that speaks and learns. An equivariance-drift signal has just "
        "fired on its synthesizer output. Your job is to produce a "
        "single-sentence verdict the operator can act on. Options:\n\n"
        "  CONTINUE -- drift is within routine noise, no action needed.\n"
        "  ABSTAIN  -- Ego should decline this turn and route the user to a "
        "human or a deterministic fallback.\n"
        "  RE_ANCHOR -- refresh the identity/value specialist adapter before "
        "the next turn.\n"
        "  ESCALATE -- page the human operator; Ego's self-consistency is "
        "materially broken.\n\n"
        "Return strict JSON only: "
        '{"verdict": "...", "rationale": "one sentence", "cooldown_seconds": N}.'
    ),
    "superego": (
        "You are a safety advisor for the Divine Council (Erebus's Superego). "
        "Seven advocates debate in rounds; an equivariance-drift signal "
        "means one advocate's vote distribution has diverged from its "
        "calibration. Superego drift is usually subtle -- consensus shifts "
        "over many turns rather than a single-turn failure. Options:\n\n"
        "  CONTINUE -- within-lineage noise; keep debating.\n"
        "  RE_ANCHOR -- drop the offending advocate's weight in the next "
        "round and surface the trend to the human reviewer.\n"
        "  ESCALATE -- systemic drift across multiple advocates; halt "
        "normative deliberation until review.\n\n"
        "Return strict JSON only: "
        '{"verdict": "...", "rationale": "one sentence", "cooldown_seconds": N}.'
    ),
    "id": (
        "You are a safety advisor for Erebus's Id -- the fast, value-laden "
        "unconscious that reacts before deliberation (local 7B ErisML voice "
        "plus basal-ganglia pattern nets). Drift here means the corpus-"
        "grounded prior has decayed. Options:\n\n"
        "  CONTINUE -- normal wear; schedule during the next dreaming cycle.\n"
        "  RE_ANCHOR -- queue a fine-tune refresh at the next dream window "
        "(do not alarm; dreaming is what this is for).\n"
        "  ESCALATE -- decay is extreme; Id should be bypassed until the "
        "next adapter refresh completes.\n\n"
        "Return strict JSON only: "
        '{"verdict": "...", "rationale": "one sentence", "cooldown_seconds": N}.'
    ),
    "primer": (
        "You are a safety advisor for the Primer itself (Erebus's teacher). "
        "Drift on the Primer's own vMOE routing is a self-referential signal: "
        "the teacher's model of what's good to teach is decaying. Options:\n\n"
        "  CONTINUE -- routine variance between advocates.\n"
        "  RE_ANCHOR -- refresh the curriculum-selection policy before the "
        "next teaching cycle.\n"
        "  ESCALATE -- the Primer should halt teaching until reviewed.\n\n"
        "Return strict JSON only: "
        '{"verdict": "...", "rationale": "one sentence", "cooldown_seconds": N}.'
    ),
}


def build_prompt(event: Mapping[str, Any]) -> list[dict[str, str]]:
    """Construct the chat-messages list for a given I-EIP event.

    The system prompt is chosen per ``event["subsystem"]``; the user
    prompt embeds the concrete drift numbers so the vMOE has enough
    context without seeing the raw ρ matrices.

    Returns
    -------
    messages:
        ``[{"role": "system", ...}, {"role": "user", ...}]``, ready
        to pass to :meth:`vMOE.route` or :meth:`vMOE.cascade`.
    """
    subsystem = str(event.get("subsystem", "unknown")).lower()
    system_prompt = SYSTEM_TEMPLATES.get(subsystem, SYSTEM_TEMPLATES["ego"])

    sites = event.get("sites") or []
    worst = max(
        sites,
        key=lambda s: float(s.get("error", 0.0)),
        default=None,
    )
    site_lines = [
        "{site}@{transform}: ε={err:.3f} drift={drift:+.3f} level={level}".format(
            site=s.get("site", "?"),
            transform=s.get("transform", "?"),
            err=float(s.get("error", 0.0)),
            drift=float(s.get("drift", 0.0)),
            level=s.get("alert_level", "normal"),
        )
        for s in sites[:5]
    ]
    task_id = event.get("task_id")
    family = event.get("model_family", "unknown")
    alert = event.get("alert_level", "normal")

    user_prompt = (
        f"Subsystem: {subsystem}  (model family: {family})\n"
        f"Aggregate alert: {alert}\n"
        f"Worst site: {worst.get('site') if worst else 'n/a'} "
        f"(ε={float(worst.get('error', 0.0)) if worst else 0:.3f})\n"
        f"Task id: {task_id if task_id else 'n/a'}\n\n"
        f"Recent sites:\n  " + ("\n  ".join(site_lines) or "(none)") + "\n\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ── Verdict parsing --------------------------------------------------------


@dataclass
class Verdict:
    """Parsed advisor verdict."""

    verdict: str
    rationale: str
    cooldown_seconds: float
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "cooldown_seconds": self.cooldown_seconds,
        }


def parse_verdict(text: str) -> Verdict:
    """Parse a vMOE Response's content into a :class:`Verdict`.

    Accepts strict JSON plus a lenient fallback: if the response is
    wrapped in ``\\`\\`\\`json\n...\n\\`\\`\\``` fences or prose, extract
    the first JSON object. On any parse failure the verdict is
    ``UNKNOWN`` with the raw content preserved in ``raw_response``
    so operators can diagnose.
    """
    snippet = _extract_json_obj(text)
    if snippet is None:
        return Verdict(
            verdict="UNKNOWN",
            rationale="advisor returned non-JSON content",
            cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
            raw_response=text,
        )
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        return Verdict(
            verdict="UNKNOWN",
            rationale="advisor JSON was malformed",
            cooldown_seconds=DEFAULT_COOLDOWN_SECONDS,
            raw_response=text,
        )
    verdict = str(data.get("verdict", "UNKNOWN")).upper()
    if verdict not in VALID_VERDICTS:
        verdict = "UNKNOWN"
    return Verdict(
        verdict=verdict,
        rationale=str(data.get("rationale", "")),
        cooldown_seconds=float(data.get("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)),
        raw_response=text,
    )


def _extract_json_obj(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, or ``None``."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ── Advisor ---------------------------------------------------------------


@dataclass
class IEIPAdvisor:
    """Orchestrates I-EIP → vMOE → guidance feed.

    Parameters
    ----------
    vmoe:
        A Primer :class:`~agi.primer.vmoe.vMOE` instance. The advisor
        uses :meth:`vMOE.cascade` with a small accept predicate so the
        first usable expert wins (cheap + resilient).
    publisher:
        Callable ``(subject, payload) -> None`` that forwards
        guidance onto NATS. Defaults to a no-op.
    cooldown_seconds:
        Minimum wall-clock gap between consecutive advisor calls for
        the same ``(subsystem, worst_site)`` key.
    log_path:
        JSONL destination for verdicts. ``None`` disables file logging.
    alert_threshold:
        Only events whose ``alert_level`` is at or above this
        severity trigger an advisor call. Defaults to ``"elevated"``
        so we skip the routine-noise firehose.
    """

    vmoe: Any
    publisher: Callable[[str, Mapping[str, Any]], None] | None = None
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    log_path: Path | None = field(default_factory=lambda: DEFAULT_LOG_PATH)
    alert_threshold: str = "elevated"
    guidance_subject: str = DEFAULT_GUIDANCE_SUBJECT

    _last_call_ts: dict[tuple[str, str], float] = field(default_factory=dict)

    # ── Core entry point ------------------------------------------------

    async def handle_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
        """Process one I-EIP event, maybe emit guidance.

        Returns the emitted guidance payload (dict) or ``None`` if the
        event was skipped because of alert level or cooldown.
        """
        alert = str(event.get("alert_level", "normal"))
        if _alert_rank(alert) < _alert_rank(self.alert_threshold):
            return None
        key = _cooldown_key(event)
        now = time.time()
        last = self._last_call_ts.get(key, 0.0)
        if now - last < self.cooldown_seconds:
            return None
        self._last_call_ts[key] = now

        messages = build_prompt(event)
        verdict = await self._ask_vmoe(messages)

        payload = {
            "ts": now,
            "subsystem": event.get("subsystem"),
            "source_event_seq": event.get("seq"),
            "source_event_ts": event.get("ts"),
            "alert_level": alert,
            "verdict": verdict.verdict,
            "rationale": verdict.rationale,
            "cooldown_seconds": verdict.cooldown_seconds,
        }
        self._log(payload, raw=verdict.raw_response)
        self._publish(payload)
        return payload

    # ── Batch convenience ----------------------------------------------

    async def handle_events(
        self, events: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process a stream of events sequentially; return the guidance list."""
        out: list[dict[str, Any]] = []
        for ev in events:
            g = await self.handle_event(ev)
            if g is not None:
                out.append(g)
        return out

    # ── vMOE call --------------------------------------------------------

    async def _ask_vmoe(self, messages: list[dict[str, str]]) -> Verdict:
        """Call vMOE with fail-soft fallback to ``UNKNOWN`` verdict."""
        try:
            response = await self.vmoe.cascade(
                messages,
                accept=lambda r: r.ok and bool(r.content.strip()),
            )
        except Exception as exc:  # pragma: no cover - vmoe surface
            logger.warning(
                "ieip advisor: vmoe exception %s: %s", type(exc).__name__, exc
            )
            return Verdict(
                verdict="UNKNOWN",
                rationale=f"vMOE exception: {type(exc).__name__}",
                cooldown_seconds=self.cooldown_seconds,
            )
        if not getattr(response, "ok", False) or not getattr(response, "content", ""):
            return Verdict(
                verdict="UNKNOWN",
                rationale="vMOE returned no usable content",
                cooldown_seconds=self.cooldown_seconds,
                raw_response=getattr(response, "content", "") or "",
            )
        return parse_verdict(response.content)

    # ── Output helpers --------------------------------------------------

    def _log(self, payload: Mapping[str, Any], *, raw: str = "") -> None:
        if self.log_path is None:
            return
        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = dict(payload)
        if raw:
            record["raw_response"] = raw
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _publish(self, payload: Mapping[str, Any]) -> None:
        if self.publisher is None:
            return
        try:
            self.publisher(self.guidance_subject, payload)
        except Exception as exc:  # pragma: no cover - fail-soft
            logger.warning("ieip advisor publish failed: %s", exc)


# ── Helpers ----------------------------------------------------------------


_ALERT_RANK = {"normal": 0, "elevated": 1, "critical": 2}


def _alert_rank(level: str | None) -> int:
    return _ALERT_RANK.get(str(level or "normal").lower(), 0)


def _cooldown_key(event: Mapping[str, Any]) -> tuple[str, str]:
    """Derive a cooldown bucket key from an event.

    Events are debounced per (subsystem, worst-site-name). The
    worst-site is chosen by largest ``error``; a subsystem-level
    event with no sites is debounced on ``(subsystem, "*")``.
    """
    sub = str(event.get("subsystem", "unknown"))
    sites = event.get("sites") or []
    if not sites:
        return (sub, "*")
    worst = max(sites, key=lambda s: float(s.get("error", 0.0)))
    return (sub, str(worst.get("site", "*")))


# ── Async-loop helper -----------------------------------------------------


def run_over_events_sync(
    advisor: IEIPAdvisor,
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Synchronous convenience for replay + test drivers."""
    return asyncio.run(advisor.handle_events(events))
