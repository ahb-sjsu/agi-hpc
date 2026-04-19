"""Expert health tracking for the vMOE.

Managed LLM endpoints have time-of-day load variance. When a heavy
reasoning model (kimi, glm-4.7) has timed out its last N calls, the
next N calls are overwhelmingly likely to time out too. Keep calling
them and we just burn wall-clock per tick for nothing. Mark them
degraded; prefer the canary (fastest model) during a bad window.

Usage sketch:

    tracker = HealthTracker(window=5, slow_s=180.0, cooldown_s=3600)
    ...
    tracker.record("kimi", latency=302.1, ok=False)
    ...
    if tracker.healthy("kimi"):
        use_kimi = True

``healthy(name)`` returns False iff the expert was previously marked
degraded and the cooldown hasn't expired.

Mark logic: an expert is marked degraded when at least ``min_failures``
of the most-recent ``window`` calls were unhealthy. An "unhealthy"
call is either a failure (``ok=False``, e.g. timeout or error) OR a
success whose latency exceeded ``slow_s``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _CallRecord:
    ts: float
    latency_s: float
    ok: bool

    @property
    def unhealthy_at(self) -> bool:
        return (not self.ok) or self.latency_s > 999_999.0  # overridden by tracker


@dataclass
class HealthTracker:
    """Per-expert rolling health state.

    Thread-safe only for single-process use (the Primer daemon is
    single-process). Methods are fast: constant-time per update.
    """

    window: int = 5
    """Number of recent calls considered when deciding degradation."""

    slow_s: float = 180.0
    """Call latency (seconds) above which we count the call as unhealthy
    even if it returned ok. For thinking models on bad-NRP nights, this
    is what the 'degraded' state actually captures — they still return,
    just too slow to be useful in an ensemble."""

    min_failures: int = 3
    """How many of the last ``window`` calls must be unhealthy before
    we flip the expert to degraded."""

    cooldown_s: int = 3600
    """How long to keep an expert out of rotation once marked."""

    _history: dict[str, deque] = field(default_factory=dict)
    _degraded_until: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, latency_s: float, ok: bool) -> None:
        """Append an outcome. Re-evaluates degradation on each call."""
        d = self._history.setdefault(name, deque(maxlen=self.window))
        d.append(_CallRecord(ts=time.time(), latency_s=latency_s, ok=ok))
        if self._count_unhealthy(name) >= self.min_failures:
            self._degraded_until[name] = time.time() + self.cooldown_s

    def healthy(self, name: str) -> bool:
        """True if expert isn't in an active degradation window."""
        du = self._degraded_until.get(name)
        if du is None:
            return True
        return time.time() >= du

    def clear(self, name: str) -> None:
        """Manually reset an expert's degradation (for tests / probes)."""
        self._degraded_until.pop(name, None)

    def summary(self) -> dict[str, dict]:
        """Dict suitable for JSON export (e.g. /api/primer/status)."""
        now = time.time()
        out: dict[str, dict] = {}
        for name, d in self._history.items():
            hist = list(d)
            unhealthy_count = self._count_unhealthy(name)
            du = self._degraded_until.get(name, 0)
            avg_latency = sum(c.latency_s for c in hist) / len(hist) if hist else 0.0
            out[name] = {
                "healthy": now >= du,
                "degraded_until_s": max(0, int(du - now)) if du > now else 0,
                "window_size": len(hist),
                "unhealthy_in_window": unhealthy_count,
                "avg_latency_s": round(avg_latency, 1),
            }
        return out

    def _count_unhealthy(self, name: str) -> int:
        d = self._history.get(name)
        if d is None:
            return 0
        return sum(1 for c in d if (not c.ok) or c.latency_s > self.slow_s)
