"""vMOE — virtual Mixture-of-Experts over multiple LLM services.

Each `Expert` is an OpenAI-compatible HTTP endpoint (e.g. NRP's ELLM
hosted models, Atlas's local Kirk, or a future self-hosted fine-tuned
pod). `vMOE` provides three orchestration policies:

1. ``route(hint)`` — pick a single expert by role tag and call it.
2. ``cascade(experts)`` — try experts in order; first non-error wins.
   Useful for resilience (Kimi → GLM → Qwen if Kimi times out).
3. ``ensemble(experts, verify_fn)`` — fire all experts in parallel,
   run ``verify_fn`` against each response, return the first that
   passes. Useful for code-gen tasks where "any solution that
   passes the tests" beats "the best single model's attempt."

The default expert pool is the four NRP frontier services + Atlas
Kirk as a local fallback. Override by passing ``experts=[...]``
or by env var ``EREBUS_VMOE_EXPERTS``.

This module is the substrate for both the Primer (teaching layer)
and — eventually — Erebus's chat handler (ego as routing policy
rather than a single model).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from .health import HealthTracker

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Expert:
    """One LLM service slot in the vMOE pool."""

    name: str
    """Friendly name used for logging and hint routing (e.g. ``'kimi'``)."""

    model: str
    """API-level model identifier passed as ``model=`` to the endpoint."""

    base_url: str
    """OpenAI-compatible base URL (``https://.../v1``)."""

    api_key_env: str
    """Environment variable from which to read the bearer token."""

    role_hints: frozenset[str] = field(default_factory=frozenset)
    """Tags this expert is particularly good at (``{'code','agentic'}``)."""

    timeout_s: float = 120.0
    """Per-call wall-clock budget."""

    priority: int = 100
    """Lower = preferred in cascade/default picks."""


@dataclass
class Response:
    """Normalized response from any vMOE expert call."""

    expert: str
    """Which expert actually produced this response."""

    model: str
    """Underlying model ID reported by the endpoint."""

    content: str
    """Assistant message text (possibly empty on error)."""

    ok: bool
    """True iff we received a completion without exception or timeout."""

    latency_s: float
    """Wall time from call to response."""

    usage: dict[str, Any] = field(default_factory=dict)
    """Token usage fields as reported by the API."""

    error: str = ""
    """Error message if ``ok`` is False."""


def default_experts() -> list[Expert]:
    """Return the standard Erebus expert pool.

    Four NRP frontier services plus Atlas-local Kirk as a resilience
    fallback. Override via ``EREBUS_VMOE_EXPERTS`` (comma-separated
    expert names to include; unset = all).
    """
    nrp = "https://ellm.nrp-nautilus.io/v1"
    nrp_key = "NRP_LLM_TOKEN"
    all_experts = [
        Expert(
            name="kimi",
            model="kimi",
            base_url=nrp,
            api_key_env=nrp_key,
            role_hints=frozenset({"code", "agentic", "long_context", "default"}),
            timeout_s=300.0,
            priority=10,
        ),
        Expert(
            name="glm-4.7",
            model="glm-4.7",
            base_url=nrp,
            api_key_env=nrp_key,
            role_hints=frozenset({"reason", "chat", "long_context"}),
            timeout_s=300.0,
            priority=20,
        ),
        Expert(
            name="qwen3",
            model="qwen3",
            base_url=nrp,
            api_key_env=nrp_key,
            role_hints=frozenset({"structured", "fast", "code"}),
            timeout_s=240.0,
            priority=30,
        ),
        Expert(
            name="minimax-m2",
            model="minimax-m2",
            base_url=nrp,
            api_key_env=nrp_key,
            role_hints=frozenset({"tool_use", "long_context"}),
            timeout_s=300.0,
            priority=40,
        ),
        Expert(
            name="kirk-local",
            model="qwen",
            base_url="http://localhost:8080/v1",
            api_key_env="ATLAS_KIRK_TOKEN",
            role_hints=frozenset({"local", "fallback"}),
            timeout_s=90.0,
            priority=100,
        ),
    ]
    include = os.environ.get("EREBUS_VMOE_EXPERTS", "").strip()
    if include:
        wanted = {n.strip() for n in include.split(",") if n.strip()}
        return [e for e in all_experts if e.name in wanted]
    return all_experts


class vMOE:
    """Router + ensemble coordinator over an expert pool.

    All methods are async. The class itself is cheap to construct
    (no network calls at init); clients are lazy-created per expert.
    """

    def __init__(
        self,
        experts: Iterable[Expert] | None = None,
        *,
        health: HealthTracker | None = None,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package is required; install with `pip install openai`"
            )
        self.experts: dict[str, Expert] = {
            e.name: e for e in (experts or default_experts())
        }
        self._clients: dict[str, AsyncOpenAI] = {}
        self.health: HealthTracker = health or HealthTracker()

    # ── expert lookup ──────────────────────────────────────────────

    def get(self, name: str) -> Expert:
        if name not in self.experts:
            raise KeyError(f"unknown expert: {name!r} (have: {list(self.experts)})")
        return self.experts[name]

    def by_hint(self, hint: str | None, *, only_healthy: bool = False) -> list[Expert]:
        """Return experts matching the hint, ordered by priority.

        ``hint=None`` returns all experts sorted by priority (lowest
        = most preferred). ``hint="code"`` returns only those with
        that tag. ``only_healthy=True`` filters out experts currently
        in a health cooldown (see ``agi.primer.health.HealthTracker``).
        Empty result if no expert matches.
        """
        pool = list(self.experts.values())
        if hint:
            pool = [e for e in pool if hint in e.role_hints]
        if only_healthy:
            pool = [e for e in pool if self.health.healthy(e.name)]
        pool.sort(key=lambda e: e.priority)
        return pool

    def healthy_subset(self, names: list[str]) -> list[str]:
        """Filter a name list to just those currently marked healthy."""
        return [n for n in names if n in self.experts and self.health.healthy(n)]

    # ── low-level call ────────────────────────────────────────────

    async def call(
        self,
        expert_name: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        timeout_s: float | None = None,
    ) -> Response:
        """Call a single expert. Catches network/timeout errors and
        surfaces them as ``Response(ok=False, error=...)``."""
        expert = self.get(expert_name)
        client = self._get_client(expert)
        budget = timeout_s if timeout_s is not None else expert.timeout_s
        t0 = time.time()
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=expert.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=budget,
            )
            content = (
                completion.choices[0].message.content
                if completion.choices and completion.choices[0].message
                else ""
            ) or ""
            usage = getattr(completion, "usage", None)
            usage_dict = usage.model_dump() if usage is not None else {}
            latency = time.time() - t0
            self.health.record(expert.name, latency, True)
            return Response(
                expert=expert.name,
                model=expert.model,
                content=content,
                ok=True,
                latency_s=latency,
                usage=usage_dict,
            )
        except asyncio.TimeoutError:
            latency = time.time() - t0
            self.health.record(expert.name, latency, False)
            return Response(
                expert=expert.name,
                model=expert.model,
                content="",
                ok=False,
                latency_s=latency,
                error=f"timeout after {budget:.1f}s",
            )
        except Exception as e:  # noqa: BLE001 — convert any client error to Response
            latency = time.time() - t0
            self.health.record(expert.name, latency, False)
            return Response(
                expert=expert.name,
                model=expert.model,
                content="",
                ok=False,
                latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )

    # ── orchestration policies ────────────────────────────────────

    async def route(
        self,
        messages: list[dict[str, Any]],
        *,
        hint: str | None = None,
        **opts: Any,
    ) -> Response:
        """Pick one expert by hint (or lowest priority if hint is None)
        and call it. No fallback — if the chosen expert fails, you
        get the failure Response back."""
        pool = self.by_hint(hint)
        if not pool:
            raise ValueError(f"no expert matches hint {hint!r}")
        return await self.call(pool[0].name, messages, **opts)

    async def cascade(
        self,
        messages: list[dict[str, Any]],
        *,
        hint: str | None = None,
        accept: Callable[[Response], bool] | None = None,
        **opts: Any,
    ) -> Response:
        """Try experts in priority order; first accepted Response wins.

        ``accept`` defaults to ``lambda r: r.ok`` (first non-error).
        Pass a stricter predicate for quality gating (e.g. non-empty
        content, structured JSON parseable, verification pass)."""
        accept = accept or (lambda r: r.ok)
        pool = self.by_hint(hint)
        last: Response | None = None
        for expert in pool:
            last = await self.call(expert.name, messages, **opts)
            if accept(last):
                return last
            log.warning(
                "cascade: expert=%s rejected (ok=%s err=%r); trying next",
                expert.name,
                last.ok,
                last.error,
            )
        assert last is not None  # by_hint non-empty guaranteed above
        return last

    async def ensemble(
        self,
        messages: list[dict[str, Any]],
        *,
        experts: list[str] | None = None,
        hint: str | None = None,
        verify: (
            Callable[[Response], Awaitable[bool]] | Callable[[Response], bool] | None
        ) = None,
        return_all: bool = False,
        **opts: Any,
    ) -> list[Response]:
        """Fire the chosen experts in parallel.

        - ``experts``: explicit list of expert names. Overrides ``hint``.
        - ``hint``: if ``experts`` is None, use all matching experts.
        - ``verify``: async or sync predicate. If given, responses
          failing it are filtered out; the returned list preserves
          the remaining in completion order (fastest verified-true
          first).
        - ``return_all``: if True, return every response (ok or not,
          verified or not) for inspection. ``verify`` is ignored.

        Returns a list of ``Response`` objects. Empty if every expert
        failed and no ``return_all`` override."""
        if experts:
            names = [e for e in experts if e in self.experts]
        else:
            names = [e.name for e in self.by_hint(hint)]
        if not names:
            return []
        tasks = [self.call(name, messages, **opts) for name in names]
        results: list[Response] = []
        for coro in asyncio.as_completed(tasks):
            r = await coro
            if return_all:
                results.append(r)
                continue
            if not r.ok:
                log.info("ensemble: %s failed: %s", r.expert, r.error)
                continue
            if verify is not None:
                v = verify(r)
                passed = await v if asyncio.iscoroutine(v) else bool(v)
                if not passed:
                    log.info("ensemble: %s did not verify", r.expert)
                    continue
            results.append(r)
        return results

    async def first_verified(
        self,
        messages: list[dict[str, Any]],
        verify: Callable[[Response], Awaitable[bool]] | Callable[[Response], bool],
        *,
        experts: list[str] | None = None,
        hint: str | None = None,
        **opts: Any,
    ) -> Response | None:
        """Convenience wrapper: run ensemble with verification and
        return only the first-to-pass response (or None if none pass).

        Unlike ``ensemble``, this cancels pending calls as soon as a
        verified response arrives — saves tokens when the leader
        answers quickly. The callable is awaited if async."""
        if experts:
            names = [e for e in experts if e in self.experts]
        else:
            names = [e.name for e in self.by_hint(hint)]
        if not names:
            return None
        tasks = {
            asyncio.create_task(self.call(name, messages, **opts)): name
            for name in names
        }
        try:
            while tasks:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    r = t.result()
                    name = tasks.pop(t)
                    if not r.ok:
                        log.info("first_verified: %s failed: %s", name, r.error)
                        continue
                    v = verify(r)
                    passed = await v if asyncio.iscoroutine(v) else bool(v)
                    if passed:
                        for p in pending:
                            p.cancel()
                        return r
                    log.info("first_verified: %s did not verify", name)
            return None
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    # ── internal ───────────────────────────────────────────────────

    def _get_client(self, expert: Expert) -> AsyncOpenAI:
        if expert.name not in self._clients:
            key = os.environ.get(expert.api_key_env, "") or "none"
            self._clients[expert.name] = AsyncOpenAI(
                api_key=key, base_url=expert.base_url
            )
        return self._clients[expert.name]
