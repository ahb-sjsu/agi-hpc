# vMOE — Virtual Mixture-of-Experts

A thin orchestration layer over multiple LLM services, where each service is an "expert" slot and the routing policy is explicit Python rather than a gating network. Source: [`src/agi/primer/vmoe.py`](../src/agi/primer/vmoe.py).

---

## Why not a single model?

We already have four frontier-class open models available as OpenAI-compatible services on NRP (Kimi K2.5 1T, Qwen 3.5 397B, GLM-4.7 358B, MiniMax M2.7), plus a local Qwen 32B on Atlas GPU 1 as a resilience fallback. These models have genuinely different strengths:

- **Kimi K2.5** — best-in-class agentic coding, long-horizon tasks, 1 M token context.
- **GLM-4.7** — strong pure reasoning, 200 K context, good at math / logic.
- **Qwen 3.5** — fast, structured, good at code generation.
- **MiniMax M2.7** — good at tool use, long context.
- **Atlas Kirk (Qwen 32B)** — local resilience, always available when NRP is unreachable.

Collapsing these into "whichever one we happen to be using this week" loses the specialization and the redundancy. vMOE embraces the reality: many experts, one orchestration layer, explicit policy.

It also maps cleanly onto the existing cognitive architecture. The Ego doesn't have to *be* a single model; it can be a *routing policy* over the expert pool. A self-hosted fine-tuned slot (see Phase 2 of [`AGI_ROADMAP.md`](AGI_ROADMAP.md)) becomes one expert *in the pool*, not a replacement for it.

---

## Primitives

### `Expert` — one service slot

```python
@dataclass(frozen=True)
class Expert:
    name: str                    # "kimi"
    model: str                   # "kimi" (API-level model ID)
    base_url: str                # "https://ellm.nrp-nautilus.io/v1"
    api_key_env: str             # "NRP_LLM_TOKEN"
    role_hints: frozenset[str]   # {"code", "agentic", "long_context"}
    timeout_s: float             # 300.0
    priority: int                # 10 (lower = preferred)
```

Role hints drive `by_hint()` filtering; priority drives `cascade()` order and `route()` selection.

### `Response` — normalized completion

```python
@dataclass
class Response:
    expert: str
    model: str
    content: str
    ok: bool             # True iff the call returned a completion without exception / timeout
    latency_s: float
    usage: dict          # token usage reported by the API
    error: str           # message if ok is False
```

Every orchestration policy returns `Response`s. Network failures and timeouts are surfaced as `ok=False, error="..."` rather than raised — callers never handle raw exceptions.

### `HealthTracker` — rolling per-expert state

```python
@dataclass
class HealthTracker:
    window: int = 5           # recent calls considered
    slow_s: float = 180.0     # latency (seconds) above which a successful call counts as unhealthy
    min_failures: int = 3     # how many unhealthy calls flip the expert to degraded
    cooldown_s: int = 3600    # how long to skip a degraded expert
```

Every `vMOE.call()` records into the tracker. `vMOE.healthy_subset(names)` filters a name list to just those not in cooldown.

---

## Default expert pool

From `default_experts()`:

| Expert | Model | Base URL | Role hints | Timeout | Priority |
|---|---|---|---|---|---|
| `kimi` | `kimi` | NRP ellm | `{code, agentic, long_context, default}` | 300 s | 10 |
| `glm-4.7` | `glm-4.7` | NRP ellm | `{reason, chat, long_context}` | 300 s | 20 |
| `qwen3` | `qwen3` | NRP ellm | `{structured, fast, code}` | 240 s | 30 |
| `minimax-m2` | `minimax-m2` | NRP ellm | `{tool_use, long_context}` | 300 s | 40 |
| `kirk-local` | `qwen` | `http://localhost:8080/v1` | `{local, fallback}` | 90 s | 100 |

Override via `EREBUS_VMOE_EXPERTS` env var (comma-separated names to include; unset = all).

---

## Orchestration policies

### `route(hint)` — single expert, no fallback

Pick the highest-priority expert matching `hint` and call it. If the call fails, you get the failure `Response` back. Useful when you want a specific model's response and nothing else.

```python
r = await moe.route(messages, hint="code")  # highest-priority "code" expert
```

### `cascade(hint, accept=...)` — priority-ordered fallback

Try experts in priority order; first to satisfy `accept` wins. Default `accept = lambda r: r.ok`. Custom predicates let you gate on quality (non-empty content, parseable JSON, verification pass). Returns the last attempted Response if nothing satisfied.

```python
r = await moe.cascade(
    messages,
    hint="code",
    accept=lambda resp: resp.ok and len(resp.content) > 100,
)
```

### `ensemble(experts=, hint=, verify=, return_all=)` — parallel fan-out

Fire all matching experts in parallel. Optionally filter with `verify`. Returns a list of surviving Responses in completion order (fastest first). With `return_all=True`, returns every Response regardless of ok-ness (useful for Primer-style "inspect all options, validate each separately").

```python
# Fire all experts, collect every response for per-candidate validation
rs = await moe.ensemble(
    messages,
    experts=["kimi", "glm-4.7", "qwen3"],
    return_all=True,
)

# Fire all "code" experts, keep only those whose response parses as JSON
rs = await moe.ensemble(
    messages,
    hint="code",
    verify=lambda r: r.ok and r.content.strip().startswith("{"),
)
```

### `first_verified(verify, experts=, hint=)` — parallel with cancel-on-win

Like `ensemble`, but returns the *first* Response to pass `verify` and *cancels the pending calls*. Saves tokens when the leader answers quickly. `verify` may be sync or async.

```python
async def passes_train(r):
    code = extract_code(r.content)
    return bool(code) and validate(code, task).all_pass

winner = await moe.first_verified(
    messages,
    verify=passes_train,
    experts=["kimi", "glm-4.7", "qwen3"],
)
```

---

## Health-aware routing

```python
candidate = moe.healthy_subset(["kimi", "glm-4.7", "qwen3"])
if not candidate:
    candidate = ["qwen3"]  # canary probe — one call to detect recovery
responses = await moe.ensemble(messages, experts=candidate, return_all=True)
```

The Primer uses this pattern to avoid burning 5 min × 3 experts per task on NRP-slow nights when all three would time out. See [`docs/THE_PRIMER.md`](THE_PRIMER.md) §"Expert health tracking".

---

## Usage in the chat ego (Phase 3, planned)

Today's chat handler (`telemetry_server.py:_erebus_chat`) hardcodes `model="kimi"`. The planned cutover (see [`AGI_ROADMAP.md`](AGI_ROADMAP.md) Phase 3) replaces that with a vMOE cascade:

```python
moe = vMOE()
r = await moe.cascade(
    messages,
    hint="chat",
    accept=lambda r: r.ok and r.content,
    # cascade order by priority: kimi (if healthy) → glm-4.7 → qwen3 → kirk-local
)
return r.content
```

The cascade automatically benefits from the health tracker — a slow Kimi doesn't block the chat, it just fails fast and the cascade moves on.

---

## Testing

Unit tests at [`tests/unit/test_primer_vmoe.py`](../tests/unit/test_primer_vmoe.py) and [`tests/unit/test_primer_health.py`](../tests/unit/test_primer_health.py) use a stubbed `call()` implementation (no network) to verify:

- hint filtering + priority ordering
- route with no match raises
- cascade custom `accept` predicates
- cascade returns last failure when all fail
- ensemble all successes, `verify` filter
- `first_verified` returns fastest accepted, handles async predicates, returns None if nothing passes
- health degradation trigger + cooldown expiry + manual clear + summary schema

Plus a live smoke via Atlas paramiko — every time the module changes, run an ensemble of 3 experts against a trivial prompt and confirm all return.

---

## Design notes

**No gating network.** A learned gate would be overfit to the current expert pool; explicit policy is cheaper and more debuggable. Future: could add a light pre-classifier (1-token prompt to qwen3-fast: "is this query best served by a code model or a reasoning model?") if hints become insufficient.

**Every call records health.** The tracker is internal to `vMOE`. Callers can't forget to update it; it's woven into `call()` directly. Record-then-return on every path (success, timeout, exception).

**Responses, not exceptions.** This was a deliberate choice. Callers of `ensemble()` want to iterate over results and validate each; raising on the first failure would force try/except in every orchestration. Surfacing all errors as `Response(ok=False, ...)` keeps call sites linear.

**Async throughout.** `AsyncOpenAI` + `asyncio.gather` + `asyncio.wait_for`. Cancellation works — `first_verified` genuinely cancels pending calls when a leader wins. Tests verify this.

**Cheap construction.** `vMOE(experts=...)` does no network calls at init; clients are lazy-created per expert on first `call()`. Safe to build one per request if needed.
