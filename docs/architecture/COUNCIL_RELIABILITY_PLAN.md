# Divine Council Reliability Plan

**Status:** In progress · **Date:** 2026-04-13 · **Author:** Andrew H. Bond

## Problem

`src/agi/reasoning/divine_council.py` runs Gemma 4 26B-A4B on a single `llama-server --parallel 8` process on CPU. Field reports (r/LocalLLaMA, 2026-04-12) and internal observation confirm Gemma 4 crashes frequently — some users report daily, ours has been crashing under load (context fill >75%). When the Ego server goes down, the entire Divine Council goes down. All 7 member calls fail simultaneously; callers receive `(error: ConnectionError)` responses; consensus logic degenerates.

Existing mitigation is a systemd `Restart=always` on `atlas-ego.service`. That restores the process eventually but says nothing to in-flight requests.

## Goal

Make the Divine Council resilient to Gemma 4 instability AND fix correctness bugs in the consensus logic, without adding hardware or redesigning the architecture. Specifically:

1. **Reduce crash rate** by applying the Reddit-validated `llama-server` flags (`--cache-ram 4096`, `--no-mmap`, `--jinja`, bumped `--ctx-size`).
2. **Survive crashes gracefully** by adding health-check + retry + circuit-breaker logic to the Python client.
3. **Degrade rather than fail** by routing council requests to the Spock backend (Qwen 2.5 72B on GPU 0) when Ego is unhealthy, producing a single-model council verdict.
4. **Fix the silent-error-counts-as-approval bug** so a failed member (backend error / timeout) does not count toward consensus approval. Votes get three-valued outcomes: `approve`, `challenge`, `abstain`.
5. **Make Ethicist flags a veto**, not a counted challenge. Any `moderate` or `serious` ethical flag sets `consensus=False` regardless of approval count.
6. **Untruncate member responses** so Historian and Futurist have room to cite precedent and trace consequences. `max_tokens` raised per-member; display truncation moved from data-model to presentation layer.
7. **Add correlation IDs** so a user query's path through Superego → Id → Council is traceable in logs.
8. **Observe the whole thing** via Prometheus metrics (optional dep) and structured logs.
9. **Verify the hypothesis** via a 20-query eval harness: we claim a 7-member council produces better answers than Spock alone. Harness measures that claim on a fixed query set with blind A/B-comparable output.

Explicitly out of scope for this work:

- Multi-GPU Ego (we don't have the VRAM).
- Full Grafana dashboard JSON (separate ticket).
- Automated nightly eval cron (separate ticket; eval harness is manually runnable for now).
- **Architectural redesign.** The Ego-as-Council is currently the structural workhorse — it receives Superego + Id outputs, has 7 perspectives on them, and produces the synthesis. A tighter metaphor would make Superego / Id / Ego parallel with a lightweight mediator producing the final answer and the Council as an escalation path. That's a real conversation for a future architecture review; this pass is strictly hardening + bug fixing.

## Design

### Backend abstraction

```
Caller  ──►  DivineCouncil.deliberate()
                  │
                  ▼
          CouncilBackend (Protocol)
                  │
                  ├─► LlamaServerBackend(Gemma 4, :8084)   ← primary
                  │       · health probe (GET /health, cached 5 s)
                  │       · per-request retry (3 attempts, exponential)
                  │       · circuit breaker (open after 5 consecutive fails)
                  │
                  └─► LlamaServerBackend(Spock/Qwen 72B, :8080)   ← fallback
                          · same health/retry semantics
                          · used in "degraded" mode: one model, all 7 roles
```

`FallbackBackend(primary, fallback)` composes them: it tries primary; if primary's circuit is open OR health check fails, it routes to fallback and sets a `degraded_mode=True` flag on the returned vote. Verdicts carry this flag so callers can log/display the degradation.

### Retry policy

Per-member request:
- Attempt 1: immediate
- Attempt 2: 500 ms delay + full jitter
- Attempt 3: 1500 ms delay + full jitter
- Each attempt is wrapped in the configured request timeout (default 120 s).
- On attempt 3 failure: return `CouncilVote(response="(backend unavailable after retries)", score=5.0, flags=["backend_error"])`.

Retries are only performed for transport failures (ConnectionError, Timeout) and 5xx responses. 4xx (auth, bad request) fail immediately.

### Circuit breaker

A per-backend counter:
- Increment on any failed request (including after retries).
- Reset to zero on any successful request.
- After 5 consecutive failures, circuit opens for 30 s.
- Health probe can close the circuit early if it succeeds.

While open, the backend returns a synthetic failure immediately rather than firing a request. `FallbackBackend` interprets open-circuit as "try fallback".

### Metrics

Optional `prometheus_client` integration (`_council_metrics.py`). If the library is not installed, all metric operations are no-ops and nothing breaks. Metrics exposed:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `council_request_latency_seconds` | Histogram | `member`, `backend`, `outcome` | Per-member latency |
| `council_request_total` | Counter | `member`, `backend`, `outcome` | Request count |
| `council_backend_health` | Gauge | `backend` | 1=healthy, 0=unhealthy |
| `council_circuit_open` | Gauge | `backend` | 1=open, 0=closed |
| `council_fallback_active_total` | Counter | — | Count of fallback activations |
| `council_deliberation_latency_seconds` | Histogram | `consensus`, `degraded` | End-to-end deliberation latency |
| `council_consensus_rate` | Gauge | — | Rolling rate of consensus verdicts |

### Service-level changes

`deploy/systemd/atlas-ego.service`:

- `--ctx-size 16384` (from 4096) — gives each of 8 parallel slots ~2 K tokens of effective context.
- `--cache-ram 4096` — per `aldegr`'s r/LocalLLaMA fix; prevents the checkpoint-paging OOM that kills Gemma 4 at high context fill.
- `--no-mmap` — already used by other services; Gemma 4 inherits the same treatment.
- `--jinja` — lets Gemma 4 use its official chat template, which in our testing reduces weird outputs.

Tuning rationale documented inline in the service file.

### Eval harness

`scripts/council_eval.py`:

- 20 fixed queries (safety-relevant, coding, reasoning, ethics edge cases)
- Runs each query against: (a) full Gemma 4 council, (b) Spock-fallback council (force degraded mode), (c) no-council baseline (just Superego + Id)
- Writes `/archive/council_eval/{timestamp}.parquet` with: query, backend, latency, verdict, approval_count, ethical_flags, synthesis_preview
- Prints a summary table: success rate per backend, median latency, consensus agreement
- Run manually from any notebook or CLI; no cron yet.

## Rollout

1. **Land this PR on main** (code + service file + docs + tests).
2. **Deploy service change on Atlas**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart atlas-ego.service
   sudo systemctl status atlas-ego.service  # verify it started
   ```
3. **Smoke-test from the chat endpoint** — one interactive query that exercises the council.
4. **Run the eval harness** to establish baseline:
   ```bash
   python scripts/council_eval.py --runs 3
   ```
5. **Monitor for 24 h** — watch `journalctl -u atlas-ego.service`, note crash frequency vs pre-change baseline.

## Rollback

If the new ego.service config is worse:

```bash
sudo git -C /home/claude/source/agi-hpc checkout HEAD~1 -- deploy/systemd/atlas-ego.service
sudo systemctl daemon-reload
sudo systemctl restart atlas-ego.service
```

If the new Python code is worse: revert the relevant commit on `main` and `git pull` on Atlas. `divine_council.py`'s public API (`DivineCouncil.deliberate`) is unchanged, so no caller rewrites are needed either way.

## Measurements of success

After 1 week of production running with the changes:

- Gemma 4 crash frequency (from journalctl) drops by ≥50%.
- No user-visible errors from DivineCouncil during the crash window — fallback activates and logs are clear.
- Eval harness p50 latency per backend documented, used as a reference for future tuning.
- Consensus rate (via Prometheus gauge) stays within ±5% of pre-change baseline — we're hardening reliability, not changing deliberation quality.

## References

- [r/LocalLLaMA "Best Local LLMs - Apr 2026" megathread](https://www.reddit.com/r/LocalLLaMA/) — user reports on Gemma 4 crashes and mitigations
- `src/agi/reasoning/divine_council.py` — existing implementation
- `deploy/systemd/atlas-ego.service` — service definition
- `.claude/rules/atlas-operations.md` — hardware and service operations rules
