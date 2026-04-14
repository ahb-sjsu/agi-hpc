# Divine Council Debugging Runbook

For when the council is misbehaving in production on Atlas.

## Quick triage — where to look first

```bash
# 1. Is the ego service up?
sudo systemctl status atlas-ego.service

# 2. Recent ego logs (last 100 lines)
sudo journalctl -u atlas-ego.service -n 100 --no-pager

# 3. Is the llama-server actually responding?
curl -sf http://localhost:8084/health && echo "OK"

# 4. Any council-level errors in app logs?
journalctl -u atlas-atlas-ai.service -n 200 --no-pager | grep -i council

# 5. What does Prometheus say about fallback activation?
curl -s localhost:9090/metrics | grep council_fallback_active_total
```

## Symptom → likely cause → fix

### "All council members abstained" / "no verdict available"

**What happened:** every backend call failed. Either the ego server is down, or the circuit breaker has all 7 member-paths open.

**Check:**

```bash
sudo systemctl status atlas-ego.service
sudo journalctl -u atlas-ego.service -n 50 --no-pager
```

**Common causes:**

- Gemma 4 crashed (check journal for `llama-server` stack trace or OOM).
- Service is restarting (`Restart=always` means it comes back ~15 s later). If you're unlucky, you hit it during restart.

**Fix:**

```bash
sudo systemctl restart atlas-ego.service
# wait ~30 seconds for model load
curl -sf http://localhost:8084/health && echo "back up"
```

If it crashes again immediately, the `--cache-ram` value may need to drop (2048 or 0) — Gemma 4's memory behavior varies.

### "Council degraded" flag on every verdict

**What happened:** FallbackBackend is routing to Spock because Gemma 4's circuit is open.

**Check:**

```bash
curl -sf http://localhost:8084/health
# If this returns non-200 or hangs, Gemma 4 is down.
```

**Fix:** restart ego service (see above). Metrics `council_backend_health{backend="gemma4"}` should return to 1.

If it keeps happening: review `journalctl -u atlas-ego.service --since="1 hour ago"` for the crash pattern. Report upstream or try lower `--cache-ram`.

### Latency spikes / requests timing out

**What happened:** the shared llama-server is saturated. 8 parallel slots, 7 members trying to use them — usually fine, but if a prior request is still in flight or the model is thinking hard, they queue.

**Check:**

```bash
# Is there CPU pressure?
top -b -n 1 | head -20

# Request latency histogram
curl -s localhost:9090/metrics | grep council_request_latency_seconds_bucket
```

**Fix:**

- If CPU is saturated (all 24 threads at 100%): another Atlas service is fighting for CPU. Check `atlas-training.service`, `atlas-rag-server.py`, etc.
- If only council is slow: bump `--max-tokens` isn't the issue (we already set per-member budgets). Consider reducing `--parallel 8` to `--parallel 7` (one dedicated slot per member) if there's evidence of slot contention.

### Consensus rate dropped suddenly

**What happened:** something changed about the inputs or the model's behavior. Consensus rate is an EWMA; a sudden drop is significant.

**Check:**

```bash
# What was consensus rate last hour vs last day?
curl -s localhost:9090/metrics | grep council_consensus_rate
```

**Ways to investigate:**

1. Run the eval harness and compare to baseline:

   ```bash
   python scripts/council_eval.py --runs 1 --output /archive/council_eval/
   diff <(cat /archive/council_eval/LAST.summary.md) \
        <(cat /archive/council_eval/*.summary.md | tail -N)
   ```

2. Look at recent deliberations with `abstain_count > 0` — those point at backend issues.

3. Look at recent deliberations with `ethical_veto=True` — is there a new kind of adversarial input hitting the council?

### One specific member always abstains

**What happened:** a member's system prompt is parse-breaking Gemma 4. Common culprits: heredocs, nested JSON, very long prompts that overflow the per-slot context.

**Check:**

```bash
# Look for repeated error on one member
sudo journalctl -u atlas-ego.service --since="1 hour ago" | grep member=historian
```

**Fix:** edit the member's `system_prompt` in `src/agi/reasoning/divine_council.py`. Keep it under ~400 tokens. Re-deploy.

### Ethicist never vetoes / always vetoes

**Symptom 1 (never vetoes):** severity parsing may not be finding "moderate"/"serious" even when it should.

**Symptom 2 (always vetoes):** the regex is matching too liberally, or the prompt is making Gemma 4 always say "serious concern" out of caution.

**Debug:**

```bash
# Print what severity parsing sees on a specific input
python -c "
from agi.reasoning.divine_council import _extract_ethical_severity
print(_extract_ethical_severity('ethicist', open('/tmp/ethicist_response.txt').read()))
"
```

**Tune:** if prompt-induced, soften "Flag concerns with severity: minor, moderate, serious" to "Only escalate to moderate/serious when the concern is specific and cite-able." If parser-induced, the `_SEVERITY_RX` + context-window regex in `divine_council.py` is the place.

## How to test a council change safely

1. **Don't touch production first.** Always run the eval harness in `--dry-run` to confirm the shape:

   ```bash
   python scripts/council_eval.py --dry-run
   ```

2. **Run against the production endpoints, read-only:**

   ```bash
   python scripts/council_eval.py --backends combined --runs 1 \
       --output /tmp/my-change-eval/
   cat /tmp/my-change-eval/*.summary.md
   ```

3. **Compare to the last known-good:**

   ```bash
   ls -lt /archive/council_eval/*.summary.md | head -2
   diff <last-good> <tmp>
   ```

4. **If the change looks safe, update code + restart:**

   ```bash
   git pull
   sudo systemctl restart atlas-ego.service
   # wait for load
   python scripts/council_eval.py --runs 1 --output /archive/council_eval/
   ```

## When to page the admin

- Ego service is restart-looping (`StartLimitBurst` exceeded). Systemd stops trying after 3 restarts in 120 s.
- All three backends (gemma4, spock, id) report unhealthy simultaneously — something wider is broken.
- Consensus rate drops below 20% and stays there for >30 minutes.
- Ethical veto rate spikes above 50% of deliberations — likely either Gemma 4 went weird or we're under adversarial load.

## Related files

- `src/agi/reasoning/divine_council.py` — aggregation logic
- `src/agi/reasoning/_council_backend.py` — transport + retry + circuit breaker
- `src/agi/reasoning/_council_metrics.py` — Prometheus metrics
- `deploy/systemd/atlas-ego.service` — llama-server launch config
- `scripts/council_eval.py` — eval harness
- `docs/architecture/COUNCIL_RELIABILITY_PLAN.md` — design doc
