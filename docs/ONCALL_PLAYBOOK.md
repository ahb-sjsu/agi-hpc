# On-Call Playbook — Atlas AI / agi-hpc

**Resolves part of:** #35
**Scope:** "Something is broken; what do I look at." Companion to [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) (happy-path deploys).

There is no 24/7 pager rotation. This is the playbook for the operator (usually the maintainer, or a contributor with sudo on Atlas) when a symptom shows up.

---

## Triage flowchart

```
  SYMPTOM
    │
    ├─► Dashboard blank / errors          → §1 Dashboard
    ├─► Chat says "cortex timed out"      → §2 Chat / NRP ELLM
    ├─► Primer not publishing notes        → §3 Primer
    ├─► ARC Scientist stopped              → §4 ARC Scientist
    ├─► CI failing for days                → §5 CI
    ├─► Atlas unreachable (ssh timeout)    → §6 Atlas host
    ├─► NATS leaf disconnected             → §7 Fabric
    └─► Everything is slow / hot           → §8 Thermal / resource
```

---

## 1. Dashboard symptoms

### 1.1 "Dashboard looks empty / missing panels"

1. **Hard-refresh first.** Ctrl+Shift+R / Cmd+Shift+R. Browser cache is the #1 cause of "dashboard looks broken."
2. Compare the footer stamp to git HEAD:
   ```bash
   curl -sk https://atlas-sjsu.duckdns.org/schematic.html | grep -oE "ui:[a-f0-9]+ · [^<]+"
   git -C /path/to/agi-hpc rev-parse --short HEAD
   ```
   If they don't match, the deploy is stale. Go to [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) §1.2.
3. Check the Dashboard Render CI workflow:
   ```bash
   gh run list --repo ahb-sjsu/agi-hpc --workflow "Dashboard Render" --limit 5
   ```
   Recent failures with uploaded Playwright traces pinpoint which widget is broken.

### 1.2 "Specific widget is blank (e.g. NATS Topology, NRP Burst)"

This is the exact class of failure we've hit historically (deploy-drift post-mortem). Check:

```bash
# Is the backend actually returning data for that widget?
curl -sk https://atlas-sjsu.duckdns.org/api/nats-live | jq '.connections | length'     # NATS topo
curl -sk https://atlas-sjsu.duckdns.org/api/nrp-burst | jq '.jobs | length'            # Burst
curl -sk https://atlas-sjsu.duckdns.org/api/erebus/status | jq '.running'              # Erebus card
curl -sk https://atlas-sjsu.duckdns.org/api/primer/status | jq '.running'              # Primer
```

If the API returns good data but the widget is still blank: browser console errors are the next signal. Open devtools, Console tab, look for red.

### 1.3 "Dashboard unreachable (502 / 504 / connection refused)"

```bash
# Is Caddy up?
ssh claude@100.68.134.21 'sudo systemctl is-active atlas-caddy.service'

# Is telemetry up?
ssh claude@100.68.134.21 'sudo systemctl is-active atlas-telemetry.service'
ssh claude@100.68.134.21 'sudo ss -tlnp | grep 8085'
```

Restart whichever is down per [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) §3.

---

## 2. Chat symptoms

### 2.1 "Connection error: Request timed out" in chat

Root cause: NRP ELLM is slow. The chat handler falls through timeouts in cascade (120 s agentic → 90 s simple → error surfaced to user).

Diagnosis:

```bash
ssh claude@100.68.134.21 <<'EOF'
sudo journalctl -u atlas-telemetry.service --since "5 minutes ago" | grep -iE "chat|timeout|retry" | tail -20
EOF
```

Look for patterns:

- `Retrying request to /chat/completions` — NRP is responding but slowly; wait it out or bump timeouts
- `BadRequestError: max context length` — prompt too large; likely a new failure mode, file an issue
- `Connection refused` — NRP ELLM endpoint is actually down; fall back to Atlas local models temporarily

### 2.2 "Chat responded with empty content"

Usually a thinking-model budget issue — the model burned its `max_tokens` on chain-of-thought before emitting `content`. Not ops-actionable; it'll self-correct as the handler retries. If persistent, bump `max_tokens` in `_erebus_chat()` in `scripts/telemetry_server.py`.

### 2.3 "Chat errored mid-conversation" ⇒ telemetry restart mid-chat

Don't do this unless chat is already broken. Restarting `atlas-telemetry.service` drops any in-flight connection and surfaces as `JSON.parse: unexpected end of data` in the browser. See `feedback_no_mid_chat_restarts` in durable memory.

---

## 3. Primer symptoms

### 3.1 "Primer hasn't published a note in hours"

This is often expected, not a bug. Check in order:

1. Is it running?
   ```bash
   ssh claude@100.68.134.21 'sudo systemctl is-active atlas-primer.service'
   ```
2. Is every expert in cooldown?
   ```bash
   ssh claude@100.68.134.21 'cat /archive/neurogolf/primer_health.json | jq ".[] | select(.healthy==false)"'
   ```
   If all three show `healthy: false`, NRP is load-spiked. The Primer is correctly backing off. Wait for `degraded_until_s` to count down.
3. Is the candidate pool empty?
   ```bash
   ssh claude@100.68.134.21 'sudo journalctl -u atlas-primer.service --since "30 minutes ago" | grep "candidate stuck tasks"'
   ```
   `candidate stuck tasks: 0` means either the Scientist isn't generating new stuck tasks or cooldown has masked everything. Normal steady-state behaviour.

### 3.2 "Primer published a note with wrong rule"

By design, this *shouldn't happen* — the validator runs the code against `task.train` and requires 100% pass. If you spot one:

1. Read the note. Check whether the code actually matches the claimed rule.
2. Read `primer_health.json` to see which expert it came from and at what latency.
3. Revert the commit per [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) §4.4.
4. File an issue: likely means the train set didn't exercise the failure, which is *also* true for the Scientist — genuine edge case, not a Primer bug.

### 3.3 "Primer won't re-process a task I want it to retry"

It's in the 6-hour cooldown. See [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) §5.5 for the force-retry procedure.

---

## 4. ARC Scientist symptoms

### 4.1 "Scientist stopped (no recent log lines)"

```bash
ssh claude@100.68.134.21 <<'EOF'
systemctl is-active atlas-scientist.service
systemctl status atlas-scientist.service --no-pager | head -20
journalctl -u atlas-scientist.service -n 30 --no-pager
tail -5 /archive/neurogolf/preflight.log
EOF
```

Common causes:
- `active` but quiet → cycle running a long task, not actually stuck.
- `inactive (dead)` with exit 0 → cycle completed normally. Restart per [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) §5.3.
- `failed` → unit hit the crash-loop limit (`StartLimitBurst=5` in 10 min). Check the journal + preflight log for the root cause, fix it, then `sudo systemctl reset-failed atlas-scientist && sudo systemctl start atlas-scientist`.
- Preflight log shows `ABORT another arc_scientist process is already running` → leftover nohup process. `pkill -TERM -f 'agi/autonomous/arc_scientist.py'`, wait 30 s, retry.
- Preflight shows `ABORT sentinel file present` → operator disabled Erebus intentionally. `rm /archive/neurogolf/.erebus_disabled` only after confirming it's safe to resume.
- Preflight shows `ABORT nvidia-smi hung` → GPU driver wedged. `nvidia-smi` from a shell to confirm; if still wedged, escalate before rebooting.
- OOM killer → check `dmesg | grep -i oom` and the unit's memory peak via `systemctl status`.

### 4.2 "Scientist keeps attempting the same task and failing"

Expected for hard tasks. The thrash-detection logic deprioritizes tasks with ≥15 attempts and no score improvement. If one specific task dominates the queue:

1. Check if the Primer has produced a sensei note for it yet.
2. If not, force the Primer to attempt it (§3.3) or write one by hand.
3. If yes and still failing, the note may be wrong — read it carefully and verify against `task.train` manually.

### 4.3 "Solve count went down"

Impossible — `total_solves` is monotonic (memory is only appended). If you're seeing a drop in the dashboard card, that's a display issue. Raw count is in `/archive/neurogolf/arc_scientist_memory.json`.

---

## 5. CI symptoms

### 5.1 "CI failing on main"

```bash
gh run list --repo ahb-sjsu/agi-hpc --limit 5
gh run view <failed-run-id> --repo ahb-sjsu/agi-hpc --log-failed | tail -40
```

Common failure classes:
- Missing dependency in `pyproject.toml` — add it, commit, push.
- Flaky test (rare on our suite) — re-run via `gh run rerun <id>`.
- Atlas unreachable during deploy step — see §6, then `gh run rerun`.

### 5.2 "Deploy Smoke failing with SHA mismatch"

Live SHA ≠ main HEAD. The deploy didn't land. Causes:
- Atlas SSH credentials wrong (secret rotation) — update `ATLAS_PASSWORD` in GH secrets.
- Atlas unreachable during the deploy window — manual re-deploy per [`DEPLOYMENT_RUNBOOK.md`](DEPLOYMENT_RUNBOOK.md) §2.
- Services restarted but serving cached content — very unlikely with symlinks but possible if someone broke the symlink chain. Check `/home/claude/atlas-chat/*.html` symlinks point at `infra/local/atlas-chat/`.

### 5.3 "Dashboard Render (Playwright) failing"

The failing assertion tells you which widget. Fetch the uploaded trace artifact:

```bash
gh run view <failed-run-id> --repo ahb-sjsu/agi-hpc --log | grep -E "expect|assertion"
```

If this is a new legitimate failure (e.g. version-stamp format changed), update the Playwright assertion in `tests/dashboard/smoke.spec.js`.

---

## 6. Atlas host symptoms

### 6.1 "SSH timeout to 100.68.134.21"

```bash
# From anywhere with Tailscale
tailscale status | grep atlas
tailscale ping atlas

# If Tailscale itself is broken, fall back to LAN
ssh claude@192.168.0.7
```

If both are down, Atlas is offline. Physical access or remote power-cycle (IPMI) is the only recourse.

### 6.2 "Atlas is up but services are down"

```bash
ssh claude@100.68.134.21 'sudo systemctl list-units --state=failed'
ssh claude@100.68.134.21 'sudo systemctl status atlas.target'
```

Restart all Atlas services:

```bash
ssh claude@100.68.134.21 'sudo systemctl restart atlas.target'
# or individually:
ssh claude@100.68.134.21 'sudo systemctl restart atlas-telemetry atlas-rag-server atlas-primer atlas-nats'
```

### 6.3 "Disk filling up on /archive"

```bash
ssh claude@100.68.134.21 'df -h /archive'
ssh claude@100.68.134.21 'du -sh /archive/* 2>/dev/null | sort -hr | head -10'
```

Common culprits:
- `/archive/neurogolf/scientist.log` — grows unbounded during a long cycle. Safe to truncate during a restart window: `truncate -s 0 /archive/neurogolf/scientist.log` after stopping the Scientist.
- Dream consolidation artifacts in `/archive/neurogolf/dreams/` — rotate older ones.
- Model weight caches under `/archive/models/` — inspect before deleting.

### 6.4 "PostgreSQL errors in service logs"

```bash
ssh claude@100.68.134.21 'sudo systemctl status postgresql'
ssh claude@100.68.134.21 'sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"'
```

Connection saturation is the usual cause; restart the service that leaked connections rather than Postgres itself.

---

## 7. Fabric / NATS symptoms

### 7.1 "NATS leaf disconnected"

```bash
ssh claude@100.68.134.21 'sudo systemctl is-active atlas-nats-leaf.service'
ssh claude@100.68.134.21 'curl -s http://localhost:8222/leafz | jq ".leafnodes"'
```

If `leafnodes: 0`, the leaf connection died. Restart:

```bash
ssh claude@100.68.134.21 'sudo systemctl restart atlas-nats-leaf.service'
```

Common cause: NRP hub rotated certificates. Renew via the standard TLS rotation procedure (see `/home/claude/.nats-leaf/` on Atlas for the current cert + `atlas-nats-leaf.service` for the mount path).

### 7.2 "Subject not seeing messages"

Subscriptions are asymmetric over the leaf: hub→spoke works, spoke→hub doesn't auto-propagate for all subjects. See `feedback_nats_leaf_asymmetry` durable memory. Workarounds:
- HTTP webhook bridge (`/api/erebus/result`) for worker → arc_scientist result propagation
- Explicit cross-leaf subscriptions for new subjects you want bi-directional

### 7.3 "JetStream stream corrupt / not replaying"

```bash
ssh claude@100.68.134.21 'curl -s http://localhost:8222/jsz | jq'
```

If stream state looks off, the nuclear option (7-day loss of in-flight events) is `nats stream purge AGI_EVENTS`. Rarely needed.

---

## 8. Thermal / resource symptoms

### 8.1 "CPU temp critical"

```bash
ssh claude@100.68.134.21 'sensors | grep Package'
```

If ≥82 °C for a sustained period:

```bash
# Throttle CPU-bound work
ssh claude@100.68.134.21 'sudo systemctl stop atlas-ego.service'   # Divine Council --parallel 8
ssh claude@100.68.134.21 'tmux kill-session -t <any CPU-heavy session>'
# Thread caps already set at OMP/MKL/OPENBLAS via atlas-operations.md rules
```

### 8.2 "GPU VRAM exhausted"

```bash
ssh claude@100.68.134.21 'nvidia-smi'
```

Kill orphans on GPU 1 while keeping Spock (GPU 0) running:

```bash
ssh claude@100.68.134.21 <<'EOF'
sudo systemctl stop atlas-id.service
kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v $(pgrep -f 'port 8080'))
EOF
```

See [`ATLAS_OPERATIONS.md`](ATLAS_OPERATIONS.md) §"GPU 1 Maintenance" for the full protocol.

### 8.3 "Load average > 24"

```bash
ssh claude@100.68.134.21 'uptime && ps aux --sort=-%cpu | head -10'
```

Runaway Python processes are the most common culprit. Correlate the top offender with a service log. If legitimate work, see §8.1 thermal protocol; if a bug, kill and file an issue.

---

## 9. Escalation

No pager rotation today. If you can't resolve and it's beyond experimentation time:

1. Open a GitHub issue with `type: incident` label.
2. Tag `@ahb-sjsu` (repo owner).
3. Include: symptom, what you've tried, the relevant log excerpts, current state.

For really fast feedback: `https://atlas-sjsu.duckdns.org/schematic.html` shows live state without needing ssh.

---

## 10. Post-incident

After resolving any non-trivial incident:

- [ ] If you learned something durable (e.g. a new failure mode, a root-cause pattern), add it to the relevant durable-memory file (`feedback_*.md` in the operator's personal memory, or the commit message on the fix).
- [ ] If the symptom wasn't covered by this playbook, add a subsection here.
- [ ] If a test would have caught it, open an issue to add that test.
- [ ] If a workflow would have caught it earlier, extend deploy-smoke or dashboard-render.

The playbook grows by post-incident contribution. It's the cheapest form of institutional knowledge.
