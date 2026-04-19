# Metrics, Logs, and Evaluation Inventory

**Resolves:** #36
**Last audit:** 2026-04-19
**Scope:** What Atlas AI / agi-hpc is actually measuring and logging today, where it goes, and who consumes it.

This document is the *descriptive* inventory. For the *prescriptive* side (what we should be measuring, SLOs, conventions), see [`SLOS_AND_KPIS.md`](SLOS_AND_KPIS.md) and [`METRICS_CONTRIBUTOR_GUIDE.md`](METRICS_CONTRIBUTOR_GUIDE.md).

---

## TL;DR

| Category | Where it lives | State |
|---|---|---|
| **System metrics** (CPU, GPU, RAM, disk, net) | `scripts/telemetry_server.py` → `/api/telemetry` + VictoriaMetrics scrape | Comprehensive; 2-year retention |
| **Per-service logs** | `journalctl -u atlas-*.service` | Structured enough to grep, not JSON-formatted |
| **Event bus traffic** | NATS JetStream `AGI_EVENTS` stream, 7-day retention | Every `agi.>` subject captured |
| **Erebus activity** | `/archive/neurogolf/scientist.log` + `/api/erebus/activity` ring buffer | Tail-based; line classifier categorizes by event kind |
| **Competition scoring** | `/archive/neurogolf/arc_scientist_memory.json` + `/api/erebus/status` | Per-task attempt history, strategy win rates, solve counts |
| **Primer health** | `/archive/neurogolf/primer_health.json` + `/api/primer/status` | Rolling per-expert latency window |
| **CI health** | GitHub Actions across 3 workflows | Deploy-smoke + dashboard-render runs every 30 min |
| **Explicit application metrics** | `scripts/telemetry_server.py` _get_system_deep | Ad-hoc collection, no Prometheus client |
| **Structured tracing** | — | Not implemented |
| **OpenTelemetry / Prometheus instrumentation** | — | Not implemented |

Overall: **Atlas has a lot of *signals* being captured.** The gap is that they're served by a heterogeneous set of endpoints rather than a single standards-based observability stack. Most are human-readable; few are machine-scrapeable via a standard protocol.

---

## 1. Hardware & system metrics

### Collector — `scripts/telemetry_server.py` on port 8085

The telemetry server scrapes local system state every request to `/api/telemetry`. It shells out to:

| Source | Metric coverage |
|---|---|
| `nvidia-smi --query-gpu=...` | per-GPU: name, temp, util%, VRAM used, VRAM total |
| `sensors` | CPU package temperatures |
| `free -b` | RAM total / used / free / buffers / cache / available + percentages |
| `/proc/sys/fs/file-nr` | File-descriptor allocation + max |
| `ss -s` | TCP / UDP socket counts (total / established / closed) |
| `/proc/loadavg` | 1m / 5m / 15m load, running threads |
| `/proc/stat` | Context switches, total forks |
| `df -BG /` `/mnt/raid5` | Disk total / used / available + % |
| `/proc/net/dev` | Net RX / TX GB since boot |
| `/proc/uptime` | Uptime days / hours |
| `/proc` listing | Process / thread count |
| `ps aux \| grep -c Z` | Zombie count |

### Storage — VictoriaMetrics

`atlas-victoriametrics.service` at `:8428`, 2-year retention (`-retentionPeriod=2y`). Scrapes telemetry and exposes Prometheus-compatible query endpoints.

### Frontend — dashboard at `atlas-sjsu.duckdns.org/schematic.html`

Reads `/api/telemetry` + `/api/events` + `/api/nrp-burst` every 5 s (system) / 15 s (events) / 30 s (NRP), renders:

- GPU gauges (temp, VRAM, util)
- CPU temperature + mini-spark (load, tcp, fds, context switches, netrx, nettx)
- Memory tiers panel
- NATS topology SVG
- NATS Live (wireshark-style message stream)
- NRP Burst Jobs + Worker Pools card
- Erebus — NeuroGolf 2026 card

All of the above is described in [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md).

---

## 1b. Structured lifecycle events (2026-04-19 onwards)

Shipped in #38 as the foundation for #78 (full JSON-log migration). Runs in **parallel** to the pretty-printed logs in §2 — doesn't replace them.

### Emitters

- `arc_scientist.py` — emits `cycle_start`, `attempt_end` (with outcome, correct, total, error_type), `cycle_end` via `LifecycleLogger("scientist")`.
- `agi.primer.service` — emits `tick_start`, `tick_empty`, `ensemble_complete`, `publish`, `verify_fail`, `tick_complete` via `LifecycleLogger("primer")`.
- Other services can opt in by constructing a `LifecycleLogger("<name>")`.

### Storage

- One JSONL file per subsystem under `/archive/neurogolf/lifecycle/`:
  - `/archive/neurogolf/lifecycle/scientist.jsonl`
  - `/archive/neurogolf/lifecycle/primer.jsonl`
- Each line is one event: `{ts, subsystem, event, seq, ...}` with monotonic `seq` that survives process restarts.
- Thread-safe (`threading.Lock` on file-write).

### API

- `GET /api/jobs/recent?subsystem=scientist&limit=100` — tails the JSONL, returns `{events: [...]}` newest-first.
- Default `limit=100`, cap 500.
- Cheap: reads only the tail of the file.

### Frontend

- Not yet surfaced on the dashboard — a "Recent lifecycle events" panel is the obvious next step (planned follow-up to #38).

## 2. Service logs (systemd → journalctl)

All Atlas services log to journald:

```bash
sudo journalctl -u atlas-telemetry.service -f
sudo journalctl -u atlas-primer.service -f
sudo journalctl -u atlas-nats.service -f
# etc.
```

### Log format

Most services log via Python's `logging` module with the default format:

    2026-04-19 05:32:05,684 [INFO] primer: Primer online. ...

- **Timestamp** — local time.
- **Level** — `INFO` / `WARNING` / `ERROR`.
- **Logger name** — `primer`, `telemetry`, `council`, etc.
- **Message** — free text.

**Not JSON.** Grep-able, not easily machine-parseable. Categorical fields (e.g. "task_id", "expert") are interpolated into the message text rather than structured.

### Service-specific logs

| Service | Log notes |
|---|---|
| `atlas-telemetry.service` | HTTP access log + `[INFO] "GET /api/..."` per request; also embeds curriculum / NRP-watchdog / etc. sub-system lines |
| `atlas-primer.service` | One line per tick + per-expert call outcome; raw-peek on verify-fail |
| `atlas-rag-server.service` | Request log + search query traces (Tier 0–4 pick) |
| `atlas-nats.service` | Standard NATS server log (config, subs, leaf links) |
| `atlas-dreaming.service` | Consolidation steps, wiki writes |
| `atlas-watchdog.service` | Service health probes + restart decisions |

### Accessory log files

| Path | Written by | Purpose |
|---|---|---|
| `/archive/neurogolf/scientist.log` | `arc_scientist.py` via nohup | Full attempt trace + per-cycle strategy report |
| `/archive/neurogolf/dreaming_schedule.log` | `dreaming_schedule.py` | Idle-dream consolidation trace |
| `/archive/neurogolf/onnx_scientist.log` | `onnx_scientist.py` | ONNX direct-solver attempts |
| `/tmp/atlas/rag_server.log` | fallback tmux RAG server | When systemd unit unavailable |
| `/tmp/atlas/telemetry.log` | fallback tmux telemetry | When systemd unit unavailable |

### Log rotation

Not currently configured explicitly; journald applies its defaults (disk-limit based). The `/archive/neurogolf/*.log` files grow unbounded until `arc_scientist.py` restarts (truncates `>` if launched with `>>`). Should be hardened — see [`SLOS_AND_KPIS.md`](SLOS_AND_KPIS.md) §"Operational maturity".

---

## 3. NATS event bus

### Stream config — `AGI_EVENTS`

Configured in `src/agi/core/events/nats_fabric.py` and the NATS server settings:

- Subjects: `agi.>` (wildcard — every agi.* subject captured)
- Max size: 1 GB
- Retention: 7 days
- Storage: file-backed JetStream

### Subject hierarchy

From [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md):

| Subject prefix | Owner |
|---|---|
| `agi.rh.*` | Right hemisphere (Ego slot) |
| `agi.lh.*` | Left hemisphere |
| `agi.safety.*` | Safety gateway |
| `agi.ego.deliberate` | Divine Council |
| `agi.meta.monitor.*` | Metacognition monitor |
| `agi.memory.*` | Memory broker |
| `agi.dreaming.*` | Dreaming consolidator |
| `agi.autonomous.*` | ARC Scientist |
| `agi.primer.teach` | The Primer (synthetic subject, not yet published) |
| `agi.integration.*` | Integration orchestrator |
| `agi.dht.*` | DHT registry |

### Monitoring

- **HTTP monitoring** at `:8222` — `/varz`, `/connz`, `/subsz`, `/leafz`, `/jsz`.
- **Dashboard NATS Live panel** — reads `/api/nats-live` which tails the latest N messages and the leaf-node RTT.

---

## 4. Erebus / ARC Scientist observability

### Memory file — `/archive/neurogolf/arc_scientist_memory.json`

Written by `arc_scientist.py` after every attempt. Schema:

```jsonc
{
  "total_attempts": 1364,
  "total_solves": 100,
  "tasks": {
    "<task_num>": {
      "solved": bool,
      "attempts": [ {timestamp, strategy, model, correct, total, code, error, error_type}, ... ],
      "strategies_tried": [...],
      "best_correct": int,
      "best_total": int,
      "error_types": [...],
      "failure_patterns": [...]
    }
  },
  "strategies": {
    "<name>": {"attempts": int, "successes": int, "name": "..."}
  },
  "global_insights": [...],
  "last_updated": ISO-8601
}
```

### Help queue — `/archive/neurogolf/erebus_help_queue.json`

Tasks the Scientist has flagged as stuck. Consumed by The Primer.

### Vision pool

K8s Jobs created by `scripts/render_erebus_vision_job.py`. Dashboard `/api/primer/status` and `/api/nrp-burst` both inspect `kubectl get jobs` for labels `app=erebus-vision`.

### API endpoints

| Endpoint | Payload | Consumer |
|---|---|---|
| `/api/erebus/status` | `{running, cycle, attempt, current, memory_summary, recent_solves, vision_pool, help_queue_count, this_cycle}` | Erebus NeuroGolf 2026 dashboard card |
| `/api/erebus/activity?since=N&limit=N` | Ring buffer of recent log lines classified by kind (attempt / solved / help / meta / dream / error / info) | Erebus chat UI sidebar |
| `/api/erebus/help` | Help queue JSON | Primer, dashboard |
| `/api/erebus/memory` | Full memory file passthrough | Chat handler for episodic context |
| `/api/erebus/result` | POST — burst worker → arc_scientist memory | Vision pool, future NRP workers |

---

## 5. The Primer observability

### Cooldown state — `/archive/neurogolf/primer_cooldown.json`

`{task_num_str: unix_ts}` — last time each task was attempted by the Primer, regardless of outcome. Enforces `PRIMER_COOLDOWN_S` (default 6 h).

### Health state — `/archive/neurogolf/primer_health.json`

Written each tick by `_save_health(moe)`. Per-expert:

```jsonc
{
  "kimi": {
    "healthy": true,
    "degraded_until_s": 0,
    "window_size": 5,
    "unhealthy_in_window": 1,
    "avg_latency_s": 112.4
  },
  // ...
}
```

### API — `/api/primer/status`

Exposes process status (`pgrep -f agi.primer.service`), `tasks_touched`, `last_touched_task`, `last_touched_age_s`, plus the full `expert_health` summary. Consumed by the NATS-topology synthetic Primer node (green/red live indicator) and, future, a Primer dashboard card.

### Git commits as audit trail

Every verified sensei note becomes a git commit on `main`:

    primer: verified sensei note for task 020 (symmetry-completion)

Author `agi.hpc@gmail.com`. The commit log IS the Primer's public output log.

---

## 6. CI / CD observability

Three GitHub Actions workflows. Each has its own job timeline visible at `actions.github.com/.../agi-hpc/actions`.

| Workflow | What it reports |
|---|---|
| `ci.yaml` — Atlas AI — CI/CD | Lint result, test results, auto-deploy + smoke result |
| `deploy-smoke.yaml` — Deploy Smoke | Live SHA vs main HEAD, per-widget presence on rendered page |
| `dashboard-render.yaml` — Dashboard Render | Playwright: topology populated, burst rows present, stamp format, no console errors |

### Version stamp — `/api/version`

Returns `{sha, stamp, repo_dir, static_dir}` where `sha` is `git rev-parse --short HEAD` of the repo at serve time. The dynamic UI stamp in `schematic.html` footer (`ui:<sha> · <mtime>`) comes from the same source. This is the ground-truth "which commit is live" signal.

---

## 7. Testing & evaluation artifacts

### Unit tests — `tests/unit/`

| File | Covers |
|---|---|
| `test_primer_vmoe.py` | vMOE router / cascade / ensemble / first_verified |
| `test_primer_validator.py` | Sandboxed code verification against train examples |
| `test_primer_health.py` | Expert health tracker degradation + cooldown |
| `test_dashboard_panels.py` | Dashboard HTML structure + element IDs |
| `test_cascade.py` | Cascade classifier |
| `test_attention_filter.py` | Distractor detection |
| `test_creative_dreaming.py` | Dreaming synthesis |
| `test_curriculum.py`, `test_curriculum_planner.py` | Curriculum manager |
| `test_daily_training.py` | Training scheduler |
| `test_cicd.py` | CI config sanity |

Total roughly 700+ unit tests; run in CI via the `ci.yaml` `Unit tests` step.

### End-to-end tests — `tests/dashboard/`

Playwright-based Chromium render test. Assertions documented in [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) §"CI / CD".

### Benchmarks — `benchmarks/`

- `attention/` — attention mechanism benchmarks
- `social_cognition/` — Kaggle-style evaluation suite with writeup

Run ad-hoc; not currently wired into CI.

### Per-attempt evaluation (built into arc_scientist)

Every generated `transform()` is scored against task.train examples via the same sandbox the Primer validator uses. Results populate `arc_scientist_memory.json`. This is the real-time evaluation loop; reasoning about quality trends happens by reading the memory file or the log.

---

## 8. What's *not* currently measured

Honest list of gaps, for planning purposes:

- **End-to-end request latency percentiles** — no p50/p90/p99 aggregation. VictoriaMetrics has the raw signals but no pre-computed SLIs.
- **Per-subsystem error rates** — only the raw log count of `ERROR`-level messages; no structured error-rate panel.
- **Service-level dependencies** — no explicit dependency tracking (X depends on Y being up).
- **Request tracing (OpenTelemetry spans)** — not instrumented. A chat request touching telemetry → RAG → vMOE → NATS is not traceable as a single span tree.
- **Business-level KPIs** — solve-rate trend over time *is* implicitly tracked (memory file), but there's no visualization of the time-series.
- **Cost metrics** — NRP is free, but NRP ELLM throughput-per-token would still be useful for capacity planning.
- **Alerting** — the only alert path is systemd restarting crashed services; no PagerDuty / email / Slack hooks.
- **Structured JSON logs** — would enable much better filtering and aggregation in a log-index backend.

These gaps inform the prescriptive docs: [`SLOS_AND_KPIS.md`](SLOS_AND_KPIS.md) proposes the first set of success metrics, and [`METRICS_CONTRIBUTOR_GUIDE.md`](METRICS_CONTRIBUTOR_GUIDE.md) describes conventions for filling them in.
