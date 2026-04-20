# Atlas AI — Architecture Overview

This document describes the production architecture of Atlas AI as deployed at SJSU: a psychoanalytic multi-agent cognitive architecture coordinated by NATS JetStream, backed by PostgreSQL + pgvector semantic memory, with frontier LLM access via NRP Nautilus and an always-on Claude-style teaching layer (The Primer) that auto-generates verified reference implementations for the autonomous ARC Scientist.

---

## System overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         ATLAS AI — PRODUCTION DEPLOYMENT                                │
│                         HP Z840 · 2× GV100 32 GB · 251 GB RAM                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌───────────────────────────────┐   │
│  │ EGO (Kirk)          │  │ SUPEREGO (Spock)     │  │ DIVINE COUNCIL                │   │
│  │ Balanced decision-  │  │ Logic, rules,        │  │ 7 advocate agents (CPU)       │   │
│  │ maker; the self     │  │ ethical evaluation   │  │                               │   │
│  │                     │  │                      │  │ Judge · Advocate               │   │
│  │ Qwen 3 32B Q5_K_M   │  │ Gemma 4 31B Q5_K_M   │  │ Synthesizer · Ethicist        │   │
│  │ GPU 1 · :8082       │  │ GPU 0 · :8080        │  │ Historian · Futurist          │   │
│  │ (local server)      │  │                      │  │ Pragmatist                     │   │
│  │                     │  │                      │  │                               │   │
│  │ + NRP fallback:     │  │ + NRP fallback:      │  │ Kimi K2.5 1T on NRP           │   │
│  │ Qwen 3.5 397B       │  │ Gemma 4              │  │ + atlas-ego.service --par 8   │   │
│  │                     │  │                      │  │                               │   │
│  │ NATS: agi.rh.*      │  │ NATS: agi.safety.*   │  │ NATS: agi.ego.deliberate      │   │
│  └─────────┬───────────┘  └──────────┬───────────┘  └───────────────┬───────────────┘   │
│            │                         │                              │                    │
│  ┌─────────▼─────────────────────────▼──────────────────────────────▼───────────────┐   │
│  │                  NATS JETSTREAM EVENT FABRIC                                      │   │
│  │                  :4222 (local) · :7422 (leaf to NRP, TLS + compression)           │   │
│  │                                                                                   │   │
│  │   Subjects: agi.rh.* · agi.safety.* · agi.ego.deliberate                          │   │
│  │             agi.meta.monitor.* · agi.memory.* · agi.dreaming.*                    │   │
│  │             agi.autonomous.* · agi.primer.teach                                   │   │
│  └──┬────────────────────┬────────────────────┬──────────────────────┬───────────────┘   │
│     │                    │                    │                      │                    │
│  ┌──▼─────────────┐ ┌────▼───────────┐ ┌──────▼──────────────┐ ┌─────▼─────────────┐   │
│  │ AUTONOMOUS     │ │ SAFETY         │ │ MEMORY              │ │ SUPPORT            │   │
│  │                │ │                │ │                     │ │                    │   │
│  │ ARC Scientist  │ │ 3-layer:       │ │ PostgreSQL +        │ │ RAG server (:8081) │   │
│  │  closed loop   │ │  Reflex <100µs │ │  pgvector           │ │ Telemetry (:8085)  │   │
│  │  mentor        │ │  Tactical 10ms │ │                     │ │ Caddy + OAuth2     │   │
│  │  preamble      │ │  Strategic pol │ │ 3.3 M PCA-384       │ │ Thermal guardian   │   │
│  │                │ │                │ │  IVFFlat vectors    │ │ Watchdog           │   │
│  │ The Primer     │ │ ErisML:        │ │                     │ │ Operations dash    │   │
│  │  vMOE tutor    │ │  Bond Index    │ │ 5-tier retrieval    │ │  /schematic.html   │   │
│  │  verify-only   │ │  Hohfeld       │ │  L0 Dream           │ │                    │   │
│  │  publish       │ │  hash chains   │ │  L1 Sensei wiki     │ │ Dreaming (idle     │   │
│  │                │ │                │ │  L2 Vector          │ │  consolidation)    │   │
│  │ Vision burst   │ │ Input/Output/  │ │  L3 FTS             │ │                    │   │
│  │  GLM-4.1V on   │ │  Privilege     │ │  L4 Episodic        │ │ Backup (daily)     │   │
│  │  NRP Jobs      │ │  Gates         │ │                     │ │                    │   │
│  └────────────────┘ └────────────────┘ └─────────────────────┘ └────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## NATS topology and NRP burst

```
                        ┌────────────────────────────────────────┐
                        │      NRP NAUTILUS                      │
                        │      (100+ university nodes)           │
                        │                                        │
                        │  Managed LLM API (https://ellm.nrp-    │
                        │  nautilus.io/v1) — OpenAI-compatible   │
                        │  and /anthropic proxy:                 │
                        │    Kimi K2.5 1T                        │
                        │    Qwen 3.5 397B                       │
                        │    GLM-4.7 358B                        │
                        │    MiniMax M2.7                        │
                        │    Gemma 4                             │
                        │    (zero marginal cost, shared pool)   │
                        │                                        │
                        │  Burst Jobs (ephemeral K8s Jobs):      │
                        │    GLM-4.1V vision (4× L40/L40S/A10)   │
                        │                                        │
                        │  Worker Pools (Deployments):           │
                        │    erebus-workers (8× nats-bursting)   │
                        │                                        │
                        │  Storage: CephFS PVC erebus-ego-models │
                        │           300 Gi, rook-cephfs class    │
                        │                                        │
                        │  NATS leaf node :7422 (TLS, s2_fast)   │
                        └───────────────────┬────────────────────┘
                                            │
                                   Tailscale VPN mesh
                                   100.68.134.21
                                            │
                        ┌───────────────────▼────────────────────┐
                        │        ATLAS WORKSTATION                │
                        │        HP Z840 · SJSU                   │
                        │                                         │
                        │  NATS JetStream :4222                   │
                        │                                         │
                        │  ┌────────────────────────────────┐    │
                        │  │ GPU 0 (Superego): Gemma 4 31B   │    │
                        │  │   llama.cpp · :8080              │    │
                        │  ├────────────────────────────────┤    │
                        │  │ GPU 1 (Ego): Qwen 3 32B         │    │
                        │  │   llama.cpp · :8082              │    │
                        │  ├────────────────────────────────┤    │
                        │  │ CPU (Council): Gemma 4 26B-A4B  │    │
                        │  │   llama.cpp · :8084 ·  --par 8   │    │
                        │  ├────────────────────────────────┤    │
                        │  │ PostgreSQL + pgvector           │    │
                        │  │ RAG Server · :8081              │    │
                        │  │ Telemetry · :8085               │    │
                        │  ├────────────────────────────────┤    │
                        │  │ atlas-primer.service            │    │
                        │  │   (CPU-only, vMOE tutor)        │    │
                        │  │ arc_scientist.py                │    │
                        │  │   (ARC closed loop)             │    │
                        │  └────────────────────────────────┘    │
                        │                                         │
                        │  atlas-sjsu.duckdns.org                │
                        │  Caddy → oauth2-proxy → service        │
                        └─────────────────────────────────────────┘
```

---

## Core components

### Freudian agents

The system implements a psychoanalytic decision model. Star Trek analogues show the functional mapping:

**Ego (Kirk)** — balanced decision-maker, the self that speaks and learns. Qwen 3 32B on GPU 1 (`atlas-id.service`, historical name); fallback to Qwen 3.5 397B on NRP when local GPU 1 is busy or fine-tuning. Subject: `agi.rh.request.>`.

**Superego (Spock)** — logic, rules, ethical evaluation. Gemma 4 31B on GPU 0 (`atlas-superego.service`); NRP Gemma 4 fallback. Subject: `agi.safety.*`.

**Divine Council** — multi-perspective deliberation. Seven concurrent advocate roles (Judge, Advocate, Synthesizer, Ethicist, Historian, Futurist, Pragmatist) run on Kimi K2.5 1T via NRP, with an `atlas-ego.service --parallel 8` CPU fallback. Subject: `agi.ego.deliberate`.

**The Id** — currently unfilled at the LLM slot. Fast instinct / pattern-match is served by subcortical local-GPU procedural memory (pattern nets + A* search) rather than a dedicated language model.

Location: `src/agi/reasoning/`

| File | Purpose |
|---|---|
| `divine_council.py` | Council orchestration and debate protocol |
| `_council_backend.py` | LLM backend integration for council agents |
| `_council_metrics.py` | Council performance + reliability |
| `tree_of_thought.py` | Tree-of-Thought reasoning for structured debate |
| `nats_service.py` | NATS pub/sub for council coordination |

### Autonomous learning — ARC Scientist

Closed-loop scientific reasoning against the NeuroGolf 2026 / ARC-AGI task set. Location: `src/agi/autonomous/`.

```
  OBSERVE ─► HYPOTHESIZE ─► EXPERIMENT ─► EVALUATE ─► REFLECT ─► ADAPT ─► LEARN ─┐
                                                                                  │
                                  ◄───────────────────────────────────────────────┘
```

| Stage | Mechanism |
|---|---|
| Observe | Tier-weighted task pick (similarity to solved / near-miss exploitation / exploration) |
| Hypothesize | LLM prompt in one of 5 strategies: `direct`, `failure_aware`, `example_chain`, `diagnostic`, `primitives_guided` |
| Experiment | Generate candidate Python `transform(grid)` |
| Evaluate | Verify against **every** training example; any mismatch rejects the candidate |
| Learn | Store attempt with structured error classification (`reasoning` / `execution` / `perception` / `specification`) |
| Reflect | LLM diagnoses *why* a transform failed; result becomes context for the next attempt |
| Adapt | Thompson sampling shifts strategy weights by Bayesian evidence |

**Mentor preamble** — if `wiki/sensei_task_NNN.md` exists, its YAML-stripped body is prepended to every prompt for that task. Notes are either human-written or Primer-generated.

**Vision burst** — when ≥ 10 perception-error tasks accumulate, `arc_scientist._dispatch_vision_burst()` fires a K8s Job with 4 GLM-4.1V-9B-Thinking vision pods on NRP (L40 / L40S / A10 nodeAffinity). Results write back via `/api/erebus/result` webhook.

### The Primer — auto-sensei

Always-on Claude-style tutor. Watches the Scientist's help queue, uses a **virtual MoE ensemble** of frontier NRP models to propose verified reference implementations for stuck tasks, publishes them as wiki articles. Design principle: frontier reasoning + project memory + verify-before-publish.

- Location: `src/agi/primer/` — `vmoe.py`, `service.py`, `validator.py`, `health.py`.
- Systemd: `atlas-primer.service` on Atlas (CPU-only, `MemoryMax=4G`).
- Poll: 10 min. Tier-1 (partial progress) > Tier-2 (zero progress), most-attempted first.
- Safety invariant: **only code that passes 100 % of `task.train` is published.** See [`THE_PRIMER.md`](THE_PRIMER.md).
- Orchestration substrate: [`VMOE.md`](VMOE.md).

### Left Hemisphere — deliberative

Planning, reasoning, metacognition. Location: `src/agi/lh/`.

| Component | Purpose |
|---|---|
| Planner | Hierarchical planning with `PlanGraph` / `PlanStep` |
| Plan Service | gRPC orchestration |
| Memory Client | Interface to memory services for context |
| Safety Client | Interface to Safety Gateway |
| Metacognition | Self-monitoring + plan review |
| Performance | LRU cache with TTL, async operation batcher |
| HPC Deploy | Slurm launcher, Apptainer runner |
| Resilience | Circuit breaker, retry, graceful degradation |

### Right Hemisphere — reactive

Perception, world model, motor control. Location: `src/agi/rh/`.

| Component | Directory | Purpose |
|---|---|---|
| Perception | `perception/` | Sensor fusion, object recognition |
| World Model | `world_model/` | Physics simulation, state prediction |
| Control | `control/` | Motor primitives, trajectory, realtime |

### Safety subsystem

Location: `src/agi/safety/`. Three-layer architecture:

| Layer | Latency | Function |
|---|---|---|
| Reflex | < 100 µs | Emergency stops, thermal limits, PII / prompt-injection gates |
| Tactical | 10–100 ms | ErisML `MoralVector`, Bond Index, Hohfeldian rights analysis |
| Strategic | > 100 ms | Policy enforcement, SHA-256 hash-chained decision proofs, human oversight |

### Memory subsystem

Location: `src/agi/memory/`. 5-tier retrieval over PostgreSQL + pgvector + sensei wiki:

| Tier | Source | Latency | Typical use |
|---|---|---|---|
| L0 | Dream-consolidated summaries | < 1 ms | Idle-time synthesized insights |
| L1 | Sensei wiki (`wiki/sensei_*.md`) | < 1 ms | Human + Primer-written teaching articles |
| L2 | PCA-384 IVFFlat vectors (3.3 M) | 1–5 ms | Semantic similarity |
| L3 | tsvector full-text search | 1–10 ms | Keyword / phrase queries |
| L4 | Episodic memory (per-session) | 5–20 ms | Interaction history, debate traces |

**Primer writes to L1.** Every verified sensei note becomes Tier-1 knowledge for RAG and for the Scientist's mentor-preamble mechanism.

### Metacognition + dreaming

Location: `src/agi/metacognition/` + `src/agi/dreaming/`.

- **Monitor** — latency percentiles, hemisphere ratio, veto rate, throughput.
- **Reflector** — periodic self-assessment every 10 interactions.
- **Adjuster** — auto-tunes `max_tokens`, safety thresholds, routing balance.
- **Dreaming consolidator** — runs during idle windows (not fixed 2–4 am). QLoRA fine-tuning produces adapters on `/archive/neurogolf/adapters`, cached on NRP PVC. Currently blocked on chat-ego cutover for a fine-tunable base model (see [`AGI_ROADMAP.md`](AGI_ROADMAP.md) Phase 2–3).

### Knowledge Gap Mapping (planned)

Roadmap item 2.3 — v1 spec locked, not yet implemented. Detects user dissatisfaction at conversation-finalization time, clusters failures by topic, and feeds the dreaming prioritizer. Storage is the existing Unified Knowledge Graph (`type=gap`, `source="dissatisfaction"`) plus a sidecar `/archive/neurogolf/dissatisfaction_events.jsonl` for raw audit. See [`KNOWLEDGE_GAP_MAPPING_v1_spec.md`](KNOWLEDGE_GAP_MAPPING_v1_spec.md) for the full design, locked decisions, and 7-phase delivery plan.

---

## Infrastructure

### Fabric

- **NATS JetStream** — global workspace at `:4222`, monitoring at `:8222`. `AGI_EVENTS` stream with `agi.>` wildcard, 1 GB max, 7-day retention. Leaf node at `:7422` bridges to NRP (TLS, `s2_fast` compression, 24 ms RTT).
- **Event dataclass** — `src/agi/common/event.py` with serialization.
- **Subject hierarchy** — 10 subsystems: `rh`, `safety`, `ego`, `meta`, `memory`, `dreaming`, `autonomous`, `primer`, `dht`, `integration`.

### Storage

- **PostgreSQL 15 + pgvector** — 3.3 M vectors in IVFFlat index.
- **SQLite** — procedural memory.
- **CephFS PVC** `erebus-ego-models` on NRP — 300 Gi, `rook-cephfs` class, RWX. Sized for a future self-hosted ego pod + LoRA adapter artifacts from dreaming.
- **`/archive`** — 15 TB local: episodic memory JSON, task JSONs, solution artifacts, scientist log.

### Service stack (systemd, `atlas.target`)

| Service | Purpose |
|---|---|
| `atlas-nats.service` | JetStream broker |
| `atlas-nats-leaf.service` | Leaf connection to NRP |
| `atlas-telemetry.service` | `/api/*` endpoints + dashboard static serving |
| `atlas-rag-server.service` | RAG search + LLM proxy |
| `atlas-id.service` | Ego backend (Qwen 32B on GPU 1) — historical service name, functional role is Ego |
| `atlas-superego.service` | Superego backend (Gemma 31B on GPU 0) |
| `atlas-ego.service` | Divine Council CPU backend |
| `atlas-dreaming.service` | Idle consolidation |
| `atlas-metacognition.service` | Monitor / reflector / adjuster |
| `atlas-memory.service` | Memory NATS broker |
| `atlas-safety.service` | DEME safety NATS |
| `atlas-victoriametrics.service` | Time-series metrics backend |
| **`atlas-primer.service`** | The Primer tutor daemon (new 2026-04-19) |
| `atlas-thermal.service` | Thermal guardian |
| `atlas-watchdog.service` | Health monitor |
| `atlas-backup.timer` | Nightly backup |
| `atlas-training.timer` | Training window (midnight–8 am) |
| `research-portal.service` | Atlas portal dashboard |
| `atlas-oauth2-proxy.service` | Google OAuth |
| `atlas-caddy.service` | Reverse proxy + HTTPS |

### Dashboard

- URL: `https://atlas-sjsu.duckdns.org/schematic.html`
- Served by `atlas-telemetry.service` on `:8085`, reverse-proxied by `atlas-caddy.service`.
- Live panels: NATS topology (with synthetic Ego/Superego/Council/Scientist/**Primer** nodes), NATS Live (wireshark-style message stream), NRP Burst Jobs + Worker Pools (Jobs and Deployments combined), Erebus — NeuroGolf 2026 (competition score, strategy bars, vision pool, help queue, recent solves), GPU gauges, CPU temperature, memory tiers.
- **Dynamic version stamp** — footer shows `ui:<git short sha> · <file mtime UTC>`. `telemetry_server.py` substitutes `{{UI_VERSION}}` at serve time. Exposed as JSON via `/api/version`.

---

## CI / CD

Three workflows at `.github/workflows/`:

| Workflow | Trigger | What it catches |
|---|---|---|
| **Atlas AI — CI/CD** (`ci.yaml`) | push, PR | Lint (ruff), unit tests, build; on main push also SSHes to Atlas and runs `git pull` + restarts telemetry / RAG. |
| **Deploy Smoke** (`deploy-smoke.yaml`) | push to main + 30-min cron | Compares live `/api/version.sha` to `git rev-parse HEAD`. Asserts key widgets are present on the rendered page. |
| **Dashboard Render** (`dashboard-render.yaml`) | dashboard code changes + 30-min cron | Playwright headless Chromium: loads page, asserts topology SVG populated, burst table has rows, version stamp format matches, no console errors. |

Both drift-detector workflows are `continue-on-error: true` — they surface state, they don't gate PRs.

### Deploy-drift history

Before 2026-04-19, the CI deploy step ran `cp -f atlas-chat-*.html $STATIC_PATH/*.html`. The destination was a symlink into the git tree, so `cp` followed the symlink and wrote stale content back into the working tree after every push. Sixteen commits of dashboard features were silently reverted. Fix: `ln -sfn`. Post-mortem + durable memory entry at `feedback_dashboard_deploy_drift`.

---

## NRP Nautilus integration

### Accounts + policy

- **Namespace:** `ssu-atlas-ai`
- **Hard ResourceQuota:** `a100-limit` = 0 (A100 gated behind reservation form; we default to A10 × 8, abundantly available)
- **Burst pod limit:** 4 heavy-GPU Pods per namespace (Deployments + Jobs combined, per `nautilus.io/hardware=large-gpu` taint-tolerating pods)
- **Policy:** GPU pods must sustain > 40 % utilization; interactive-only pods violate this (motivates Job-shape with idle-exit for bursts, continuous-batched serving for persistent pods)

### Hardware availability (2026-04-19)

| GPU type | Per node | Nodes | Accessible to `ssu-atlas-ai`? |
|---|---|---|---|
| A10 | 8 | 33 | Yes — abundant |
| A40 | 8 | 1 | Yes |
| L40 | 4 | 17 | **No** — 15 carry `csu-tide` reservation taint; rest have no free GPUs |
| L40S | 4 | 3 | Partial — depends on current load |
| A100 80 GB | 4 or 8 | 21 | **No** — namespace quota = 0 |
| H100 / H200 | varies | 6 | No — reservation required |
| V100 32 GB | 8 | 7 | Yes |
| A6000 | 4 or 8 | 6 | Yes |

### Storage

- **CephFS** (`rook-cephfs` class) — recommended for large model weights. 16 TB per-file ceiling, parallel-read throughput.
- **RBD** (`rook-ceph-block` default) — higher IOPS for small-files workloads.
- **S3** (NRP Ceph S3 endpoints) — for cross-region, > 5 TiB single-file cases.

### Managed LLM service

- `https://ellm.nrp-nautilus.io/v1` — OpenAI-compatible. Models: `kimi`, `qwen3`, `qwen3-27b`, `glm-4.7`, `gpt-oss`, `gemma`, `gemma-4-e4b`, `minimax-m2`, `olmo`, `qwen3-embedding`, `qwen3-small`.
- `https://ellm.nrp-nautilus.io/anthropic` — **protocol-translation proxy** (not Claude). Lets Claude Code tooling talk to the open-model pool via the Anthropic messages API.
- Auth: Bearer token from `~/.llmtoken` (issued via `/llmtoken`).

---

## Regression guards summary

| Failure mode | Caught by |
|---|---|
| Python bug in Atlas service | `ci.yaml` unit tests, ruff |
| Deployed code doesn't match `main` | `deploy-smoke.yaml` (compare live SHA) |
| HTML renders but JS breaks | `dashboard-render.yaml` (Playwright) |
| Wiki note publishes wrong rule | Primer validator (subprocess run-against-train; `all_pass=True` required) |
| Expert timeout cascades | vMOE `HealthTracker` degradation + cooldown |
| Working tree drift on deploy | `ln -sfn` replaces `cp -f`; symlinks are idempotent |
| Chat handler hangs on slow NRP | 120 s agentic / 90 s simple timeouts; eventual vMOE cascade (Phase 3) |

---

## Documentation map

| Doc | Topic |
|---|---|
| [`../README.md`](../README.md) | Front door, hardware, quickstart |
| [`THE_PRIMER.md`](THE_PRIMER.md) | The Primer — teaching daemon, safety invariants, operation |
| [`VMOE.md`](VMOE.md) | Virtual Mixture-of-Experts — routing, cascade, ensemble, health |
| [`ATLAS_OPERATIONS.md`](ATLAS_OPERATIONS.md) | Workstation ops: systemd, thermal, backup, SSH |
| [`AGI_ROADMAP.md`](AGI_ROADMAP.md) | Phase status, pending work, long-horizon plan |
| [`CHANGELOG.md`](CHANGELOG.md) | Per-date ship log |
| [`HPC_DEPLOYMENT.md`](HPC_DEPLOYMENT.md) | NRP / Slurm / Apptainer deployment |
| [`API_REFERENCE.md`](API_REFERENCE.md) | gRPC + NATS subject reference |
| [`ERISML_API.md`](ERISML_API.md) | Ethical reasoning API |

---

## Design principles

- **Cortex = frontier LLMs (NRP), subcortical = local GPUs.** Reasoning goes cloud-ward, pattern / procedural memory stays on metal.
- **Global workspace = NATS.** Any subsystem can see any event on `agi.>`.
- **Multiple agents, explicit policy.** vMOE router > learned gate. Divine Council > single LLM. Each slot is debuggable and replaceable.
- **Verified before published.** Primer never writes an unverified note. CI never claims a deploy without comparing live state.
- **Drift detectors, not gates.** Regression workflows surface truth about the deployed state; they don't block code velocity.
- **Teach, don't fish for.** Primer articulates rules and demonstrates with verified code; the Scientist learns to apply the technique next time. The goal is the Scientist becoming able to solve the task itself.
