<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/atlas_mark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/brand/atlas_mark_light.svg">
    <img src="docs/brand/atlas_mark.svg" width="280" alt="Atlas AI — sphere + Eris apple">
  </picture>
</p>

# Atlas AI

### Neuroscience-inspired cognitive architecture with distributed compute

![CI](https://github.com/ahb-sjsu/agi-hpc/actions/workflows/ci.yaml/badge.svg)
![Deploy Smoke](https://github.com/ahb-sjsu/agi-hpc/actions/workflows/deploy-smoke.yaml/badge.svg)
![Dashboard Render](https://github.com/ahb-sjsu/agi-hpc/actions/workflows/dashboard-render.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Live dashboard:** [atlas-sjsu.duckdns.org/schematic.html](https://atlas-sjsu.duckdns.org/schematic.html) · **Brand assets:** [erisml.org/brand/](https://erisml.org/brand/)

Atlas AI is a **production cognitive architecture** that mirrors the vertebrate brain's cortical / subcortical split. High-level reasoning runs on frontier cloud LLMs (the *cortex*), fast pattern learning and procedural memory run on local GPUs (the *subcortical brain*), and the two tiers are coordinated by NATS JetStream acting as a global workspace (Baars, 1988). Persistent memory lives in PostgreSQL + pgvector.

A Freudian agent model — Ego, Superego, and the Divine Council — negotiates decisions through structured debate. An autonomous learning loop (ARC Scientist) improves its own problem-solving strategies over time, and **The Primer** (inspired by Stephenson's *Young Lady's Illustrated Primer*) is an always-on Claude-style tutor that teaches the Scientist through verified reference implementations written back to the wiki.

---

## Architecture at a glance

```
                         ╔═══════════════════════════════════════════════════╗
                         ║        CORTEX  —  NRP Nautilus Cloud               ║
                         ║        Frontier LLMs via managed API               ║
                         ╠═══════════════════════════════════════════════════╣
                         ║                                                    ║
                         ║  Ego       Kirk        Qwen 3.5 397B              ║
                         ║  Superego  Spock       Gemma 4                    ║
                         ║  Council   7 agents    Kimi K2.5 1T               ║
                         ║  Scientist ARC loop    Kimi / Qwen / GLM rotation ║
                         ║  Primer    vMOE tutor  Kimi + GLM-4.7 + Qwen3     ║
                         ║                                                    ║
                         ║  + Vision pods (GLM-4.1V on L40 / A10)            ║
                         ║  + Worker pools (nats-bursting on 33-node A10)    ║
                         ╚═══════════════════════╤════════════════════════════╝
                                                 │  NATS leaf :7422 (TLS)
                         ╔═══════════════════════╧════════════════════════════╗
                         ║   GLOBAL WORKSPACE  —  NATS JetStream :4222        ║
                         ║   Subjects: agi.*                                  ║
                         ╚═══════════════════════╤════════════════════════════╝
           ┌─────────────────────────────────────┼─────────────────────────────┐
           │                                     │                             │
  ╔════════╧════════════╗   ╔════════════════════╧════════╗   ╔═══════════════╧═══════╗
  ║ SUBCORTICAL BRAIN    ║   ║ MEMORY  — PostgreSQL +      ║   ║  BRAINSTEM             ║
  ║ 2× Quadro GV100 32GB ║   ║ pgvector · 3.3M PCA-384 vecs║   ║                        ║
  ║                      ║   ║                             ║   ║  Thermal guardian     ║
  ║  Conv training       ║   ║  L0  Dream-consolidated     ║   ║  Watchdog             ║
  ║  Pattern learning    ║   ║  L1  Sensei wiki (auto +    ║   ║  Telemetry (:8085)    ║
  ║  Procedural memory   ║   ║       human-written)        ║   ║  Caddy proxy + OAuth2 ║
  ║  A* search           ║   ║  L2  PCA-384 IVFFlat        ║   ║  Backup (daily)       ║
  ║  Kirk (local ego     ║   ║  L3  tsvector full-text     ║   ║  atlas-target systemd ║
  ║    fallback)         ║   ║  L4  Episodic memory        ║   ║                        ║
  ╚══════════════════════╝   ╚═════════════════════════════╝   ╚════════════════════════╝
```

See [`docs/ARCHITECTURE_OVERVIEW.md`](docs/ARCHITECTURE_OVERVIEW.md) for the full systems view including deployment, CI/CD, and regression guards.

---

## The Freudian agents

Three psychoanalytic agents negotiate decisions through structured debate. The Star Trek analogues show the functional mapping — not the reverse.

| Agent | Role | Primary backend | Current status |
|---|---|---|---|
| **Ego (Kirk)** | Balanced decision-maker — the self that speaks and learns | Qwen 32B on local GPU 1, Qwen 3.5 397B on NRP as ego-scale fallback | Active; chat currently routes via Kimi on NRP pending ego-slot cutover (`Phase 3`) |
| **Superego (Spock)** | Logic, rules, ethical evaluation | Gemma 4 31B on local GPU 0, Gemma 4 on NRP | Active |
| **Divine Council** | Multi-perspective deliberation (7 advocate agents: Judge, Advocate, Synthesizer, Ethicist, Historian, Futurist, Pragmatist) | Kimi K2.5 1T on NRP, `atlas-ego.service --parallel 8` CPU backend | Active |

The **Id** role is currently unfilled; the fast-instinct pattern-matching function is served by the basal-ganglia-style local-GPU procedural memory (pattern nets + A*) rather than a dedicated LLM slot.

**Fallback path:** when NRP is unavailable, Council and Ego fall back to local llama.cpp on the two GV100s.

---

## Autonomous learning

```
   ┌───────────────────────┐           ┌─────────────────────────────┐
   │   ARC SCIENTIST       │           │        THE PRIMER           │
   │   (agi.autonomous)    │           │        (agi.primer)         │
   │                       │           │                             │
   │   OBSERVE             │           │   watch help queue          │
   │   HYPOTHESIZE ──┐     │           │   read wiki + memory        │
   │   EXPERIMENT    │     │◄──────────┤   vMOE ensemble             │
   │   EVALUATE      │     │  verified │   verify vs train           │
   │   REFLECT       │     │  sensei   │   publish sensei note       │
   │   ADAPT ────────┘     │   notes   │   git commit + push         │
   │                       │           │                             │
   └───────────────────────┘           └─────────────────────────────┘
```

### ARC Scientist (`src/agi/autonomous/arc_scientist.py`)

Closed-loop scientific reasoning against the NeuroGolf 2026 / ARC-AGI task set:

1. **Observe** — pick an unsolved task (tier-weighted by similarity to prior solves and near-miss exploitation).
2. **Hypothesize** — form a theory via LLM prompt strategy (direct / failure_aware / example_chain / diagnostic).
3. **Experiment** — generate candidate Python `transform(grid)` implementations.
4. **Evaluate** — verify against every training example; any failure = reject.
5. **Learn** — store what worked + what failed with structured error classification (reasoning / execution / perception / specification).
6. **Reflect** — LLM diagnoses *why* a transform failed; result becomes prompt context for the next attempt.
7. **Adapt** — Thompson sampling shifts strategy weights based on evidence.
8. Mentor preambles — task-specific sensei notes from the wiki (either human-written or Primer-generated) are prepended to relevant prompts.

Current state: **100 / 400 tasks solved (25.0 %)** as of 2026-04-19, 1,364 lifetime attempts.

### The Primer (`src/agi/primer/`) — new 2026-04-19

An always-on teaching daemon that meets Erebus at his current confusion and writes verified sensei notes. Details in [`docs/THE_PRIMER.md`](docs/THE_PRIMER.md).

- **Watches** the help queue for tasks where the Scientist has ≥ 10 attempts and no solve.
- **Reads** task JSON + attempt history + relevant wiki articles to assemble context.
- **Calls** a **vMOE ensemble** of frontier models (Kimi, GLM-4.7, Qwen3) in parallel; see [`docs/VMOE.md`](docs/VMOE.md).
- **Verifies** each candidate's `transform()` against every training example. **Only verified code is published** — a wrong mentor note is worse than no note.
- **Publishes** a `sensei_task_NNN.md` to the wiki, git-commits and pushes; the existing CI deploy picks it up.
- **Tracks expert health** — if a model has been consistently slow/erroring, it's skipped for a cooldown window to avoid burning tokens on foregone conclusions.

Runs as `atlas-primer.service` with a 10-minute polling cadence.

---

## Infrastructure

| Component | Description |
|---|---|
| **NATS JetStream** | Global workspace at `:4222`. Leaf node at `:7422` bridges to NRP. |
| **PostgreSQL + pgvector** | 3.3 M PCA-384 vectors. 5-tier retrieval (L0–L4). |
| **RAG server** | Flask at `:8081` with dual-hemisphere proxy + hybrid search (BM25 + dense + HyDE + RRF). |
| **Telemetry** | At `:8085`. NATS stats, NRP pod metrics, live GPU / VRAM via `kubectl exec nvidia-smi`. Serves `/schematic.html` with a dynamic `ui:<sha> · <mtime>` version stamp. |
| **Caddy** | Reverse proxy + OAuth2. Serves `atlas-sjsu.duckdns.org`. |
| **Thermal guardian** | CPU temp monitoring (82 °C warn, 100 °C critical). |
| **Watchdog** | Health checks + automatic service restart. |
| **Primer daemon** | `atlas-primer.service` — always-on tutor; CPU-only. |

## NRP Nautilus integration

NATS leaf node bridges Atlas to [NRP Nautilus](https://nrp.ai) (namespace `ssu-atlas-ai`):

- **Managed LLM API** (`https://ellm.nrp-nautilus.io/v1`): Kimi K2.5, Qwen 3.5 397B, GLM-4.7, MiniMax M2.7, Gemma 4 — shared, OpenAI-compatible, zero marginal cost.
- **Anthropic-compatible proxy** (`.../anthropic`) for Claude Code tooling.
- **Burst Jobs** — ephemeral K8s Jobs (vision pool 4× L40/L40S/A10, GLM-4.1V). See [`deploy/k8s/`](deploy/k8s/).
- **Worker Pools** — persistent K8s Deployments (`erebus-workers`, 8× nats-bursting workers). See [nats-bursting](https://github.com/ahb-sjsu/nats-bursting).
- **Live monitoring** — dashboard shows pod count, GPU model, VRAM used/total, utilization. Worker pools and burst jobs appear side-by-side in the *NRP Burst Jobs + Worker Pools* card.
- **L40 reservation caveat** — 15 of 17 L40 nodes carry `csu-tide` reservation taints; our namespace defaults to A10 × 8 (33 abundantly-available nodes) for self-hosted pods.

## Safety

Three-layer architecture:

| Layer | Latency | Function |
|---|---|---|
| **Reflex** | < 100 µs | Emergency stops, thermal limits |
| **Tactical** | 10–100 ms | ErisML ethical evaluation, Bond Index |
| **Strategic** | > 100 ms | Policy enforcement, human oversight |

ErisML provides mathematically grounded ethical reasoning with Hohfeldian analysis and SHA-256 hash-chained decision proofs.

## Regression guards (CI/CD)

| Workflow | Trigger | What it checks |
|---|---|---|
| **Atlas AI — CI/CD** (`ci.yaml`) | push to main | Lint, tests, build, auto-deploy to Atlas (git pull + restart telemetry/rag). |
| **Deploy Smoke** (`deploy-smoke.yaml`) | push to main + 30-min cron | Compares live `/api/version` SHA to `main` HEAD. Asserts `NATS Topology`, `NRP Burst Jobs`, `NATS Live`, `Erebus Cognitive Architecture` are present on the rendered page. |
| **Dashboard Render** (`dashboard-render.yaml`) | dashboard path changes + 30-min cron | Playwright headless Chromium loads `schematic.html`, asserts topology SVG populates, burst table has rows, version stamp matches `ui:<sha> · <date>T<time>Z`, no console errors. |

These exist because silent dashboard regressions (stale deploy, empty widget, JS error) previously hid under green Python CI. See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for the post-mortem on the `cp -f`-through-symlink drift that motivated them.

---

## Project layout

```
agi-hpc/
├── src/agi/
│   ├── autonomous/          # ARC Scientist (self-improving solver)
│   ├── primer/              # The Primer — vMOE + validator + service
│   ├── core/                # gRPC, NATS/ZMQ/UCX, DHT, LLM providers
│   ├── reasoning/           # Divine Council, debate, NATS service
│   ├── lh/                  # Left hemisphere: planning, metacognition
│   ├── rh/                  # Right hemisphere: perception, world model
│   ├── memory/              # Episodic, semantic, procedural, knowledge
│   ├── safety/              # 3-layer safety, ErisML, privilege gates
│   ├── metacognition/       # Ego monitor, consistency, anomaly
│   ├── dreaming/            # Memory consolidation via wiki synthesis
│   ├── training/            # Dungeon Master, gym environment, curriculum
│   ├── attention/           # Distractor detection and filtering
│   ├── thermal/             # Thermal management, job queue
│   ├── integration/         # Cross-subsystem orchestration
│   ├── env/                 # Gymnasium-compatible (MuJoCo / Unity)
│   └── meta/                # LLM-based metacognitive reflection
│
├── configs/                 # Service YAML
├── deploy/
│   ├── systemd/             # 20 service units under atlas.target (incl. atlas-primer)
│   └── k8s/                 # NRP pod manifests (erebus-ego PVC, vision burst)
├── proto/                   # Protocol Buffer definitions
├── infra/
│   ├── hpc/                 # Apptainer, Slurm, Docker
│   └── local/atlas-chat/    # Dashboard (schematic.html, erebus.html)
├── scripts/                 # Watchdog, telemetry, utilities
├── tests/
│   ├── unit/                # Python unit tests (primer_vmoe, primer_validator, primer_health, dashboard_panels, …)
│   └── dashboard/           # Playwright render tests
├── wiki/                    # Sensei notes (Tier-1 RAG) — Primer writes here
├── docs/                    # Architecture, operations, design docs
└── .github/workflows/       # CI/CD pipeline
```

## Hardware

| Resource | Spec |
|---|---|
| **Atlas** | HP Z840, 2× Xeon E5-2690v3 (48 threads), 251 GB RAM |
| **Local GPUs** | 2× Quadro GV100 32GB (Volta, no NVLink — separate CPU sockets) |
| **Storage** | 15 TB at `/archive` |
| **Network** | Tailscale at `100.68.134.21` |
| **NRP** | L40 48 GB, L40S 48 GB, A10 24 GB, A40 48 GB across 100+ university nodes (A100/H100 gated behind reservation form) |
| **NRP LLMs** | Kimi K2.5 1T, Qwen 3.5 397B, GLM-4.7 358B, MiniMax M2.7, Gemma 4 (managed, zero marginal cost) |

---

## Quickstart

```bash
git clone https://github.com/ahb-sjsu/agi-hpc.git
cd agi-hpc
pip install -e ".[dev]"

# Local development (no network)
export AGI_FABRIC_MODE=local
python src/agi/lh/service.py

# Production (NATS + NRP)
export AGI_FABRIC_MODE=nats
export NRP_LLM_TOKEN=<your-token>
python src/agi/autonomous/arc_scientist.py --task-dir /path/to/tasks

# Run The Primer (CPU-only)
export NRP_LLM_TOKEN=<your-token>
python -m agi.primer.service
```

For production deployment on Atlas, see [`docs/ATLAS_OPERATIONS.md`](docs/ATLAS_OPERATIONS.md). For The Primer specifically, [`docs/THE_PRIMER.md`](docs/THE_PRIMER.md).

---

## Documentation

| Doc | Topic |
|---|---|
| [`docs/ARCHITECTURE_OVERVIEW.md`](docs/ARCHITECTURE_OVERVIEW.md) | Full systems view: agents, fabric, memory, NRP, CI/CD |
| [`docs/THE_PRIMER.md`](docs/THE_PRIMER.md) | The Primer — design, safety invariants, operation |
| [`docs/VMOE.md`](docs/VMOE.md) | Virtual Mixture-of-Experts — routing policies, health |
| [`docs/ATLAS_OPERATIONS.md`](docs/ATLAS_OPERATIONS.md) | Atlas workstation ops: systemd, thermal, backup |
| [`docs/AGI_ROADMAP.md`](docs/AGI_ROADMAP.md) | Phase status, pending work, long-horizon plan |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | Per-date ship log |

---

## License

MIT (c) 2025 Andrew Bond
