# Atlas AI -- Architecture Overview

This document describes the production architecture of Atlas AI as deployed at SJSU: a psychoanalytic multi-agent cognitive architecture running three local LLMs coordinated by NATS JetStream, backed by PostgreSQL + pgvector semantic memory, with burst compute via NRP Nautilus.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                          ATLAS AI — PRODUCTION DEPLOYMENT                                │
│                          HP Z840 · 2x GV100 32GB · 251GB RAM                             │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────────────┐  │
│   │   ID (Kirk)          │  │ SUPEREGO (Spock)      │  │  EGO (McCoy)                 │  │
│   │   System 1: Fast     │  │ System 2: Deliberate  │  │  Arbitrator: Debate          │  │
│   │                      │  │                       │  │                              │  │
│   │   Qwen 3 32B Q5_K_M  │  │ Gemma 4 31B Q5_K_M   │  │  Gemma 4 26B-A4B MoE Q4_K_XL│  │
│   │   GPU 1 · :8082      │  │ GPU 0 · :8080         │  │  CPU x24 · :8084             │  │
│   │   12 threads          │  │ 12 threads            │  │  8 parallel slots            │  │
│   │                      │  │                       │  │                              │  │
│   │   NATS: agi.rh.*     │  │ NATS: agi.safety.*    │  │  NATS: agi.ego.deliberate    │  │
│   └──────────┬───────────┘  └──────────┬────────────┘  └──────────────┬───────────────┘  │
│              │                         │                              │                   │
│              └─────────────────────────┼──────────────────────────────┘                   │
│                                        │                                                  │
│                          ┌─────────────▼──────────────┐                                  │
│                          │   DIVINE COUNCIL            │                                  │
│                          │   7 Advocate Agents (CPU)   │                                  │
│                          │                             │                                  │
│                          │   Judge · Advocate           │                                  │
│                          │   Synthesizer · Ethicist     │                                  │
│                          │   Historian · Futurist       │                                  │
│                          │   Pragmatist                 │                                  │
│                          │                             │                                  │
│                          │   Heartbeat: agi.meta.*     │                                  │
│                          └─────────────┬───────────────┘                                  │
│                                        │                                                  │
│   ┌────────────────────────────────────▼────────────────────────────────────────────────┐ │
│   │                     NATS JETSTREAM EVENT FABRIC                                      │ │
│   │                     :4222 (local) · :7422 (leaf node to NRP)                         │ │
│   │                                                                                      │ │
│   │   Subjects: agi.rh.request · agi.safety.* · agi.ego.deliberate                      │ │
│   │             agi.meta.monitor.* · agi.memory.* · agi.dreaming.*                      │ │
│   └───────┬────────────────────────┬──────────────────────────┬──────────────────────────┘ │
│           │                        │                          │                            │
│   ┌───────▼──────────┐    ┌───────▼──────────┐    ┌─────────▼────────────────────────┐   │
│   │ SAFETY SUBSYSTEM │    │ MEMORY SUBSYSTEM │    │ SUPPORT SERVICES                 │   │
│   │                  │    │                  │    │                                   │   │
│   │ 3-Layer Safety:  │    │ PostgreSQL +     │    │ RAG Server (:8081)               │   │
│   │  Reflex (<100us) │    │   pgvector       │    │ Telemetry (:8085)                │   │
│   │  Tactical (10ms) │    │                  │    │ Caddy (reverse proxy)            │   │
│   │  Strategic (pol) │    │ 3.3M PCA-384     │    │ OAuth2 Proxy (:4180)             │   │
│   │                  │    │ IVFFlat vectors  │    │ Thermal Guardian                 │   │
│   │ ErisML:          │    │                  │    │ Watchdog                         │   │
│   │  Bond Index      │    │ 5-Tier Retrieval │    │ Operations Dashboard             │   │
│   │  Hohfeld rights  │    │  L1: Dream-wiki  │    │   /schematic.html                │   │
│   │  Decision proofs │    │  L2: Wiki        │    │                                   │   │
│   │                  │    │  L3: Vector       │    │ Dreaming (idle consolidation)    │   │
│   │ Input/Output/    │    │  L4: FTS          │    │                                   │   │
│   │ Privilege Gates  │    │  L5: Live         │    │                                   │   │
│   └──────────────────┘    └──────────────────┘    └───────────────────────────────────┘   │
│                                                                                           │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## NATS Topology and NRP Bursting

```
                    ┌──────────────────────────────┐
                    │      NRP NAUTILUS             │
                    │      (100+ university nodes)  │
                    │                               │
                    │   ┌───────────────────────┐   │
                    │   │ CPU Pods               │   │
                    │   │  Prompt fuzzing (AFL)  │   │
                    │   │  Kimi K2.5 1T API      │   │
                    │   │  Qwen3.5 397B API      │   │
                    │   └───────────────────────┘   │
                    │                               │
                    │   ┌───────────────────────┐   │
                    │   │ GPU Pods               │   │
                    │   │  A100 80GB             │   │
                    │   │  L4 24GB               │   │
                    │   │  L40 48GB / L40S       │   │
                    │   │  Conv training         │   │
                    │   └───────────────────────┘   │
                    │                               │
                    │   NATS leaf node :7422        │
                    └──────────────┬────────────────┘
                                   │
                         Tailscale VPN mesh
                         100.68.134.21
                                   │
                    ┌──────────────▼────────────────┐
                    │      ATLAS WORKSTATION        │
                    │      HP Z840 · SJSU           │
                    │                               │
                    │   NATS JetStream :4222        │
                    │   ┌───────────────────────┐   │
                    │   │ GPU 0: Superego       │   │
                    │   │   Gemma 4 31B · :8080 │   │
                    │   ├───────────────────────┤   │
                    │   │ GPU 1: Id             │   │
                    │   │   Qwen 3 32B · :8082  │   │
                    │   ├───────────────────────┤   │
                    │   │ CPU: Ego              │   │
                    │   │   Gemma 4 26B · :8084 │   │
                    │   ├───────────────────────┤   │
                    │   │ PostgreSQL + pgvector  │   │
                    │   │ RAG Server · :8081     │   │
                    │   │ Telemetry · :8085      │   │
                    │   └───────────────────────┘   │
                    │                               │
                    │   atlas-sjsu.duckdns.org      │
                    │   Caddy -> OAuth2 -> RAG      │
                    └───────────────────────────────┘
```

---

## Core Components

### Divine Council -- Freudian Psyche

The system implements a psychoanalytic model where three LLMs serve as Freudian agents:

**Id (Kirk)** -- System 1 processing. Fast, intuitive, pattern-matching reasoning. Runs Qwen 3 32B on GPU 1 via llama.cpp. Subscribes to `agi.rh.request` via NATS.

**Superego (Spock)** -- System 2 processing. Slow, deliberative, ethical reasoning. Runs Gemma 4 31B on GPU 0 via llama.cpp. Subscribes to `agi.safety.*` via NATS.

**Ego (McCoy)** -- Arbitrator. Runs Gemma 4 26B-A4B MoE on CPU with 8 parallel slots. Hosts 7 council advocate roles (Judge, Advocate, Synthesizer, Ethicist, Historian, Futurist, Pragmatist) that run concurrent structured debates. Subscribes to `agi.ego.deliberate`. Publishes heartbeat to `agi.meta.monitor.ego`.

**Location:** `src/agi/reasoning/`

| File | Description |
|------|-------------|
| `divine_council.py` | Council orchestration and debate protocol |
| `_council_backend.py` | LLM backend integration for council agents |
| `_council_metrics.py` | Council performance and reliability metrics |
| `tree_of_thought.py` | Tree-of-Thought reasoning for structured debate |
| `nats_service.py` | NATS JetStream pub/sub for council coordination |

### Left Hemisphere -- Deliberative Processing

Handles planning, reasoning, and metacognition pipeline.

**Location:** `src/agi/lh/`

| Component | Description |
|-----------|-------------|
| Planner | Hierarchical planning with PlanGraph/PlanStep |
| Plan Service | gRPC service orchestrating the planning pipeline |
| Memory Client | Interface to memory services for context |
| Safety Client | Interface to Safety Gateway |
| Metacognition | Self-monitoring and plan review |
| LLM Integration | Language model backends |
| Performance | LRU cache with TTL, async operation batcher |
| HPC Deploy | Slurm launcher, Apptainer container runner |
| Resilience | Circuit breaker, retry, graceful degradation |

### Right Hemisphere -- Reactive Processing

Handles perception, world modeling, and motor control.

**Location:** `src/agi/rh/`

| Component | Directory | Description |
|-----------|-----------|-------------|
| Perception | `perception/` | Sensor fusion, object recognition |
| World Model | `world_model/` | Physics simulation, state prediction |
| Control | `control/` | Motor primitives, trajectory, realtime control, simulation |

### Safety Subsystem

Three-layer safety architecture with ErisML ethical reasoning.

**Location:** `src/agi/safety/`

```
                           ┌─────────────────────────────┐
                           │     SAFETY GATEWAY          │
                           │     gateway.py              │
                           ├─────────────────────────────┤
                           │                             │
                           │  ┌────────┐  ┌──────────┐  │
    Input Gate ────────────>│  │ Reflex │  │ Tactical │  │
    input_gate.py          │  │ <100us │  │  10-100ms│  │
                           │  └────┬───┘  └────┬─────┘  │
    Output Gate <──────────│       │           │        │
    output_gate.py         │  ┌────▼───────────▼─────┐  │
                           │  │     Strategic         │  │
    Privilege Gate         │  │     (policy)          │  │
    privilege_gate.py      │  └────┬─────────────────┘  │
                           │       │                    │
                           │  ┌────▼─────────────────┐  │
                           │  │  ErisML              │  │
                           │  │  Bond Index          │  │
                           │  │  Hohfeldian rights   │  │
                           │  │  Decision proofs     │  │
                           │  └──────────────────────┘  │
                           │                             │
                           │  NATS: agi.safety.*         │
                           └─────────────────────────────┘
```

| Layer | Latency | Function |
|-------|---------|----------|
| Reflex | <100us | Emergency stops, collision avoidance |
| Tactical | 10-100ms | ErisML ethical evaluation, Bond Index |
| Strategic | >100ms | Policy enforcement, human oversight triggers |

Additional modules:
- `learning/` -- Bayesian rule weight updates, anomaly detection
- `rules/` -- Rule engine for banned tools, constraints, risk scoring
- `dcgm_attestation.py` -- GPU attestation
- `adapter.py`, `deme_gateway.py` -- Safety adapters and DEME pipeline

### Memory Subsystem

PostgreSQL + pgvector with 3.3M PCA-384 IVFFlat vectors. Five-tier retrieval hierarchy.

**Location:** `src/agi/memory/`

```
  Query ──────────────────────────────────────────────────────────┐
    │                                                              │
    ▼                                                              │
  ┌─────────────────────────────────────────────────────────────┐  │
  │                   5-TIER RETRIEVAL                           │  │
  │                                                             │  │
  │   L1: Dream-wiki articles (1.5x relevance boost)           │  │
  │   L2: Wiki corpus (encyclopedic knowledge)                 │  │
  │   L3: pgvector semantic search (3.3M PCA-384 IVFFlat)      │  │
  │   L4: Full-text search (PostgreSQL tsvector)               │  │
  │   L5: Live retrieval (real-time web/API)                   │  │
  │                                                             │  │
  └─────────────────────────────────────────────────────────────┘  │
    │                                                              │
    ▼                                                              │
  ┌─────────────────────────────────────────────────────────────┐  │
  │                   MEMORY TYPES                               │  │
  │                                                             │  │
  │   Episodic     Event sequences, experiences, decision proofs│  │
  │   Semantic     Facts, concepts, relationships (pgvector)    │  │
  │   Procedural   Skills, learned behaviors, motor programs    │  │
  │   Knowledge    Structured knowledge base                    │  │
  │   Unified      Cross-type query interface                   │  │
  │                                                             │  │
  └─────────────────────────────────────────────────────────────┘  │
    │                                                              │
    ▼                                                              │
  ┌─────────────────────────────────────────────────────────────┐  │
  │   DREAMING (idle consolidation)                              │  │
  │   Synaptic plasticity + wiki article synthesis               │──┘
  │   Location: src/agi/dreaming/                                │
  └─────────────────────────────────────────────────────────────┘
```

### Metacognition

Monitors cognitive processes, detects anomalies, and manages executive function.

**Location:** `src/agi/metacognition/`

| Module | Description |
|--------|-------------|
| `ego_monitor.py` | Monitors Ego performance and decision quality |
| `consistency_checker.py` | Detects contradictions across agents |
| `anomaly_detector.py` | Identifies unusual patterns in reasoning |
| `executive_function.py` | Goal management and attention allocation |
| `adaptive_router.py` | Routes tasks to appropriate agents |
| `reasoning_analyzer.py` | Evaluates reasoning chain quality |
| `research_loop.py` | Autonomous research and learning |
| `curriculum_planner.py` | Curriculum planning for training |
| `reflector.py` | Self-reflection and introspection |
| `temporal.py` | Temporal reasoning and prediction |
| `nats_service.py` | NATS integration for monitoring events |

### Core Infrastructure

**Location:** `src/agi/core/`

**Event Fabric** (`events/`):

| Backend | File | Use Case |
|---------|------|----------|
| Local | `fabric.py` (LocalBackend) | In-process testing |
| ZeroMQ | `fabric.py` (ZmqBackend) | Multi-process dev |
| Redis | `redis_backend.py` | Persistent streams |
| NATS JetStream | `nats_backend.py`, `nats_fabric.py` | **Production** -- at-least-once delivery, durable consumers |
| Broker | `broker.py` | Event routing and fan-out |

Set via `AGI_FABRIC_MODE` environment variable: `local`, `zmq`, `redis`, `nats`, `ucx`.

**Other core modules:**
- `api/` -- Base gRPC server infrastructure
- `dht/` -- Distributed hash table (observability, HPC transport via UCX, mTLS security)
- `llm/` -- Shared LLM client with provider abstraction and middleware

---

## Deployment Architecture

### Systemd Services

All 19 services run under `atlas.target`:

```
atlas.target
├── atlas-nats.service              NATS JetStream :4222
├── atlas-id.service                Qwen 3 32B on GPU 1 :8082
├── atlas-superego.service          Gemma 4 31B on GPU 0 :8080
├── atlas-ego.service               Gemma 4 26B-A4B MoE on CPU :8084
├── atlas-rag-server.service        Flask + pgvector :8081
├── atlas-telemetry.service         Metrics + event stream :8085
├── atlas-caddy.service             Reverse proxy
├── atlas-oauth2-proxy.service      Authentication :4180
├── atlas-watchdog.service          Health monitoring
├── atlas-backup.service            Backup (timer-driven)
├── atlas-backup.timer
├── atlas-training.service          Training env (timer-driven)
├── atlas-training.timer
├── atlas-llm-kirk.service          Kirk LLM service
├── atlas-llm-spock.service         Spock LLM service
└── atlas-llm-dm.service            Dungeon Master LLM service
```

### CI/CD Pipeline

```
  Developer                GitHub Actions                  Atlas
  ────────                 ──────────────                  ─────
      │                          │                           │
      │── git push main ────────>│                           │
      │                          │── ruff check ────────>    │
      │                          │── black --check ─────>    │
      │                          │── pytest tests/unit/ ─>   │
      │                          │                           │
      │                          │   (all pass)              │
      │                          │                           │
      │                          │── SSH via Tailscale ─────>│
      │                          │   git pull                │
      │                          │   pip install -e .[nats]  │
      │                          │   systemctl restart       │
      │                          │                           │
      │                          │── smoke test ────────────>│
      │                          │   /health on :8080-8085   │
      │                          │                           │
```

### Network Architecture

```
  Internet
      │
      ▼
  atlas-sjsu.duckdns.org
      │
      ▼
  Caddy (reverse proxy)
      │
      ▼
  OAuth2 Proxy (:4180)
      │
      ├──> RAG Server (:8081)
      │        │
      │        ├──> Id/Kirk (:8082)
      │        ├──> Superego/Spock (:8080)
      │        └──> PostgreSQL + pgvector
      │
      ├──> Telemetry (:8085)
      │
      └──> Static files
           /schematic.html (operations dashboard)
           /events.html (event stream)
```

---

## Data Flow

### Decision Pipeline

```
  1. Request arrives via RAG Server (:8081)
     └── User query or system event

  2. Id (Kirk) generates fast response
     └── Qwen 3 32B on GPU 1
     └── System 1: intuitive, pattern-matching

  3. Superego (Spock) evaluates safety and ethics
     └── Gemma 4 31B on GPU 0
     └── System 2: deliberative, rule-following

  4. Ego (McCoy) arbitrates if conflict exists
     └── Gemma 4 26B-A4B MoE on CPU
     └── Runs Divine Council debate:
         7 advocates deliberate in parallel
         └── Judge, Advocate, Synthesizer, Ethicist,
             Historian, Futurist, Pragmatist

  5. Safety Gateway checks decision
     ├── Reflex layer: immediate constraints
     ├── Tactical layer: ErisML Bond Index
     └── Strategic layer: policy enforcement

  6. Memory consolidation
     └── Log to episodic memory
     └── Update semantic vectors if novel
     └── Dream-wiki synthesis during idle
```

### Safety Decision Flow

```
  PlanStep ──> PlanStepToEthicalFacts ──> EthicalFactsProto
                                                │
                                                ▼
                                        ErisMLService.EvaluateStep()
                                                │
                                                ▼
                                   ┌────────────┴────────────┐
                                   │    MoralVector          │
                                   │  (8+1 dimensions)       │
                                   └────────────┬────────────┘
                                                │
                                   ┌────────────▼────────────┐
                                   │   Verdict               │
                                   │   strongly_prefer       │
                                   │   prefer                │
                                   │   neutral               │
                                   │   avoid                 │
                                   │   forbid (VETO)         │
                                   └────────────┬────────────┘
                                                │
                                   ┌────────────▼────────────┐
                                   │   DecisionProof         │
                                   │   (hash-chained)        │
                                   └─────────────────────────┘
```

---

## Protocol Buffers

All inter-service communication uses gRPC with Protocol Buffers.

**Definitions:** `proto/`
**Generated code:** `src/agi/proto_gen/`

| Proto File | Package | Description |
|------------|---------|-------------|
| `plan.proto` | `agi.plan.v1` | Plan requests, responses, steps |
| `erisml.proto` | `agi.erisml.v1` | Ethical facts, moral vectors, proofs |
| `safety.proto` | `agi.safety.v1` | Safety checks, decisions, outcomes |
| `memory.proto` | `agi.memory.v1` | Memory queries and storage |
| `lh.proto` | `agi.lh.v1` | LH-specific messages |
| `rh.proto` | `agi.rh.v1` | RH-specific messages |
| `meta.proto` | `agi.meta.v1` | Metacognition messages |
| `env.proto` | `agi.env.v1` | Environment interface |

---

## Hardware Specifications

### Atlas Workstation

| Component | Specification |
|-----------|---------------|
| Chassis | HP Z840 |
| CPU | 2x Intel Xeon E5-2690 v3 (48 threads, 2.60 GHz) |
| RAM | 251 GB DDR4 |
| GPU 0 | NVIDIA Quadro GV100 32GB (Volta, compute 7.0) |
| GPU 1 | NVIDIA Quadro GV100 32GB (Volta, compute 7.0) |
| NVLink | Not available (GPUs on different CPU sockets) |
| Storage | 15 TB at `/archive` |
| Network | Tailscale VPN at 100.68.134.21 |
| OS | Ubuntu Linux |
| Python | 3.12 (venv at `/home/claude/env`) |
| ML Stack | PyTorch 2.10.0+cu128, transformers 5.3.0, CuPy 14.0.1, bitsandbytes 0.49.2 |
| LLM Runtime | llama.cpp (CUDA build) |

### NRP Nautilus

| Resource | Access |
|----------|--------|
| Namespace | `ssu-atlas-ai` |
| Portal | [nrp.ai](https://nrp.ai) |
| GPU Types | A100 80GB, L4 24GB, L40 48GB, L40S |
| Nodes | 100+ across participating universities |
| Bridge | NATS leaf node at `:7422` |

### Thermal Constraints

| Threshold | Temperature | Action |
|-----------|-------------|--------|
| Normal | < 82C | Full operation |
| High | 82C | Alert, reduce threads |
| Critical | 100C | Emergency throttle |
| Thread cap | -- | 20 threads maximum (48 available) |

---

## Key Design Principles

### 1. Psychoanalytic Multi-Agent Architecture
Three LLMs implement a Freudian model: the Id generates fast intuitive responses, the Superego enforces ethical reasoning, and the Ego arbitrates conflict through structured multi-advocate debate.

### 2. Safety-First Design
Safety is not a filter at the end but woven throughout. Three layers (reflex, tactical, strategic) operate at different timescales. ErisML provides mathematically grounded ethical reasoning with auditable decision proofs.

### 3. Hybrid Memory Architecture
Five-tier retrieval hierarchy combines dream-synthesized articles, wiki knowledge, vector similarity, full-text search, and live retrieval. Memory consolidation occurs during idle cycles via the dreaming subsystem.

### 4. Event-Driven Coordination
NATS JetStream provides the "global workspace" (per Baars 1988) -- a shared broadcast medium where all agents publish and subscribe. Durable consumers ensure at-least-once delivery.

### 5. Burst Compute via NRP
Local GPUs handle inference; NRP Nautilus provides burst capacity for training and large-scale fuzzing. NATS leaf node bridges the two environments transparently.

### 6. Graceful Degradation
The system remains safe when components fail. The safety gateway works without ErisML. The Ego works without the full council. All services have fallback behaviors and are auto-restarted by the watchdog.

---

## Related Documentation

- [ATLAS_OPERATIONS.md](ATLAS_OPERATIONS.md) -- Production operations guide
- [ERISML_API.md](ERISML_API.md) -- ErisML integration API reference
- [ERISML_INTEGRATION_SKETCH.md](ERISML_INTEGRATION_SKETCH.md) -- Detailed integration design
- [HPC_DEPLOYMENT.md](HPC_DEPLOYMENT.md) -- HPC cluster deployment guide
- [LH_SPRINT_PLAN.md](LH_SPRINT_PLAN.md) -- Left Hemisphere development plan
- [RH_SPRINT_PLAN.md](RH_SPRINT_PLAN.md) -- Right Hemisphere development plan
