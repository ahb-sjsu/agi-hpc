<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/atlas_mark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/brand/atlas_mark_light.svg">
    <img src="docs/brand/atlas_mark.svg" width="280" alt="Atlas AI — sphere + Eris apple">
  </picture>
</p>

# Atlas AI

### Psychoanalytic cognitive architecture on commodity hardware

![CI](https://github.com/ahb-sjsu/agi-hpc/actions/workflows/ci.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Brand assets:** [erisml.org/brand/](https://erisml.org/brand/) | local: [`docs/brand/`](docs/brand/)

Atlas AI is a **production multi-agent cognitive architecture** running on an HP Z840 workstation at SJSU, with burst compute via [NRP Nautilus](https://nrp.ai). It implements a Freudian psychoanalytic model where three local LLMs (Id, Superego, Ego) negotiate decisions through structured debate, coordinated by NATS JetStream and backed by PostgreSQL + pgvector semantic memory.

---

## The Divine Council

Atlas runs three on-premise LLMs as psychoanalytic agents, served by llama.cpp:

| Agent | Role | Model | Hardware | Port |
|-------|------|-------|----------|------|
| **Id** (Kirk) | System 1 -- fast, intuitive reasoning | Qwen 3 32B Q5_K_M | GPU 1 (GV100 32GB) | 8082 |
| **Superego** (Spock) | System 2 -- deliberative, ethical reasoning | Gemma 4 31B Q5_K_M | GPU 0 (GV100 32GB) | 8080 |
| **Ego** (McCoy) | Arbitrator -- runs structured debates | Gemma 4 26B-A4B MoE Q4_K_XL | CPU (24 threads, 8 parallel slots) | 8084 |

The Ego hosts 7 concurrent council roles (Judge, Advocate, Synthesizer, Ethicist, Historian, Futurist, Pragmatist) that deliberate on decisions requiring multi-perspective analysis.

## Infrastructure

| Component | Description |
|-----------|-------------|
| **NATS JetStream** | Production event fabric at `:4222`. Leaf node at `:7422` bridges to NRP Nautilus. |
| **PostgreSQL + pgvector** | 3.3M PCA-384 IVFFlat vectors. 5-tier semantic retrieval (L1 dream-wiki 1.5x boost, L2 wiki, L3 vector, L4 FTS, L5 live). |
| **RAG Server** | Flask at `:8081` -- dual-hemisphere proxy + pgvector search. |
| **Caddy** | Reverse proxy with OAuth2 proxy at `:4180`. Serves `atlas-sjsu.duckdns.org`. |
| **Telemetry Server** | At `:8085`. NATS stats, NRP pod metrics, system health. Drives the operations dashboard. |
| **Operations Dashboard** | Real-time schematic at `/schematic.html` with NATS topology, NRP pod GPU/VRAM monitoring, service status, memory tiers. |
| **Thermal Guardian** | CPU temp monitoring (82C high, 100C critical). Caps compute at 20 threads. |
| **Watchdog** | Health checks and automatic service restarts. |

## NRP Nautilus Integration

NATS leaf node bridges Atlas to NRP's Kubernetes cluster (namespace `ssu-atlas-ai`) for burst compute:

- **CPU pods** -- LLM-guided prompt fuzzing via managed APIs (Kimi K2.5 1T, Qwen3.5 397B)
- **GPU pods** -- Conv training on A100 80GB, L4 24GB, L40 48GB via node affinity
- **Live monitoring** -- Pod GPU model, VRAM, CPU/RAM utilization via `kubectl exec nvidia-smi`
- **Coverage-guided fuzzer** -- AFL-style prompt fuzzer for ARC task solving

## Active Projects

- **NeuroGolf 2026** -- Kaggle competition. ONNX golf for ARC-AGI tasks. DSL-to-ONNX compiler, VLM-guided solver. Private repo: `ahb-sjsu/neurogolf-2026`.
- **ErisML** -- Ethical reasoning framework. Bond Index, Hohfeldian analysis, hash-chained decision proofs.
- **Dreaming** -- Memory consolidation via wiki article synthesis during idle cycles.

---

## Project Layout

```
agi-hpc/
├── src/agi/
│   ├── core/                # gRPC server, event fabric, DHT, LLM client
│   │   ├── api/             # Base gRPC server infrastructure
│   │   ├── events/          # Local, ZMQ, Redis, NATS JetStream backends
│   │   ├── dht/             # Hash ring, observability, HPC transport, security
│   │   └── llm/             # Client, providers, middleware
│   ├── reasoning/           # Divine Council, Tree-of-Thought debate, NATS service
│   ├── lh/                  # Left hemisphere: planning, metacognition, HPC deploy
│   ├── rh/                  # Right hemisphere: perception, world model, control
│   ├── memory/              # Episodic, semantic, procedural, knowledge, unified
│   ├── safety/              # 3-layer safety, ErisML, input/output/privilege gates
│   ├── metacognition/       # Ego monitor, consistency checker, anomaly detector
│   ├── dreaming/            # Synaptic plasticity, wiki consolidation
│   ├── training/            # Dungeon Master, gym environment, curriculum
│   ├── attention/           # Distractor detection and filtering
│   ├── thermal/             # Thermal management and job queue
│   ├── integration/         # Cross-subsystem orchestration
│   ├── env/                 # Gymnasium-compatible environment (MuJoCo/Unity)
│   ├── meta/                # LLM-based metacognitive reflection
│   └── common/              # Shared utilities
│
├── deploy/
│   ├── systemd/             # 19 service units under atlas.target
│   └── llm/                 # Kubernetes production config
│
├── proto/                   # Protocol Buffer definitions
├── configs/                 # Service configuration YAML
├── infra/
│   ├── hpc/                 # Apptainer, Slurm, Docker, monitoring
│   └── local/               # Local development setup
│
├── scripts/                 # Watchdog, telemetry, utilities
├── tests/                   # Unit and integration tests
├── docs/                    # Architecture docs, sprint plans, runbooks
├── .github/workflows/       # CI/CD (ruff + black + pytest + auto-deploy)
└── Caddyfile                # Reverse proxy config
```

## Systemd Services

All services run under `atlas.target` and are managed by systemd:

| Service | Description |
|---------|-------------|
| `atlas-nats` | NATS JetStream event fabric |
| `atlas-id` | Kirk / Id (Qwen 3 32B on GPU 1) |
| `atlas-superego` | Spock / Superego (Gemma 4 31B on GPU 0) |
| `atlas-ego` | Divine Council (Gemma 4 26B-A4B MoE, 8 parallel slots on CPU) |
| `atlas-rag-server` | RAG server (Flask + pgvector search) |
| `atlas-telemetry` | Telemetry server (metrics + event stream) |
| `atlas-caddy` | Caddy reverse proxy |
| `atlas-oauth2-proxy` | OAuth2 authentication proxy |
| `atlas-watchdog` | Health monitoring and auto-restart |
| `atlas-backup` | Scheduled backup (timer-driven) |
| `atlas-training` | Training environment (timer-driven) |
| `atlas-llm-kirk` | Kirk LLM service |
| `atlas-llm-spock` | Spock LLM service |
| `atlas-llm-dm` | Dungeon Master LLM service |

---

## CI/CD

GitHub Actions pipeline on every push to `main`:

1. **Lint** -- `ruff check` + `black --check`
2. **Test** -- `pytest tests/unit/`
3. **Deploy** -- SSH to Atlas via Tailscale, `git pull`, `pip install -e ".[nats]"`, restart services
4. **Smoke test** -- Verify RAG server and telemetry endpoints respond

---

## Hardware

| Resource | Specification |
|----------|---------------|
| **Atlas Workstation** | HP Z840, 2x Xeon E5-2690v3 (48 threads), 251GB RAM |
| **GPU 0** | Quadro GV100 32GB (Volta) -- Superego |
| **GPU 1** | Quadro GV100 32GB (Volta) -- Id |
| **Storage** | 15TB at `/archive` |
| **Network** | Tailscale VPN at 100.68.134.21 |
| **NRP Nautilus** | A100 80GB, L4 24GB, L40 48GB, L40S across 100+ university nodes |

---

## Quickstart

```bash
# Clone
git clone https://github.com/ahb-sjsu/agi-hpc.git
cd agi-hpc

# Install
pip install -e ".[dev]"
pre-commit install

# Generate protobuf stubs
python generate_protos.py --clean

# Run locally (event fabric mode: local, zmq, nats, ucx)
export AGI_FABRIC_MODE=local
python src/agi/lh/service.py
python src/agi/rh/service.py
```

For production deployment on Atlas, see [`docs/ATLAS_OPERATIONS.md`](docs/ATLAS_OPERATIONS.md).

---

## Dev Teams

- Cognitive Architecture (LH/RH)
- Memory Systems
- EventBus and Messaging
- HPC Deployment
- Maritime Digital Twin
- Unity Simulation
- Evaluation and Metrics

---

## License

MIT (c) 2025 Andrew Bond
