# AGI-HPC Architecture Overview

This document describes the high-level architecture of the AGI-HPC system—a cognitive architecture for safe, embodied artificial general intelligence designed for high-performance computing deployments.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGI-HPC COGNITIVE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────┐         ┌─────────────────────────┐          │
│   │    LEFT HEMISPHERE      │         │    RIGHT HEMISPHERE     │          │
│   │    (Deliberative)       │         │    (Reactive)           │          │
│   │                         │         │                         │          │
│   │  ┌─────────────────┐   │         │  ┌─────────────────┐    │          │
│   │  │    Planner      │   │         │  │  World Model    │    │          │
│   │  │  - Goal decomp  │   │         │  │  - Physics sim  │    │          │
│   │  │  - Plan graphs  │   │         │  │  - Prediction   │    │          │
│   │  └────────┬────────┘   │         │  └────────┬────────┘    │          │
│   │           │            │         │           │             │          │
│   │  ┌────────▼────────┐   │         │  ┌────────▼────────┐    │          │
│   │  │  Metacognition  │   │         │  │   Perception    │    │          │
│   │  │  - Self-monitor │   │         │  │  - Sensor fusion│    │          │
│   │  │  - Plan review  │   │         │  │  - Object recog │    │          │
│   │  └─────────────────┘   │         │  └─────────────────┘    │          │
│   │                         │         │                         │          │
│   │  Port: 50051            │         │  Port: 50057            │          │
│   └───────────┬─────────────┘         └───────────┬─────────────┘          │
│               │                                   │                        │
│               └───────────────┬───────────────────┘                        │
│                               │                                            │
│   ┌───────────────────────────▼────────────────────────────────────────┐   │
│   │                      SAFETY GATEWAY                                 │   │
│   │  ┌──────────────────────────────────────────────────────────────┐  │   │
│   │  │                   ERISML INTEGRATION                          │  │   │
│   │  │   ┌──────────┐    ┌──────────┐    ┌──────────┐               │  │   │
│   │  │   │ Reflex   │ →  │ Tactical │ →  │Strategic │               │  │   │
│   │  │   │ (<100μs) │    │(10-100ms)│    │ (policy) │               │  │   │
│   │  │   └──────────┘    └──────────┘    └──────────┘               │  │   │
│   │  │        │               │               │                      │  │   │
│   │  │        ▼               ▼               ▼                      │  │   │
│   │  │   Emergency       DEME Pipeline    Governance                 │  │   │
│   │  │    Stops          Bond Index       Policies                   │  │   │
│   │  └──────────────────────────────────────────────────────────────┘  │   │
│   │  Ports: 50055 (Gateway), 50060 (ErisML)                            │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                               │                                            │
│   ┌───────────────────────────▼────────────────────────────────────────┐   │
│   │                       MEMORY SERVICES                               │   │
│   │   ┌──────────┐    ┌──────────┐    ┌──────────┐                     │   │
│   │   │ Episodic │    │ Semantic │    │Procedural│                     │   │
│   │   │  Memory  │    │  Memory  │    │  Memory  │                     │   │
│   │   │  :50052  │    │  :50053  │    │  :50054  │                     │   │
│   │   └──────────┘    └──────────┘    └──────────┘                     │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Left Hemisphere (LH) - Deliberative Processing

The Left Hemisphere handles high-level reasoning, planning, and goal management.

**Location:** `src/agi/lh/`

| Component | File | Description |
|-----------|------|-------------|
| Planner | `planner.py` | Hierarchical planning with PlanGraph/PlanStep |
| Plan Service | `plan_service.py` | gRPC service orchestrating the planning pipeline |
| Memory Client | `memory_client.py` | Interface to memory services for context |
| Safety Client | `safety_client.py` | Interface to Safety Gateway |
| Metacognition | `metacog_client.py` | Self-monitoring and plan review |
| LLM Integration | `llm/` | Language model backends (OpenAI, Anthropic, local) |
| Observability | `observability.py` | Metrics, logging, request context |

**Key Data Structures:**
- `PlanGraph`: Hierarchical plan representation
- `PlanStep`: Individual action with safety tags, tool references
- `PlanRequest/Response`: gRPC API messages

### Right Hemisphere (RH) - Reactive Processing

The Right Hemisphere handles perception, world modeling, and motor control.

**Location:** `src/agi/rh/`

| Component | Directory | Description |
|-----------|-----------|-------------|
| Perception | `perception/` | Sensor fusion, object recognition |
| World Model | `world_model/` | Physics simulation, state prediction |
| Control | `control/` | Motor control, trajectory execution |

### Safety Subsystem

Three-layer safety architecture ensuring safe operation at all timescales.

**Location:** `src/agi/safety/`

| Layer | Latency | Function |
|-------|---------|----------|
| Reflex | <100μs | Hardware-level emergency stops, collision avoidance |
| Tactical | 10-100ms | ErisML ethical evaluation, Bond Index verification |
| Strategic | >100ms | Policy enforcement, human oversight triggers |

**ErisML Integration:**
- `erisml/service.py`: gRPC service for ethical evaluation
- `erisml/facts_builder.py`: Converts PlanStep → EthicalFacts
- `gateway.py`: Safety Gateway with pre/in/post-action checking

### Memory Services

Distributed memory architecture for different types of knowledge.

**Location:** `src/agi/memory/`

| Memory Type | Port | Description |
|-------------|------|-------------|
| Episodic | 50052 | Event sequences, experiences, decision proofs |
| Semantic | 50053 | Facts, concepts, relationships |
| Procedural | 50054 | Skills, learned behaviors, motor programs |

### Core Infrastructure

**Location:** `src/agi/core/`

| Component | Directory | Description |
|-----------|-----------|-------------|
| gRPC Server | `api/` | Base gRPC server infrastructure |
| Event Fabric | `events/` | Pub/sub event system (local or distributed) |
| DHT | `dht/` | Distributed hash table for HPC deployments |
| LLM | `llm/` | Shared LLM client infrastructure |

## Service Ports

| Service | Port | Proto File |
|---------|------|------------|
| LH (Plan Service) | 50051 | `plan.proto`, `lh.proto` |
| Episodic Memory | 50052 | `memory.proto` |
| Semantic Memory | 50053 | `memory.proto` |
| Procedural Memory | 50054 | `memory.proto` |
| Safety Gateway | 50055 | `safety.proto` |
| In-Action Safety | 50056 | `safety.proto` |
| RH (World Model) | 50057 | `rh.proto` |
| Post-Action Safety | 50058 | `safety.proto` |
| ErisML Service | 50060 | `erisml.proto` |
| Metacognition | 50070 | `meta.proto` |

## Data Flow

### Planning Pipeline

```
1. PlanRequest arrives at LH
   └── Goal: "Pick up the red cube"

2. Memory Enrichment
   └── Query episodic/semantic memory for context

3. Plan Generation (Planner)
   └── Decompose goal → PlanGraph with hierarchical steps

4. Safety Check (Safety Gateway)
   ├── Rule-based checks (banned tools, constraints)
   ├── ErisML evaluation (MoralVector, Bond Index)
   └── Decision: ALLOW / BLOCK / REVISE

5. Metacognition Review
   └── Self-check: ACCEPT / REJECT / REVISE

6. Plan Execution
   ├── Publish steps to Event Fabric
   ├── RH simulates and executes
   └── In-action safety monitors continuously

7. Post-Action Learning
   └── Log outcomes to episodic memory
```

### Safety Decision Flow

```
PlanStep → PlanStepToEthicalFacts → EthicalFactsProto
                                          │
                                          ▼
                                   ErisMLService.EvaluateStep()
                                          │
                                          ▼
                              ┌───────────┴───────────┐
                              │    MoralVector        │
                              │  (8+1 dimensions)     │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   Verdict Decision    │
                              │ strongly_prefer       │
                              │ prefer                │
                              │ neutral               │
                              │ avoid                 │
                              │ forbid (VETO)         │
                              └───────────┬───────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   DecisionProof       │
                              │  (hash-chained)       │
                              └───────────────────────┘
```

## Protocol Buffers

All inter-service communication uses gRPC with Protocol Buffers.

**Location:** `proto/`

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

**Generated Code:** `src/agi/proto_gen/`

## Configuration

Configuration files are in `configs/`:

| File | Description |
|------|-------------|
| `lh.yaml` | Left Hemisphere service config |
| `lh_config.yaml` | LH detailed parameters |
| `rh_config.yaml` | Right Hemisphere config |
| `memory_config.yaml` | Memory services config |
| `safety_config.yaml` | Safety thresholds and policies |
| `meta_config.yaml` | Metacognition parameters |
| `env_config.yaml` | Environment interface config |

## Key Design Principles

### 1. Safety-First Architecture
Safety is not a filter at the end but woven throughout:
- Pre-action: Check before execution
- In-action: Monitor during execution
- Post-action: Learn from outcomes

### 2. Dual-Hemisphere Design
Inspired by cognitive neuroscience:
- LH: Slow, deliberative, symbolic reasoning
- RH: Fast, reactive, subsymbolic processing

### 3. Formal Ethics Integration
ErisML provides mathematically grounded ethical reasoning:
- Bond Index: Quantifies ethical consistency
- Hohfeldian Analysis: Rights/duties verification
- Decision Proofs: Auditable governance

### 4. HPC-Ready Architecture
Designed for distributed deployment:
- Stateless services with external state stores
- Event-driven communication
- Horizontal scaling via DHT

### 5. Graceful Degradation
System remains safe when components fail:
- Safety Gateway works without ErisML
- LH works without Memory services
- All services have fallback behaviors

## Related Documentation

- [ERISML_API.md](ERISML_API.md) - ErisML integration API reference
- [ERISML_INTEGRATION_SKETCH.md](ERISML_INTEGRATION_SKETCH.md) - Detailed integration design
- [LH_SPRINT_PLAN.md](LH_SPRINT_PLAN.md) - Left Hemisphere development plan
- [HPC_DEPLOYMENT.md](HPC_DEPLOYMENT.md) - HPC cluster deployment guide
