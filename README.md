# 🧠 AGI-HPC  
### Embodied, safety-aware AGI architecture for High Performance Computing clusters

![CI](https://github.com/ahb-sjsu/agi-hpc/actions/workflows/ci.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

AGI-HPC is a **modular, distributed, safety-gated cognitive architecture** designed to run on **university-scale HPC clusters** (SJSU CoE HPC) using:

- **Dual-hemisphere cognition** (LH + RH)  
- **Tiered memory subsystem** (semantic, episodic, procedural)  
- **Multi-layer safety system** (pre-action, in-action, post-action)  
- **Virtual embodiment environment** (Unity/MuJoCo)  
- **Event-driven coordination** (UCX/ZeroMQ fabric)  
- **gRPC services** across all components  
- **Apptainer/Singularity containers** for reproducible HPC deployment

This repository provides the **full scaffolding and automation tools** necessary to build, test, and deploy the system incrementally.

---

# 📐 Architecture Overview

Below is a high-level diagram of the full AGI-HPC multi-service system:
```mermaid
flowchart LR
    %% ============================================================
    %%  TOP LEVEL SYSTEM
    %% ============================================================
    subgraph HPC_Cluster["🏔️ HPC Cluster (SJSU CoE)"]
    direction LR

        %% ========================================================
        %% Left Hemisphere
        %% ========================================================
        subgraph LH["🧠 Left Hemisphere (Reasoning + Planning)"]
        direction TB
            LH_RPC["gRPC Server\nPlanService / MetaService"]
            LH_MEM["Semantic Memory Queries"]
            LH_SAFETY["Pre-Action Safety RPC"]
            LH_META["Metacognition Client\n(ReviewPlan)"]
            LH_EVENTS["EventFabric Publisher\n(plan.step_ready)"]
        end

        %% ========================================================
        %% Right Hemisphere
        %% ========================================================
        subgraph RH["👁️ Right Hemisphere (Perception + World Model + Control)"]
        direction TB
            RH_RPC["gRPC Server\nSimulatePlan / ControlService"]
            RH_PERCEPTION["Perception Pipeline\n(vision encoders, object detection)"]
            RH_WM["World Model\n(short-horizon physics & prediction)"]
            RH_SAFETY["In-Action Safety RPC"]
            RH_POST["Post-Action Safety RPC"]
            RH_EVENTS["EventFabric Publisher\n(perception.state_update,\n simulation.result)"]
        end

        %% ========================================================
        %% Memory Subsystem
        %% ========================================================
        subgraph MEM["💾 Memory Subsystem"]
        direction TB
            subgraph SEM["📚 Semantic Memory"]
                SEM_RPC["SemanticService RPC"]
                SEM_STORE["Vector Store (Qdrant/FAISS)\nConcepts, skills, facts"]
            end
            subgraph EPI["🎞️ Episodic Memory"]
                EPI_RPC["EpisodicService RPC"]
                EPI_LOG["Append-only Logs\n(JSONL/Parquet)"]
            end
            subgraph PROC["⚙️ Procedural Memory"]
                PROC_RPC["ProceduralService RPC"]
                PROC_SKILLS["Skill Catalog\n(pre/postconditions,\n policies)"]
            end
        end

        %% ========================================================
        %% Safety Subsystem
        %% ========================================================
        subgraph SAFETY["🛡️ Safety Subsystem"]
        direction TB
            PRE["Pre-Action Safety\n(CheckPlan)"]
            INACT["In-Action Safety\n(CheckStep)"]
            POST["Post-Action Safety\n(AnalyzeOutcome)"]
            RULES["Rule Engine\n(banned tools, constraints,\n risk scoring, thresholds)"]
        end

        %% ========================================================
        %% Metacognition
        %% ========================================================
        subgraph META["🔍 Metacognition"]
        direction TB
            META_RPC["MetacognitionService RPC"]
            META_ENGINE["Evaluation Engine\n(confidence, issues,\n ACCEPT/REVISE/REJECT)"]
        end

        %% ========================================================
        %% Event Fabric
        %% ========================================================
        subgraph FABRIC["🔌 Event Fabric (UCX/ZeroMQ)"]
        direction TB
            TOPICS["Topics:\nperception.state_update\nplan.step_ready\nsimulation.result\nsafety.*\nmeta.review"]
        end

    end

    %% ============================================================
    %% External Environment
    %% ============================================================
    subgraph ENV["🌍 Virtual Environment\n(Unity or MuJoCo)\n(runs on laptop)"]
    direction TB
        CAM["Camera Frames\nRGB-D / metadata"]
        STEP_API["HTTP/gRPC API\nStep(), Reset(), GetState()"]
    end

    %% ============================================================
    %%  CONNECTIONS
    %% ============================================================

    %% Environment → RH
    CAM --> RH_PERCEPTION
    STEP_API <--> RH_RPC

    %% Event Fabric links
    RH_EVENTS --> FABRIC
    LH_EVENTS --> FABRIC
    FABRIC --> LH_MEM
    FABRIC --> RH_WM

    FABRIC --> LH
    FABRIC --> RH

    %% LH RPC to memory & safety & meta
    LH_MEM --> SEM_RPC
    LH ---> SAFETY
    LH -- gRPC ReviewPlan --> META

    %% RH RPC to safety
    RH --> INACT
    RH --> POST

    %% Memory internal wiring
    SEM_RPC --> SEM_STORE
    EPI_RPC --> EPI_LOG
    PROC_RPC --> PROC_SKILLS

    %% Safety uses rule engine
    PRE --> RULES
    INACT --> RULES
    POST --> RULES

    META_RPC --> META_ENGINE

    %% LH -> RH plan dispatch
    LH_EVENTS --> RH
```
### Major subsystems

| Subsystem | Responsibilities | Implementation |
|----------|------------------|----------------|
| **Left Hemisphere (LH)** | Planning, reasoning, metacognition pipeline, symbolic tool use | gRPC service + event fabric |
| **Right Hemisphere (RH)** | Perception, world-model simulation, motor control | gRPC service + event fabric |
| **Semantic Memory** | Vector DB for concepts, embeddings, world knowledge | Qdrant/FAISS (placeholder in-memory for now) |
| **Episodic Memory** | Append-only event logs for replay + analysis | JSONL/Parquet logs |
| **Procedural Memory** | Skills, policies, reusable action graphs | In-memory → SQL later |
| **Safety (3-layer)** | Pre-action, in-action, post-action verification | gRPC services + rule engine |
| **Metacognition** | Cross-check plans, confidence estimation, revise/reject loop | gRPC service + trace analyzer |
| **Environment** | Virtual embodiment (Unity/MuJoCo) | WebSocket/gRPC interface |
| **Event Fabric** | Topic-based low-latency message bus | UCX/ZeroMQ stub (pluggable) |

---

# 🧭 Project Layout

```
agi-hpc/
│
├── src/agi/
│   ├── lh/                  # Left Hemisphere service
│   ├── rh/                  # Right Hemisphere service
│   ├── memory/              # Semantic, episodic, procedural memory
│   ├── safety/              # Safety subsystem (pre/in/post + rule engine)
│   ├── meta/                # Metacognition service
│   ├── core/                # RPC, event fabric, config loader
│   ├── proto_gen/           # Auto-generated protobuf stubs
│   └── env_client/          # Unity/MuJoCo environment client
│
├── proto/                   
├── configs/                 
├── infra/
│   ├── hpc/                
│   └── local/              
│
├── docs/                    
├── design/                  
│
├── generate_protos.py       
├── generate_services.py     
├── generate_memory_services.py
├── generate_safety_services.py
├── generate_metacog_service.py
│
└── .github/workflows/ci.yaml
```

---

# ⚡ Quickstart: Local Development

### 1. Clone:
```bash
git clone https://github.com/ahb-sjsu/agi-hpc.git
cd agi-hpc
```

### 2. Install dev environment:
```bash
pip install -e ".[dev]"
pre-commit install
```

### 3. Generate protobuf stubs:
```bash
python generate_protos.py --clean
```

### 4. Run LH/RH locally:
```bash
python src/agi/lh/service.py
python src/agi/rh/service.py
```

---

# 📜 License
MIT © 2025 Andrew Bond
