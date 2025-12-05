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

```
Left Hemisphere  <---->  Memory Subsystem <---->  Right Hemisphere
       |                    |                         |
       |                    |                         |
   Pre-Action Safety   <----|---->  In-Action Safety  ---->  Post-Action Safety
       |
   Metacognition
       |
 Virtual Environment (Unity/MuJoCo)
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
