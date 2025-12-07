# AGI-HPC Maritime System – Glue Architecture & Integration Contracts

**Version:** 0.1 (Draft)  
**Owner:** Thesis Advisor / System Architect  
**Last Updated:** 2025-12-07  

---

## 1. Purpose & Scope

This document defines the **interfaces, contracts, and integration rules** for the AGI-HPC + Digital Twin + Unity Maritime Simulation project.

It is **not** an implementation guide. Individual theses and teams are free to choose internal designs, as long as they respect the contracts defined here.

### Goals

- Allow independent development of:
  - Maritime **Digital Twin & Unity Simulation**
  - **AGI-HPC cognitive stack** (LH/RH, memory, EventBus)
  - **HPC deployment & infrastructure**
- Ensure a **single end-to-end system** can be integrated with minimal friction:
  - Unity → AGI-HPC → Unity loop
  - Local and HPC/cluster modes

### Non-goals

- Detailed equations for vessel hydrodynamics (see `DIGITAL_TWIN.md`)
- Detailed cognitive algorithm design (see `AGENT_ARCHITECTURE.md`)
- Detailed HPC tuning and cluster specifics (see `HPC_DEPLOYMENT.md`)

---

## 2. System Overview

### 2.1 Components

1. **Unity Maritime Simulation**
   - 3D environment, vessel model, sensors  
   - Runs physics and graphics  
   - Sends sensor data out, applies control commands  

2. **Bridge Process** (`bridge_service`)
   - Sits between Unity and AGI-HPC  
   - Handles network IO, encoding/decoding of messages  
   - Normalizes transport so Unity and AGI-HPC don’t depend on each other’s tech stack  

3. **AGI-HPC Core**
   - **Left Hemisphere (LH):** perception, world modeling  
   - **Right Hemisphere (RH):** planning, control, behavior selection  
   - **Memory Services:** episodic, semantic, working, memory manager  
   - **EventBus:** publish/subscribe messaging layer  

4. **HPC Infrastructure**
   - Containerized services (Docker → Apptainer/Singularity)  
   - SLURM job scripts for distributed deployment  
   - Shared logging and data directories  

### 2.2 High-Level Data Flow

Conceptual pipeline (single mission step):

```text
Unity Simulation
    |
    v
Bridge Service  <-->  AGI-HPC EventBus
    |                      |
    |                +------------+
    |                | Memory     |
    v                | LH / RH    |
Control Commands <---+------------+
