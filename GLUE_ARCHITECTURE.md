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

    Unity Simulation
        |
        v
    Bridge Service  <-->  AGI-HPC EventBus
        |                      |
        |                +------------+
        |                | Memory     |
        v                | LH / RH    |
    Control Commands <---+------------+

More explicitly:

    Unity
      -> SensorPacket (UDP/ZeroMQ)
      -> bridge_service
      -> EventBus topic: sensors.raw
      -> LH (perception) -> EventBus (perception.state)
      -> RH (control)    -> EventBus (control.command)
      -> bridge_service
      -> ControlCommand (UDP/ZeroMQ)
      -> Unity

---

## 3. Timing & Control Loop Contracts

### 3.1 Time Master

- **Unity is the time master.**
  - Unity’s fixed timestep (e.g., 0.05 s → 20 Hz) drives the control loop.
  - At each physics tick:
    1. Unity gathers sensor data.
    2. Unity sends a `SensorPacket` to the bridge.
    3. Unity waits up to a timeout for a `ControlCommand`.
    4. If command is late/missing:
       - Unity applies the **last valid command** (or a safe default).

### 3.2 Nominal Rates

- **Physics / control rate:** 10–20 Hz (to be fixed and documented).  
- The **Bridge** and **AGI-HPC** must be capable of handling this rate with headroom.

### 3.3 Time Stamps

All messages that leave a process must include:

- **Sim time** (float seconds since sim start, `sim_time_s`)  
- Optionally: **wall-clock time** (`unix_time_ms`)  

Sim time is used for replay and training; wall-clock is used for debugging.

---

## 4. Unity ↔ Bridge Interface

This is the **external IO** between Unity and an out-of-process controller.

### 4.1 Transport

**Required (MVP):**

- **Protocol:** UDP or ZeroMQ (REQ/REP or PUB/SUB) over TCP  
- **Direction:**
  - Unity → Bridge: sensor stream  
  - Bridge → Unity: control stream  

For MVP simplicity, the recommended pattern is:

- Unity: **client**  
- Bridge: **server**  

Example (UDP):

- Unity sends `SensorPacket` to `BRIDGE_HOST:BRIDGE_SENSOR_PORT`.  
- Bridge sends `ControlCommand` back to `UNITY_HOST:UNITY_CONTROL_PORT`.  

Exact choice (UDP vs ZeroMQ) to be frozen at **v1.0** and documented here.

### 4.2 Encoding

Two layers:

1. **Logical schema:** defined here (fields, types, units).  
2. **Physical encoding:**
   - MVP: **JSON** over the wire (easy debugging).  
   - Production/stretch: **Protobuf** using `.proto` files under `proto/`.  

All teams must keep JSON and Protobuf logically isomorphic.

### 4.3 Message: `SensorPacket`

Logical fields (JSON example, shown here as pseudocode):

    {
      "version": "1.0",
      "sim_time_s": 12.34,
      "unix_time_ms": 1733592345123,
      "vessel": {
        "lat_deg": 37.123456,
        "lon_deg": -122.123456,
        "heading_deg": 45.0,
        "u_surge_mps": 4.5,
        "v_sway_mps": 0.2,
        "yaw_rate_dps": 2.0
      },
      "imu": {
        "roll_deg": 1.2,
        "pitch_deg": 0.5,
        "yaw_deg": 45.0,
        "gyro_dps": { "x": 0.1, "y": 0.0, "z": 2.0 },
        "accel_mps2": { "x": 0.01, "y": 0.0, "z": -9.8 }
      },
      "gps": {
        "sog_mps": 4.6,
        "cog_deg": 46.0,
        "valid": true
      },
      "speed_log": {
        "stw_mps": 4.4
      },
      "engines": {
        "port_rpm": 1500.0,
        "stbd_rpm": 1500.0,
        "gear_port": "FWD",
        "gear_stbd": "FWD"
      },
      "env": {
        "wind_speed_mps": 6.0,
        "wind_dir_deg": 210.0,
        "current_speed_mps": 0.5,
        "current_dir_deg": 90.0,
        "wave_height_m": 0.8
      },
      "proximity": {
        "min_range_m": 50.0,
        "sectors": [
          { "bearing_deg": -45.0, "range_m": 60.0 },
          { "bearing_deg": 0.0,  "range_m": 40.0 },
          { "bearing_deg": 45.0, "range_m": 80.0 }
        ]
      },
      "mode": {
        "control_mode": "AUTONOMOUS",
        "mission_id": "wp_demo_001"
      }
    }

#### Required Fields (v1.0)

- `version`  
- `sim_time_s`  
- `vessel.lat_deg`, `vessel.lon_deg`, `vessel.heading_deg`  
- `vessel.u_surge_mps`  
- At least one velocity measurement (`gps.sog_mps` or `speed_log.stw_mps`)  
- `mode.control_mode`  

All other fields optional but recommended.

### 4.4 Message: `ControlCommand`

Logical schema:

    {
      "version": "1.0",
      "sim_time_s": 12.39,
      "unix_time_ms": 1733592345173,
      "command_id": "cmd_000123",
      "actuators": {
        "throttle_port": 0.7,
        "throttle_stbd": 0.7,
        "rudder_angle_deg": 5.0
      },
      "gear": {
        "port": "FWD",
        "stbd": "FWD"
      },
      "mode": {
        "control_mode": "AUTONOMOUS",
        "hold_heading_deg": 45.0,
        "target_waypoint": {
          "lat_deg": 37.124000,
          "lon_deg": -122.124000
        }
      },
      "safety": {
        "e_stop": false
      }
    }

#### Required Fields (v1.0)

- `version`  
- `sim_time_s` (time at which command is intended to be applied)  
- `actuators.throttle_port` ∈ [-1.0, 1.0]  
- `actuators.throttle_stbd` ∈ [-1.0, 1.0]  
- `actuators.rudder_angle_deg` ∈ [min_rudder, max_rudder] (e.g. [-35, 35])  
- `mode.control_mode` ∈ {`MANUAL`, `AUTONOMOUS`, `SAFE`}  

### 4.5 Unity Responsibilities

Unity team must:

- Provide a **single script** (e.g., `NetworkBridge.cs`) that:
  - Listens for `ControlCommand` and updates:
    - Engine commands  
    - Rudder angle  
  - Sends `SensorPacket` at each physics tick.  
- Ensure physical behavior is consistent with:
  - Internal dynamics model (documented in `DIGITAL_TWIN.md`).  

---

## 5. Bridge ↔ AGI-HPC Interface

The bridge connects external IO (Unity) to the internal **EventBus**.

### 5.1 EventBus Basics

- **Transport:** Implementation-dependent (e.g., ZeroMQ, in-process bus, Redis, etc.).  
- **Contract:** Pub/sub topics and message schemas are shared and fixed here.

### 5.2 Topics

Required topics (v1.0):

- `sensors.raw`  
  - Payload: full `SensorPacket` as received from Unity.  
- `perception.state`  
  - LH publishes processed/normalized state.  
- `control.command`  
  - RH publishes `ControlCommand` objects to be sent to Unity.  
- `memory.store`  
  - Events to store experiences.  
- `memory.query` / `memory.result`  
  - Asynchronous memory query/response.  
- `episodes.log`  
  - Time-indexed logs for training/evaluation.  

Topic names are strings; underlying tech can map them as needed.

### 5.3 Bridge Behavior

- Subscribes to **`control.command`**:
  - On each new command:
    - Encodes as JSON/Protobuf.  
    - Sends to Unity via the external transport.  

- Publishes to **`sensors.raw`**:
  - On each incoming Unity `SensorPacket`:
    - Decodes.  
    - Adds metadata (`source = "unity"`).  
    - Publishes to EventBus.  

- May optionally:
  - Log raw IO for diagnostics.  
  - Implement simple rate limiting / smoothing.  

---

## 6. Internal AGI-HPC Contracts

Teams implementing LH/RH, memory, and EventBus must agree on:

1. **Message envelopes**  
2. **Core message types**  
3. **RPC APIs for memory**  

### 6.1 Event Envelope

Every message on the EventBus should conform to a common envelope:

    {
      "envelope_version": "1.0",
      "event_type": "PerceptionUpdate",
      "event_id": "uuid-1234",
      "sim_time_s": 12.34,
      "unix_time_ms": 1733592345123,
      "source": "LH",
      "payload": { ... }
    }

- `event_type`: string enum (see below).  
- `payload`: type-specific schema.  

### 6.2 Event Types & Payloads (Core Set)

#### 6.2.1 `PerceptionUpdate`

- Published by **LH** to `perception.state`.  
- Payload example:

    {
      "pose": {
        "x_m": 12.3,
        "y_m": -4.5,
        "heading_deg": 45.0
      },
      "vel": {
        "speed_mps": 4.6,
        "yaw_rate_dps": 2.0
      },
      "env": {
        "wind_speed_mps": 6.0,
        "current_speed_mps": 0.5
      },
      "obstacles": [
        { "bearing_deg": 0.0, "range_m": 40.0 }
      ],
      "mission_state": {
        "waypoint_index": 1,
        "distance_to_waypoint_m": 120.0
      }
    }

#### 6.2.2 `ControlDecision`

- Published by **RH** to `control.command`.  
- `payload` is the logical `ControlCommand` body (without envelope).  

#### 6.2.3 `MemoryStore`

- Published by LH/RH/Bridge to `memory.store`.  
- Payload:

    {
      "store_type": "EPISODIC",
      "episode_id": "ep_20251207_0001",
      "sim_time_s": 12.34,
      "sensor_snapshot": { ... },
      "control_snapshot": { ... },
      "labels": {
        "mission_id": "wp_demo_001",
        "phase": "APPROACH",
        "success": null
      }
    }

#### 6.2.4 `MemoryQuery` & `MemoryResult`

- `MemoryQuery` payload:

    {
      "query_id": "q_0001",
      "query_type": "EPISODIC_NEAREST",
      "filters": {
        "mission_id": "wp_demo_001"
      },
      "criteria": {
        "desired_heading_deg": 45.0,
        "speed_mps": 4.6
      }
    }

- `MemoryResult` payload:

    {
      "query_id": "q_0001",
      "results": [
        {
          "episode_id": "ep_20251115_0123",
          "sim_time_s": 345.6,
          "similarity": 0.91,
          "metadata": { ... }
        }
      ]
    }

Exact fields can evolve, but the **pattern (query/result) and `event_type`s** must be maintained.

---

## 7. Memory Service RPC API

Memory services run as separate processes with a stable API.

### 7.1 Transport

- **gRPC** or similar RPC framework.  
- `.proto` files under `proto/memory.proto`.  

### 7.2 Core RPCs

Logical signatures (language-agnostic):

- `StoreEpisode(EpisodeRecord) -> StoreResponse`  
- `QueryEpisodes(QueryRequest) -> QueryResponse`  
- `GetEpisode(EpisodeId) -> EpisodeRecord`  

Where an example Protobuf sketch might look like:

    message EpisodeRecord {
      string episode_id = 1;
      double sim_time_s = 2;
      string mission_id = 3;
      bytes compressed_sensor_snapshot = 4;
      bytes compressed_control_snapshot = 5;
      map<string, string> labels = 6;
    }

    message QueryRequest {
      string query_id = 1;
      string mission_id = 2;
      // Additional fields TBD
    }

    message QueryResponse {
      string query_id = 1;
      repeated EpisodeRecord results = 2;
    }

(Exact `.proto` types can be filled in, but must preserve these logical roles.)

---

## 8. Configuration & Environment Contracts

### 8.1 Environment Variables

All services must honor these core env vars:

- `AGI_CONFIG_PATH` – path to a YAML/JSON config file.  
- `EVENTBUS_HOST`, `EVENTBUS_PORT`  
- `BRIDGE_HOST`, `BRIDGE_SENSOR_PORT`, `BRIDGE_CONTROL_PORT`  
- `LOG_DIR` – base directory for logs.  
- `DATA_DIR` – base directory for episodes and datasets.  
- `HPC_MODE` – `"true"` or `"false"`.  

### 8.2 CLI Entry Points

Each service must provide a CLI entry consistent with:

- `left_hemisphere --config $AGI_CONFIG_PATH`  
- `right_hemisphere --config $AGI_CONFIG_PATH`  
- `memory_service --config $AGI_CONFIG_PATH`  
- `eventbus_service --config $AGI_CONFIG_PATH`  
- `bridge_service --config $AGI_CONFIG_PATH`  

Other options allowed, but **these must work** in both local and HPC modes.

---

## 9. HPC Deployment Contracts

Details go into `HPC_DEPLOYMENT.md`, but high-level contracts are:

### 9.1 Containers

- Single **base image** provides:
  - Python 3.11  
  - Required dependencies for AGI-HPC  
  - Runtime for EventBus, Memory, LH, RH, Bridge  

### 9.2 SLURM Scripts

Standard job names and expectations:

- `memory_service.slurm`  
- `eventbus.slurm`  
- `left_hemisphere.slurm`  
- `right_hemisphere.slurm`  
- `full_system.slurm`  

Each script must:

- Set `LOG_DIR` and `DATA_DIR` in a predictable way.  
- Launch each process using the CLIs in §8.2.  
- Document which nodes/partitions they expect.  

---

## 10. Logging & Episodes

### 10.1 Log Format

At minimum, logs must be **line-oriented JSON** or structured text.

Each log entry should include:

- `unix_time_ms`  
- `sim_time_s`  
- `component` (e.g., `LH`, `RH`, `bridge`, `unity`, `memory`)  
- `level` (`INFO`, `WARN`, `ERROR`)  
- `message` and/or structured `fields`  

### 10.2 Episode Recording

A **Bridge** or dedicated **EpisodeRecorder** process will:

- Subscribe to:
  - `sensors.raw`  
  - `control.command`  
  - Optional: `perception.state`  
- Write a time-indexed record to:
  - A file in `$DATA_DIR/episodes/EPISODE_ID.*`  

All teams must ensure messages contain enough info for offline replay and training.

---

## 11. Versioning & Change Management

### 11.1 Interface Versions

- Messages carry `version` fields (`"1.0"` initially).  
- Any **breaking change** increments the major version (e.g., `"2.0"`).  
- Backward-compatible extensions (adding optional fields) may increment minor version.  

### 11.2 Change Process

- Any proposed interface change:
  - Must be documented in a short section at the end of this file (CHANGELOG).  
  - Must be communicated to all teams.  
  - Should provide a deprecation window if possible.  

---

## 12. Integration & Acceptance Tests

### 12.1 Reference Harnesses

To de-risk integration, the **advisor/glue** maintains:

1. `bridge/fake_unity_sim.py`  
   - Generates `SensorPacket`s from a simple kinematic model.  
   - Accepts `ControlCommand`s from AGI-HPC.  

2. `bridge/reference_controller.py`  
   - Reads `SensorPacket`s (from Unity or fake sim).  
   - Outputs basic `ControlCommand`s (heading/waypoint hold).  

These must always be kept in sync with the schemas in this document.

### 12.2 Minimum Integration Criteria

- **Local Integration:**
  - AGI-HPC stack controlling **real Unity** via bridge.  
  - Demonstrate:
    - Straight-line waypoint navigation  
    - Basic turning maneuver  

- **HPC Integration:**
  - LH, RH, Memory, EventBus on **multiple nodes**.  
  - Running (at least) with fake Unity sim; stretch goal: real Unity.  

---

## 13. Changelog (Interface-Level)

- **v0.1 (Draft)** – Initial glue architecture drafted; defines:
  - Unity ↔ Bridge schemas  
  - EventBus topics and envelopes  
  - Memory RPC pattern  
  - Basic configs and entrypoints  
