# RH (Right Hemisphere) Sprint Plan

## Current State Assessment

### Implemented (Scaffolding Complete)
| File | Status | Description |
|------|--------|-------------|
| `perception.py` | **Stub** | Perception pipeline with correct API, returns placeholder state |
| `world_model.py` | **Stub** | Short-horizon predictive model with heuristic rollouts |
| `control_service.py` | **Stub** | PlanStep → control action translation, rule-based |
| `simulation_service.py` | **Done** | gRPC SimulationService with full pipeline integration |
| `rh_event_loop.py` | **Done** | EventFabric subscriptions, async step handling |
| `service.py` | **TODO** | Just prints "RH service running" - needs gRPC server |
| `rh_service.py` | **TODO** | Empty file |
| `semantic_service.py` | **TODO** | Semantic features for RH |

### Proto Definitions
- `plan.proto` - SimulationRequest/SimulationResult defined
- `rh.proto` - Minimal placeholder (needs expansion)

### Key Dependencies
- `agi.core.events.fabric` - EventFabric for LH↔RH communication
- `agi.proto_gen.plan_pb2` - PlanStep, SimulationRequest/Result

---

## Sprint 1: Core RH Service Bootstrap

**Goal**: Get RH running as a real gRPC server that can receive SimulationRequests and execute plan steps.

### Tasks

#### 1.1 Wire up `service.py` as gRPC server
- [ ] Import `GRPCServer` from `agi.core.api.grpc_server`
- [ ] Import `EventFabric` from `agi.core.events.fabric`
- [ ] Instantiate `Perception`, `WorldModel`, `ControlService`
- [ ] Create `SimulationService` instance with all dependencies
- [ ] Register `SimulationService` with gRPC server on port 50057
- [ ] Add CLI argument parsing (port, config file)
- [ ] Add graceful shutdown handling

#### 1.2 Expand `rh.proto` definitions
- [ ] Define `PerceptionState` message (objects, poses, features)
- [ ] Define `WorldModelQuery` / `WorldModelResponse` messages
- [ ] Define `ControlCommand` / `ControlFeedback` messages
- [ ] Define `RHStatus` for health checking

#### 1.3 Add configuration loading
- [ ] Create `configs/rh.yaml` with service parameters
- [ ] Support environment variable overrides (`AGI_RH_PORT`, etc.)
- [ ] Configure perception model settings
- [ ] Configure world model horizon and physics parameters

#### 1.4 EventFabric integration
- [ ] Start `RHEventLoop` as asyncio task
- [ ] Verify `plan.step_ready` subscriptions work
- [ ] Publish `perception.state_update` events
- [ ] Add logging for all fabric events

### Acceptance Criteria
```bash
# RH should start and accept gRPC requests
python src/agi/rh/service.py --port 50057

# Test with grpcurl or Python client
grpcurl -plaintext localhost:50057 agi.plan.v1.SimulationService/Simulate
```

---

## Sprint 2: Unit Tests for RH Components

**Goal**: Achieve 80%+ test coverage for RH module.

### Tasks

#### 2.1 Test `perception.py`
- [ ] `test_perception_update_observation`
- [ ] `test_perception_current_state_returns_dict`
- [ ] `test_perception_extract_features`
- [ ] `test_perception_detect_objects`
- [ ] `test_perception_build_state_representation`
- [ ] `test_perception_handles_invalid_frame`

#### 2.2 Test `world_model.py`
- [ ] `test_world_model_rollout_returns_result`
- [ ] `test_world_model_risk_estimation`
- [ ] `test_world_model_violation_detection`
- [ ] `test_world_model_horizon_limit`
- [ ] `test_world_model_predicted_states_trajectory`
- [ ] `test_world_model_overspeed_violation`
- [ ] `test_world_model_forbidden_action_violation`

#### 2.3 Test `control_service.py`
- [ ] `test_control_translate_move_step`
- [ ] `test_control_translate_manipulation_step`
- [ ] `test_control_translate_scan_step`
- [ ] `test_control_default_fallback`
- [ ] `test_control_disabled_mode`
- [ ] `test_control_execute_actions_async`

#### 2.4 Test `simulation_service.py`
- [ ] `test_simulation_service_simulate_rpc`
- [ ] `test_simulation_service_uses_perception`
- [ ] `test_simulation_service_uses_world_model`
- [ ] `test_simulation_service_aggregates_risk`
- [ ] `test_simulation_service_publishes_event`
- [ ] `test_simulation_service_handles_errors`

#### 2.5 Test `rh_event_loop.py`
- [ ] `test_event_loop_subscribes_to_fabric`
- [ ] `test_event_loop_handles_plan_step_ready`
- [ ] `test_event_loop_handles_perception_update`
- [ ] `test_event_loop_shutdown`

### Test Infrastructure
```python
# tests/rh/conftest.py
@pytest.fixture
def mock_fabric():
    """In-memory EventFabric for testing."""
    return EventFabric(mode="local")

@pytest.fixture
def perception():
    """Perception instance for testing."""
    return Perception(model_name="test_encoder")

@pytest.fixture
def world_model():
    """WorldModel instance for testing."""
    return WorldModel(model_name="test_model", horizon=5)

@pytest.fixture
def control():
    """ControlService instance for testing."""
    return ControlService(controller_type="rule_based")

@pytest.fixture
def simulation_service(mock_fabric, perception, world_model, control):
    """SimulationService with mocked dependencies."""
    return SimulationService(
        world_model=world_model,
        perception=perception,
        control=control,
        fabric=mock_fabric,
    )
```

---

## Sprint 3: Integration Testing & E2E Pipeline

**Goal**: Verify RH works end-to-end with LH and Safety services.

### Tasks

#### 3.1 LH → RH simulation flow
- [ ] LH sends `SimulationRequest` to RH
- [ ] RH runs rollout and returns `SimulationResult`
- [ ] Verify risk scores propagate to LH
- [ ] Verify violations are detected

#### 3.2 RH → Safety integration
- [ ] Connect RH to In-Action Safety service
- [ ] Real-time safety checks during control execution
- [ ] Emergency stop on safety violations

#### 3.3 EventFabric LH ↔ RH communication
- [ ] LH publishes `plan.step_ready`
- [ ] RH receives via `RHEventLoop`
- [ ] RH executes steps and publishes results
- [ ] LH receives `simulation.result` events

#### 3.4 Docker Compose local stack
- [ ] Create `infra/local/docker-compose.rh.yaml`
- [ ] Include: RH, LH, Safety (mock), ZMQ fabric
- [ ] One-command startup: `docker-compose up rh-stack`

---

## Sprint 4: Advanced Perception Pipeline

**Goal**: Replace stub perception with real ML models.

### Tasks

#### 4.1 Vision encoder integration
- [ ] Define `VisionEncoder` protocol/interface
- [ ] Implement `CLIPEncoder` for semantic features
- [ ] Implement `DINOv2Encoder` for visual features
- [ ] Implement `ViTEncoder` for general embeddings
- [ ] GPU acceleration with PyTorch/TensorRT

#### 4.2 Object detection
- [ ] Implement `YOLODetector` for real-time detection
- [ ] Implement `SAMSegmenter` for instance segmentation
- [ ] Implement `GroundingDINO` for open-vocabulary detection
- [ ] 3D bounding box estimation
- [ ] Object tracking (SORT/DeepSORT)

#### 4.3 Depth and pose estimation
- [ ] Depth estimation (MiDaS, ZoeDepth)
- [ ] 6-DOF pose estimation
- [ ] Point cloud generation
- [ ] Spatial relationship inference

#### 4.4 Multi-modal fusion
- [ ] Camera + LiDAR fusion
- [ ] Temporal smoothing
- [ ] Uncertainty quantification
- [ ] Confidence calibration

#### 4.5 Configuration
- [ ] `AGI_RH_PERCEPTION_MODEL` env var
- [ ] Model selection (clip/dino/yolo/sam)
- [ ] Device selection (cpu/cuda/mps)
- [ ] Batch size, resolution settings

---

## Sprint 5: Physics-Based World Model

**Goal**: Replace stub world model with real physics simulation.

### Tasks

#### 5.1 Physics engine integration
- [ ] Define `PhysicsEngine` protocol/interface
- [ ] Implement `MuJoCoEngine` adapter
- [ ] Implement `PyBulletEngine` adapter
- [ ] Implement `IsaacGymEngine` adapter (GPU)
- [ ] Implement `UnityMLEngine` adapter

#### 5.2 Predictive dynamics
- [ ] Forward dynamics simulation
- [ ] Contact/collision detection
- [ ] Rigid body physics
- [ ] Soft body/deformable objects
- [ ] Fluid dynamics (optional)

#### 5.3 Learned world models
- [ ] Implement `DreamerV3` world model
- [ ] Implement `IRIS` (transformer world model)
- [ ] Train on domain-specific data
- [ ] Uncertainty-aware predictions

#### 5.4 Risk estimation
- [ ] Collision risk from physics
- [ ] Stability analysis
- [ ] Reachability analysis
- [ ] Constraint satisfaction checking

#### 5.5 World state persistence (PostgreSQL + PostGIS)
- [ ] Schema design for spatial world state
- [ ] Object positions as PostGIS `POINT` / `POINTZ`
- [ ] Object bounding boxes as PostGIS `BOX3D`
- [ ] Trajectory storage as PostGIS `LINESTRINGZ`
- [ ] Spatial indexing (R-tree)
- [ ] Temporal versioning (valid_from, valid_to)
- [ ] Query interface for spatial relationships

```sql
-- Example schema for world state persistence
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE world_objects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id VARCHAR(255) NOT NULL,
    label VARCHAR(255),
    confidence FLOAT,
    position GEOMETRY(POINTZ, 4326),
    bounding_box GEOMETRY(POLYGON, 4326),
    pose JSONB,  -- 6-DOF pose as JSON
    properties JSONB,
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    valid_to TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_world_objects_position ON world_objects USING GIST (position);
CREATE INDEX idx_world_objects_bbox ON world_objects USING GIST (bounding_box);
CREATE INDEX idx_world_objects_temporal ON world_objects (valid_from, valid_to);

CREATE TABLE agent_trajectory (
    id SERIAL PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    trajectory GEOMETRY(LINESTRINGZ, 4326),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX idx_trajectory_geom ON agent_trajectory USING GIST (trajectory);

-- Spatial query example: find objects within 1 meter of agent
SELECT * FROM world_objects
WHERE ST_DWithin(position, ST_MakePoint(0, 0, 0)::geography, 1.0)
  AND valid_to IS NULL;
```

---

## Sprint 6: Advanced Control & Robotics

**Goal**: Replace stub control with real motor control and robotics interfaces.

### Tasks

#### 6.1 Motor primitive library
- [ ] Define `MotorPrimitive` protocol
- [ ] Implement reach/grasp/place primitives
- [ ] Implement navigation primitives
- [ ] Implement manipulation sequences
- [ ] Behavior tree integration

#### 6.2 Trajectory planning
- [ ] RRT/RRT* path planning
- [ ] Trajectory optimization (CHOMP, TrajOpt)
- [ ] Motion planning with MoveIt2
- [ ] Collision-free trajectory generation

#### 6.3 Real-time control
- [ ] PID controllers
- [ ] Model Predictive Control (MPC)
- [ ] Impedance/admittance control
- [ ] Force-torque feedback

#### 6.4 Robot interface adapters
- [ ] ROS2 bridge adapter
- [ ] Direct URDF loading
- [ ] Hardware abstraction layer
- [ ] Sensor driver integration

#### 6.5 Simulation environments
- [ ] MuJoCo environment wrapper
- [ ] Isaac Sim environment wrapper
- [ ] Unity ML-Agents wrapper
- [ ] Gazebo wrapper

---

## Sprint 7: World Persistence Layer

**Goal**: Full PostgreSQL/PostGIS integration for spatial world state.

### Tasks

#### 7.1 Database infrastructure
- [ ] PostgreSQL 16+ with PostGIS 3.4
- [ ] Connection pooling (PgBouncer)
- [ ] Schema migrations (Alembic)
- [ ] Read replicas for query scaling

#### 7.2 Python ORM/adapters
- [ ] SQLAlchemy + GeoAlchemy2 models
- [ ] Async database access (asyncpg)
- [ ] Connection pool management
- [ ] Query builders for spatial operations

```python
# src/agi/rh/persistence/models.py
from geoalchemy2 import Geometry
from sqlalchemy import Column, String, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID

class WorldObject(Base):
    __tablename__ = 'world_objects'

    id = Column(UUID, primary_key=True)
    object_id = Column(String(255), nullable=False)
    label = Column(String(255))
    confidence = Column(Float)
    position = Column(Geometry('POINTZ', srid=4326))
    bounding_box = Column(Geometry('POLYGON', srid=4326))
    pose = Column(JSON)
    properties = Column(JSON)
    valid_from = Column(DateTime)
    valid_to = Column(DateTime)
```

#### 7.3 World state service
- [ ] `WorldPersistence` class
- [ ] `save_state()` - persist current perception state
- [ ] `load_state()` - restore state from DB
- [ ] `query_nearby()` - spatial proximity queries
- [ ] `get_history()` - temporal queries
- [ ] `replay_trajectory()` - trajectory playback

```python
# src/agi/rh/persistence/service.py
class WorldPersistence:
    async def save_state(self, perception_state: Dict) -> None:
        """Persist current perception state to PostGIS."""
        pass

    async def load_state(self, timestamp: datetime = None) -> Dict:
        """Load world state, optionally at specific time."""
        pass

    async def query_nearby(
        self,
        position: Tuple[float, float, float],
        radius: float,
        label: str = None,
    ) -> List[WorldObject]:
        """Find objects within radius of position."""
        pass

    async def get_trajectory(
        self,
        agent_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Tuple[float, float, float]]:
        """Get agent trajectory between times."""
        pass
```

#### 7.4 Spatial analytics
- [ ] Reachability analysis (what can agent reach?)
- [ ] Visibility analysis (what can agent see?)
- [ ] Path feasibility (can agent navigate to X?)
- [ ] Scene change detection

#### 7.5 Configuration
- [ ] `AGI_RH_DB_URL` env var
- [ ] Connection pool settings
- [ ] Persistence interval
- [ ] History retention policy

---

## Sprint 8: Production Hardening

**Goal**: Prepare RH for HPC deployment.

### Tasks

#### 8.1 Observability
- [ ] Prometheus metrics (perception latency, simulation time, control loop frequency)
- [ ] Structured logging with correlation IDs
- [ ] Distributed tracing (OpenTelemetry)
- [ ] GPU utilization metrics

#### 8.2 Error handling & resilience
- [ ] Retry logic for transient failures
- [ ] Circuit breakers for perception models
- [ ] Graceful degradation (fallback perception)
- [ ] Watchdog for control loop health

#### 8.3 Performance optimization
- [ ] GPU batching for perception
- [ ] Parallel rollouts in world model
- [ ] CUDA streams for overlapping compute
- [ ] Memory-efficient trajectory storage

#### 8.4 Real-time guarantees
- [ ] Control loop timing (100Hz+)
- [ ] Perception pipeline budget (<50ms)
- [ ] World model rollout budget (<100ms)
- [ ] Priority scheduling

#### 8.5 HPC deployment
- [ ] Apptainer/Singularity container with CUDA
- [ ] SLURM job scripts for RH nodes
- [ ] Multi-GPU world model training
- [ ] UCX fabric configuration

---

## File Structure After Completion

```
src/agi/rh/
├── __init__.py
├── service.py              # gRPC server entrypoint
├── simulation_service.py   # SimulationService implementation
├── rh_event_loop.py        # EventFabric event loop
├── perception/
│   ├── __init__.py
│   ├── perception.py       # Main perception class
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── base.py         # VisionEncoder protocol
│   │   ├── clip.py         # CLIP encoder
│   │   ├── dino.py         # DINOv2 encoder
│   │   └── vit.py          # ViT encoder
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py         # Detector protocol
│   │   ├── yolo.py         # YOLO detector
│   │   ├── sam.py          # SAM segmenter
│   │   └── grounding_dino.py
│   └── depth/
│       ├── __init__.py
│       ├── midas.py
│       └── zoe.py
├── world_model/
│   ├── __init__.py
│   ├── world_model.py      # Main world model class
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── base.py         # PhysicsEngine protocol
│   │   ├── mujoco.py       # MuJoCo adapter
│   │   ├── pybullet.py     # PyBullet adapter
│   │   └── isaac.py        # Isaac Gym adapter
│   └── learned/
│       ├── __init__.py
│       ├── dreamer.py      # DreamerV3
│       └── iris.py         # Transformer world model
├── control/
│   ├── __init__.py
│   ├── control_service.py  # Main control class
│   ├── primitives/
│   │   ├── __init__.py
│   │   ├── base.py         # MotorPrimitive protocol
│   │   ├── reach.py
│   │   ├── grasp.py
│   │   └── navigate.py
│   ├── planners/
│   │   ├── __init__.py
│   │   ├── rrt.py
│   │   └── trajopt.py
│   └── robots/
│       ├── __init__.py
│       ├── ros2.py         # ROS2 bridge
│       └── urdf.py         # URDF loader
├── persistence/
│   ├── __init__.py
│   ├── models.py           # SQLAlchemy/GeoAlchemy models
│   ├── service.py          # WorldPersistence service
│   ├── migrations/         # Alembic migrations
│   └── queries.py          # Spatial query builders
└── config.py               # RH configuration dataclass

tests/rh/
├── __init__.py
├── conftest.py             # Fixtures
├── test_perception.py
├── test_world_model.py
├── test_control_service.py
├── test_simulation_service.py
├── test_rh_event_loop.py
├── test_persistence.py
└── integration/
    ├── test_lh_rh_flow.py
    └── test_full_pipeline.py

configs/
├── rh.yaml                 # RH service configuration
├── rh_config.yaml          # RH detailed parameters
└── rh_db.yaml              # PostgreSQL/PostGIS config
```

---

## PostgreSQL/PostGIS Configuration

### Docker Compose for Development

```yaml
# docker-compose.rh-db.yaml
version: '3.8'

services:
  rh-postgres:
    image: postgis/postgis:16-3.4
    container_name: rh-world-db
    environment:
      POSTGRES_USER: agi_rh
      POSTGRES_PASSWORD: ${RH_DB_PASSWORD:-secret}
      POSTGRES_DB: world_state
    ports:
      - "5432:5432"
    volumes:
      - rh-world-data:/var/lib/postgresql/data
      - ./init-postgis.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agi_rh -d world_state"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  rh-world-data:
```

### Connection Configuration

```yaml
# configs/rh_db.yaml
database:
  url: "postgresql+asyncpg://agi_rh:secret@localhost:5432/world_state"
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 1800

persistence:
  save_interval_ms: 100  # Persist state every 100ms
  history_retention_days: 7
  spatial_index_refresh_interval_s: 60
```

---

## Priority Order

1. **Sprint 1** - Critical: RH must be runnable as gRPC server
2. **Sprint 2** - High: Tests enable safe iteration
3. **Sprint 3** - High: Validates LH↔RH architecture
4. **Sprint 7** - High: World persistence (PostGIS) foundational
5. **Sprint 4** - Medium: Real perception models
6. **Sprint 5** - Medium: Real physics simulation
7. **Sprint 6** - Medium: Robotics integration
8. **Sprint 8** - Low (for now): Production concerns

---

## Quick Start Command (After Sprint 1 + 7)

```bash
# Terminal 1: Start PostgreSQL/PostGIS
docker-compose -f docker-compose.rh-db.yaml up -d

# Terminal 2: Start RH
cd agi-hpc
export AGI_FABRIC_MODE=local
export AGI_RH_DB_URL="postgresql+asyncpg://agi_rh:secret@localhost:5432/world_state"
python src/agi/rh/service.py --port 50057

# Terminal 3: Test simulation request
python -c "
import grpc
from agi.proto_gen import plan_pb2, plan_pb2_grpc

channel = grpc.insecure_channel('localhost:50057')
stub = plan_pb2_grpc.SimulationServiceStub(channel)

# Create a test step
step = plan_pb2.PlanStep(
    step_id='test-step-001',
    index=0,
    kind='action',
    description='Move to red cube',
    tool_id='navigation',
)

request = plan_pb2.SimulationRequest(
    plan_id='test-plan-001',
)
request.candidate_steps.append(step)

result = stub.Simulate(request)
print(f'Plan ID: {result.plan_id}')
print(f'Overall Risk: {result.overall_risk:.3f}')
print(f'Approved: {result.approved}')
print(f'Step Risks: {list(result.step_risk)}')
"
```

---

## Spatial Query Examples (PostGIS)

```python
# Find objects within 2 meters of agent's current position
nearby = await world_persistence.query_nearby(
    position=(1.5, 0.0, 0.5),
    radius=2.0,
)

# Find all red cubes in the scene
red_cubes = await world_persistence.query_by_label(
    label="red_cube",
    valid_at=datetime.now(),
)

# Get agent trajectory for the last 10 seconds
trajectory = await world_persistence.get_trajectory(
    agent_id="robot_1",
    start_time=datetime.now() - timedelta(seconds=10),
    end_time=datetime.now(),
)

# Check if path between two points is clear of obstacles
is_clear = await world_persistence.check_path_clear(
    start=(0, 0, 0),
    end=(2, 3, 0),
    clearance=0.5,  # 50cm clearance
)
```

---

## Dependencies

```toml
# pyproject.toml additions for RH
[project.optional-dependencies]
rh = [
    # Database
    "asyncpg>=0.29.0",
    "sqlalchemy[asyncio]>=2.0",
    "geoalchemy2>=0.14.0",
    "alembic>=1.13.0",

    # Perception
    "torch>=2.0",
    "torchvision>=0.15",
    "transformers>=4.35",
    "ultralytics>=8.0",  # YOLO
    "segment-anything>=1.0",  # SAM

    # Physics
    "mujoco>=3.0",
    "pybullet>=3.2",

    # Robotics
    "numpy>=1.24",
    "scipy>=1.11",
    "open3d>=0.18",  # Point clouds
]

rh-gpu = [
    "agi-hpc[rh]",
    "cupy>=12.0",
    "tensorrt>=8.6",
]
```
