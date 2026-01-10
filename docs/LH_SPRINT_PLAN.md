# LH (Left Hemisphere) Sprint Plan

## Current State Assessment

### Implemented (Scaffolding Complete)
| File | Status | Description |
|------|--------|-------------|
| `planner.py` | **Done** | Hierarchical planning engine with PlanGraph/PlanStep dataclasses |
| `plan_service.py` | **Done** | Full gRPC PlanService with pipeline orchestration |
| `memory_client.py` | **Stub** | Returns request unchanged, gRPC wiring present |
| `safety_client.py` | **Stub** | Returns mock approval, gRPC wiring present |
| `metacog_client.py` | **Stub** | Returns ACCEPT, gRPC wiring present |
| `service.py` | **TODO** | Just prints "LH service running" - needs gRPC server |

### Proto Definitions
- `plan.proto` - Comprehensive: PlanRequest, PlanResponse, PlanStep, PlanGraphProto, SimulationRequest/Result
- `lh.proto` - Minimal placeholder (needs expansion or merge with plan.proto)

---

## Sprint 1: Core LH Service Bootstrap

**Goal**: Get LH running as a real gRPC server that can accept PlanRequests and return PlanResponses.

### Tasks

#### 1.1 Wire up `service.py` as gRPC server
- [ ] Import `GRPCServer` from `agi.core.api.grpc_server`
- [ ] Import `EventFabric` from `agi.core.events.fabric`
- [ ] Instantiate `Planner`, `MemoryClient`, `SafetyClient`, `MetacognitionClient`
- [ ] Create `PlanService` instance with all dependencies
- [ ] Register `PlanService` with gRPC server
- [ ] Add CLI argument parsing (port, config file)
- [ ] Add graceful shutdown handling

#### 1.2 Add configuration loading
- [ ] Create `configs/lh.yaml` with service parameters
- [ ] Support environment variable overrides (`AGI_LH_PORT`, etc.)
- [ ] Configure client addresses (memory, safety, metacog)

#### 1.3 Local-mode EventFabric integration
- [ ] Verify `AGI_FABRIC_MODE=local` works
- [ ] Test plan step publishing to local fabric
- [ ] Add logging for published events

### Acceptance Criteria
```bash
# LH should start and accept gRPC requests
python src/agi/lh/service.py --port 50100

# Test with grpcurl or Python client
grpcurl -plaintext localhost:50100 agi.plan.v1.PlanService/Plan
```

---

## Sprint 2: Unit Tests for LH Components

**Goal**: Achieve 80%+ test coverage for LH module.

### Tasks

#### 2.1 Test `planner.py`
- [ ] `test_planner_generates_valid_plan_graph`
- [ ] `test_planner_creates_hierarchical_structure` (mission → subgoal → step)
- [ ] `test_planner_extracts_goal_text_from_request`
- [ ] `test_planner_handles_empty_request`
- [ ] `test_plan_step_has_required_fields`
- [ ] `test_plan_graph_metadata_populated`

#### 2.2 Test `plan_service.py`
- [ ] `test_plan_service_returns_valid_response`
- [ ] `test_plan_service_calls_memory_enrichment`
- [ ] `test_plan_service_calls_safety_check`
- [ ] `test_plan_service_calls_metacognition_review`
- [ ] `test_plan_service_publishes_steps_to_fabric`
- [ ] `test_plan_service_handles_safety_rejection`
- [ ] `test_plan_service_handles_metacog_reject`
- [ ] `test_plan_service_handles_metacog_revise`

#### 2.3 Test client stubs
- [ ] `test_memory_client_passthrough_when_unavailable`
- [ ] `test_safety_client_mock_approval_when_unavailable`
- [ ] `test_metacog_client_mock_accept_when_unavailable`

### Test Infrastructure
```python
# tests/lh/conftest.py
@pytest.fixture
def mock_fabric():
    """In-memory EventFabric for testing."""
    return EventFabric(mode="local")

@pytest.fixture
def plan_service(mock_fabric):
    """PlanService with mocked dependencies."""
    return PlanService(
        planner=Planner(),
        memory=MockMemoryClient(),
        safety=MockSafetyClient(approved=True),
        metacog=MockMetacogClient(decision="ACCEPT"),
        fabric=mock_fabric,
    )
```

---

## Sprint 3: Integration Testing & E2E Pipeline

**Goal**: Verify LH works end-to-end with other services (mock or real).

### Tasks

#### 3.1 LH ↔ Safety integration
- [ ] Start Safety service (or mock)
- [ ] LH sends PlanGraphProto to Safety
- [ ] Verify safety rejection blocks plan
- [ ] Verify safety approval allows plan

#### 3.2 LH ↔ Metacognition integration
- [ ] Start Metacognition service (or mock)
- [ ] LH sends plan for review
- [ ] Verify REJECT aborts pipeline
- [ ] Verify REVISE triggers revision loop
- [ ] Verify ACCEPT proceeds to publication

#### 3.3 LH ↔ RH simulation request
- [ ] LH publishes `plan.step_ready` events
- [ ] Verify RH receives events via EventFabric
- [ ] Test simulation request/result flow

#### 3.4 Docker Compose local stack
- [ ] Create `infra/local/docker-compose.lh.yaml`
- [ ] Include: LH, Safety (mock), Metacog (mock), ZMQ fabric
- [ ] One-command startup: `docker-compose up lh-stack`

---

## Sprint 4: LLM Integration for Real Planning

**Goal**: Replace deterministic scaffold planner with LLM-powered planning.

### Tasks

#### 4.1 LLM adapter interface
- [ ] Define `LLMAdapter` protocol/interface
- [ ] Implement `OllamaAdapter` for local models
- [ ] Implement `AnthropicAdapter` for Claude API
- [ ] Implement `OpenAIAdapter` for GPT API

#### 4.2 LLM-powered planner
- [ ] Add `llm_planner.py` module
- [ ] Prompt engineering for hierarchical plan generation
- [ ] Parse LLM output into PlanGraph structure
- [ ] Fallback to deterministic planner on LLM failure

#### 4.3 Tool/skill integration
- [ ] Define tool schemas in procedural memory
- [ ] LLM generates tool calls in plan steps
- [ ] Validate tool references against schema

#### 4.4 Configuration
- [ ] `AGI_LH_LLM_PROVIDER` env var (ollama/anthropic/openai)
- [ ] `AGI_LH_LLM_MODEL` env var
- [ ] Temperature, max tokens, etc.

---

## Sprint 5: Memory Integration

**Goal**: Connect LH to real semantic/episodic/procedural memory.

### Tasks

#### 5.1 Semantic memory enrichment
- [ ] Query semantic memory for domain facts
- [ ] Inject relevant facts into planning context
- [ ] Cache frequent queries

#### 5.2 Episodic memory lookup
- [ ] Find similar past tasks/episodes
- [ ] Use past successes to inform planning
- [ ] Learn from past failures

#### 5.3 Procedural memory skills
- [ ] Lookup available skills for task type
- [ ] Match skills to plan steps
- [ ] Verify skill preconditions

---

## Sprint 6: Production Hardening

**Goal**: Prepare LH for HPC deployment.

### Tasks

#### 6.1 Observability
- [ ] Prometheus metrics (request latency, plan size, safety rejections)
- [ ] Structured logging with correlation IDs
- [ ] Distributed tracing (OpenTelemetry)

#### 6.2 Error handling & resilience
- [ ] Retry logic for transient failures
- [ ] Circuit breakers for downstream services
- [ ] Graceful degradation modes

#### 6.3 Performance optimization
- [ ] Connection pooling for gRPC clients
- [ ] Async operations where possible
- [ ] Caching layer for memory queries

#### 6.4 HPC deployment
- [ ] Apptainer/Singularity container definition
- [ ] SLURM job script for LH service
- [ ] UCX fabric configuration

---

## File Structure After Completion

```
src/agi/lh/
├── __init__.py
├── service.py              # gRPC server entrypoint
├── plan_service.py         # PlanService implementation
├── planner.py              # Hierarchical planner (deterministic)
├── llm_planner.py          # LLM-powered planner (Sprint 4)
├── memory_client.py        # Memory subsystem client
├── safety_client.py        # Safety subsystem client
├── metacog_client.py       # Metacognition client
├── llm/
│   ├── __init__.py
│   ├── adapter.py          # LLMAdapter protocol
│   ├── ollama.py           # Ollama adapter
│   ├── anthropic.py        # Anthropic adapter
│   └── openai.py           # OpenAI adapter
└── config.py               # LH configuration dataclass

tests/lh/
├── __init__.py
├── conftest.py             # Fixtures
├── test_planner.py
├── test_plan_service.py
├── test_memory_client.py
├── test_safety_client.py
├── test_metacog_client.py
└── test_llm_planner.py

configs/
└── lh.yaml                 # LH service configuration
```

---

## Priority Order

1. **Sprint 1** - Critical: LH must be runnable
2. **Sprint 2** - High: Tests enable safe iteration
3. **Sprint 3** - High: Validates architecture
4. **Sprint 4** - Medium: Core intelligence layer
5. **Sprint 5** - Medium: Memory integration
6. **Sprint 6** - Low (for now): Production concerns

---

## Quick Start Command (After Sprint 1)

```bash
# Terminal 1: Start LH
cd agi-hpc
export AGI_FABRIC_MODE=local
python src/agi/lh/service.py --port 50100

# Terminal 2: Test request
python -c "
import grpc
from agi.proto_gen import plan_pb2, plan_pb2_grpc

channel = grpc.insecure_channel('localhost:50100')
stub = plan_pb2_grpc.PlanServiceStub(channel)

request = plan_pb2.PlanRequest(
    task=plan_pb2.Task(
        goal_id='test-001',
        description='Navigate to the red cube and pick it up',
        task_type='manipulation',
    ),
    environment=plan_pb2.EnvironmentDescriptor(
        scenario_id='tabletop-v1',
    ),
)

response = stub.Plan(request)
print(f'Plan ID: {response.plan_id}')
print(f'Steps: {len(response.steps)}')
for step in response.steps:
    print(f'  {step.index}: {step.description}')
"
```
