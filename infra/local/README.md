# Local Development Stack

Docker Compose configuration for running the AGI-HPC LH service locally with mock dependencies.

## Quick Start

```bash
# From repository root
docker-compose -f infra/local/docker-compose.lh.yaml up
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| lh | 50100 | Left Hemisphere planning service |
| safety-mock | 50120 | Mock safety service (always approves) |
| metacog-mock | 50130 | Mock metacognition service (always accepts) |
| memory-mock | 50110 | Mock memory service (passthrough) |

## Testing the LH Service

Once the stack is running, test with a Python client:

```python
import grpc
from agi.proto_gen import plan_pb2, plan_pb2_grpc

channel = grpc.insecure_channel('localhost:50100')
stub = plan_pb2_grpc.PlanServiceStub(channel)

request = plan_pb2.PlanRequest(
    task=plan_pb2.Task(
        goal_id='test-001',
        description='Navigate to waypoint A',
        task_type='navigation',
    ),
)

response = stub.Plan(request)
print(f'Plan ID: {response.plan_id}')
print(f'Steps: {len(response.steps)}')
```

## Environment Variables

### LH Service
- `AGI_LH_PORT`: gRPC port (default: 50100)
- `AGI_FABRIC_MODE`: Event fabric mode (local, zmq, ucx)
- `AGI_LH_MEMORY_ADDR`: Memory service address
- `AGI_LH_SAFETY_ADDR`: Safety service address
- `AGI_LH_META_ADDR`: Metacognition service address

### Mock Services
- `MOCK_SERVICE`: Which service to mock (safety, metacog, memory)
- `MOCK_PORT`: Port to listen on
- `MOCK_BEHAVIOR`: Response behavior (approve, reject, accept, revise, passthrough)

## Building Images

```bash
# Build LH image
docker build -f infra/local/Dockerfile.lh -t agi-hpc-lh .

# Build mock image
docker build -f infra/local/Dockerfile.mock -t agi-hpc-mock .
```

## Development

For local development without Docker:

```bash
# Terminal 1: Start LH service
export AGI_FABRIC_MODE=local
python -m agi.lh.service --port 50100

# Terminal 2: Test client
python -c "
import grpc
from agi.proto_gen import plan_pb2, plan_pb2_grpc
channel = grpc.insecure_channel('localhost:50100')
stub = plan_pb2_grpc.PlanServiceStub(channel)
request = plan_pb2.PlanRequest(task=plan_pb2.Task(goal_id='test', description='Test'))
print(stub.Plan(request))
"
```
