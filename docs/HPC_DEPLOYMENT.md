# HPC Deployment Guide

This guide describes how to deploy the AGI-HPC cognitive architecture on high-performance computing clusters.

## Overview

AGI-HPC is designed for distributed deployment across HPC infrastructure. The architecture supports:

- **Horizontal scaling** of all cognitive services
- **Stateless services** with external state stores
- **Event-driven communication** via distributed pub/sub
- **Graceful degradation** when components fail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HPC CLUSTER TOPOLOGY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        COMPUTE NODES                                 │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│   │   │  LH Node 1  │  │  LH Node 2  │  │  LH Node N  │  (Planners)     │   │
│   │   │  :50051     │  │  :50051     │  │  :50051     │                 │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│   │                                                                      │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│   │   │  RH Node 1  │  │  RH Node 2  │  │  RH Node N  │  (World Model)  │   │
│   │   │  :50057     │  │  :50057     │  │  :50057     │                 │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                      SAFETY LAYER (HA)                               │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│   │   │ Gateway 1   │  │ Gateway 2   │  │  ErisML 1   │                 │   │
│   │   │ :50055      │  │ :50055      │  │  :50060     │                 │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│   ┌────────────────────────────────▼────────────────────────────────────┐   │
│   │                       STORAGE LAYER                                  │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│   │   │ Episodic    │  │ Semantic    │  │ Procedural  │                 │   │
│   │   │ Memory      │  │ Memory      │  │ Memory      │                 │   │
│   │   │ :50052      │  │ :50053      │  │ :50054      │                 │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU Cores (per node) | 8 | 32+ |
| RAM (per node) | 32 GB | 128 GB |
| GPU (for LLM inference) | 1x A100 | 4x H100 |
| Network | 10 Gbps | 100 Gbps InfiniBand |
| Storage | 500 GB SSD | 2 TB NVMe |

### Software Requirements

- Python 3.10+
- gRPC 1.78.0+
- Kubernetes 1.28+ (for container deployment)
- SLURM (for traditional HPC)
- Redis or etcd (for distributed state)

### Network Ports

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| LH (Plan Service) | 50051 | gRPC | Left Hemisphere planning |
| Episodic Memory | 50052 | gRPC | Event/experience storage |
| Semantic Memory | 50053 | gRPC | Facts/concepts storage |
| Procedural Memory | 50054 | gRPC | Skills/behaviors storage |
| Safety Gateway | 50055 | gRPC | Pre-action safety |
| In-Action Safety | 50056 | gRPC | Real-time monitoring |
| RH (World Model) | 50057 | gRPC | Right Hemisphere |
| Post-Action Safety | 50058 | gRPC | Outcome logging |
| ErisML Service | 50060 | gRPC | Ethical evaluation |
| Metacognition | 50070 | gRPC | Self-monitoring |
| Prometheus | 9090 | HTTP | Metrics |
| Event Fabric | 6379 | Redis | Pub/sub events |

## Deployment Options

### Option 1: Kubernetes Deployment

#### Namespace Setup

```bash
kubectl create namespace agi-hpc
kubectl config set-context --current --namespace=agi-hpc
```

#### Deploy ConfigMaps

```yaml
# configs/k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agi-hpc-config
  namespace: agi-hpc
data:
  lh_config.yaml: |
    service:
      host: "0.0.0.0"
      port: 50051
    safety:
      gateway_address: "safety-gateway:50055"
      bond_index_threshold: 0.30
    llm:
      provider: "anthropic"
      model: "claude-3-opus"

  safety_config.yaml: |
    gateway:
      port: 50055
      erisml_address: "erisml-service:50060"
    thresholds:
      bond_index_warn: 0.25
      bond_index_block: 0.30
      physical_harm_threshold: 0.9
```

```bash
kubectl apply -f configs/k8s/configmap.yaml
```

#### Deploy Services

```yaml
# deploy/k8s/lh-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lh-planner
  namespace: agi-hpc
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lh-planner
  template:
    metadata:
      labels:
        app: lh-planner
    spec:
      containers:
      - name: lh-planner
        image: agi-hpc/lh-planner:latest
        ports:
        - containerPort: 50051
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "32Gi"
            cpu: "16"
            nvidia.com/gpu: "1"
        env:
        - name: SAFETY_GATEWAY_ADDRESS
          value: "safety-gateway:50055"
        - name: EPISODIC_MEMORY_ADDRESS
          value: "episodic-memory:50052"
        volumeMounts:
        - name: config
          mountPath: /app/configs
      volumes:
      - name: config
        configMap:
          name: agi-hpc-config
---
apiVersion: v1
kind: Service
metadata:
  name: lh-planner
  namespace: agi-hpc
spec:
  selector:
    app: lh-planner
  ports:
  - port: 50051
    targetPort: 50051
  type: ClusterIP
```

```yaml
# deploy/k8s/safety-gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: safety-gateway
  namespace: agi-hpc
spec:
  replicas: 2  # HA for safety-critical component
  selector:
    matchLabels:
      app: safety-gateway
  template:
    metadata:
      labels:
        app: safety-gateway
    spec:
      containers:
      - name: safety-gateway
        image: agi-hpc/safety-gateway:latest
        ports:
        - containerPort: 50055
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: ERISML_ADDRESS
          value: "erisml-service:50060"
        - name: BOND_INDEX_THRESHOLD
          value: "0.30"
        livenessProbe:
          grpc:
            port: 50055
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          grpc:
            port: 50055
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: safety-gateway
  namespace: agi-hpc
spec:
  selector:
    app: safety-gateway
  ports:
  - port: 50055
    targetPort: 50055
  type: ClusterIP
```

```yaml
# deploy/k8s/erisml-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: erisml-service
  namespace: agi-hpc
spec:
  replicas: 2
  selector:
    matchLabels:
      app: erisml-service
  template:
    metadata:
      labels:
        app: erisml-service
    spec:
      containers:
      - name: erisml
        image: agi-hpc/erisml-service:latest
        ports:
        - containerPort: 50060
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "8"
        env:
        - name: PROFILE_NAME
          value: "agi_hpc_safety_v1"
---
apiVersion: v1
kind: Service
metadata:
  name: erisml-service
  namespace: agi-hpc
spec:
  selector:
    app: erisml-service
  ports:
  - port: 50060
    targetPort: 50060
  type: ClusterIP
```

#### Deploy All Services

```bash
kubectl apply -f deploy/k8s/
```

#### Verify Deployment

```bash
# Check pod status
kubectl get pods -n agi-hpc

# Check service endpoints
kubectl get endpoints -n agi-hpc

# Check logs
kubectl logs -l app=safety-gateway -n agi-hpc

# Test gRPC connectivity
kubectl run grpc-test --rm -it --image=fullstorydev/grpcurl -- \
  -plaintext safety-gateway:50055 list
```

### Option 2: SLURM Deployment

For traditional HPC clusters using SLURM:

#### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=agi-hpc
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# Load modules
module load python/3.10
module load cuda/12.0
module load openmpi/4.1

# Activate virtual environment
source /path/to/venv/bin/activate

# Get node list
NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODE_ARRAY=($NODES)

# Assign roles to nodes
LH_NODES="${NODE_ARRAY[0]} ${NODE_ARRAY[1]}"
RH_NODES="${NODE_ARRAY[2]} ${NODE_ARRAY[3]}"
SAFETY_NODE="${NODE_ARRAY[4]}"
ERISML_NODE="${NODE_ARRAY[5]}"
MEMORY_NODES="${NODE_ARRAY[6]} ${NODE_ARRAY[7]}"

# Start services
echo "Starting Safety Gateway on $SAFETY_NODE..."
srun --nodes=1 --nodelist=$SAFETY_NODE \
  python -m agi.safety.gateway --port 50055 &

echo "Starting ErisML Service on $ERISML_NODE..."
srun --nodes=1 --nodelist=$ERISML_NODE \
  python -m agi.safety.erisml.service --port 50060 &

echo "Starting Memory Services on $MEMORY_NODES..."
for i in 0 1; do
  MEM_NODE=$(echo $MEMORY_NODES | cut -d' ' -f$((i+1)))
  srun --nodes=1 --nodelist=$MEM_NODE \
    python -m agi.memory.service --type episodic --port 50052 &
done

echo "Starting LH Planners on $LH_NODES..."
for node in $LH_NODES; do
  srun --nodes=1 --nodelist=$node \
    python -m agi.lh.plan_service \
      --port 50051 \
      --safety-gateway $SAFETY_NODE:50055 &
done

echo "Starting RH World Model on $RH_NODES..."
for node in $RH_NODES; do
  srun --nodes=1 --nodelist=$node \
    python -m agi.rh.world_model --port 50057 &
done

# Wait for all services
wait
```

#### Submit Job

```bash
sbatch slurm/agi-hpc-job.sh
```

### Option 3: Docker Compose (Development/Testing)

```yaml
# docker-compose.yaml
version: '3.8'

services:
  safety-gateway:
    build:
      context: .
      dockerfile: docker/Dockerfile.safety
    ports:
      - "50055:50055"
    environment:
      - ERISML_ADDRESS=erisml:50060
      - BOND_INDEX_THRESHOLD=0.30
    depends_on:
      - erisml
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:50055"]
      interval: 10s
      timeout: 5s
      retries: 3

  erisml:
    build:
      context: .
      dockerfile: docker/Dockerfile.erisml
    ports:
      - "50060:50060"
    environment:
      - PROFILE_NAME=agi_hpc_safety_v1

  lh-planner:
    build:
      context: .
      dockerfile: docker/Dockerfile.lh
    ports:
      - "50051:50051"
    environment:
      - SAFETY_GATEWAY_ADDRESS=safety-gateway:50055
      - EPISODIC_MEMORY_ADDRESS=episodic-memory:50052
    depends_on:
      - safety-gateway
      - episodic-memory

  rh-world-model:
    build:
      context: .
      dockerfile: docker/Dockerfile.rh
    ports:
      - "50057:50057"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  episodic-memory:
    build:
      context: .
      dockerfile: docker/Dockerfile.memory
    ports:
      - "50052:50052"
    environment:
      - MEMORY_TYPE=episodic
    volumes:
      - episodic-data:/data

  semantic-memory:
    build:
      context: .
      dockerfile: docker/Dockerfile.memory
    ports:
      - "50053:50053"
    environment:
      - MEMORY_TYPE=semantic
    volumes:
      - semantic-data:/data

  procedural-memory:
    build:
      context: .
      dockerfile: docker/Dockerfile.memory
    ports:
      - "50054:50054"
    environment:
      - MEMORY_TYPE=procedural
    volumes:
      - procedural-data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  episodic-data:
  semantic-data:
  procedural-data:
  redis-data:
```

```bash
docker-compose up -d
```

## Distributed Hash Table (DHT)

For large-scale deployments, AGI-HPC uses a DHT for service discovery and distributed state.

### DHT Configuration

```yaml
# configs/dht_config.yaml
dht:
  bootstrap_nodes:
    - "node1.cluster:5000"
    - "node2.cluster:5000"
    - "node3.cluster:5000"
  replication_factor: 3
  virtual_nodes: 128
  consistency: "eventual"  # or "strong" for safety-critical

  # Service registration
  services:
    - name: "lh-planner"
      port: 50051
      health_check_interval: 5s
    - name: "safety-gateway"
      port: 50055
      health_check_interval: 1s  # More frequent for safety
```

### Service Discovery

```python
from agi.core.dht import DHTClient

# Initialize DHT client
dht = DHTClient(bootstrap_nodes=["node1:5000", "node2:5000"])

# Register service
dht.register_service("lh-planner", "10.0.0.5:50051")

# Discover services
safety_gateways = dht.discover("safety-gateway")
# Returns: ["10.0.0.10:50055", "10.0.0.11:50055"]

# Get nearest instance (latency-aware)
nearest_gateway = dht.get_nearest("safety-gateway")
```

## Scaling Configuration

### Horizontal Pod Autoscaling (Kubernetes)

```yaml
# deploy/k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lh-planner-hpa
  namespace: agi-hpc
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lh-planner
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: grpc_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

### Safety-Aware Scaling

The Safety Gateway should never be a bottleneck:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: safety-gateway-hpa
  namespace: agi-hpc
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: safety-gateway
  minReplicas: 2  # Always HA
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50  # Scale earlier for safety
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300  # Scale down slowly
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# configs/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'agi-hpc-services'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ['agi-hpc']
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: $1

rule_files:
  - /etc/prometheus/alerts/*.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Critical Alerts

```yaml
# configs/prometheus/alerts/safety.yml
groups:
  - name: safety-alerts
    rules:
      - alert: SafetyGatewayDown
        expr: up{job="safety-gateway"} == 0
        for: 10s
        labels:
          severity: critical
        annotations:
          summary: "Safety Gateway is down"
          description: "Safety Gateway {{ $labels.instance }} has been down for more than 10 seconds."

      - alert: HighBondIndex
        expr: agi_hpc_bond_index_value > 0.25
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Bond Index exceeding warning threshold"
          description: "Bond Index is {{ $value }}, approaching block threshold of 0.30"

      - alert: ErisMLLatencyHigh
        expr: histogram_quantile(0.99, rate(erisml_evaluation_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ErisML evaluation latency is high"
          description: "99th percentile latency is {{ $value }}s, exceeding 100ms target"

      - alert: SafetyCheckFailureRate
        expr: rate(safety_check_failures_total[5m]) / rate(safety_checks_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High safety check failure rate"
          description: "{{ $value | humanizePercentage }} of safety checks are failing"
```

### Grafana Dashboard

Key metrics to visualize:

| Panel | Metric | Description |
|-------|--------|-------------|
| Request Rate | `rate(grpc_server_handled_total[5m])` | Requests per second |
| Latency P99 | `histogram_quantile(0.99, ...)` | 99th percentile latency |
| Bond Index | `agi_hpc_bond_index_value` | Current Bond Index |
| Safety Decisions | `agi_hpc_safety_decisions_total` | ALLOW/BLOCK/REVISE counts |
| Memory Usage | `process_resident_memory_bytes` | Per-service memory |
| GPU Utilization | `nvidia_gpu_utilization` | GPU usage for inference |

## Safety Considerations for HPC

### Fail-Safe Defaults

```yaml
# configs/safety_config.yaml
failsafe:
  # If ErisML is unavailable, default to blocking
  erisml_unavailable_action: "BLOCK"

  # If Bond Index cannot be computed, assume worst case
  bond_index_fallback: 1.0

  # Maximum time to wait for safety check
  safety_check_timeout_ms: 100

  # If timeout, default action
  timeout_action: "BLOCK"
```

### Network Partition Handling

```python
# Safety-first behavior during network partitions
class PartitionAwareSafetyGateway(SafetyGateway):
    def check_plan(self, plan, world_state=None):
        try:
            return super().check_plan(plan, world_state)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                # Network partition - fail safe
                logger.critical("Network partition detected, blocking all actions")
                return SafetyCheckResult(
                    decision=SafetyDecision.BLOCK,
                    risk_score=1.0,
                    reasons=["Network partition - failing safe"],
                )
            raise
```

### Consensus for Critical Decisions

For distributed safety decisions, use quorum:

```python
async def distributed_safety_check(plan, gateways: List[str]) -> SafetyDecision:
    """Require majority consensus for safety decisions."""
    results = await asyncio.gather(*[
        check_with_gateway(plan, gw) for gw in gateways
    ])

    block_votes = sum(1 for r in results if r.decision == SafetyDecision.BLOCK)

    # Require majority to ALLOW
    if block_votes >= len(gateways) // 2 + 1:
        return SafetyDecision.BLOCK

    return SafetyDecision.ALLOW
```

## Troubleshooting

### Common Issues

#### Service Discovery Failure

```bash
# Check if services are registered
kubectl get endpoints -n agi-hpc

# Check DNS resolution
kubectl run dns-test --rm -it --image=busybox -- nslookup safety-gateway

# Check gRPC connectivity
grpcurl -plaintext safety-gateway:50055 grpc.health.v1.Health/Check
```

#### High Latency

```bash
# Check network latency between nodes
kubectl exec -it lh-planner-xxx -- ping safety-gateway

# Profile gRPC calls
GRPC_TRACE=all python -m agi.lh.plan_service

# Check for resource contention
kubectl top pods -n agi-hpc
```

#### Bond Index Threshold Exceeded

```bash
# Check current Bond Index
grpcurl -plaintext erisml-service:50060 agi.erisml.v1.ErisMLService/GetBondIndex

# Review recent decisions
kubectl logs -l app=safety-gateway --tail=100 | grep "Bond Index"

# Adjust threshold if needed (with caution)
kubectl set env deployment/safety-gateway BOND_INDEX_THRESHOLD=0.35
```

#### Memory Exhaustion

```bash
# Check memory usage
kubectl top pods -n agi-hpc

# Increase limits
kubectl set resources deployment/lh-planner --limits=memory=64Gi

# Enable garbage collection tuning
kubectl set env deployment/lh-planner PYTHONMALLOC=malloc
```

### Health Checks

```bash
#!/bin/bash
# scripts/health_check.sh

SERVICES=(
  "lh-planner:50051"
  "safety-gateway:50055"
  "erisml-service:50060"
  "episodic-memory:50052"
  "semantic-memory:50053"
  "procedural-memory:50054"
)

for service in "${SERVICES[@]}"; do
  if grpcurl -plaintext $service grpc.health.v1.Health/Check > /dev/null 2>&1; then
    echo "✓ $service is healthy"
  else
    echo "✗ $service is unhealthy"
  fi
done
```

## Performance Tuning

### gRPC Configuration

```python
# Optimized gRPC channel options
channel_options = [
    ('grpc.keepalive_time_ms', 10000),
    ('grpc.keepalive_timeout_ms', 5000),
    ('grpc.keepalive_permit_without_calls', True),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),
    ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
]

channel = grpc.insecure_channel(address, options=channel_options)
```

### Thread Pool Sizing

```python
# Size thread pools based on workload
import os

# For CPU-bound work (planning, inference)
CPU_WORKERS = os.cpu_count() * 2

# For I/O-bound work (gRPC, memory access)
IO_WORKERS = os.cpu_count() * 4

# Safety services need dedicated threads
SAFETY_WORKERS = max(10, os.cpu_count())
```

## Related Documentation

- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) - System architecture
- [ERISML_API.md](ERISML_API.md) - ErisML integration API
- [ERISML_INTEGRATION_SKETCH.md](ERISML_INTEGRATION_SKETCH.md) - Integration design
- [LH_SPRINT_PLAN.md](LH_SPRINT_PLAN.md) - Development plan
