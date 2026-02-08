# Memory Subsystem Sprint Plan

## Current State Assessment

### Implemented (Scaffolding Complete)
| Service | Port | File | Status | Description |
|---------|------|------|--------|-------------|
| Semantic Memory | 50053 | `semantic/service.py` | **Stub** | In-memory dict, no real vector search |
| Episodic Memory | 50052 | `episodic/service.py` | **Stub** | JSONL append-only log, no filtering |
| Procedural Memory | 50054 | `procedural/service.py` | **Stub** | In-memory skill catalog, no search |
| Unified Memory | - | **TODO** | Not implemented |

### Proto Definitions (Comprehensive)
`memory.proto` defines complete APIs:
- `SemanticMemoryService` - Facts, entities, relations, tool schemas
- `EpisodicMemoryService` - Episodes, plans, outcomes, insights
- `ProceduralMemoryService` - Skills, parameters, proficiency tracking
- `UnifiedMemoryService` - Combined planning context enrichment

### Key Gaps
1. **No vector search** - Semantic memory uses dict lookup
2. **No filtering** - Episodic queries return all events
3. **No embeddings** - No embedding model integration
4. **No persistence** - All data in-memory or simple files
5. **No Decision Proof storage** - Required for governance audit trail

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY SUBSYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    UNIFIED MEMORY SERVICE                            │   │
│   │                  EnrichPlanningContext()                             │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                             │
│   ┌───────────────────────────┼─────────────────────────────────────────┐   │
│   │                           │                                          │   │
│   │   ┌───────────────┐  ┌────┴────────┐  ┌───────────────┐             │   │
│   │   │   SEMANTIC    │  │  EPISODIC   │  │  PROCEDURAL   │             │   │
│   │   │   MEMORY      │  │  MEMORY     │  │  MEMORY       │             │   │
│   │   │   :50053      │  │  :50052     │  │  :50054       │             │   │
│   │   │               │  │             │  │               │             │   │
│   │   │  Facts        │  │  Episodes   │  │  Skills       │             │   │
│   │   │  Concepts     │  │  Events     │  │  Actions      │             │   │
│   │   │  Relations    │  │  Outcomes   │  │  Proficiency  │             │   │
│   │   │  Tool Schemas │  │  Proofs     │  │  Statistics   │             │   │
│   │   └───────┬───────┘  └──────┬──────┘  └───────┬───────┘             │   │
│   │           │                 │                 │                      │   │
│   └───────────┼─────────────────┼─────────────────┼──────────────────────┘   │
│               │                 │                 │                          │
│   ┌───────────┴─────────────────┴─────────────────┴──────────────────────┐   │
│   │                        STORAGE LAYER                                  │   │
│   │                                                                       │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│   │   │   Qdrant    │  │ PostgreSQL  │  │   Redis     │                  │   │
│   │   │  (vectors)  │  │ (structured)│  │  (cache)    │                  │   │
│   │   │   :6333     │  │   :5432     │  │   :6379     │                  │   │
│   │   └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│   │                                                                       │   │
│   └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Sprint 1: Semantic Memory with Vector Search

**Goal**: Replace in-memory stub with real vector database using Qdrant.

### Tasks

#### 1.1 Qdrant integration
- [ ] Add `qdrant-client` dependency
- [ ] Create `QdrantVectorStore` class
- [ ] Collection schema: `semantic_facts` with metadata
- [ ] Batch upsert for `StoreFact()`
- [ ] Vector similarity search for `SemanticSearch()`
- [ ] Metadata filtering (domain, entity_type)

```python
# src/agi/memory/semantic/qdrant_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantVectorStore:
    def __init__(self, url: str = "localhost:6333"):
        self.client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self):
        self.client.recreate_collection(
            collection_name="semantic_facts",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    async def store_fact(self, fact_id: str, embedding: List[float], metadata: dict):
        self.client.upsert(
            collection_name="semantic_facts",
            points=[PointStruct(id=fact_id, vector=embedding, payload=metadata)],
        )

    async def search(self, query_embedding: List[float], limit: int = 10, **filters):
        return self.client.search(
            collection_name="semantic_facts",
            query_vector=query_embedding,
            limit=limit,
            query_filter=self._build_filter(filters),
        )
```

#### 1.2 Embedding model integration
- [ ] Define `EmbeddingModel` protocol
- [ ] Implement `SentenceTransformerEmbedder`
- [ ] Implement `OpenAIEmbedder` (text-embedding-3-small)
- [ ] Implement `CohereEmbedder`
- [ ] Batch embedding for efficiency
- [ ] Caching layer (Redis)

```python
# src/agi/memory/semantic/embedders/base.py
from typing import Protocol, List

class EmbeddingModel(Protocol):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text list."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...
```

#### 1.3 Knowledge graph support
- [ ] PostgreSQL schema for entities and relations
- [ ] Entity CRUD operations
- [ ] Relation CRUD operations
- [ ] Graph traversal queries
- [ ] Integration with vector search

```sql
-- Schema for knowledge graph
CREATE TABLE entities (
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100),
    properties JSONB,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE relations (
    relation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id UUID REFERENCES entities(entity_id),
    predicate VARCHAR(100) NOT NULL,
    object_id UUID REFERENCES entities(entity_id),
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_embedding ON entities USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_relations_subject ON relations(subject_id);
CREATE INDEX idx_relations_object ON relations(object_id);
CREATE INDEX idx_relations_predicate ON relations(predicate);
```

#### 1.4 Tool schema registry
- [ ] Load tool schemas from YAML/JSON
- [ ] Store in PostgreSQL
- [ ] `GetToolSchema()` implementation
- [ ] Schema validation
- [ ] Version management

#### 1.5 Configuration
- [ ] `AGI_SEMANTIC_QDRANT_URL` env var
- [ ] `AGI_SEMANTIC_EMBEDDING_MODEL` env var
- [ ] Collection settings
- [ ] Cache TTL settings

### Acceptance Criteria
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Semantic Memory
python -m agi.memory.semantic.service --port 50053

# Test vector search
grpcurl -plaintext -d '{"text": "robot navigation", "max_results": 5}' \
  localhost:50053 agi.memory.v1.SemanticMemoryService/SemanticSearch
```

---

## Sprint 2: Episodic Memory with Temporal Storage

**Goal**: Replace JSONL stub with proper temporal database supporting Decision Proof storage.

### Tasks

#### 2.1 PostgreSQL schema for episodes
- [ ] Design episode storage schema
- [ ] Temporal indexing (timestamp ranges)
- [ ] JSONB for flexible payload
- [ ] Decision Proof hash chain storage
- [ ] Efficient range queries

```sql
-- Episodic memory schema
CREATE TABLE episodes (
    episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_description TEXT NOT NULL,
    task_type VARCHAR(100),
    scenario_id VARCHAR(255),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    outcome_success BOOLEAN,
    outcome_description TEXT,
    completion_percentage FLOAT,
    total_duration_ms BIGINT,
    insights TEXT[],
    metadata JSONB,
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE episode_steps (
    id SERIAL PRIMARY KEY,
    episode_id UUID REFERENCES episodes(episode_id) ON DELETE CASCADE,
    step_index INT NOT NULL,
    step_id VARCHAR(255),
    description TEXT,
    tool_id VARCHAR(255),
    succeeded BOOLEAN,
    failure_reason TEXT,
    duration_ms BIGINT,
    params JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE episode_events (
    id SERIAL PRIMARY KEY,
    episode_id UUID REFERENCES episodes(episode_id) ON DELETE CASCADE,
    step_index INT,
    event_type VARCHAR(100) NOT NULL,
    timestamp_ms BIGINT NOT NULL,
    payload JSONB,
    tags JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Decision Proof storage (for governance)
CREATE TABLE decision_proofs (
    proof_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID REFERENCES episodes(episode_id),
    step_id VARCHAR(255),
    timestamp_ms BIGINT NOT NULL,
    decision VARCHAR(50) NOT NULL,  -- ALLOW, BLOCK, REVISE
    bond_index FLOAT,
    moral_vector JSONB,
    previous_proof_hash VARCHAR(64),
    proof_hash VARCHAR(64) NOT NULL,
    signature BYTEA,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_episodes_time ON episodes(start_time, end_time);
CREATE INDEX idx_episodes_task_type ON episodes(task_type);
CREATE INDEX idx_episodes_scenario ON episodes(scenario_id);
CREATE INDEX idx_episodes_embedding ON episodes USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_decision_proofs_episode ON decision_proofs(episode_id);
CREATE INDEX idx_decision_proofs_hash ON decision_proofs(proof_hash);
```

#### 2.2 Episode service implementation
- [ ] `StoreEpisode()` with full persistence
- [ ] `EpisodicSearch()` with similarity + filters
- [ ] `GetEpisode()` with steps and events
- [ ] Episode embedding generation
- [ ] Temporal range queries

#### 2.3 Decision Proof integration
- [ ] Store proofs from Safety Gateway
- [ ] Hash chain verification
- [ ] Query proofs by episode/step
- [ ] Audit trail export
- [ ] Proof integrity checks

```python
# src/agi/memory/episodic/proof_store.py
import hashlib
from dataclasses import dataclass

@dataclass
class DecisionProof:
    proof_id: str
    episode_id: str
    step_id: str
    timestamp_ms: int
    decision: str
    bond_index: float
    moral_vector: dict
    previous_proof_hash: str
    proof_hash: str

    @staticmethod
    def compute_hash(
        step_id: str,
        timestamp_ms: int,
        decision: str,
        bond_index: float,
        previous_hash: str,
    ) -> str:
        data = f"{step_id}:{timestamp_ms}:{decision}:{bond_index}:{previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

class ProofStore:
    async def store_proof(self, proof: DecisionProof) -> None:
        """Store decision proof with hash chain verification."""
        pass

    async def get_proof_chain(self, episode_id: str) -> List[DecisionProof]:
        """Get all proofs for episode in order."""
        pass

    async def verify_chain(self, episode_id: str) -> bool:
        """Verify hash chain integrity."""
        pass
```

#### 2.4 Event streaming
- [ ] EventFabric subscription for real-time events
- [ ] Batch event ingestion
- [ ] Event replay for training
- [ ] Event export (Parquet)

#### 2.5 Configuration
- [ ] `AGI_EPISODIC_DB_URL` env var
- [ ] Retention policies
- [ ] Proof verification settings

---

## Sprint 3: Procedural Memory with Skill Learning

**Goal**: Implement skill catalog with proficiency tracking and version management.

### Tasks

#### 3.1 PostgreSQL schema for skills
- [ ] Skill definition storage
- [ ] Version history
- [ ] Proficiency metrics
- [ ] Execution statistics
- [ ] Action sequence storage

```sql
-- Procedural memory schema
CREATE TABLE skills (
    skill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    version INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    preconditions TEXT[],
    postconditions TEXT[],
    proficiency FLOAT DEFAULT 0.5,
    execution_count INT DEFAULT 0,
    success_count INT DEFAULT 0,
    avg_duration_ms BIGINT,
    embedding VECTOR(768),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE skill_parameters (
    id SERIAL PRIMARY KEY,
    skill_id UUID REFERENCES skills(skill_id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    param_type VARCHAR(50),
    description TEXT,
    required BOOLEAN DEFAULT FALSE,
    default_value TEXT,
    constraints JSONB
);

CREATE TABLE skill_actions (
    id SERIAL PRIMARY KEY,
    skill_id UUID REFERENCES skills(skill_id) ON DELETE CASCADE,
    sequence INT NOT NULL,
    action_type VARCHAR(100),
    description TEXT,
    tool_id VARCHAR(255),
    parameters JSONB
);

CREATE TABLE skill_executions (
    id SERIAL PRIMARY KEY,
    skill_id UUID REFERENCES skills(skill_id),
    episode_id UUID,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    succeeded BOOLEAN,
    duration_ms BIGINT,
    error_message TEXT,
    context JSONB
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_embedding ON skills USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_skill_executions_skill ON skill_executions(skill_id);
```

#### 3.2 Skill service implementation
- [ ] `RegisterSkill()` with validation
- [ ] `SkillSearch()` with semantic matching
- [ ] `GetSkill()` with full details
- [ ] `UpdateSkillStats()` with proficiency update

#### 3.3 Proficiency learning
- [ ] Bayesian skill proficiency updates
- [ ] Success rate tracking
- [ ] Duration statistics
- [ ] Confidence intervals

```python
# src/agi/memory/procedural/proficiency.py
import math

class ProficiencyTracker:
    """Bayesian proficiency tracking for skills."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.alpha = prior_alpha
        self.beta = prior_beta

    def update(self, success: bool) -> float:
        """Update proficiency after execution."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1
        return self.proficiency

    @property
    def proficiency(self) -> float:
        """Expected proficiency (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """Confidence in proficiency estimate."""
        n = self.alpha + self.beta - 2  # observations
        return 1 - 1 / math.sqrt(n + 1)
```

#### 3.4 Skill versioning
- [ ] Version increment on modification
- [ ] Version history queries
- [ ] Rollback support
- [ ] A/B testing between versions

#### 3.5 Behavior tree integration
- [ ] Store behavior trees as skills
- [ ] Tree node serialization
- [ ] Composite skill composition
- [ ] Skill inheritance

---

## Sprint 4: Unified Memory Service

**Goal**: Implement combined query interface for planning context enrichment.

### Tasks

#### 4.1 UnifiedMemoryService implementation
- [ ] `EnrichPlanningContext()` RPC
- [ ] Parallel queries to all memory types
- [ ] Result aggregation and ranking
- [ ] Tool schema resolution

```python
# src/agi/memory/unified/service.py
class UnifiedMemoryService:
    def __init__(
        self,
        semantic: SemanticMemoryClient,
        episodic: EpisodicMemoryClient,
        procedural: ProceduralMemoryClient,
    ):
        self.semantic = semantic
        self.episodic = episodic
        self.procedural = procedural

    async def enrich_planning_context(
        self,
        task_description: str,
        task_type: str,
        scenario_id: str,
        include_semantic: bool = True,
        include_episodic: bool = True,
        include_procedural: bool = True,
    ) -> PlanningContext:
        """Query all memory types in parallel for planning context."""
        tasks = []

        if include_semantic:
            tasks.append(self.semantic.search(task_description))
        if include_episodic:
            tasks.append(self.episodic.search(task_description, task_type))
        if include_procedural:
            tasks.append(self.procedural.search(task_description))

        results = await asyncio.gather(*tasks)

        return self._aggregate_results(results)
```

#### 4.2 Cross-memory reasoning
- [ ] Link episodes to skills used
- [ ] Link facts to related episodes
- [ ] Skill recommendation from episodes
- [ ] Failure pattern detection

#### 4.3 Caching layer
- [ ] Redis caching for frequent queries
- [ ] Cache invalidation strategies
- [ ] TTL configuration
- [ ] Cache warming

#### 4.4 LH integration
- [ ] Update `MemoryClient` in LH
- [ ] Planning context injection
- [ ] Memory-augmented prompts
- [ ] Retrieval-augmented generation

---

## Sprint 5: Unit Tests for Memory Services

**Goal**: Achieve 80%+ test coverage for all memory services.

### Tasks

#### 5.1 Semantic Memory tests
- [ ] `test_store_fact`
- [ ] `test_semantic_search_returns_similar`
- [ ] `test_semantic_search_filters_by_domain`
- [ ] `test_entity_crud`
- [ ] `test_relation_crud`
- [ ] `test_tool_schema_retrieval`
- [ ] `test_embedding_generation`

#### 5.2 Episodic Memory tests
- [ ] `test_store_episode`
- [ ] `test_episodic_search_by_similarity`
- [ ] `test_episodic_search_by_time_range`
- [ ] `test_get_episode_with_steps`
- [ ] `test_decision_proof_storage`
- [ ] `test_proof_chain_verification`
- [ ] `test_event_streaming`

#### 5.3 Procedural Memory tests
- [ ] `test_register_skill`
- [ ] `test_skill_search`
- [ ] `test_get_skill`
- [ ] `test_update_skill_stats`
- [ ] `test_proficiency_update`
- [ ] `test_skill_versioning`

#### 5.4 Unified Memory tests
- [ ] `test_enrich_planning_context`
- [ ] `test_parallel_query_execution`
- [ ] `test_cross_memory_linking`
- [ ] `test_caching`

### Test Infrastructure
```python
# tests/memory/conftest.py
import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.qdrant import QdrantContainer

@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:16") as pg:
        yield pg

@pytest.fixture(scope="session")
def qdrant_container():
    with QdrantContainer("qdrant/qdrant:latest") as q:
        yield q

@pytest.fixture
async def semantic_memory(qdrant_container):
    """Semantic memory with real Qdrant."""
    return SemanticMemoryService(qdrant_url=qdrant_container.get_connection_url())

@pytest.fixture
async def episodic_memory(postgres_container):
    """Episodic memory with real PostgreSQL."""
    return EpisodicMemoryService(db_url=postgres_container.get_connection_url())
```

---

## Sprint 6: Integration Testing

**Goal**: Verify memory services work with LH, RH, and Safety subsystems.

### Tasks

#### 6.1 LH ↔ Memory integration
- [ ] Memory enrichment in planning pipeline
- [ ] Skill lookup for plan steps
- [ ] Episode storage after execution
- [ ] Learning from past episodes

#### 6.2 Safety ↔ Episodic integration
- [ ] Decision Proof storage
- [ ] Audit trail queries
- [ ] Governance reports
- [ ] Chain integrity verification

#### 6.3 RH ↔ Memory integration
- [ ] World state queries
- [ ] Skill retrieval for control
- [ ] Episode logging during execution

#### 6.4 Docker Compose stack
- [ ] `docker-compose.memory.yaml`
- [ ] Qdrant, PostgreSQL, Redis
- [ ] All three memory services
- [ ] Health checks

```yaml
# docker-compose.memory.yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: agi_memory
      POSTGRES_PASSWORD: ${MEMORY_DB_PASSWORD:-secret}
      POSTGRES_DB: memory
    ports:
      - "5433:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-memory.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - redis-data:/data

  semantic-memory:
    build:
      context: .
      dockerfile: docker/Dockerfile.memory
    command: python -m agi.memory.semantic.service
    ports:
      - "50053:50053"
    environment:
      - QDRANT_URL=qdrant:6333
      - REDIS_URL=redis://redis:6379
    depends_on:
      - qdrant
      - redis

  episodic-memory:
    build:
      context: .
      dockerfile: docker/Dockerfile.memory
    command: python -m agi.memory.episodic.service
    ports:
      - "50052:50052"
    environment:
      - DATABASE_URL=postgresql://agi_memory:secret@postgres:5432/memory
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  procedural-memory:
    build:
      context: .
      dockerfile: docker/Dockerfile.memory
    command: python -m agi.memory.procedural.service
    ports:
      - "50054:50054"
    environment:
      - DATABASE_URL=postgresql://agi_memory:secret@postgres:5432/memory
      - QDRANT_URL=qdrant:6333
    depends_on:
      - postgres
      - qdrant

volumes:
  qdrant-data:
  postgres-data:
  redis-data:
```

---

## Sprint 7: Advanced Features

**Goal**: Add advanced memory capabilities for production use.

### Tasks

#### 7.1 Memory consolidation
- [ ] Periodic summarization of episodes
- [ ] Fact extraction from episodes
- [ ] Skill refinement from executions
- [ ] Forgetting/pruning old memories

#### 7.2 Distributed memory
- [ ] Sharded vector storage
- [ ] Read replicas for queries
- [ ] Cross-node consistency
- [ ] DHT integration

#### 7.3 Memory analytics
- [ ] Query latency metrics
- [ ] Hit rate tracking
- [ ] Memory utilization
- [ ] Grafana dashboards

#### 7.4 Export/Import
- [ ] Memory snapshot export
- [ ] Parquet export for training
- [ ] Memory migration tools
- [ ] Backup/restore

---

## Sprint 8: Production Hardening

**Goal**: Prepare memory services for HPC deployment.

### Tasks

#### 8.1 Observability
- [ ] Prometheus metrics per service
- [ ] Query latency histograms
- [ ] Cache hit/miss rates
- [ ] Vector search performance

#### 8.2 High availability
- [ ] PostgreSQL replication
- [ ] Qdrant clustering
- [ ] Redis Sentinel/Cluster
- [ ] Failover handling

#### 8.3 Performance optimization
- [ ] Connection pooling
- [ ] Batch operations
- [ ] Async everywhere
- [ ] Memory-efficient embeddings

#### 8.4 Security
- [ ] Encryption at rest
- [ ] TLS for connections
- [ ] Access control
- [ ] Audit logging

---

## File Structure After Completion

```
src/agi/memory/
├── __init__.py
├── semantic/
│   ├── __init__.py
│   ├── service.py           # gRPC service
│   ├── qdrant_store.py      # Qdrant integration
│   ├── pg_store.py          # PostgreSQL for entities/relations
│   ├── embedders/
│   │   ├── __init__.py
│   │   ├── base.py          # EmbeddingModel protocol
│   │   ├── sentence_transformer.py
│   │   ├── openai.py
│   │   └── cohere.py
│   └── schema_registry.py   # Tool schema management
├── episodic/
│   ├── __init__.py
│   ├── service.py           # gRPC service
│   ├── pg_store.py          # PostgreSQL storage
│   ├── proof_store.py       # Decision Proof chain
│   ├── event_stream.py      # Event ingestion
│   └── export.py            # Parquet export
├── procedural/
│   ├── __init__.py
│   ├── service.py           # gRPC service
│   ├── pg_store.py          # PostgreSQL storage
│   ├── qdrant_store.py      # Vector search for skills
│   ├── proficiency.py       # Bayesian proficiency
│   └── behavior_tree.py     # BT serialization
├── unified/
│   ├── __init__.py
│   ├── service.py           # UnifiedMemoryService
│   ├── aggregator.py        # Result aggregation
│   └── cache.py             # Redis caching
├── common/
│   ├── __init__.py
│   ├── db.py                # Database utilities
│   ├── redis_client.py      # Redis utilities
│   └── migrations/          # Alembic migrations
└── config.py                # Memory configuration

tests/memory/
├── __init__.py
├── conftest.py              # Fixtures with testcontainers
├── semantic/
│   ├── test_service.py
│   ├── test_qdrant_store.py
│   └── test_embedders.py
├── episodic/
│   ├── test_service.py
│   ├── test_proof_store.py
│   └── test_event_stream.py
├── procedural/
│   ├── test_service.py
│   ├── test_proficiency.py
│   └── test_skill_versioning.py
├── unified/
│   └── test_service.py
└── integration/
    ├── test_lh_memory.py
    └── test_full_pipeline.py

configs/
└── memory_config.yaml       # Memory service configuration
```

---

## Storage Technology Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Vector Store | **Qdrant** | Fast, feature-rich, good filtering |
| Structured Data | **PostgreSQL 16** | Reliable, pgvector, JSON support |
| Caching | **Redis 7** | Fast, pub/sub for events |
| Embeddings | **sentence-transformers** | Local, fast, good quality |

### Alternatives Considered

| Component | Alternative | When to use |
|-----------|-------------|-------------|
| Vector Store | Milvus | Very large scale (>1B vectors) |
| Vector Store | Pinecone | Managed service preference |
| Vector Store | Weaviate | Need GraphQL API |
| Embeddings | OpenAI | Higher quality, API dependency OK |
| Embeddings | Cohere | Multilingual requirements |

---

## Priority Order

1. **Sprint 1** - Critical: Semantic memory enables LH context
2. **Sprint 2** - Critical: Episodic memory for Decision Proofs
3. **Sprint 4** - High: Unified service for LH integration
4. **Sprint 3** - High: Procedural memory for skill lookup
5. **Sprint 5** - High: Tests for reliability
6. **Sprint 6** - Medium: Integration verification
7. **Sprint 7** - Low: Advanced features
8. **Sprint 8** - Low: Production concerns

---

## Quick Start (After Sprint 1-4)

```bash
# Terminal 1: Start storage stack
docker-compose -f docker-compose.memory.yaml up -d qdrant postgres redis

# Terminal 2: Run migrations
cd agi-hpc
alembic -c src/agi/memory/alembic.ini upgrade head

# Terminal 3-5: Start memory services
python -m agi.memory.semantic.service --port 50053 &
python -m agi.memory.episodic.service --port 50052 &
python -m agi.memory.procedural.service --port 50054 &

# Terminal 6: Test unified query
python -c "
import grpc
from agi.proto_gen import memory_pb2, memory_pb2_grpc

channel = grpc.insecure_channel('localhost:50053')
stub = memory_pb2_grpc.SemanticMemoryServiceStub(channel)

# Store a fact
response = stub.StoreFact(memory_pb2.StoreFactRequest(
    content='Robots should not harm humans',
    source='asimov_laws',
    domains=['robotics', 'safety'],
    confidence=1.0,
))
print(f'Stored fact: {response.fact_id}')

# Query
results = stub.SemanticSearch(memory_pb2.SemanticQuery(
    text='robot safety rules',
    max_results=5,
))
for fact in results.facts:
    print(f'  {fact.content} (sim={fact.similarity:.3f})')
"
```

---

## Dependencies

```toml
# pyproject.toml additions for memory
[project.optional-dependencies]
memory = [
    # Vector DB
    "qdrant-client>=1.7.0",

    # Database
    "asyncpg>=0.29.0",
    "sqlalchemy[asyncio]>=2.0",
    "alembic>=1.13.0",
    "pgvector>=0.2.0",

    # Redis
    "redis>=5.0",

    # Embeddings
    "sentence-transformers>=2.2.0",
    "openai>=1.0",  # optional

    # Utilities
    "numpy>=1.24",
    "pyarrow>=14.0",  # Parquet export
]
```
