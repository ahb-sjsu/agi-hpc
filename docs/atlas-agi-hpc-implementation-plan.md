# Atlas AI: AGI-HPC Cognitive Architecture Implementation Plan

## Overview

Rebuild Atlas AI as a full showcase of the AGI-HPC cognitive architecture -- all 10 subsystems running on the HP Z840 workstation with dual GV100 GPUs.

## Architecture: Subsystem Mapping

| AGI-HPC Subsystem | Atlas Implementation | Port |
|---|---|---|
| **Event Fabric** | NATS JetStream | 4222 |
| **Left Hemisphere** | Qwen-72B with analytical prompt (temp=0.3), RAG | 50100-50101 |
| **Right Hemisphere** | Qwen-72B with creative prompt (temp=0.8) | 50200-50201 |
| **Memory (Semantic)** | PostgreSQL + pgvector (existing RAG index) | 50301 |
| **Memory (Episodic)** | PostgreSQL (conversation history + embeddings) | 50302 |
| **Memory (Procedural)** | SQLite (learned procedures, routing preferences) | 50303 |
| **Safety Gateway** | ErisML DEME pipeline (3-layer: Reflex/Tactical/Strategic) | 50400-50401 |
| **Metacognition** | Self-monitoring, reflection loop, parameter adjustment | 50500-50501 |
| **Environment** | System sensors (GPU/CPU/RAM), repo watcher, shell/web actuators | 50600-50601 |
| **DHT** | Local service registry + config store (Kademlia interface) | 50800 |
| **Integration** | Orchestrator: request classification, hemisphere routing, response merging | 50700-50701 |

## Hardware Resource Allocation

```
CPU Socket 0 (24 threads): LH, Memory, Safety, Metacognition, NATS
CPU Socket 1 (24 threads): RH, Environment, Integration, PostgreSQL, Indexer
GPU 0 (32GB): Qwen-72B layers 0-39
GPU 1 (32GB): Qwen-72B layers 40-79
RAM: ~50GB Qwen + 8GB Postgres + 4GB BGE-M3 + 4GB services = ~66GB used, ~158GB free
```

## NATS Event Fabric Subjects

```
agi.lh.request.{chat,plan,reason}     agi.lh.response.{chat,plan}
agi.rh.request.{pattern,spatial,creative}  agi.rh.response.*
agi.memory.{store,query}.{semantic,episodic,procedural}
agi.safety.check.{input,output,action}    agi.safety.{veto,audit}
agi.meta.monitor.{latency,quality,confidence}  agi.meta.adjust.*
agi.env.sensor.{system,repos,network}     agi.env.actuator.{shell,web}
agi.integration.{route,merge,session}
```

## Request Flow (Post-Implementation)

```
User -> Integration -> Safety(input) -> Memory(context) -> Router
  -> LH (analytical) and/or RH (creative) -> Safety(output)
  -> Memory(store episode) -> Metacognition(observe) -> User
```

LH and RH use the SAME Qwen-72B model but with different system prompts and sampling parameters -- like biological hemispheres using the same neural substrate with different connectivity patterns.

## Phased Implementation (7 Sprints, 14 Weeks)

### Phase 0: Foundation (Week 1-2)
- Bootstrap repo structure with `bootstrap_agi_hpc_repo_v2.py`
- Install NATS, configure JetStream
- Write `Event` dataclass and `EventFabric` NATS wrapper
- Verify publish/subscribe works
- **Deliverable**: Event fabric running, directory structure created

### Phase 1: LLM Integration + Left Hemisphere (Week 3-4)
- Write `LLMClient` wrapping llama-server's OpenAI API
- Write `InferenceConfig` and prompt template registry
- Refactor `atlas-rag-server.py` into `src/agi/lh/rag.py`
- Write LH service: subscribes to `agi.lh.request.*`, retrieves RAG, calls LLM with analytical prompt
- **Deliverable**: Chat flows through NATS -> LH -> LLM, functionally equivalent to current Atlas

### Phase 2: Memory Subsystem (Week 5-6)
- Semantic: wrap existing pgvector (already done)
- Episodic: new PostgreSQL table for conversation episodes with embeddings
- Procedural: SQLite store for learned routing preferences, prompt patterns
- Memory Service Broker routes to appropriate sub-store
- **Deliverable**: "What did we discuss yesterday?" returns actual recall

### Phase 3: Safety Gateway (Week 7-8)
- Integrate `erisml-lib` DEME pipeline as a dependency
- Write EthicalFacts adapter for chat interactions
- Input gate (pre-LLM), output gate (post-LLM), action gate (pre-actuator)
- Configure DEME profile for research-assistant domain
- Audit log with full DecisionProof
- **Deliverable**: Every interaction safety-checked with 3-layer pipeline, dashboard at :50401

### Phase 4: Right Hemisphere + Integration (Week 9-10)
- RH service: same model, different prompt (creative, analogical, temp=0.8)
- Request classifier: routes to LH (analytical), RH (creative), or both
- Response merger for dual-hemisphere queries
- Session manager with episodic memory linking
- Chat UI upgraded: shows LH/RH/Both badges, safety status, memory hits
- **Deliverable**: "What patterns do you see?" goes to RH, "Debug this" goes to LH

### Phase 5: Metacognition + Environment (Week 11-12)
- Monitor: subscribes to all events, tracks latency/throughput/quality/veto-rate
- Reflector: periodically asks LLM to evaluate its own performance
- Adjuster: tunes temperature, max_tokens, routing based on metrics
- System sensor (GPU/CPU/RAM via research-portal discovery functions)
- Repo sensor (watches /archive/ahb-sjsu/ for changes, triggers re-index)
- Shell/web actuators (sandboxed, safety-gated)
- **Deliverable**: Atlas can answer "How are you performing?" with real data

### Phase 6: DHT + Dashboard + Polish (Week 13-14)
- DHT service registry with Kademlia interface (ready for multi-node scaling)
- Unified dashboard: chat + system status + cognitive state + safety audit + memory + metacognition
- Process supervisor: `start_atlas.sh` starts all services in correct order
- Health check script
- **Deliverable**: Complete 10-subsystem architecture running, full dashboard

## Key Integration Points

- **erisml-lib** (`C:/source/erisml-lib/`): DEME pipeline, MoralVectors, EthicalFacts, DecisionProof -> Safety Gateway
- **research-portal** (`C:/source/atlas-portal/`): `discovery.py` hardware functions -> Environment sensors
- **bootstrap script** (`C:/source/bootstrap_agi_hpc_repo_v2.py`): creates canonical directory layout
- **DEME whitepaper** (`C:/source/erisml-lib/docs/guides/deme_whitepaper_nist.md`): authoritative architecture reference

## Risk Mitigations

1. **Single LLM bottleneck**: LH/RH share llama-server. Serialize by default, parallelize only for dual-hemisphere queries. Use 2-3 concurrent slots.
2. **NATS SPOF**: Each service also exposes direct HTTP endpoints as fallback. Graceful degradation to current architecture if NATS is down.
3. **Memory growth**: TTL-based episodic cleanup (90 days). Procedural memory has success/failure counters for pruning.
4. **Safety overhead**: Reflex layer < 1ms. Full Tactical layer can be async for non-critical interactions. Cache decisions for similar inputs.


## Tiered Memory Architecture

Atlas implements a hardware-mapped memory hierarchy inspired by CPU cache levels:

| Tier | Medium | Latency | Capacity (Atlas) | Contents |
|---|---|---|---|---|
| **L1** | VRAM (KV cache) | <1ms | ~8GB per GPU | Current conversation context, active inference state |
| **L2** | RAM | ~1ms | 256GB | Recent sessions, hot embeddings, procedural cache, mmap indexes |
| **L3** | SSD (PostgreSQL) | ~5ms | 1.8TB NVMe | Episodic history, semantic chunks (pgvector), procedural DB |
| **L4** | HDD/RAID (Archive) | ~50ms | 15TB RAID5 | Full repo clones, old episodes, cold storage, training data |
| **L5** | Network | ~100ms+ | Unlimited | GitHub API, web fetch, external knowledge bases |

### Promotion / Eviction Policies

- **L3 to L2**: Frequently accessed episodes and embeddings promoted to RAM (mmap or Redis)
- **L2 to L3**: Stale sessions evict from RAM after idle timeout (default 30 min)
- **L3 to L4**: Episodes older than 90 days archived to /archive
- **L4 to L3**: On-demand recall triggers archive search
- **L5 to L3**: External knowledge fetched and cached locally on first access

### Cache Controller

The Memory Service Broker acts as the cache controller:
- Tracks access frequency and recency per memory item
- Manages promotion/eviction across tiers
- Pre-warms L2 cache on session start (load recent episodes + user preferences)
- Reports cache hit/miss rates to metacognition via agi.meta.monitor.memory

### Hardware Mapping

- L1: GPU 0 KV cache (Gemma 4) + GPU 1 KV cache (Qwen 3)
- L2: 256GB DDR4 (16x 16GB RDIMM, upgradeable to 1TB LRDIMM)
- L3: 1.8TB NVMe (root filesystem)
- L4: 15TB RAID5 (/archive) + 916GB RAID1 (/mnt/newhome)
- L5: Tailscale + Comcast WAN (atlas-sjsu.duckdns.org)

### Implementation Phases

- **Phase 2 (current)**: L1 (existing KV cache) + L3 (PostgreSQL episodic/procedural)
- **Phase 2b**: L2 RAM cache layer (Redis or mmap) for hot data
- **Phase 5 (Environment)**: L4 archive tiering + L5 network fetcher
- **Phase 6 (Polish)**: Full cache controller with promotion/eviction metrics in dashboard


## Database Normalization (3NF)

All AGI-HPC persistent storage follows Third Normal Form to eliminate redundancy,
enable efficient FK-based queries, and support file-level deduplication.

### Semantic Memory Schema (RAG)

```sql
-- 1NF: atomic values, no repeating groups
-- 2NF: no partial dependencies (all non-key columns depend on full PK)
-- 3NF: no transitive dependencies

CREATE TABLE repos (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    url TEXT,
    last_indexed TIMESTAMPTZ,
    chunk_count INT DEFAULT 0
);

CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    repo_id INT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    size_bytes INT,
    last_modified TIMESTAMPTZ,
    UNIQUE(repo_id, path)
);
CREATE INDEX idx_files_repo ON files(repo_id);
CREATE INDEX idx_files_hash ON files(file_hash);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    file_id INT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_offset INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024)
);
CREATE INDEX idx_chunks_file ON chunks(file_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### Episodic Memory Schema

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_active TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_message TEXT,
    atlas_response TEXT,
    hemisphere TEXT CHECK (hemisphere IN ('lh', 'rh', 'both')),
    safety_flags JSONB DEFAULT '{}',
    quality_score FLOAT,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024)
);
CREATE INDEX idx_episodes_session ON episodes(session_id);
CREATE INDEX idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX idx_episodes_embedding ON episodes USING ivfflat (embedding vector_cosine_ops);
```

### Procedural Memory Schema (SQLite)

```sql
CREATE TABLE procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    trigger_pattern TEXT NOT NULL,
    procedure_steps TEXT NOT NULL,  -- JSON array
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT DEFAULT '{}'  -- JSON
);
CREATE INDEX idx_procedures_trigger ON procedures(trigger_pattern);
```

### Benefits of Normalization

1. **Repo filtering via FK join** instead of string scan on 44K+ rows
2. **File-level dedup**: skip re-embedding unchanged files by checking file_hash
3. **Cascade deletes**: removing a repo cleans up all files and chunks automatically
4. **Clean stats**: `SELECT COUNT(*) FROM files WHERE repo_id = ?`
5. **Episodic sessions**: group episodes by session for conversation recall
6. **Schema evolution**: add columns to repos/files without touching the chunks table

### Migration Path

The existing flat `chunks` table will be migrated incrementally:
1. Create new normalized tables
2. Populate repos and files from existing chunk metadata
3. Update chunks with file_id FK references
4. Drop old string columns (repo, file_path) from chunks
5. Rebuild indexes
