#!/usr/bin/env python3
"""
generate_memory_services.py

Creates service skeletons for:
- Semantic Memory
- Episodic Memory
- Procedural Memory

Each service includes:
- gRPC server
- EventFabric integration
- Config loader
- Appendix-A compliant RPC stubs
- Logging + TODO placeholders

Run:
    python generate_memory_services.py
"""

from pathlib import Path
from textwrap import dedent

def write(path: Path, content: str, overwrite=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"[skip] {path}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"[write] {path}")

# =====================================================================
# SEMANTIC MEMORY SERVICE
# =====================================================================

SEMANTIC_SERVICE = dedent(r'''
"""
Semantic Memory Service (AGI-HPC)

Responsibilities:
- Store vector embeddings + metadata
- Support filtered nearest-neighbor search
- Provide Write() + Query() over gRPC
- EventFabric notifications (optional)

Currently uses an in-memory dictionary. Later: Qdrant, FAISS, Milvus, etc.
"""

from typing import Dict, List
from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

from agi.proto_gen.memory_pb2 import (
    SemanticWriteResponse, SemanticQueryResponse, SemanticHit, SemanticEntry
)
from agi.proto_gen.memory_pb2_grpc import (
    SemanticServiceServicer,
    add_SemanticServiceServicer_to_server
)

class InMemoryVectorStore:
    """A tiny placeholder vector DB implementation."""
    def __init__(self):
        self.entries: Dict[str, SemanticEntry] = {}

    def write(self, entries: List[SemanticEntry]):
        for e in entries:
            self.entries[e.id] = e

    def query(self, qvec: bytes, top_k: int):
        # TODO: true vector similarity
        # Currently: returns first K entries as dummy.
        results = []
        for i, e in enumerate(self.entries.values()):
            if len(results) >= top_k:
                break
            results.append(SemanticHit(entry=e, score=1.0))
        return results

class SemanticMemServicer(SemanticServiceServicer):
    def __init__(self, store: InMemoryVectorStore, fabric: EventFabric):
        self.store = store
        self.fabric = fabric

    def Write(self, request, context):
        self.store.write(request.entries)
        self.fabric.publish("memory.semantic.write", {
            "count": len(request.entries)
        })
        return SemanticWriteResponse(ids=[e.id for e in request.entries])

    def Query(self, request, context):
        hits = self.store.query(request.query_embedding, request.top_k)
        return SemanticQueryResponse(hits=hits)

class SemanticMemoryService:
    def __init__(self, config_path="configs/memory_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.store = InMemoryVectorStore()
        self.grpc = GRPCServer(self.config.rpc_port)

    def run(self):
        serv = SemanticMemServicer(self.store, self.fabric)
        add_SemanticServiceServicer_to_server(serv, self.grpc.server)
        print("[MEM-SEM] Semantic Memory service running...")
        self.grpc.start()
        self.grpc.wait()

def main():
    SemanticMemoryService().run()

if __name__ == "__main__":
    main()
''')

# =====================================================================
# EPISODIC MEMORY SERVICE
# =====================================================================

EPISODIC_SERVICE = dedent(r'''
"""
Episodic Memory Service (AGI-HPC)

Responsibilities:
- Append observations/actions/simulations/safety events into Parquet logs
- Query episodes or ranges
- Provide Append() + Query() RPCs
- Useful for replay + training data + safety analysis

This skeleton uses a naive JSONL-based log for now.
"""

import json
import time
from pathlib import Path

from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

from agi.proto_gen.memory_pb2 import (
    EpisodicAppendResponse, EpisodicQueryResponse, EpisodicEvent
)
from agi.proto_gen.memory_pb2_grpc import (
    EpisodicServiceServicer,
    add_EpisodicServiceServicer_to_server
)

class AppendOnlyLog:
    """Simple append-only JSONL-based store for episodic events."""
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, events):
        ts = int(time.time() * 1000)
        fname = self.root / f"events_{ts}.jsonl"
        with fname.open("w", encoding="utf8") as f:
            for e in events:
                f.write(json.dumps({
                    "episode_id": e.episode_id,
                    "step_index": e.step_index,
                    "event_type": e.event_type,
                    "timestamp_ms": e.timestamp_ms,
                    "payload": e.payload_uri,
                    "tags": dict(e.tags),
                }) + "\n")

    def query_all(self):
        results = []
        for file in self.root.glob("*.jsonl"):
            with file.open() as f:
                for line in f:
                    results.append(json.loads(line))
        return results

class EpisodicMemServicer(EpisodicServiceServicer):
    def __init__(self, log: AppendOnlyLog, fabric: EventFabric):
        self.log = log
        self.fabric = fabric

    def Append(self, request, context):
        self.log.append(request.events)
        self.fabric.publish("memory.episodic.append", {
            "count": len(request.events)
        })
        return EpisodicAppendResponse()

    def Query(self, request, context):
        raw = self.log.query_all()
        # TODO: filtering by tag / time window / event type
        # Convert dict -> EpisodicEvent
        events = []
        for r in raw[: request.limit or 1000]:
            events.append(EpisodicEvent(
                episode_id=r["episode_id"],
                step_index=r["step_index"],
                event_type=r["event_type"],
                timestamp_ms=r["timestamp_ms"],
                payload_uri=r["payload"],
                tags=r["tags"],
            ))
        return EpisodicQueryResponse(events=events)

class EpisodicMemoryService:
    def __init__(self, config_path="configs/memory_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.log = AppendOnlyLog(Path("data/episodic"))
        self.grpc = GRPCServer(self.config.rpc_port + 1)

    def run(self):
        serv = EpisodicMemServicer(self.log, self.fabric)
        add_EpisodicServiceServicer_to_server(serv, self.grpc.server)
        print("[MEM-EPI] Episodic Memory service running...")
        self.grpc.start()
        self.grpc.wait()

def main():
    EpisodicMemoryService().run()

if __name__ == "__main__":
    main()
''')

# =====================================================================
# PROCEDURAL MEMORY SERVICE
# =====================================================================

PROCEDURAL_SERVICE = dedent(r'''
"""
Procedural Memory Service (AGI-HPC)

Responsibilities:
- Store skill definitions (preconditions, postconditions, policy refs)
- Search by tags or metadata
- Update & version skill catalog
- gRPC: SkillGet, SkillSearch, SkillPut

Real implementation will use SQL/Vector DB.
"""

from typing import Dict
from agi.common.config_loader import load_config
from agi.core.events.fabric import EventFabric
from agi.core.api.grpc_server import GRPCServer

from agi.proto_gen.memory_pb2 import (
    Skill, SkillGetResponse, SkillGetRequest,
    SkillSearchResponse, SkillSearchRequest,
)
from agi.proto_gen.memory_pb2_grpc import (
    ProceduralServiceServicer,
    add_ProceduralServiceServicer_to_server
)

class SkillCatalog:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def put(self, skill: Skill):
        self.skills[skill.skill_id] = skill

    def get(self, skill_id: str):
        return self.skills.get(skill_id, None)

    def search(self, tags):
        # TODO: real filtering
        return [s for s in self.skills.values() if True]

class ProceduralMemServicer(ProceduralServiceServicer):
    def __init__(self, catalog: SkillCatalog, fabric: EventFabric):
        self.catalog = catalog
        self.fabric = fabric

    def Get(self, request: SkillGetRequest, context):
        s = self.catalog.get(request.skill_id)
        if s:
            return SkillGetResponse(skill=s)
        return SkillGetResponse()  # empty reply

    def Search(self, request: SkillSearchRequest, context):
        # Placeholder: return all skills
        results = self.catalog.search(request.domain_tags)
        return SkillSearchResponse(skills=results)

    # Optional: SkillPut (not in Appendix A but useful)
    # def Put(self, request, context):
    #     self.catalog.put(request.skill)
    #     return Empty()

class ProceduralMemoryService:
    def __init__(self, config_path="configs/memory_config.yaml"):
        self.config = load_config(config_path)
        self.fabric = EventFabric()
        self.catalog = SkillCatalog()
        self.grpc = GRPCServer(self.config.rpc_port + 2)

    def run(self):
        serv = ProceduralMemServicer(self.catalog, self.fabric)
        add_ProceduralServiceServicer_to_server(serv, self.grpc.server)
        print("[MEM-PROC] Procedural Memory service running...")
        self.grpc.start()
        self.grpc.wait()

def main():
    ProceduralMemoryService().run()

if __name__ == "__main__":
    main()
''')

# =====================================================================
# Main Script: Write Files
# =====================================================================

def main():
    root = Path("src/agi/memory")

    write(root / "semantic" / "service.py", SEMANTIC_SERVICE)
    write(root / "episodic" / "service.py", EPISODIC_SERVICE)
    write(root / "procedural" / "service.py", PROCEDURAL_SERVICE)

    print("\nMemory subsystem skeleton generated.")

if __name__ == "__main__":
    main()
