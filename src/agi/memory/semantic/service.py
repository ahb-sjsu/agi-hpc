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
    SemanticWriteResponse,
    SemanticQueryResponse,
    SemanticHit,
    SemanticEntry,
)
from agi.proto_gen.memory_pb2_grpc import (
    SemanticServiceServicer,
    add_SemanticServiceServicer_to_server,
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
        self.fabric.publish("memory.semantic.write", {"count": len(request.entries)})
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
