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
    EpisodicAppendResponse,
    EpisodicQueryResponse,
    EpisodicEvent,
)
from agi.proto_gen.memory_pb2_grpc import (
    EpisodicServiceServicer,
    add_EpisodicServiceServicer_to_server,
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
                f.write(
                    json.dumps(
                        {
                            "episode_id": e.episode_id,
                            "step_index": e.step_index,
                            "event_type": e.event_type,
                            "timestamp_ms": e.timestamp_ms,
                            "payload": e.payload_uri,
                            "tags": dict(e.tags),
                        }
                    )
                    + "\n"
                )

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
        self.fabric.publish("memory.episodic.append", {"count": len(request.events)})
        return EpisodicAppendResponse()

    def Query(self, request, context):
        raw = self.log.query_all()
        # TODO: filtering by tag / time window / event type
        # Convert dict -> EpisodicEvent
        events = []
        for r in raw[: request.limit or 1000]:
            events.append(
                EpisodicEvent(
                    episode_id=r["episode_id"],
                    step_index=r["step_index"],
                    event_type=r["event_type"],
                    timestamp_ms=r["timestamp_ms"],
                    payload_uri=r["payload"],
                    tags=r["tags"],
                )
            )
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
