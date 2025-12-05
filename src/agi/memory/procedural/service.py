
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
