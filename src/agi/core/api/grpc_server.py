\
"""
Reusable gRPC server utilities for AGI-HPC services.
"""

import grpc
from concurrent import futures

class GRPCServer:
    def __init__(self, port: int):
        self.port = port
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))

    def add_servicer(self, servicer, add_fn):
        add_fn(servicer, self.server)

    def start(self):
        self.server.add_insecure_port(f"[::]:{self.port}")
        self.server.start()
        print(f"[grpc] Running on port {self.port}")
        return self.server

    def wait(self):
        self.server.wait_for_termination()
