#!/usr/bin/env python3
"""
Mock gRPC server for AGI-HPC services.

This simple server provides mock implementations of Safety, Metacognition,
and Memory services for local development and testing.

Environment Variables:
    MOCK_SERVICE: Which service to mock (safety, metacog, memory)
    MOCK_PORT: Port to listen on (default: 50120)
    MOCK_BEHAVIOR: How to respond (approve, reject, accept, revise, passthrough)
"""

import os
import time
import logging
from concurrent import futures

import grpc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def serve():
    """Start the mock gRPC server."""
    service = os.environ.get("MOCK_SERVICE", "safety")
    port = int(os.environ.get("MOCK_PORT", "50120"))
    behavior = os.environ.get("MOCK_BEHAVIOR", "approve")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    logger.info(f"Starting mock {service} server on port {port}")
    logger.info(f"Behavior: {behavior}")

    # Note: In a real implementation, we would register actual gRPC servicers
    # For now, this is a placeholder that demonstrates the architecture

    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info(f"Mock {service} server listening on port {port}")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
