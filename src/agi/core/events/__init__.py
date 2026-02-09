# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""
Event Fabric module for AGI-HPC.

Provides pub/sub messaging infrastructure with multiple backends:
- local: In-process (testing)
- zmq: ZeroMQ (multi-node)
- ucx: UCX (HPC-grade RDMA)
- redis: Redis Streams (persistent)
"""

from agi.core.events.fabric import (
    EventFabric,
    EventHandler,
    FabricBackend,
    LocalBackend,
    ZmqBackend,
    UcxBackend,
)

__all__ = [
    "EventFabric",
    "EventHandler",
    "FabricBackend",
    "LocalBackend",
    "ZmqBackend",
    "UcxBackend",
]

# Optional Redis exports
try:
    from agi.core.events.redis_backend import (
        RedisBackend,
        AsyncRedisBackend,
        RedisBackendConfig,
        StreamMessage,
    )

    __all__.extend(
        [
            "RedisBackend",
            "AsyncRedisBackend",
            "RedisBackendConfig",
            "StreamMessage",
        ]
    )
except ImportError:
    pass

# Optional broker exports
try:
    from agi.core.events.broker import (
        FabricBroker,
        BrokerClient,
        BrokerConfig,
        BrokerMetrics,
    )

    __all__.extend(
        [
            "FabricBroker",
            "BrokerClient",
            "BrokerConfig",
            "BrokerMetrics",
        ]
    )
except ImportError:
    pass
