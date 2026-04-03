# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.
# You may obtain a copy of the License at the root of this repository,
# or by contacting the author(s).
#
# You may use, modify, and distribute this file for non-commercial
# research and educational purposes, subject to the conditions in
# the License. Commercial use, high-risk deployments, and autonomous
# operation in safety-critical domains require separate written
# permission and must include appropriate safety and governance controls.
#
# Unless required by applicable law or agreed to in writing, this
# software is provided "AS IS", without warranties or conditions
# of any kind. See the License for the specific language governing
# permissions and limitations.

"""
DHT Service Registry subpackage for AGI-HPC.

Provides a local service registry using a simplified Kademlia-like
interface, backed by PostgreSQL for persistence. Integrates with
the NATS Event Fabric for distributed discovery and heartbeats.

Phase 6 (DHT Service Registry + Final Polish).
"""

from __future__ import annotations

__all__ = [
    "ConfigStore",
    "DhtNatsService",
    "ServiceInfo",
    "ServiceRegistry",
]

try:
    from agi.meta.dht.registry import ServiceInfo, ServiceRegistry
    from agi.meta.dht.nats_service import DhtNatsService
    from agi.meta.dht.config_store import ConfigStore
except ImportError:
    pass
