# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""halyard-state HTTP + WebSocket API.

See :mod:`.app` for the aiohttp application factory.
"""

from .app import WsBroadcaster, build_app

__all__ = ["build_app", "WsBroadcaster"]
