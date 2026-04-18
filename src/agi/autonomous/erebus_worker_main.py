"""Entry point for Erebus worker pods.

The pod's bash launch script installs ``nats-bursting`` + Erebus deps,
clones the bundle, then execs this module. All the pool/NATS mechanics
live in ``nats-bursting``; all we do here is register Erebus handlers.
"""

from __future__ import annotations

from nats_bursting import run_worker

from agi.autonomous.erebus_handlers import HANDLERS


if __name__ == "__main__":
    run_worker(handlers=HANDLERS)
