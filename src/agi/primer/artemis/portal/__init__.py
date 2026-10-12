# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS GM Portal — keeper cockpit backend.

HTTP + NATS service that powers the in-session Keeper UI at
``deploy/web/gm/``. Endpoints:

  GET  /api/healthz
  GET  /api/chat/recent?session=&limit=
  GET  /api/chat/thread?session=&participant=&limit=
  GET  /api/chat/search?q=&session=&limit=
  POST /api/chat/whisper            {session, to_id, body}
  POST /api/chat/broadcast          {session, body}
  GET  /api/sheet?name=characters
  GET  /api/wiki/search?q=&limit=
  GET  /api/scene                   (last seen scene state)
  POST /api/scene                   {name, flags}
  POST /api/say                     {text}  — silent-GM narration

Every endpoint except ``/api/healthz`` requires a bearer token whose
identity starts with ``keeper:`` — the player UI and the avatar agent
do NOT call this service; Keeper clients only.
"""

from __future__ import annotations

from .cache import SheetCache
from .server import build_app, run_forever

__all__ = ["SheetCache", "build_app", "run_forever"]
