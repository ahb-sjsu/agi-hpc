# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Cognition adapter — route a Discord message through Erebus's existing
chat pipeline (Ego / vMOE / RAG) via the local telemetry endpoint.

Uses stdlib urllib so the faculty pulls in no HTTP dependency. The adapter
is a plain callable ``(text, conversation_id) -> reply`` so ``handler.py``
can be tested with a fake instead.
"""

from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger("discord.cognition")


def http_cognition(base_url: str, timeout: float = 120.0):
    """Return a ``respond(text, conv_id) -> str`` bound to the chat endpoint.

    POSTs ``{"message", "conversation_id"}`` to ``/api/erebus/chat`` and
    returns the ``response`` field. Raises on transport/HTTP error so the
    caller can decide to stay silent (it does)."""
    url = base_url.rstrip("/") + "/api/erebus/chat"

    def respond(text: str, conversation_id: str) -> str:
        payload = json.dumps(
            {"message": text, "conversation_id": conversation_id}
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("response", "")).strip()

    return respond
