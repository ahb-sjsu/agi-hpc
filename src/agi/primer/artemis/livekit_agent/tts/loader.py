# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Environment-driven TTS backend factory.

Resolution order (highest to lowest priority):
  1. ARTEMIS_TTS_BACKEND=piper|xtts|nats_burst|auto   (default: auto)
  2. Under ``auto``:
       if NATS_URL reachable AND a heartbeat is seen  → nats_burst
       elif XTTS reference WAV exists AND TTS lib ok  → xtts
       else                                           → piper
  3. ``nats_burst`` always carries a fallback (xtts → piper) so the
     avatar never gets stuck silent when the worker pool is empty or
     the aggregate NRP quota is saturated.

The ``auto`` mode respects the user's aggregate-quota posture: it
never provisions pods; it only routes to a burst worker if one
happens to be advertising itself on the heartbeat subject.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .backend import TTSBackend
from .nats_burst import NatsBurstBackend
from .piper import PiperBackend
from .xtts import XttsBackend

log = logging.getLogger("artemis.tts.loader")


def build_backend_from_env() -> TTSBackend:
    """Construct the configured TTS backend."""
    choice = os.environ.get("ARTEMIS_TTS_BACKEND", "auto").strip().lower()
    piper_path = os.environ.get(
        "ARTEMIS_VOICE",
        "/home/claude/piper-voices/en_US-amy-medium.onnx",
    )
    xtts_ref = os.environ.get(
        "ARTEMIS_VOICE_REF",
        "/home/claude/artemis-voice-refs/artemis_ref.wav",
    )
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    lang = os.environ.get("ARTEMIS_TTS_LANG", "en")

    if choice == "piper":
        log.info("TTS backend: piper (explicit)")
        return PiperBackend(piper_path)

    if choice == "xtts":
        log.info("TTS backend: xtts (explicit)")
        return XttsBackend(reference_wav=xtts_ref, language=lang)

    if choice == "nats_burst":
        log.info("TTS backend: nats_burst (explicit) → NATS %s", nats_url)
        # Always carry a local fallback so a full/empty burst pool
        # doesn't break speech. Preference order: local XTTS (if the
        # ref wav is present) → Piper.
        fb: TTSBackend | None = _try_xtts_fallback(xtts_ref, lang)
        if fb is None:
            fb = PiperBackend(piper_path)
        return NatsBurstBackend(
            nats_url=nats_url, voice_ref=xtts_ref, language=lang, fallback=fb,
        )

    # ── auto ────────────────────────────────────────────────────
    # Prefer burst if a worker is heartbeating; else in-process XTTS
    # if possible; else Piper. We do NOT launch pods — that's the
    # user's aggregate-quota constraint.
    if _burst_worker_available(nats_url):
        log.info("TTS backend: nats_burst (auto — heartbeat seen)")
        fb = _try_xtts_fallback(xtts_ref, lang) or PiperBackend(piper_path)
        return NatsBurstBackend(
            nats_url=nats_url, voice_ref=xtts_ref, language=lang, fallback=fb,
        )
    local_xtts = _try_xtts_fallback(xtts_ref, lang)
    if local_xtts is not None:
        log.info("TTS backend: xtts (auto — in-process)")
        return local_xtts
    log.info("TTS backend: piper (auto — XTTS prerequisites missing)")
    return PiperBackend(piper_path)


def _try_xtts_fallback(ref: str, lang: str) -> Any | None:
    """Return an XTTS backend if TTS library is importable and ref exists.

    Does not actually load the model — that happens lazily on first
    call, so the startup cost is not paid until we're sure XTTS is
    the chosen path.
    """
    if not os.path.exists(ref):
        return None
    try:
        import importlib.util

        if importlib.util.find_spec("TTS") is None:
            return None
    except Exception:  # noqa: BLE001
        return None
    return XttsBackend(reference_wav=ref, language=lang)


def _burst_worker_available(nats_url: str, *, probe_timeout_s: float = 1.0) -> bool:
    """Return True if at least one worker heartbeat lands in the probe window.

    Lightweight check — we subscribe briefly to the heartbeat wildcard
    and return as soon as one message arrives. If nats-py isn't
    installed or the connection fails, return False (the caller picks
    a local backend).
    """
    try:
        import asyncio

        import nats  # type: ignore

        from .worker import HEARTBEAT_SUBJECT
    except ImportError:
        return False

    async def _probe() -> bool:
        nc = None
        try:
            nc = await nats.connect(servers=[nats_url])
        except Exception:  # noqa: BLE001
            return False
        seen = asyncio.Event()

        async def _on_hb(_msg: Any) -> None:
            seen.set()

        sub = await nc.subscribe(f"{HEARTBEAT_SUBJECT}.>", cb=_on_hb)
        try:
            await asyncio.wait_for(seen.wait(), timeout=probe_timeout_s)
            got = True
        except asyncio.TimeoutError:
            got = False
        finally:
            try:
                await sub.unsubscribe()
            except Exception:  # noqa: BLE001
                pass
            try:
                await nc.drain()
            except Exception:  # noqa: BLE001
                pass
        return got

    try:
        return asyncio.run(_probe())
    except Exception:  # noqa: BLE001
        return False
