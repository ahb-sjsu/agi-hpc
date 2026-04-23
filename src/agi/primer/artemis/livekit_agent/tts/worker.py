# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS TTS worker — NATS-burst XTTS inference service.

Subscribes to ``agi.rh.artemis.tts.jobs`` in the ``tts-workers`` queue
group. Each job is a single utterance; the worker synthesizes it with
a locally-loaded XTTS-v2 model and publishes the PCM back on the
reply subject supplied in the request.

# NRP aggregate-quota contract  (!! read before deploying !!)
# ─────────────────────────────────────────────────────────────
# The user's NRP namespace runs a HARD aggregate cap:
#   - ≤ 4 pods can sit in the GPU>40% / CPU 20-200% / RAM 20-150% bucket.
#   - 5+ pods means ALL must stay under the thresholds.
#   - Idle-relative-to-request pods get killed by the watchdog.
# This worker is DESIGNED NOT to claim a dedicated slot. ARTEMIS code
# NEVER spawns a pod. The recommended deploy is:
#
#   (a) Co-tenant: run the worker *inside* an existing GPU pod that is
#       already doing other work (Primer inference, an ARC scientist
#       session, a training run). The worker is a background thread
#       with low steady-state CPU — activity only during utterances.
#       Example:
#         nohup python -m agi.primer.artemis.livekit_agent.tts.worker &
#
#   (b) Atlas-local: run it on Atlas itself. Uses the local GPU when
#       idle. Zero NRP quota consumption.
#
#   (c) Dedicated burst: ONLY when the user has slack (n_active < 4
#       per the telemetry server). If you add one, it must stay busy —
#       the watchdog will kill an idle XTTS pod quickly.
#
# The avatar falls back to local (Piper or in-process XTTS) whenever
# the worker doesn't reply in time, so NOT running a worker is fine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import time
import uuid

from .backend import resample_int16_to
from .nats_burst import (
    H_ERROR,
    H_JOB,
    H_MODEL,
    H_SR,
    QUEUE_GROUP,
    REPLY_PREFIX,
    SUBJECT_JOBS,
)
from .xtts import XttsBackend

# Module-level cache so per-job voice_ref overrides don't force a
# 2 GB model reload. Each entry keys on the reference path; the
# XttsBackend holds the loaded model.
_VOICE_CACHE: dict[str, XttsBackend] = {}

log = logging.getLogger("artemis.tts.worker")

HEARTBEAT_SUBJECT = "agi.rh.artemis.tts.workers"  # .<worker_id>.hb
HEARTBEAT_INTERVAL_S = 15.0

# Idle-shutdown — if co-tenant and no jobs for this long, exit so the
# scheduler doesn't flag us for under-utilization. Default 15 min,
# configurable via env.
IDLE_SHUTDOWN_S = float(os.environ.get("ARTEMIS_TTS_WORKER_IDLE_S", "900"))


async def _run() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    ref_wav = os.environ.get(
        "ARTEMIS_VOICE_REF",
        "/home/claude/artemis-voice-refs/artemis_ref.wav",
    )
    language = os.environ.get("ARTEMIS_TTS_LANG", "en")
    worker_id = (
        os.environ.get("ARTEMIS_TTS_WORKER_ID")
        or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    )

    # Load the model eagerly so the first inbound job doesn't pay the
    # ~5-10s warm-up cost on the critical path.
    xtts = XttsBackend(reference_wav=ref_wav, language=language)
    xtts._ensure_loaded()  # noqa: SLF001 — intentional pre-warm

    import nats  # type: ignore

    nc = await nats.connect(servers=[nats_url])
    log.info("worker %s connected to NATS: %s", worker_id, nats_url)

    last_activity = time.monotonic()

    async def _handle(msg: "nats.aio.msg.Msg") -> None:  # type: ignore[name-defined]
        nonlocal last_activity
        last_activity = time.monotonic()
        try:
            req = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("bad job payload: %s", e)
            return
        job_id = str(req.get("job_id") or uuid.uuid4().hex[:12])
        reply_to = str(req.get("reply_to") or f"{REPLY_PREFIX}.{job_id}")
        text = str(req.get("text") or "").strip()
        if not text:
            await _reply_error(nc, reply_to, job_id, "empty text")
            return
        # Optional per-job voice override so the keeper can narrate as
        # different NPCs later (S1g).
        voice_override = req.get("voice_ref")
        t0 = time.monotonic()
        try:
            if voice_override and voice_override != ref_wav:
                # Reuse cached backends across jobs so NPC voice
                # switches don't re-download or re-load the model.
                # The model weights are hot-cached inside TTS.api; a
                # new XttsBackend reuses them via coqui's internal
                # cache. Speaker embedding is recomputed per voice.
                backend = _VOICE_CACHE.get(voice_override)
                if backend is None:
                    backend = XttsBackend(
                        reference_wav=voice_override, language=language,
                    )
                    _VOICE_CACHE[voice_override] = backend
                sample = backend.synthesize(text)
            else:
                sample = xtts.synthesize(text)
        except Exception as e:  # noqa: BLE001
            log.exception("synth failed for %s: %s", job_id, e)
            await _reply_error(nc, reply_to, job_id, str(e)[:120])
            return
        dt = time.monotonic() - t0
        log.info(
            "job %s: %d chars → %.2fs audio in %.2fs (rtx %.2f)",
            job_id, len(text), sample.duration_s, dt,
            sample.duration_s / max(dt, 1e-3),
        )
        # The avatar resamples to 22050 again on receipt, but publishing
        # at the model's native 24 kHz preserves the most detail over
        # the wire. Keep native here and let the client resample.
        # We already have the sample at AUDIO_SR because XttsBackend
        # resamples — re-expose the native rate instead:
        pcm24 = resample_int16_to(sample.pcm, sample.sample_rate, 24000)
        hdrs = {
            H_JOB: job_id,
            H_SR: "24000",
            H_MODEL: sample.source_model or "xtts",
        }
        await nc.publish(reply_to, pcm24.tobytes(), headers=hdrs)

    async def _heartbeat() -> None:
        hb_subject = f"{HEARTBEAT_SUBJECT}.{worker_id}.hb"
        while True:
            payload = json.dumps(
                {
                    "worker_id": worker_id,
                    "model": "xtts-v2",
                    "voice_ref": os.path.basename(ref_wav),
                    "ts": time.time(),
                    "last_activity": last_activity,
                },
                separators=(",", ":"),
            ).encode()
            try:
                await nc.publish(hb_subject, payload)
            except Exception as e:  # noqa: BLE001
                log.warning("heartbeat publish failed: %s", e)
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

    await nc.subscribe(SUBJECT_JOBS, queue=QUEUE_GROUP, cb=_handle)
    hb_task = asyncio.create_task(_heartbeat())

    stop = asyncio.Event()

    def _handle_sig() -> None:
        log.info("signal received, draining worker")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            pass

    # Idle-shutdown watchdog — protects the aggregate-quota contract
    # when this worker happens to be running in a co-tenant pod.
    async def _idle_watch() -> None:
        while not stop.is_set():
            await asyncio.sleep(60)
            if (
                IDLE_SHUTDOWN_S > 0
                and time.monotonic() - last_activity > IDLE_SHUTDOWN_S
            ):
                log.info(
                    "idle for %.0fs > %.0fs — shutting down (quota hygiene)",
                    time.monotonic() - last_activity, IDLE_SHUTDOWN_S,
                )
                stop.set()

    idle_task = asyncio.create_task(_idle_watch())
    await stop.wait()

    hb_task.cancel()
    idle_task.cancel()
    try:
        await nc.drain()
    except Exception:  # noqa: BLE001
        pass
    xtts.close()
    log.info("worker %s stopped", worker_id)
    return 0


async def _reply_error(nc, reply_to: str, job_id: str, err: str) -> None:
    hdrs = {H_JOB: job_id, H_SR: "24000", H_ERROR: err[:120]}
    try:
        await nc.publish(reply_to, b"", headers=hdrs)
    except Exception as e:  # noqa: BLE001
        log.warning("error reply failed: %s", e)


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
