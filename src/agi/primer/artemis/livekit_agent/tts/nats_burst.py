# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""NATS-burst TTS backend — offload XTTS inference to worker pods.

ARTEMIS's avatar runs on Atlas; Atlas's GPU is busy with the rest of
the Primer stack. Per-utterance XTTS on CPU is ~2-3× real-time, which
is fine for short lines but collapses under a busy scene. Route the
work to NRP burst workers instead:

  Avatar (synthesize)
    ──publish──▶ agi.rh.artemis.tts.jobs
                 payload = {job_id, text, voice_ref, language, reply_to}

  Worker pool (one per NRP pod or home box)
    ──subscribe── agi.rh.artemis.tts.jobs (queue group "tts-workers")
    ──load model once, service jobs
    ──publish──▶ <reply_to>
                 headers = {job_id, sample_rate, source_model, error?}
                 data    = raw int16 PCM bytes

  Avatar waits on the unique reply subject with a timeout; on success
  the bytes become a TTSSample, on timeout we fall back to Piper.

The **queue group** ("tts-workers") gives us free load-balancing: any
number of workers can subscribe and NATS will fan out one job per
worker. Adding a second NRP pod during a session doubles throughput
without any avatar-side code change — this is the core of the burst
pattern already used elsewhere in agi-hpc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any

import numpy as np

from .backend import AUDIO_SR, TTSSample, resample_int16_to

log = logging.getLogger("artemis.tts.nats_burst")

SUBJECT_JOBS = "agi.rh.artemis.tts.jobs"
REPLY_PREFIX = "agi.rh.artemis.tts.results"
QUEUE_GROUP = "tts-workers"

# Header keys — NATS headers are stringly-typed.
H_JOB = "job_id"
H_SR = "sample_rate"
H_MODEL = "source_model"
H_ERROR = "error"


class NatsBurstBackend:
    """Thread-facing backend that forwards jobs to a NATS worker pool.

    The avatar's TTS thread is synchronous, so we own a background
    event loop on its own thread and bridge sync↔async via
    :func:`asyncio.run_coroutine_threadsafe`. The loop holds a single
    persistent NATS connection so we don't pay connect latency per
    utterance.
    """

    name = "nats_burst"

    def __init__(
        self,
        *,
        nats_url: str,
        voice_ref: str,
        language: str = "en",
        timeout_s: float = 30.0,
        fallback: Any | None = None,
        target_sr: int = AUDIO_SR,
    ) -> None:
        self.nats_url = nats_url
        self.voice_ref = voice_ref
        self.language = language
        self.timeout_s = timeout_s
        self.fallback = fallback  # optional TTSBackend used on timeout
        self.target_sr = target_sr
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._nc: Any | None = None
        self._ready = threading.Event()
        self._start_bg_loop()

    # ── lifecycle ───────────────────────────────────────────────

    def _start_bg_loop(self) -> None:
        self._loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._connect())
            self._ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, name="tts-nats", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    async def _connect(self) -> None:
        import nats  # type: ignore

        log.info("burst backend connecting to NATS: %s", self.nats_url)
        self._nc = await nats.connect(servers=[self.nats_url])
        log.info("burst backend connected")

    def close(self) -> None:
        if not self._loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._drain(), self._loop)
        try:
            fut.result(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop = None

    async def _drain(self) -> None:
        if self._nc is not None:
            try:
                await self._nc.drain()
            except Exception:  # noqa: BLE001
                pass
            self._nc = None

    # ── synth ───────────────────────────────────────────────────

    def synthesize(self, text: str) -> TTSSample:
        text = (text or "").strip()
        if not text:
            return TTSSample(
                pcm=np.zeros(0, dtype=np.int16),
                sample_rate=self.target_sr,
            )
        if self._loop is None or self._nc is None:
            return self._fallback_or_empty(text, reason="not connected")

        fut = asyncio.run_coroutine_threadsafe(
            self._request(text), self._loop
        )
        try:
            sample = fut.result(timeout=self.timeout_s)
        except Exception as e:  # noqa: BLE001
            log.warning("burst synth failed: %s", e)
            return self._fallback_or_empty(text, reason=str(e))
        return sample

    async def _request(self, text: str) -> TTSSample:
        job_id = uuid.uuid4().hex[:16]
        reply_to = f"{REPLY_PREFIX}.{job_id}"
        req_payload = json.dumps(
            {
                "job_id": job_id,
                "text": text,
                "voice_ref": self.voice_ref,
                "language": self.language,
                "reply_to": reply_to,
            },
            separators=(",", ":"),
        ).encode()

        # Subscribe on the unique reply subject first so we don't race
        # the worker. `next_msg` returns the first message.
        sub = await self._nc.subscribe(reply_to)
        try:
            await self._nc.publish(SUBJECT_JOBS, req_payload)
            t0 = time.monotonic()
            msg = await sub.next_msg(timeout=self.timeout_s)
            dt = time.monotonic() - t0
            sample = _msg_to_sample(msg, target_sr=self.target_sr)
            log.info(
                "burst synth ok: job=%s %.2fs rtx=%.2f %s",
                job_id,
                dt,
                (sample.duration_s / max(dt, 1e-3)),
                sample.source_model,
            )
            return sample
        finally:
            try:
                await sub.unsubscribe()
            except Exception:  # noqa: BLE001
                pass

    def _fallback_or_empty(self, text: str, *, reason: str) -> TTSSample:
        if self.fallback is None:
            return TTSSample(
                pcm=np.zeros(0, dtype=np.int16),
                sample_rate=self.target_sr,
                source_model=f"nats_burst:error:{reason[:40]}",
            )
        log.info("burst → fallback (%s)", reason)
        return self.fallback.synthesize(text)


def _msg_to_sample(msg: Any, *, target_sr: int) -> TTSSample:
    """Decode a worker reply into a :class:`TTSSample`.

    Worker protocol:
      headers[sample_rate]   — int as string
      headers[source_model]  — "xtts:artemis_ref" or similar
      headers[error]         — present only on failure
      data                   — raw int16 PCM (little-endian)
    """
    headers = dict(getattr(msg, "headers", None) or {})
    if headers.get(H_ERROR):
        raise RuntimeError(f"worker error: {headers[H_ERROR]}")
    src_sr = int(headers.get(H_SR, "24000"))
    model = str(headers.get(H_MODEL, "xtts"))
    pcm = np.frombuffer(msg.data or b"", dtype=np.int16)
    pcm = resample_int16_to(pcm, src_sr, target_sr)
    return TTSSample(pcm=pcm, sample_rate=target_sr, source_model=model)
