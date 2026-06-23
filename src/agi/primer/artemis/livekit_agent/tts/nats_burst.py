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
import queue
import re
import threading
import time
import uuid
from typing import Any, Iterator

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
# Chunked reply — NATS's default max payload is 1 MB, and long
# utterances (20 s+ at 24 kHz) exceed that. Replies carry a 1-based
# chunk index + total count; the client reassembles in order.
H_CHUNK_IDX = "chunk_idx"
H_CHUNK_N = "chunk_n"
# 512 KB of PCM ≈ 10.9 s @ 24 kHz int16 — well under the 1 MB limit
# with plenty of headroom for NATS's own envelope.
CHUNK_BYTES = 512 * 1024


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
        """Non-streaming API — collects the whole stream before returning."""
        text = (text or "").strip()
        if not text:
            return TTSSample(
                pcm=np.zeros(0, dtype=np.int16),
                sample_rate=self.target_sr,
            )
        pcms: list[np.ndarray] = []
        model: str = "xtts"
        for chunk in self.synthesize_stream(text):
            pcms.append(chunk.pcm)
            model = chunk.source_model or model
        pcm = np.concatenate(pcms) if pcms else np.zeros(0, dtype=np.int16)
        return TTSSample(
            pcm=pcm,
            sample_rate=self.target_sr,
            source_model=model,
        )

    def synthesize_stream(self, text: str) -> Iterator[TTSSample]:
        """Stream per-sentence samples as the worker produces them.

        Sync→async bridge: a dedicated queue is populated by a
        coroutine on the bg loop; this generator yields from the
        queue as chunks arrive. Time-to-first-audio is dominated by
        the first sentence's synth latency rather than the full
        utterance.
        """
        text = (text or "").strip()
        if not text:
            return
        if self._loop is None or self._nc is None:
            sample = self._fallback_or_empty(text, reason="not connected")
            if sample.duration_s > 0:
                yield sample
            return

        out_q: queue.Queue = queue.Queue()
        fut = asyncio.run_coroutine_threadsafe(
            self._stream_request(text, out_q),
            self._loop,
        )
        try:
            while True:
                item = out_q.get(timeout=self.timeout_s)
                if item is None:
                    break  # sentinel: done
                if isinstance(item, Exception):
                    raise item
                yield item
        except Exception as e:  # noqa: BLE001
            log.warning("burst stream failed: %s — falling back", e)
            # Drain + cancel the remote request (best effort).
            try:
                fut.cancel()
            except Exception:  # noqa: BLE001
                pass
            if self.fallback is not None:
                fb_sample = self.fallback.synthesize(text)
                if fb_sample.duration_s > 0:
                    yield fb_sample
            return
        # Ensure the future completes to surface any late error.
        try:
            fut.result(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass

    async def _request(self, text: str) -> TTSSample:
        """Legacy one-shot path: collect every chunk, return one sample.

        Used by :meth:`synthesize` and kept to preserve backwards
        compatibility for callers that need the full reply in hand.
        """
        pcms: list[np.ndarray] = []
        model: str = "xtts"
        for chunk in self.synthesize_stream(text):
            pcms.append(chunk.pcm)
            model = chunk.source_model or model
        pcm = np.concatenate(pcms) if pcms else np.zeros(0, dtype=np.int16)
        return TTSSample(
            pcm=pcm,
            sample_rate=self.target_sr,
            source_model=model,
        )

    async def _stream_request(self, text: str, out_q: queue.Queue) -> None:
        """Publish request, forward each reply chunk onto out_q.

        Runs on the backend's bg loop; out_q is drained by
        :meth:`synthesize_stream` on the TTS worker thread.
        """
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

        sub = await self._nc.subscribe(reply_to)
        try:
            await self._nc.publish(SUBJECT_JOBS, req_payload)
            t0 = time.monotonic()
            n_received = 0
            expected: int | None = None

            while expected is None or n_received < expected:
                msg = await sub.next_msg(timeout=self.timeout_s)
                headers = dict(getattr(msg, "headers", None) or {})
                if headers.get(H_ERROR):
                    out_q.put(RuntimeError(f"worker error: {headers[H_ERROR]}"))
                    return
                n = int(headers.get(H_CHUNK_N, "1"))
                idx = int(headers.get(H_CHUNK_IDX, str(n_received + 1)))
                expected = n
                if n_received == 0:
                    log.info(
                        "burst stream first chunk: job=%s %.2fs (of %d)",
                        job_id,
                        time.monotonic() - t0,
                        n,
                    )
                sr = int(headers.get(H_SR, "24000"))
                model = str(headers.get(H_MODEL, "xtts"))
                pcm = np.frombuffer(msg.data or b"", dtype=np.int16)
                pcm = resample_int16_to(pcm, sr, self.target_sr)
                out_q.put(
                    TTSSample(
                        pcm=pcm,
                        sample_rate=self.target_sr,
                        source_model=model,
                    )
                )
                n_received = max(n_received, idx)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            out_q.put(e)
            return
        finally:
            try:
                await sub.unsubscribe()
            except Exception:  # noqa: BLE001
                pass
            out_q.put(None)  # sentinel: done

    def _fallback_or_empty(self, text: str, *, reason: str) -> TTSSample:
        if self.fallback is None:
            return TTSSample(
                pcm=np.zeros(0, dtype=np.int16),
                sample_rate=self.target_sr,
                source_model=f"nats_burst:error:{reason[:40]}",
            )
        log.info("burst → fallback (%s)", reason)
        return self.fallback.synthesize(text)


_SENTENCE_SPLIT_RE = re.compile(
    # End-of-sentence punctuation followed by whitespace (or end-of-string).
    # Keeps the punctuation with the left segment.
    r"(?<=[.!?])\s+"
)


def split_sentences(text: str, *, max_chars: int = 240) -> list[str]:
    """Split a long utterance into sentence-ish segments.

    We use a simple regex rather than pysbd/spacy here — good enough
    for chat-style prose, and avoids another heavy dep on the worker.
    Segments longer than ``max_chars`` (rare for chat; common for
    narration paragraphs) are further split on commas/semicolons so
    no single chunk exceeds the NATS payload cap.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            out.append(p)
            continue
        # Long segment — break on , ; :
        sub = re.split(r"(?<=[,;:])\s+", p)
        buf = ""
        for s in sub:
            if len(buf) + 1 + len(s) <= max_chars:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    out.append(buf)
                buf = s
        if buf:
            out.append(buf)
    return out


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
