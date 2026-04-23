# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Piper TTS backend — CPU-only ONNX speech synthesis.

Kept as the fallback when XTTS is not installed or the user wants a
fast low-cost voice. The code path matches the original
``avatar_hud.py`` loop so behaviour is unchanged when this backend is
selected.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .backend import AUDIO_SR, TTSSample, float_to_int16, resample_int16_to

log = logging.getLogger("artemis.tts.piper")


class PiperBackend:
    """Lazy-loaded Piper ONNX voice.

    ``voice_path`` is an ``.onnx`` file on disk; Piper pairs it with a
    ``.onnx.json`` config file sitting next to it.
    """

    name = "piper"

    def __init__(self, voice_path: str, *, target_sr: int = AUDIO_SR) -> None:
        self.voice_path = voice_path
        self.target_sr = target_sr
        self._voice: Any | None = None

    # ── lifecycle ───────────────────────────────────────────────

    def _ensure_loaded(self) -> Any:
        if self._voice is not None:
            return self._voice
        try:
            from piper import PiperVoice  # type: ignore
        except ImportError as e:  # pragma: no cover — Atlas has piper installed
            raise RuntimeError(
                "piper-tts not installed — `pip install piper-tts` or flip "
                "ARTEMIS_TTS_BACKEND to xtts"
            ) from e
        log.info("loading piper voice: %s", self.voice_path)
        self._voice = PiperVoice.load(self.voice_path)
        log.info("piper voice loaded (sr=%s)", self._voice.config.sample_rate)
        return self._voice

    def close(self) -> None:
        # Piper holds an ONNX runtime session; drop the reference so
        # it's GC'd. Nothing blocks on shutdown.
        self._voice = None

    # ── synth ───────────────────────────────────────────────────

    def synthesize(self, text: str) -> TTSSample:
        text = (text or "").strip()
        if not text:
            return TTSSample(
                pcm=np.zeros(0, dtype=np.int16),
                sample_rate=self.target_sr,
                source_model=f"piper:{self.voice_path}",
            )
        voice = self._ensure_loaded()
        src_sr = int(voice.config.sample_rate)

        all_pcm: list[np.ndarray] = []
        for chunk in voice.synthesize(text):
            pcm = self._chunk_to_pcm(chunk)
            if len(pcm):
                all_pcm.append(pcm)
        pcm = (
            np.concatenate(all_pcm) if all_pcm else np.zeros(0, dtype=np.int16)
        )
        pcm = resample_int16_to(pcm, src_sr, self.target_sr)
        return TTSSample(
            pcm=pcm,
            sample_rate=self.target_sr,
            source_model=f"piper:{self.voice_path}",
        )

    @staticmethod
    def _chunk_to_pcm(chunk: Any) -> np.ndarray:
        """Piper's synthesize() yields AudioChunks; version-proof decode.

        Some older Piper wheels yielded raw bytes, newer ones yield an
        object with ``audio_int16_bytes``. Handle both so the code
        works across installed versions.
        """
        if hasattr(chunk, "audio_int16_bytes"):
            return np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        if hasattr(chunk, "audio_float_array"):
            return float_to_int16(chunk.audio_float_array)
        if isinstance(chunk, (bytes, bytearray)):
            return np.frombuffer(bytes(chunk), dtype=np.int16)
        if isinstance(chunk, np.ndarray):
            if chunk.dtype == np.int16:
                return chunk
            return float_to_int16(chunk)
        raise TypeError(f"unknown piper chunk type: {type(chunk).__name__}")
