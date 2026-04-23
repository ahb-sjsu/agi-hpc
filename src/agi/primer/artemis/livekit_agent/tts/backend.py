# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""TTS backend protocol shared by Piper and XTTS implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Target sample rate for the LiveKit audio publisher. All backends
# resample to this; the avatar's AudioSource expects exactly this rate.
# 24 kHz matches XTTS-v2 native output and is an Opus native mode —
# publishing here means zero resampling on the critical path for the
# default backend and one clean upsample for Piper (22050 → 24000).
AUDIO_SR = 24000


@dataclass
class TTSSample:
    """One synthesized utterance ready for the LiveKit publisher."""

    pcm: np.ndarray  # int16, mono, at AUDIO_SR
    sample_rate: int = AUDIO_SR
    source_model: str = ""  # "piper:en_US-amy-medium", "xtts:reference.wav", …

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.pcm) / float(self.sample_rate)


class TTSBackend(Protocol):
    """Synchronous TTS interface.

    Backends are expected to be stateful (model loaded once, reused
    per call) and thread-safe with respect to ``synthesize``. The
    avatar runs one TTS worker thread, so external serialization is
    not required.
    """

    name: str

    def synthesize(self, text: str) -> TTSSample:
        """Turn text into a PCM sample at :data:`AUDIO_SR`.

        Implementations MUST:
          - Return empty PCM (len 0) for empty input rather than raising.
          - Resample to :data:`AUDIO_SR` if the model's native SR differs.
          - Never return None — callers check ``.duration_s``.
        """
        ...

    def close(self) -> None:
        """Release any heavy resources (GPU memory, temp files)."""
        ...


def resample_int16_to(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interp resample. Adequate for speech; not studio-grade.

    The avatar pipeline resamples to 22050 Hz regardless of backend so
    the LiveKit publisher can stamp consistent frame sizes. For a
    higher-quality resample, swap in ``soxr`` or ``scipy.signal`` —
    measurable but not audible for monologue speech.
    """
    if src_sr == dst_sr or len(pcm) == 0:
        return pcm.astype(np.int16, copy=False)
    ratio = dst_sr / float(src_sr)
    new_len = int(round(len(pcm) * ratio))
    if new_len <= 0:
        return np.zeros(0, dtype=np.int16)
    xs = np.linspace(0, len(pcm) - 1, new_len)
    ys = np.interp(xs, np.arange(len(pcm)), pcm.astype(np.float32))
    return ys.astype(np.int16)


def float_to_int16(wave: np.ndarray, clip: bool = True) -> np.ndarray:
    """Convert a float waveform in [-1, 1] to int16 PCM.

    XTTS returns float32; Piper returns int16 directly, so only
    callers from XTTS need this helper.
    """
    arr = np.asarray(wave, dtype=np.float32)
    if clip:
        arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16)
