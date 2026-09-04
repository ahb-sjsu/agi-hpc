# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Coqui XTTS-v2 TTS backend — in-process high-quality voice clone.

XTTS-v2 is a voice-clone multilingual neural TTS model. It accepts a
short reference WAV (6-30 s of clean speech) and produces speech in
that voice for arbitrary text. Output is 24 kHz float32, resampled
here to 22.05 kHz int16 for the LiveKit audio publisher.

**License note:** XTTS-v2 is released under the Coqui Public Model
License — free for non-commercial use; attribution required. This
project is a personal TTRPG companion and fits comfortably inside
that scope.

For a GPU-accelerated deployment that doesn't pin Atlas's main GPU,
see :mod:`.nats_burst` which routes inference to NRP burst workers.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from .backend import AUDIO_SR, TTSSample, float_to_int16, resample_int16_to

log = logging.getLogger("artemis.tts.xtts")

# Coqui's default model id. Pinned so model-repo rollouts don't
# change voice timbre under a live session.
MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"


class XttsBackend:
    """Local XTTS-v2 synthesis. Heavy — loads ~2 GB into memory on first use.

    Serialization: the Coqui ``TTS`` object isn't thread-safe at the
    inference layer. A module-level lock guards :meth:`synthesize`
    so concurrent callers queue up rather than crashing the model.
    """

    name = "xtts"

    def __init__(
        self,
        *,
        reference_wav: str,
        language: str = "en",
        device: str | None = None,
        target_sr: int = AUDIO_SR,
    ) -> None:
        if not reference_wav:
            raise ValueError(
                "reference_wav is required — XTTS-v2 clones the voice "
                "from a 6-30 s reference sample"
            )
        self.reference_wav = reference_wav
        self.language = language
        self.device = device or _auto_device()
        self.target_sr = target_sr
        self._tts: Any | None = None
        self._model_sr: int | None = None
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────

    def _ensure_loaded(self) -> Any:
        if self._tts is not None:
            return self._tts
        if not os.path.exists(self.reference_wav):
            raise FileNotFoundError(
                f"reference wav not found: {self.reference_wav} — "
                "drop a 6-30 s clean speech clip here to seed the voice clone."
            )
        try:
            from TTS.api import TTS  # type: ignore
        except ImportError as e:  # pragma: no cover — dep optional on dev
            raise RuntimeError(
                "Coqui TTS not installed. Try:\n"
                "  pip install coqui-tts\n"
                "or flip ARTEMIS_TTS_BACKEND=piper as a fallback."
            ) from e
        log.info("loading XTTS-v2 on %s (this takes a few seconds)…", self.device)
        self._tts = TTS(model_name=MODEL_ID, progress_bar=False).to(self.device)
        # The synthesizer reports its output sample rate so we can
        # resample to AUDIO_SR — for XTTS-v2 it is 24000.
        self._model_sr = int(
            getattr(self._tts.synthesizer, "output_sample_rate", 24000)
        )
        log.info("XTTS-v2 ready (model_sr=%d)", self._model_sr)
        return self._tts

    def close(self) -> None:
        self._tts = None  # PyTorch will GC the model graph

    # ── synth ───────────────────────────────────────────────────

    def synthesize(self, text: str) -> TTSSample:
        text = (text or "").strip()
        if not text:
            return TTSSample(
                pcm=np.zeros(0, dtype=np.int16),
                sample_rate=self.target_sr,
                source_model=f"xtts:{os.path.basename(self.reference_wav)}",
            )
        tts = self._ensure_loaded()
        src_sr = self._model_sr or 24000

        with self._lock:
            # tts.tts() returns a list[float] (or np.ndarray) at model SR.
            wave = tts.tts(
                text=text,
                speaker_wav=self.reference_wav,
                language=self.language,
            )
        pcm16 = float_to_int16(np.asarray(wave, dtype=np.float32))
        pcm16 = resample_int16_to(pcm16, src_sr, self.target_sr)
        return TTSSample(
            pcm=pcm16,
            sample_rate=self.target_sr,
            source_model=f"xtts:{os.path.basename(self.reference_wav)}",
        )


def _auto_device() -> str:
    """Pick a sensible default device.

    On Atlas with an available CUDA GPU we use it; otherwise CPU.
    Returning a string keeps this unit-test-friendly — we don't
    import torch here unless the backend is actually loaded.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
