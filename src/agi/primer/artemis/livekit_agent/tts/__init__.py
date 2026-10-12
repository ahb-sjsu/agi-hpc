# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Pluggable text-to-speech backends for the ARTEMIS avatar.

S1c — adds Coqui XTTS-v2 as the high-quality default with Piper kept
as a zero-GPU fallback. Both backends are wrapped behind the same
synchronous ``synthesize(text) -> np.int16`` interface so the audio
pipeline in ``avatar_hud.py`` doesn't need to care which one is
running.

Backend selection is driven by ``ARTEMIS_TTS_BACKEND``:
  - ``xtts``   (default) — Coqui XTTS-v2. Cloned reference voice.
  - ``piper``  — Piper ONNX. Fast, CPU-only, lower fidelity.

Outputs are always 16-bit PCM mono at ``AUDIO_SR`` (22050 Hz); the
backend handles resampling if the model's native SR differs.
"""

from __future__ import annotations

from .backend import TTSBackend, TTSSample
from .loader import build_backend_from_env

__all__ = ["TTSBackend", "TTSSample", "build_backend_from_env"]
