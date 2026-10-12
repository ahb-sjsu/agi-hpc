# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the pluggable TTS backends.

Neither the XTTS model nor Piper ONNX is loaded here — both adapters
are covered via the module-level functions they depend on, or via
stand-in fakes. The goal is to verify the resampling, the backend
contract, the nats_burst bridge wiring, and the env-driven loader.
"""

from __future__ import annotations

import numpy as np
import pytest

from agi.primer.artemis.livekit_agent.tts.backend import (
    AUDIO_SR,
    TTSSample,
    float_to_int16,
    resample_int16_to,
)
from agi.primer.artemis.livekit_agent.tts.loader import build_backend_from_env

# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────


def test_float_to_int16_clips_at_ends() -> None:
    w = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    out = float_to_int16(w)
    assert out[0] == -32767 and out[-1] == 32767
    assert out[2] == 0


def test_resample_int16_identity_when_rates_match() -> None:
    pcm = np.array([1, 2, 3], dtype=np.int16)
    out = resample_int16_to(pcm, 22050, 22050)
    assert out is pcm or np.array_equal(out, pcm)


def test_resample_int16_halves_length_when_rate_halved() -> None:
    pcm = np.arange(1000, dtype=np.int16)
    out = resample_int16_to(pcm, 44100, 22050)
    # Allow 1-sample rounding slack.
    assert abs(len(out) - 500) <= 1


def test_resample_int16_zero_returns_empty() -> None:
    assert len(resample_int16_to(np.zeros(0, dtype=np.int16), 24000, 22050)) == 0


def test_tts_sample_duration() -> None:
    s = TTSSample(pcm=np.zeros(22050, dtype=np.int16), sample_rate=22050)
    assert abs(s.duration_s - 1.0) < 1e-3
    s2 = TTSSample(pcm=np.zeros(0, dtype=np.int16), sample_rate=22050)
    assert s2.duration_s == 0.0


# ─────────────────────────────────────────────────────────────────
# Piper backend (voice-file-free test)
# ─────────────────────────────────────────────────────────────────


class _FakePiperChunk:
    def __init__(self, arr: np.ndarray) -> None:
        self.audio_int16_bytes = arr.tobytes()


class _FakePiperVoice:
    class config:
        sample_rate = 22050

    def synthesize(self, text: str):
        for _ in range(2):
            yield _FakePiperChunk(np.ones(100, dtype=np.int16) * 1000)


def test_piper_backend_concatenates_chunks_and_resamples(monkeypatch) -> None:
    from agi.primer.artemis.livekit_agent.tts import piper as piper_mod

    backend = piper_mod.PiperBackend("/fake/voice.onnx")

    # Inject the fake voice so the real Piper loader isn't touched.
    class _Module:
        @staticmethod
        def load(path):
            return _FakePiperVoice()

    backend._voice = _FakePiperVoice()  # skip ensure_loaded

    sample = backend.synthesize("hello world")
    assert sample.sample_rate == AUDIO_SR
    assert sample.pcm.dtype == np.int16
    # 2 chunks × 100 samples @ 22050 → upsampled to 24000 kHz (AUDIO_SR)
    # for Opus's native rate. 200 samples × 24000/22050 ≈ 218.
    expected = int(round(200 * AUDIO_SR / 22050))
    assert abs(len(sample.pcm) - expected) <= 1


def test_piper_backend_empty_text_returns_empty() -> None:
    from agi.primer.artemis.livekit_agent.tts import piper as piper_mod

    backend = piper_mod.PiperBackend("/fake/voice.onnx")
    backend._voice = _FakePiperVoice()
    sample = backend.synthesize("   ")
    assert sample.duration_s == 0.0
    assert sample.pcm.dtype == np.int16


def test_piper_chunk_decoder_accepts_raw_bytes() -> None:
    from agi.primer.artemis.livekit_agent.tts.piper import PiperBackend

    arr = np.array([100, 200, -100], dtype=np.int16)
    out = PiperBackend._chunk_to_pcm(arr.tobytes())
    assert np.array_equal(out, arr)


def test_piper_chunk_decoder_accepts_float_array() -> None:
    from agi.primer.artemis.livekit_agent.tts.piper import PiperBackend

    class _Obj:
        audio_float_array = np.array([0.5, -0.25], dtype=np.float32)

    out = PiperBackend._chunk_to_pcm(_Obj())
    assert out.dtype == np.int16
    assert abs(out[0] - 16383) <= 1
    assert abs(out[1] + 8191) <= 1


# ─────────────────────────────────────────────────────────────────
# NATS burst backend (wire protocol + fallback)
# ─────────────────────────────────────────────────────────────────


class _FakeMsg:
    def __init__(self, data: bytes, headers: dict[str, str]) -> None:
        self.data = data
        self.headers = headers


def test_msg_to_sample_decodes_int16_at_claimed_sr() -> None:
    from agi.primer.artemis.livekit_agent.tts.nats_burst import _msg_to_sample

    pcm = np.arange(200, dtype=np.int16)
    m = _FakeMsg(
        data=pcm.tobytes(),
        headers={"sample_rate": "24000", "source_model": "xtts:artemis"},
    )
    s = _msg_to_sample(m, target_sr=22050)
    assert s.sample_rate == 22050
    assert s.source_model == "xtts:artemis"
    # 200 samples @ 24kHz → ~183 samples @ 22.05kHz.
    assert 180 <= len(s.pcm) <= 186


def test_msg_to_sample_raises_on_error_header() -> None:
    from agi.primer.artemis.livekit_agent.tts.nats_burst import _msg_to_sample

    m = _FakeMsg(data=b"", headers={"error": "oom", "sample_rate": "24000"})
    with pytest.raises(RuntimeError):
        _msg_to_sample(m, target_sr=22050)


class _FakeFallback:
    name = "fake_piper"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str) -> TTSSample:
        self.calls.append(text)
        return TTSSample(
            pcm=np.ones(1024, dtype=np.int16) * 500,
            sample_rate=22050,
            source_model="fake_piper",
        )

    def close(self) -> None:
        pass


def test_burst_falls_back_when_not_connected() -> None:
    from agi.primer.artemis.livekit_agent.tts.nats_burst import NatsBurstBackend

    # Subclass to skip the real NATS bg-loop startup.
    class _NoStart(NatsBurstBackend):
        def _start_bg_loop(self) -> None:
            self._loop = None
            self._nc = None

    fallback = _FakeFallback()
    b = _NoStart(
        nats_url="nats://unused",
        voice_ref="/tmp/ref.wav",
        fallback=fallback,
    )
    out = b.synthesize("hello")
    assert fallback.calls == ["hello"]
    assert len(out.pcm) > 0


def test_burst_empty_text_short_circuit() -> None:
    from agi.primer.artemis.livekit_agent.tts.nats_burst import NatsBurstBackend

    class _NoStart(NatsBurstBackend):
        def _start_bg_loop(self) -> None:
            self._loop = None
            self._nc = None

    b = _NoStart(
        nats_url="nats://unused",
        voice_ref="/tmp/ref.wav",
        fallback=_FakeFallback(),
    )
    s = b.synthesize("   ")
    assert s.duration_s == 0.0


# ─────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────


def test_loader_explicit_piper(monkeypatch) -> None:
    monkeypatch.setenv("ARTEMIS_TTS_BACKEND", "piper")
    monkeypatch.setenv("ARTEMIS_VOICE", "/fake/voice.onnx")
    b = build_backend_from_env()
    assert b.name == "piper"


def test_loader_auto_falls_back_to_piper_when_no_xtts_ref(monkeypatch) -> None:
    monkeypatch.setenv("ARTEMIS_TTS_BACKEND", "auto")
    # Point XTTS ref at a non-existent path so the auto picker rejects
    # it, and make sure the burst probe returns False deterministically.
    monkeypatch.setenv("ARTEMIS_VOICE_REF", "/does/not/exist.wav")
    import agi.primer.artemis.livekit_agent.tts.loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "_burst_worker_available",
        lambda _url: False,
    )
    b = build_backend_from_env()
    assert b.name == "piper"


def test_loader_nats_burst_explicit_carries_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ARTEMIS_TTS_BACKEND", "nats_burst")
    monkeypatch.setenv("ARTEMIS_VOICE_REF", "/does/not/exist.wav")

    import agi.primer.artemis.livekit_agent.tts.nats_burst as burst_mod

    # Stub out the bg-loop so construction doesn't actually open a
    # socket to a NATS server we don't have in CI.
    def _skip(self) -> None:
        self._loop = None
        self._nc = None

    monkeypatch.setattr(burst_mod.NatsBurstBackend, "_start_bg_loop", _skip)
    b = build_backend_from_env()
    assert b.name == "nats_burst"
    # Fallback is Piper (since the XTTS ref doesn't exist).
    assert b.fallback is not None
    assert b.fallback.name == "piper"
