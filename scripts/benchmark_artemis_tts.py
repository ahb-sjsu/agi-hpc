"""A/B benchmark the configured TTS backends on ARTEMIS-style lines.

Usage:
    python scripts/benchmark_artemis_tts.py [--out out_dir]

Runs each available backend against a fixed set of utterances and
writes the audio side-by-side so you can decide which voice clears
the "really good" bar before flipping ARTEMIS_TTS_BACKEND.

Does NOT require a running NATS broker — the loader picks local
backends when no heartbeat is seen. For burst benchmarking, spin up
the worker first and re-run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

# Ensure src/ is importable when invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agi.primer.artemis.livekit_agent.tts.loader import (  # noqa: E402
    build_backend_from_env,
)
from agi.primer.artemis.livekit_agent.tts.piper import PiperBackend  # noqa: E402
from agi.primer.artemis.livekit_agent.tts.xtts import XttsBackend  # noqa: E402

LINES = [
    "Acknowledged. Telemetry shows the Halyard is still on nominal trajectory.",
    "I would caution against approaching the Mi-go installation unarmed.",
    "Reading the crew's biometrics — Arlo's heart rate is elevated but stable.",
    "The vault's atmosphere is breathable, but the radiation count is rising.",
    "Stand by. I am cross-referencing the glyph pattern against the SCAT archive.",
]


def _write_wav(path: Path, pcm, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def bench(backend, name: str, out_dir: Path) -> None:
    print(f"\n── backend: {name} ──")
    for i, line in enumerate(LINES):
        t0 = time.monotonic()
        sample = backend.synthesize(line)
        dt = time.monotonic() - t0
        if sample.duration_s == 0:
            print(f"  [{i}] empty output ({line[:40]}…)")
            continue
        rtx = sample.duration_s / max(dt, 1e-3)
        out = out_dir / f"{name}_{i:02d}.wav"
        _write_wav(out, sample.pcm, sample.sample_rate)
        print(
            f"  [{i}] {dt:.2f}s synth · {sample.duration_s:.2f}s audio · "
            f"rtx={rtx:.2f} · {out}"
        )
    backend.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(Path.home() / "artemis-voice-refs" / "bench"),
        help="output directory for WAV files",
    )
    ap.add_argument(
        "--piper-voice",
        default=os.environ.get(
            "ARTEMIS_VOICE",
            "/home/claude/piper-voices/en_US-amy-medium.onnx",
        ),
    )
    ap.add_argument(
        "--xtts-ref",
        default=os.environ.get(
            "ARTEMIS_VOICE_REF",
            "/home/claude/artemis-voice-refs/artemis_ref.wav",
        ),
    )
    args = ap.parse_args()
    out_dir = Path(args.out)

    # Env-configured backend first (whatever the user would hit in prod).
    prod = build_backend_from_env()
    bench(prod, f"env_{prod.name}", out_dir)

    # Then each backend individually so the user can compare timbre.
    if os.path.exists(args.piper_voice):
        bench(PiperBackend(args.piper_voice), "piper", out_dir)
    else:
        print(f"skipping piper — voice file not found: {args.piper_voice}")

    if os.path.exists(args.xtts_ref):
        try:
            bench(
                XttsBackend(reference_wav=args.xtts_ref),
                "xtts",
                out_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"xtts skipped: {e}")
    else:
        print(f"skipping xtts — ref wav not found: {args.xtts_ref}")

    print(f"\nbench WAVs written under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
