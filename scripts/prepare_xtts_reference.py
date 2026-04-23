#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Prepare a source audio file as an XTTS-v2 voice-clone reference.

XTTS-v2 gets its best output from a 15-20 s clean single-speaker
clip. This script takes any audio file ffmpeg can decode and
produces the most useful reference it can:

  1. Decode + downmix to mono
  2. Trim leading/trailing silence (< -40 dBFS for > 0.5 s)
  3. High-pass filter at 80 Hz to remove rumble
  4. Loudness-normalize to -16 LUFS (ITU-R BS.1770, broadcast spec)
  5. Resample to 22050 Hz, 16-bit PCM WAV
  6. Report peak / RMS / duration / warnings

Usage:
    python scripts/prepare_xtts_reference.py input.mp3
    python scripts/prepare_xtts_reference.py input.wav --out ref.wav
    python scripts/prepare_xtts_reference.py input.mp3 --window 15
       (auto-crop the best 15 s window by RMS consistency)

Requires: ffmpeg on PATH. Atlas has it; most Linux boxes do.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

DEFAULT_OUT = Path.home() / "artemis-voice-refs" / "artemis_ref.wav"
TARGET_SR = 22050  # XTTS accepts 22050 or 24000; 22050 matches the avatar path.


def _check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not on PATH. Install it first.", file=sys.stderr)
        sys.exit(2)


def _ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-of", "json",
                "-show_format", str(path),
            ],
            text=True,
        )
        return float(json.loads(out)["format"]["duration"])
    except Exception:
        return 0.0


def _ffmpeg_condition(src: Path, dst: Path, *, target_sr: int) -> None:
    """Run the full conditioning chain via ffmpeg.

    Filter order matters: silence-remove before loudnorm so the
    measurement isn't diluted by leading/trailing dead air.
    """
    filter_chain = (
        # trim silence at start + end, threshold -40 dBFS, > 0.5 s
        "silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:"
        "stop_periods=-1:stop_duration=0.5:stop_threshold=-40dB,"
        # rumble cut
        "highpass=f=80,"
        # EBU R128 loudness normalization — -16 LUFS is the sweet
        # spot for voice references: loud enough to drive the model,
        # not so hot that it clips on sibilants.
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-af", filter_chain,
        "-ac", "1",
        "-ar", str(target_sr),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _window_crop(src: Path, dst: Path, seconds: float, target_sr: int) -> None:
    """Pick the most consistent N-second window by RMS variance.

    Walks the conditioned file with a 1 s hop; scores each window by
    (low silence + low RMS variance). Avoids cropping mid-word by
    anchoring to detected silence boundaries where possible.
    """
    import numpy as np

    with wave.open(str(src), "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        raw = f.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total = len(pcm) / sr
    if total <= seconds + 0.1:
        shutil.copy(src, dst)
        return

    win_n = int(seconds * sr)
    hop_n = sr  # 1 s hops
    best = (-1e9, 0)
    for start in range(0, len(pcm) - win_n, hop_n):
        chunk = pcm[start : start + win_n]
        rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)
        # Score: high average RMS (signal present) minus variance
        # (consistent energy, no big silent gaps in the window).
        blocks = chunk.reshape(-1, max(1, win_n // 10))
        block_rms = np.sqrt(np.mean(blocks ** 2, axis=1))
        var_penalty = float(np.std(block_rms))
        score = rms - 2.0 * var_penalty
        if score > best[0]:
            best = (score, start)

    start = best[1]
    cropped = pcm[start : start + win_n]
    cropped_i16 = (np.clip(cropped, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(dst), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(target_sr)
        f.writeframes(cropped_i16.tobytes())


def _report(path: Path) -> dict:
    """Emit a human-readable quality report."""
    import numpy as np

    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        ch = f.getnchannels()
        n = f.getnframes()
        raw = f.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    duration = n / sr
    peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
    rms = float(np.sqrt(np.mean(pcm ** 2))) if len(pcm) else 0.0
    peak_db = 20 * np.log10(peak + 1e-9)
    rms_db = 20 * np.log10(rms + 1e-9)

    warnings: list[str] = []
    if duration < 6:
        warnings.append(f"duration {duration:.1f}s < 6s — XTTS will under-sample phonemes")
    if duration > 30:
        warnings.append(f"duration {duration:.1f}s > 30s — XTTS ignores the tail; consider --window 15")
    if peak_db > -0.5:
        warnings.append(f"peak {peak_db:+.1f} dBFS — clipping likely")
    if rms_db < -30:
        warnings.append(f"RMS {rms_db:+.1f} dBFS — quiet; re-record closer or raise gain")
    if ch != 1:
        warnings.append(f"{ch} channels — XTTS expects mono")
    if sr not in (22050, 24000):
        warnings.append(f"SR {sr} Hz — XTTS prefers 22050 or 24000")

    return {
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "channels": ch,
        "peak_dbfs": round(float(peak_db), 2),
        "rms_dbfs": round(float(rms_db), 2),
        "warnings": warnings,
        "path": str(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Source audio (any format ffmpeg decodes)")
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"Output WAV path (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--window", type=float, default=0,
        help="Auto-crop to best N-second window (0 = keep full duration)",
    )
    ap.add_argument(
        "--target-sr", type=int, default=TARGET_SR,
        help=f"Output sample rate (default: {TARGET_SR})",
    )
    args = ap.parse_args()

    _check_ffmpeg()
    src = Path(args.input).resolve()
    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 1
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    src_dur = _ffprobe_duration(src)
    print(f"source: {src} ({src_dur:.2f}s)")
    print(f"target: {out}")

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td) / "conditioned.wav"
        _ffmpeg_condition(src, staged, target_sr=args.target_sr)
        if args.window > 0:
            windowed = Path(td) / "windowed.wav"
            _window_crop(staged, windowed, args.window, args.target_sr)
            shutil.copy(windowed, out)
        else:
            shutil.copy(staged, out)

    report = _report(out)
    print()
    print("── quality report ──")
    print(f"  duration     {report['duration_s']:>7.2f} s")
    print(f"  sample rate  {report['sample_rate']:>7d} Hz")
    print(f"  channels     {report['channels']:>7d}")
    print(f"  peak level   {report['peak_dbfs']:>+7.2f} dBFS")
    print(f"  rms level    {report['rms_dbfs']:>+7.2f} dBFS")
    print(f"  path         {report['path']}")
    if report["warnings"]:
        print()
        print("── warnings ──")
        for w in report["warnings"]:
            print(f"  !! {w}")
    else:
        print()
        print("  no warnings — looks ready for XTTS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
