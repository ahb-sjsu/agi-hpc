"""Synced smoke for the 3D avatar.

1. Synthesizes a line with the running XTTS NATS-burst worker.
2. Computes per-frame RMS amplitude from the PCM.
3. Renders VRM frames at 30 fps, driving setMouthOpen from that
   envelope so the mouth is in sync with what ARTEMIS is actually
   saying.
4. Muxes audio + frames into a timestamped MP4.

Run on Atlas (inside /home/claude/venvs/artemis-avatar venv) —
requires playwright, pillow, numpy, nats-py, ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import subprocess
import sys
import time
import uuid
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agi.primer.artemis.livekit_agent.avatar3d.renderer import (  # noqa: E402
    AvatarRenderer,
)
from agi.primer.artemis.livekit_agent.avatar3d.scene import SceneConfig  # noqa: E402

DEFAULT_TEXT = (
    "ARTEMIS online. This is a synchronized demonstration. "
    "You should see my mouth move in time with the audio, "
    "while my body cycles through a small library of poses. "
    "Telemetry is nominal. End of test."
)


async def synthesize_via_worker(
    text: str, timeout_s: float = 60.0
) -> tuple[np.ndarray, int]:
    """Send text to the running TTS worker; return (int16 PCM, sample_rate)."""
    import nats

    nc = await nats.connect(servers=["nats://localhost:4222"])
    job_id = uuid.uuid4().hex[:12]
    reply_to = f"agi.rh.artemis.tts.results.{job_id}"
    req = {
        "job_id": job_id,
        "text": text,
        "language": "en",
        "reply_to": reply_to,
    }
    sub = await nc.subscribe(reply_to)
    t0 = time.monotonic()
    await nc.publish("agi.rh.artemis.tts.jobs", json.dumps(req).encode())
    chunks: dict[int, bytes] = {}
    expected: int | None = None
    sr: int = 24000
    while expected is None or len(chunks) < expected:
        msg = await sub.next_msg(timeout=timeout_s)
        headers = dict(getattr(msg, "headers", None) or {})
        if headers.get("error"):
            await nc.drain()
            raise RuntimeError(f"worker error: {headers['error']}")
        sr = int(headers.get("sample_rate", sr))
        idx = int(headers.get("chunk_idx", str(len(chunks) + 1)))
        expected = int(headers.get("chunk_n", "1"))
        chunks[idx] = msg.data or b""
    dt = time.monotonic() - t0
    await nc.drain()
    assembled = b"".join(chunks[i] for i in sorted(chunks))
    pcm = np.frombuffer(assembled, dtype=np.int16)
    print(
        f"synth: {len(pcm) / sr:.2f}s audio in {dt:.2f}s "
        f"({len(chunks)} chunks @ {sr} Hz)"
    )
    return pcm, sr


def compute_mouth_envelope(pcm: np.ndarray, sr: int, fps: int) -> np.ndarray:
    """Per-frame mouth-open level 0..1, derived from audio RMS."""
    samples_per_frame = max(1, sr // fps)
    n_frames = max(1, len(pcm) // samples_per_frame)
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * samples_per_frame
        chunk = pcm[start : start + samples_per_frame].astype(np.float32)
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean((chunk / 32768.0) ** 2)))
        envelope[i] = rms
    # Normalize to [0..1] with a soft ceiling at 90th pctile so one
    # loud sibilant doesn't pin the mouth open everywhere else.
    ceil = float(np.percentile(envelope, 90)) or 1e-6
    envelope = np.clip(envelope / ceil, 0.0, 1.0)
    # Mouth open feels laggy if we feed raw RMS — gentle attack so
    # it tracks voice but doesn't twitch on noise floor.
    gate = 0.08
    envelope = np.where(envelope < gate, 0.0, (envelope - gate) / (1 - gate))
    return envelope


def write_wav(pcm: np.ndarray, sr: int, path: Path) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(arr, mode="RGBA").save(str(path), format="PNG")


def render_synced(
    envelope: np.ndarray, out_dir: Path, *, fps: int, width: int, height: int
) -> list[Path]:
    cfg = SceneConfig(fps=fps, width=width, height=height)
    paths: list[Path] = []
    with AvatarRenderer(config=cfg) as r:
        n = int(len(envelope))
        for i in range(n):
            r.set_mouth_open(float(envelope[i]))
            frame = r.capture()
            p = out_dir / f"frame_{i:04d}.png"
            save_png(frame, p)
            paths.append(p)
            if i % 30 == 0:
                print(f"  rendered {i}/{n}")
    return paths


def mux_mp4(frames_dir: Path, wav: Path, mp4: Path, *, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-i",
        str(wav),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-vf",
        "format=yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", default="/tmp/avatar3d-synced")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. synth
    pcm, sr = asyncio.run(synthesize_via_worker(args.text))
    wav = out / "audio.wav"
    write_wav(pcm, sr, wav)

    # 2. envelope
    envelope = compute_mouth_envelope(pcm, sr, args.fps)
    print(f"envelope: {len(envelope)} frames")

    # 3. render
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    # Clean stale frames from previous runs so ffmpeg glob is clean.
    for p in frames_dir.glob("frame_*.png"):
        p.unlink()
    render_synced(
        envelope,
        frames_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    # 4. mux — timestamped filename
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out / f"artemis_synced_{ts}.mp4"
    mux_mp4(frames_dir, wav, mp4, fps=args.fps)
    print(f"\nwrote: {mp4}  ({mp4.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
