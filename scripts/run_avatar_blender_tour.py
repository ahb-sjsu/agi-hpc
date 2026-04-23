"""Drive the Blender capability-tour renderer end-to-end.

1. Synthesize a narration line via the running XTTS NATS-burst worker.
2. Compute per-frame mouth amplitude envelope from the PCM.
3. Emit a JSON config + invoke ``blender -b --python blender_tour.py``.
4. Mux the rendered PNG sequence with the audio via ffmpeg + NVENC
   into a timestamped MP4.

No Playwright or three.js in the loop. Blender uses Eevee on the
Quadro GV100 for rendering, ffmpeg uses ``h264_nvenc`` for encode —
both GPU paths.
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

DEFAULT_TEXT = (
    "ARTEMIS capability tour. I will demonstrate my full range of "
    "facial expressions, mouth shapes, gaze directions, and body poses. "
    "First, expressions. This is happy. This is sad. This is angry. "
    "This is surprised. This is relaxed. Now, mouth shapes. Ah. Ee. Ih. "
    "Oh. Ou. Now, poses. Mountain. Reach. Warrior. Prayer. Ping. "
    "End of tour."
)


CAPABILITY_TIMELINE: list[dict] = [
    {"t": 0.0, "kind": "pose", "arg": "mountain"},
    {"t": 0.0, "kind": "expression", "arg": "neutral"},
    # Expressions block — each emotion peaks for ~0.3s then immediately
    # returns to neutral. Prevents ALL_Surprised etc. from slow-ramping
    # the mouth open over a full second.
    {"t": 11.0, "kind": "expression", "arg": "happy"},
    {"t": 11.3, "kind": "expression", "arg": "neutral"},
    {"t": 12.5, "kind": "expression", "arg": "sad"},
    {"t": 12.8, "kind": "expression", "arg": "neutral"},
    {"t": 14.0, "kind": "expression", "arg": "angry"},
    {"t": 14.3, "kind": "expression", "arg": "neutral"},
    {"t": 15.5, "kind": "expression", "arg": "surprised"},
    {"t": 15.8, "kind": "expression", "arg": "neutral"},
    {"t": 17.0, "kind": "expression", "arg": "relaxed"},
    {"t": 17.3, "kind": "expression", "arg": "neutral"},
    # Viseme block (explicit pulses — the audio-driven aa still runs
    # on top, these just demonstrate the other mouth shapes exist)
    {"t": 20.5, "kind": "viseme", "arg": ["ee", 1.0]},
    {"t": 21.2, "kind": "viseme", "arg": ["ee", 0.0]},
    {"t": 21.2, "kind": "viseme", "arg": ["ih", 1.0]},
    {"t": 21.9, "kind": "viseme", "arg": ["ih", 0.0]},
    {"t": 21.9, "kind": "viseme", "arg": ["oh", 1.0]},
    {"t": 22.6, "kind": "viseme", "arg": ["oh", 0.0]},
    {"t": 22.6, "kind": "viseme", "arg": ["ou", 1.0]},
    {"t": 23.3, "kind": "viseme", "arg": ["ou", 0.0]},
    # Pose block — each yoga pose is struck briefly (~1.5s), then she
    # returns to mountain between them. Avoids the "held static" feel.
    {"t": 25.0, "kind": "pose", "arg": "mountain"},
    {"t": 27.0, "kind": "pose", "arg": "reach"},
    {"t": 29.0, "kind": "pose", "arg": "mountain"},
    {"t": 30.5, "kind": "pose", "arg": "warrior"},
    {"t": 32.5, "kind": "pose", "arg": "mountain"},
    {"t": 34.0, "kind": "pose", "arg": "prayer"},
    {"t": 36.0, "kind": "pose", "arg": "mountain"},
    {"t": 37.5, "kind": "pose", "arg": "ping"},
    {"t": 39.5, "kind": "pose", "arg": "mountain"},
]


async def synthesize_via_worker(
    text: str, timeout_s: float = 180.0
) -> tuple[np.ndarray, int]:
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
    samples_per_frame = max(1, sr // fps)
    n_frames = max(1, len(pcm) // samples_per_frame)
    env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = pcm[i * samples_per_frame : (i + 1) * samples_per_frame]
        if chunk.size:
            env[i] = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768.0) ** 2)))
    ceil = float(np.percentile(env, 90)) or 1e-6
    env = np.clip(env / ceil, 0.0, 1.0)
    # Higher gate kills background-level RMS that otherwise holds the
    # mouth half-open during quiet passages.
    gate = 0.15
    env = np.where(env < gate, 0.0, (env - gate) / (1 - gate))
    # Scale to a natural mouth-open level. VRoid MTH_A at 1.0 is a
    # full "ah" gape; even 0.25 read as progressively-yawning, so
    # we hold to 0.15 (subtle conversational lip motion).
    NATURAL_MAX = 0.15
    env = env * NATURAL_MAX
    # One-pole low-pass to kill per-frame jitter.
    alpha = 0.22
    smoothed = np.zeros_like(env)
    for i, x in enumerate(env):
        smoothed[i] = alpha * x + (1 - alpha) * (smoothed[i - 1] if i else x)
    return smoothed


def write_wav(pcm: np.ndarray, sr: int, path: Path) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def run_blender(config_path: Path, blender_script: Path) -> int:
    cmd = [
        "blender",
        "-b",
        "--python",
        str(blender_script),
        "--",
        "--config",
        str(config_path),
    ]
    t0 = time.monotonic()
    r = subprocess.run(cmd)
    print(f"blender exit={r.returncode} in {time.monotonic() - t0:.1f}s")
    return r.returncode


def mux_nvenc(frames_pattern: str, wav: Path, mp4: Path, *, fps: int) -> None:
    """ffmpeg with NVIDIA hardware encoder."""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        str(fps),
        "-i",
        frames_pattern,
        "-i",
        str(wav),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "23",
        "-pix_fmt",
        "yuv420p",
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
    ap.add_argument("--vrm", default="/tmp/artemis_ref.vrm")
    ap.add_argument("--out", default="/tmp/avatar3d-blender")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument(
        "--blender-script",
        default=str(Path(__file__).with_name("blender_tour.py")),
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    # Wipe stale frames so ffmpeg's -i pattern matches a clean set.
    for p in frames_dir.glob("frame_*.png"):
        p.unlink()

    pcm, sr = asyncio.run(synthesize_via_worker(args.text))
    wav = out / "audio.wav"
    write_wav(pcm, sr, wav)

    envelope = compute_mouth_envelope(pcm, sr, args.fps)
    print(f"envelope: {len(envelope)} frames ({len(envelope)/args.fps:.1f}s)")

    cfg = {
        "vrm_path": args.vrm,
        "frames_pattern": str(frames_dir / "frame_"),
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "envelope": [float(v) for v in envelope],
        "timeline": CAPABILITY_TIMELINE,
        "blink_seed": 42,
    }
    cfg_path = out / "tour.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    rc = run_blender(cfg_path, Path(args.blender_script))
    if rc != 0:
        print(f"blender failed rc={rc}", file=sys.stderr)
        return rc

    # Frames land as frame_0001.png … frame_NNNN.png
    pattern = str(frames_dir / "frame_%04d.png")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out / f"artemis_blender_{ts}.mp4"
    mux_nvenc(pattern, wav, mp4, fps=args.fps)
    print(f"\nwrote: {mp4}  ({mp4.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
