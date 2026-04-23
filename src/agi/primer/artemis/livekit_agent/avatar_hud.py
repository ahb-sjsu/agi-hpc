#!/usr/bin/env python3
"""ARTEMIS avatar v4 — sci-fi HUD instead of cartoon face.

Replaces the 2D cartoon face with a projection-rendered rotating
wireframe icosahedron at center, radial audio-reactive spectrum bars,
a scrolling event log, and a low-amplitude waveform — amber-on-black
terminal aesthetic (Expanse / Blade Runner UI).

This matches the in-fiction concept of "the handheld's screen" far
better than a drawn face, and scales up in visual quality without
requiring 3D assets.

Still 2D-rasterized → video track. True 3D would need OpenGL
(moderngl / Panda3D) — separate sprint if we want polygonal lighting.

TTS (Piper) + phoneme-derived speech intensity are carried over. No
face = no lip-sync, but the visualization pulses with audio so it
reads as "speaking" clearly.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import math
import os
import queue
import random
import signal
import sys
import threading
import time

import numpy as np
from livekit import api, rtc
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("artemis.avatar")

# Video geometry — 960×720 gives the codec enough pixels per element
# to keep elements readable after compression. Dropping from 20→30 fps
# makes motion smoother and lets the codec spend bits on the right
# frames instead of duplicates.
W, H = 960, 720
# Dropped from 30 → 15 fps while diagnosing audio choppiness. Every
# video encode steals time from the main asyncio loop; 15 fps is
# still comfortably smooth for a HUD that doesn't have fast motion.
FPS = 15

# 24 kHz matches XTTS-v2's native output and is an Opus native rate —
# avoids one resample step and keeps Opus in its sweet spot. Piper
# (22050 Hz) is upsampled to 24 kHz in its adapter before reaching the
# audio queue, so the publisher sees a single rate regardless of
# backend.
AUDIO_SR = 24000
AUDIO_CHANNELS = 1
# 40 ms frames (vs 20 ms): halves the per-frame overhead — one
# run_coroutine_threadsafe hop per frame, one async scheduling
# boundary. Opus handles 20, 40, and 60 ms frames natively; 40 is
# the sweet spot for quality vs latency in voice contexts.
AUDIO_FRAME_MS = 40
AUDIO_SAMPLES_PER_FRAME = AUDIO_SR * AUDIO_FRAME_MS // 1000

PIPER_VOICE = os.environ.get(
    "ARTEMIS_VOICE", "/home/claude/piper-voices/en_US-amy-medium.onnx"
)

CANNED_REPLIES = [
    "Acknowledged.",
    "Noted.",
    "Scanning.",
    "The readings are nominal.",
    "Processing.",
    "Stand by.",
    "One moment.",
    "Logged.",
    "The handheld is listening.",
]

# ─────────────────────────────────────────────────────────────────
# Palette — amber-on-black terminal
# ─────────────────────────────────────────────────────────────────

BG = (6, 6, 10)
AMBER = (255, 176, 48)
AMBER_DIM = (120, 72, 10)
AMBER_DEEP = (60, 36, 4)
CYAN = (80, 220, 240)
CYAN_DIM = (30, 110, 130)
WHITE_DIM = (180, 180, 200)
GRID = (24, 24, 34)


# ─────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────


class AvatarState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_speakers: list[str] = []
        self.participants: list[str] = []
        self.speaking: bool = False
        self.amp: float = 0.0  # 0..1 — ARTEMIS's own audio amplitude
        # Ring buffer of events for the scrolling log.
        self.events: collections.deque = collections.deque(maxlen=6)
        # Spectrum energy per band (16 bands, 0..1 each), driven by FFT.
        self.spectrum: list[float] = [0.0] * 16

    def set_active(self, xs: list[str]) -> None:
        with self._lock:
            self.active_speakers = list(xs)

    def set_participants(self, xs: list[str]) -> None:
        with self._lock:
            self.participants = list(xs)

    def set_speaking(self, speaking: bool, amp: float) -> None:
        with self._lock:
            self.speaking = speaking
            self.amp = max(0.0, min(1.0, amp))

    def set_spectrum(self, bands: list[float]) -> None:
        with self._lock:
            self.spectrum = list(bands)

    def log_event(self, line: str) -> None:
        with self._lock:
            ts = time.strftime("%H:%M:%S", time.localtime())
            self.events.append(f"{ts}  {line}")

    def snap(self):
        with self._lock:
            return (
                list(self.active_speakers),
                list(self.participants),
                self.speaking,
                self.amp,
                list(self.spectrum),
                list(self.events),
            )


STATE = AvatarState()


# ─────────────────────────────────────────────────────────────────
# Icosahedron geometry
# ─────────────────────────────────────────────────────────────────

_phi = (1 + math.sqrt(5)) / 2
VERTS_3D = [
    (-1, _phi, 0),
    (1, _phi, 0),
    (-1, -_phi, 0),
    (1, -_phi, 0),
    (0, -1, _phi),
    (0, 1, _phi),
    (0, -1, -_phi),
    (0, 1, -_phi),
    (_phi, 0, -1),
    (_phi, 0, 1),
    (-_phi, 0, -1),
    (-_phi, 0, 1),
]
EDGES = [
    (0, 1),
    (0, 5),
    (0, 7),
    (0, 10),
    (0, 11),
    (1, 5),
    (1, 7),
    (1, 8),
    (1, 9),
    (2, 3),
    (2, 4),
    (2, 6),
    (2, 10),
    (2, 11),
    (3, 4),
    (3, 6),
    (3, 8),
    (3, 9),
    (4, 5),
    (4, 9),
    (4, 11),
    (5, 9),
    (5, 11),
    (6, 7),
    (6, 8),
    (6, 10),
    (7, 8),
    (7, 10),
    (8, 9),
    (10, 11),
]


def project(v, rx: float, ry: float, rz: float, scale: float, cx: int, cy: int):
    x, y, z = v
    # Rotate Y
    cy_, sy_ = math.cos(ry), math.sin(ry)
    x, z = x * cy_ + z * sy_, -x * sy_ + z * cy_
    # Rotate X
    cx_, sx_ = math.cos(rx), math.sin(rx)
    y, z = y * cx_ - z * sx_, y * sx_ + z * cx_
    # Rotate Z
    cz_, sz_ = math.cos(rz), math.sin(rz)
    x, y = x * cz_ - y * sz_, x * sz_ + y * cz_
    # Perspective
    f = scale / (z + 4)
    return (cx + x * f, cy + y * f, z)


# ─────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────


def _find_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


_FONT_H1 = _find_font(28)
_FONT_H2 = _find_font(16)
_FONT_BODY = _find_font(12)
_FONT_TINY = _find_font(10)


def _draw_grid(draw: ImageDraw.ImageDraw) -> None:
    # Sparse grid — 96 px intervals only. Fine-grain grid was destroying
    # video-codec efficiency (every pixel different per frame = codec
    # gives up and rasterizes aggressively).
    for x in range(0, W, 96):
        draw.line([(x, 0), (x, H)], fill=GRID, width=1)
    for y in range(0, H, 96):
        draw.line([(0, y), (W, y)], fill=GRID, width=1)


def _fmt_speaker(ident: str) -> str:
    return ident.split(":", 1)[-1].upper() if ":" in ident else ident.upper()


def render_frame(t: float) -> np.ndarray:
    actives, parts, speaking, amp, spectrum, events = STATE.snap()

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    _draw_grid(draw)

    cx, cy = W // 2, H // 2

    # ── Outer ring (radial spectrum bars) ────────────────────────
    n_bars = 32
    base_r = 170
    for i in range(n_bars):
        angle = 2 * math.pi * i / n_bars - math.pi / 2
        band = spectrum[i % len(spectrum)]
        length = 14 + band * 56 + (6 if speaking else 0) * math.sin(t * 3 + i * 0.3)
        x1 = cx + math.cos(angle) * base_r
        y1 = cy + math.sin(angle) * base_r
        x2 = cx + math.cos(angle) * (base_r + length)
        y2 = cy + math.sin(angle) * (base_r + length)
        color = AMBER if (speaking or band > 0.15) else AMBER_DIM
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)

    # Faint concentric guide rings
    for r in (base_r, base_r + 68):
        bbox = (cx - r, cy - r, cx + r, cy + r)
        draw.ellipse(bbox, outline=AMBER_DEEP, width=1)

    # ── Rotating wireframe icosahedron ───────────────────────────
    rx = t * 0.35
    ry = t * 0.55
    rz = math.sin(t * 0.2) * 0.15
    scale = 70 + (amp * 20)
    projected = [project(v, rx, ry, rz, scale, cx, cy) for v in VERTS_3D]
    # Draw edges; stroke color depends on back/front
    for a, b in EDGES:
        p1 = projected[a]
        p2 = projected[b]
        avg_z = (p1[2] + p2[2]) / 2
        # Back edges faded
        t_front = (avg_z + 2.5) / 5.0
        t_front = max(0.0, min(1.0, t_front))
        c = tuple(
            int(AMBER[i] * t_front + AMBER_DEEP[i] * (1 - t_front)) for i in range(3)
        )
        width = 3 if t_front > 0.5 else 1
        draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=c, width=width)

    # Central glow dot — brighter when speaking
    glow_r = 8 + int(amp * 10)
    glow_color = CYAN if speaking else CYAN_DIM
    draw.ellipse((cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r), fill=glow_color)

    # ── HUD: top-left block ─────────────────────────────────────
    draw.text((20, 18), "ARTEMIS", fill=AMBER, font=_FONT_H1)
    draw.text((20, 52), "v0.3 · HANDHELD UI", fill=AMBER_DIM, font=_FONT_BODY)

    # Status line
    if speaking:
        status = "> SPEAKING"
        status_col = CYAN
    elif actives:
        others = [a for a in actives if a != "artemis"]
        status = f"< HEARING {_fmt_speaker(others[0])}" if others else "> LISTENING"
        status_col = AMBER
    else:
        status = "> LISTENING"
        status_col = AMBER_DIM
    draw.text((20, 76), status, fill=status_col, font=_FONT_H2)

    # Top-right: timestamp + uptime
    wall_time = time.strftime("%H:%M:%S UTC", time.gmtime())
    draw.text((W - 180, 18), wall_time, fill=AMBER_DIM, font=_FONT_BODY)
    draw.text((W - 180, 34), f"T+{int(t):>4}s", fill=AMBER_DIM, font=_FONT_BODY)
    draw.text(
        (W - 180, 50), f"PARTICIPANTS {len(parts):>2}", fill=AMBER_DIM, font=_FONT_BODY
    )

    # ── Participant list (right edge) ───────────────────────────
    by = 110
    for pid in parts[:6]:
        color = CYAN if pid in actives else AMBER_DIM
        prefix = ">" if pid == "artemis" else "·"
        draw.text((W - 180, by), f"{prefix} {pid}", fill=color, font=_FONT_BODY)
        by += 16

    # ── Event log (bottom-left) ──────────────────────────────────
    log_y = H - 115
    draw.text((20, log_y), "EVENT LOG", fill=AMBER_DIM, font=_FONT_BODY)
    for i, line in enumerate(list(events)[-6:]):
        draw.text(
            (20, log_y + 16 + i * 13),
            line,
            fill=AMBER_DIM if i < len(events) - 1 else AMBER,
            font=_FONT_TINY,
        )

    # Scanlines intentionally removed. Per-3px horizontal noise was
    # catastrophic for H.264/VP8 rate-distortion — every line differed
    # from its neighbor so the codec couldn't exploit spatial
    # redundancy, producing severe rasterization artifacts at the
    # default publish bitrate.

    return np.asarray(img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────
# TTS + audio pipeline
# ─────────────────────────────────────────────────────────────────


text_queue: queue.Queue = queue.Queue()
audio_queue: queue.Queue = queue.Queue()


def _tts_worker() -> None:
    """TTS worker thread — pulls text off the queue, pushes PCM.

    Uses ``synthesize_stream`` so multi-sentence utterances start
    producing audio after only the first sentence's synth latency.
    Each chunk goes onto ``audio_queue`` as it arrives; the audio
    publisher thread drains them in order.

    Backend selection is driven by ``ARTEMIS_TTS_BACKEND``:
      - ``xtts``       Coqui XTTS-v2 local (highest quality, CPU/GPU)
      - ``nats_burst`` offload to NATS-subscribed worker pool
      - ``piper``      Piper ONNX local (fast, CPU-only fallback)
      - ``auto``       (default) — burst if a worker heartbeats, else
                       local XTTS if prerequisites present, else Piper.
    """
    from .tts import build_backend_from_env

    backend = build_backend_from_env()
    log.info("TTS backend ready: %s", backend.name)
    while True:
        text = text_queue.get()
        if text is None:
            backend.close()
            return
        try:
            log.info("synthesizing (%s): %s", backend.name, text[:80])
            STATE.log_event(f"SAY {text[:40]}")
            for chunk in backend.synthesize_stream(text):
                if chunk.duration_s > 0:
                    audio_queue.put(chunk.pcm)
        except Exception as e:  # noqa: BLE001
            log.exception("TTS error (%s): %s", backend.name, e)


def _compute_spectrum(chunk: np.ndarray, n_bands: int = 16) -> list[float]:
    """Cheap FFT-derived spectrum. Returns n_bands values in [0, 1]."""
    if len(chunk) == 0:
        return [0.0] * n_bands
    x = chunk.astype(np.float32) / 32768.0
    # Window to reduce spectral leakage
    window = np.hanning(len(x))
    spectrum = np.abs(np.fft.rfft(x * window))
    # Bin into n_bands bands (log-spaced would be better; uniform for simplicity)
    if len(spectrum) < n_bands:
        bands = list(spectrum / (spectrum.max() + 1e-9))
        return bands + [0.0] * (n_bands - len(bands))
    step = len(spectrum) // n_bands
    bands = [float(spectrum[i * step : (i + 1) * step].mean()) for i in range(n_bands)]
    m = max(bands) if bands else 1.0
    return [min(1.0, b / (m + 1e-9)) for b in bands]


def _audio_publisher_thread(
    source: rtc.AudioSource,
    loop: asyncio.AbstractEventLoop,
    running: threading.Event,
) -> None:
    """Real-time audio publisher running in its own thread.

    Audio quality with the asyncio-co-scheduled version was lossy and
    jittery: every 33 ms video-encode blocks the event loop long enough
    to miss the 20 ms audio deadline, which Opus renders as static. A
    dedicated thread with a monotonic ``time.sleep`` loop fixes the
    timing; ``capture_frame`` — an async method — is scheduled onto the
    main event loop via ``run_coroutine_threadsafe`` without blocking
    this thread.
    """
    silence = np.zeros(AUDIO_SAMPLES_PER_FRAME, dtype=np.int16)
    frame_dt = AUDIO_FRAME_MS / 1000.0
    next_t = time.monotonic()

    pending: np.ndarray | None = None
    offset = 0
    while not running.is_set():
        if pending is None:
            try:
                pending = audio_queue.get_nowait()
                offset = 0
            except queue.Empty:
                pending = None

        if pending is not None and offset < len(pending):
            chunk = pending[offset : offset + AUDIO_SAMPLES_PER_FRAME]
            offset += AUDIO_SAMPLES_PER_FRAME
            if len(chunk) < AUDIO_SAMPLES_PER_FRAME:
                chunk = np.concatenate(
                    [chunk, silence[: AUDIO_SAMPLES_PER_FRAME - len(chunk)]]
                )
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            STATE.set_speaking(True, min(1.0, rms / 4000.0))
            STATE.set_spectrum(_compute_spectrum(chunk))
            frame_bytes = chunk.tobytes()
            is_last = offset >= len(pending)
        else:
            STATE.set_speaking(False, 0.0)
            cur = STATE.snap()[4]
            STATE.set_spectrum([max(0.0, v * 0.9) for v in cur])
            frame_bytes = silence.tobytes()
            is_last = False

        frame = rtc.AudioFrame(
            frame_bytes, AUDIO_SR, AUDIO_CHANNELS, AUDIO_SAMPLES_PER_FRAME
        )
        # Submit + WAIT for the frame to be accepted. Without this,
        # frames pile up on the asyncio loop whenever video encode
        # holds it for >20 ms and get delivered in a bursty cluster,
        # which Opus renders as choppy playback. Awaiting gives real
        # backpressure: this thread only advances when the LiveKit
        # FFI has actually taken the frame.
        fut = asyncio.run_coroutine_threadsafe(
            source.capture_frame(frame), loop,
        )
        try:
            fut.result(timeout=1.0)
        except Exception as e:  # noqa: BLE001
            log.warning("capture_frame timeout/error: %s", e)

        if is_last:
            pending = None
            STATE.set_speaking(False, 0.0)

        next_t += frame_dt
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            # Fell behind — reset the phase, don't try to "catch up"
            # with a burst of frames (that's what produced the pops).
            next_t = time.monotonic()


async def _video_publisher(source: rtc.VideoSource, running: asyncio.Event) -> None:
    frame_idx = 0
    frame_dt = 1.0 / FPS
    next_t = time.monotonic()
    while not running.is_set():
        sim_t = frame_idx / FPS
        rgb = render_frame(sim_t)
        rgba = np.concatenate(
            [rgb, np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=2
        )
        frame = rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, rgba.tobytes())
        source.capture_frame(frame)
        frame_idx += 1
        next_t += frame_dt
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_t = time.monotonic()


def refresh_participants(room: rtc.Room) -> None:
    parts = [room.local_participant.identity]
    parts.extend(p.identity for p in room.remote_participants.values())
    STATE.set_participants(parts)


# ─────────────────────────────────────────────────────────────────
# Responder (canned replies on user-finishes-speaking)
# ─────────────────────────────────────────────────────────────────


class Responder:
    def __init__(self, min_gap_s: float = 4.0) -> None:
        self.last_active: list[str] = []
        self.last_reply_t: float = 0.0
        self.min_gap_s = min_gap_s

    def update(self, active: list[str]) -> None:
        others_before = [a for a in self.last_active if a != "artemis"]
        others_now = [a for a in active if a != "artemis"]
        transitioned_to_idle = bool(others_before) and not others_now
        now = time.time()
        if transitioned_to_idle and (now - self.last_reply_t) > self.min_gap_s:
            reply = random.choice(CANNED_REPLIES)
            text_queue.put(reply)
            self.last_reply_t = now
        self.last_active = list(active)


RESPONDER = Responder()


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("ARTEMIS_ROOM", "smoke-test")
    identity = os.environ.get("ARTEMIS_IDENTITY", "artemis")
    greeting = os.environ.get(
        "ARTEMIS_GREETING",
        "Greetings. ARTEMIS online. Systems nominal.",
    )

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name("ARTEMIS")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    room = rtc.Room()

    @room.on("participant_connected")
    def _on_join(p: rtc.RemoteParticipant) -> None:
        log.info("joined: %s", p.identity)
        STATE.log_event(f"JOIN {p.identity}")
        refresh_participants(room)

    @room.on("participant_disconnected")
    def _on_leave(p: rtc.RemoteParticipant) -> None:
        STATE.log_event(f"LEAVE {p.identity}")
        refresh_participants(room)

    @room.on("active_speakers_changed")
    def _on_speakers(speakers: list[rtc.Participant]) -> None:
        active = [s.identity for s in speakers]
        STATE.set_active(active)
        RESPONDER.update(active)

    await room.connect(url, token)
    refresh_participants(room)
    log.info("connected to %s as %s (room=%s)", url, identity, room_name)
    STATE.log_event("SYSTEM ONLINE")

    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()
    text_queue.put(greeting)

    v_source = rtc.VideoSource(W, H)
    v_track = rtc.LocalVideoTrack.create_video_track("artemis-face", v_source)
    # Simulcast: publisher sends 3 resolution layers, SFU routes the
    # right layer to each subscriber based on viewport + bandwidth.
    # Browser side has adaptiveStream + dynacast enabled (table.js), so
    # inactive layers are automatically paused — net CPU cost is small.
    # Top-layer cap is 2 Mbps at 960×720 @ 30 fps; LiveKit auto-derives
    # lower layers from that.
    video_opts = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=True,
        video_encoding=rtc.VideoEncoding(
            max_framerate=FPS,
            max_bitrate=2_000_000,
        ),
    )
    await room.local_participant.publish_track(v_track, video_opts)

    a_source = rtc.AudioSource(AUDIO_SR, AUDIO_CHANNELS)
    a_track = rtc.LocalAudioTrack.create_audio_track("artemis-voice", a_source)
    await room.local_participant.publish_track(
        a_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    log.info("published video + audio tracks")

    await room.local_participant.publish_data(
        json.dumps({"kind": "artemis.say", "text": "(ARTEMIS online.)"}).encode(),
        reliable=True,
    )

    running = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, running.set)
        except NotImplementedError:
            pass

    # S1b — optional NATS → LiveKit bridge for sheet updates.
    # S1e — optional DataChannel ↔ NATS bridge for chat messages.
    # S1g — optional direct-say bridge (GM narration: publish text on
    #       agi.rh.artemis.say.direct, ARTEMIS speaks it).
    # All opt-in via env so the existing Atlas deployment keeps
    # working unchanged until we flip them on.
    bridge_tasks: list[asyncio.Task] = []
    if os.environ.get("ARTEMIS_SHEETS_NATS", "") in ("1", "true", "yes", "on"):
        bridge_tasks.append(asyncio.create_task(_sheets_bridge(room, running)))
    if os.environ.get("ARTEMIS_CHAT_NATS", "") in ("1", "true", "yes", "on"):
        bridge_tasks.append(asyncio.create_task(_chat_bridge(room, running)))
    # Direct-say is always on when NATS is reachable — tiny, harmless,
    # and the GM narration console needs it.
    bridge_tasks.append(asyncio.create_task(_direct_say_bridge(running)))

    # Audio publisher runs in its own thread for hard real-time timing.
    # Mirror the asyncio shutdown event onto a threading.Event so the
    # thread can exit cleanly when the avatar is stopping.
    audio_stop = threading.Event()
    audio_thread = threading.Thread(
        target=_audio_publisher_thread,
        args=(a_source, loop, audio_stop),
        name="artemis-audio",
        daemon=True,
    )
    audio_thread.start()

    try:
        tasks = [_video_publisher(v_source, running)]
        for t in bridge_tasks:
            tasks.append(_wait_task(t))
        await asyncio.gather(*tasks)
    finally:
        text_queue.put(None)
        audio_stop.set()
        for t in bridge_tasks:
            if not t.done():
                t.cancel()
        audio_thread.join(timeout=2.0)
        await room.disconnect()
        log.info("disconnected cleanly")
    return 0


async def _wait_task(task: asyncio.Task) -> None:
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _direct_say_bridge(running: asyncio.Event) -> None:
    """Inject text onto ARTEMIS's TTS queue via NATS.

    Publish ``{"text": "..."}`` to ``agi.rh.artemis.say.direct`` and
    the avatar's TTS pipeline speaks it. Intended for the S1g GM
    narration console and for ad-hoc testing.
    """
    try:
        import nats  # type: ignore
    except ImportError:
        log.warning("nats-py not installed; direct-say bridge disabled")
        return

    url = os.environ.get("NATS_URL", "nats://localhost:4222")
    try:
        nc = await nats.connect(servers=[url])
    except Exception as e:  # noqa: BLE001
        log.warning("direct-say bridge: NATS connect failed (%s); disabled", e)
        return
    log.info("direct-say bridge online: agi.rh.artemis.say.direct")

    async def _on_msg(msg: "nats.aio.msg.Msg") -> None:  # type: ignore[name-defined]
        try:
            payload = json.loads(msg.data)
            text = str(payload.get("text") or "").strip()
        except Exception as e:  # noqa: BLE001
            log.warning("direct-say: bad payload: %s", e)
            return
        if not text:
            return
        log.info("direct-say: queuing %d chars", len(text))
        text_queue.put(text)

    try:
        await nc.subscribe("agi.rh.artemis.say.direct", cb=_on_msg)
        while not running.is_set():
            await asyncio.sleep(1)
    finally:
        try:
            await nc.drain()
        except Exception:  # noqa: BLE001
            pass
        log.info("direct-say bridge stopped")


async def _chat_bridge(room: rtc.Room, running: asyncio.Event) -> None:
    """Bridge chat messages between LiveKit DataChannel and NATS.

    Browser → NATS:
      Browser publishes {"kind": "artemis.chat.in", "msg": ChatMessage}
      on DataChannel. We re-publish the inner msg to
      ``agi.rh.artemis.chat.in.<from_id>`` so the chat service picks
      it up, persists, and routes through Primer.

    NATS → Browser:
      Subscribe to ``agi.rh.artemis.chat.out.*`` and
      ``agi.rh.artemis.chat.broadcast``; forward each as
      ``{"kind": "artemis.chat", "msg": ...}`` on DataChannel so the
      table.js handler renders it. Identity-gating already happened
      service-side when it chose the subject.
    """
    try:
        import nats  # type: ignore
    except ImportError:
        log.warning("nats-py not installed; chat bridge disabled")
        return

    url = os.environ.get("NATS_URL", "nats://localhost:4222")
    try:
        nc = await nats.connect(servers=[url])
    except Exception as e:  # noqa: BLE001
        log.warning("chat bridge: NATS connect failed (%s); disabled", e)
        return
    log.info("chat bridge online: NATS=%s", url)

    async def _on_out(msg: "nats.aio.msg.Msg") -> None:  # type: ignore[name-defined]
        try:
            inner = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("chat bridge: bad JSON on %s: %s", msg.subject, e)
            return
        out = json.dumps(
            {"kind": "artemis.chat", "msg": inner}, separators=(",", ":")
        ).encode()
        try:
            await room.local_participant.publish_data(out, reliable=True)
        except Exception as e:  # noqa: BLE001
            log.warning("chat bridge: DataChannel publish failed: %s", e)

    @room.on("data_received")
    def _on_data(data, participant, *_rest) -> None:
        # LiveKit Python SDK signature has varied across versions; we
        # accept extra positional args and pick the participant from
        # whichever slot it landed in.
        try:
            raw = data.data if hasattr(data, "data") else data
            inner = json.loads(raw)
        except Exception:  # noqa: BLE001
            return
        if inner.get("kind") != "artemis.chat.in":
            return
        chat_msg = inner.get("msg") or {}
        from_id = chat_msg.get("from_id") or "unknown"
        subj = f"agi.rh.artemis.chat.in.{from_id}"
        payload = json.dumps(chat_msg, separators=(",", ":")).encode()
        asyncio.create_task(nc.publish(subj, payload))

    try:
        await nc.subscribe("agi.rh.artemis.chat.out.*", cb=_on_out)
        await nc.subscribe("agi.rh.artemis.chat.broadcast", cb=_on_out)
        while not running.is_set():
            await asyncio.sleep(1)
    finally:
        try:
            await nc.drain()
        except Exception:  # noqa: BLE001
            pass
        log.info("chat bridge stopped")


async def _sheets_bridge(room: rtc.Room, running: asyncio.Event) -> None:
    """Subscribe to agi.rh.artemis.sheet.* and forward to DataChannel.

    Messages on NATS are already shaped as ``{"name": ..., "rows": [...]}``
    by :class:`SheetsPoller`; we wrap them as
    ``{"kind": "artemis.sheet", "rows": [...]}`` so the existing
    table.js handler (``case "artemis.sheet"``) picks them up.

    Snapshot subjects are also forwarded — the HUD treats a snapshot
    and a diff the same (apply rows by id).
    """
    try:
        import nats  # type: ignore
    except ImportError:
        log.warning("nats-py not installed; sheet bridge disabled")
        return

    url = os.environ.get("NATS_URL", "nats://localhost:4222")
    try:
        nc = await nats.connect(servers=[url])
    except Exception as e:  # noqa: BLE001
        log.warning("sheet bridge: NATS connect failed (%s); disabled", e)
        return
    log.info("sheet bridge online: %s → DataChannel", url)

    async def _on_msg(msg: "nats.aio.msg.Msg") -> None:  # type: ignore[name-defined]
        try:
            inner = json.loads(msg.data)
        except Exception as e:  # noqa: BLE001
            log.warning("sheet bridge: bad JSON on %s: %s", msg.subject, e)
            return
        rows = inner.get("rows") or []
        out = json.dumps(
            {"kind": "artemis.sheet", "name": inner.get("name"), "rows": rows},
            separators=(",", ":"),
        ).encode()
        try:
            await room.local_participant.publish_data(out, reliable=True)
        except Exception as e:  # noqa: BLE001
            log.warning("sheet bridge: publish_data failed: %s", e)

    try:
        await nc.subscribe("agi.rh.artemis.sheet.>", cb=_on_msg)
        # Stay alive until the main loop signals shutdown.
        while not running.is_set():
            await asyncio.sleep(1)
    finally:
        try:
            await nc.drain()
        except Exception:  # noqa: BLE001
            pass
        log.info("sheet bridge stopped")


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
