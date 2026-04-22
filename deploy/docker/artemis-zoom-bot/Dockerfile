# ARTEMIS Zoom Bot — Phase 3 scaffolding
#
# This is the shape of the container. Each numbered section is a
# TODO item for the Phase 3 build-out (see docs/ARTEMIS.md §11).
# The Dockerfile does NOT currently produce a working image — it is
# committed so the structure is visible and reviewable before the
# actual build work lands.

# ── base ───────────────────────────────────────────────────────────
# Ubuntu 22.04 + CUDA 12.1 runtime — matches the NRP A10 images we've
# used successfully for other bursts, and is what the Zoom Linux SDK
# is tested against.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── OS deps ────────────────────────────────────────────────────────
# - xvfb: Zoom SDK needs a display, even headless.
# - portaudio/alsa: audio routing to/from the virtual sink.
# - libssl/curl: Zoom SDK + NATS client runtime.
# - python3.11: matches Atlas's primer.service.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
      xvfb \
      portaudio19-dev libasound2 libasound2-plugins pulseaudio-utils \
      libssl-dev libcurl4-openssl-dev \
      libglib2.0-0 libx11-6 libxcb1 libxkbcommon0 libxkbcommon-x11-0 \
      python3.11 python3.11-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python

# ── Zoom Meeting SDK (Linux, C++) ──────────────────────────────────
# TODO(phase3): download + install Zoom Meeting SDK binaries.
# The SDK distribution is gated behind Zoom developer account login;
# we likely need to download it offline, stash in an artifact repo,
# and COPY it in here rather than curling during build.
#
# Reference: https://developers.zoom.us/docs/meeting-sdk/linux/
#
# COPY thirdparty/zoom-meeting-sdk-linux /opt/zoom-sdk/
# ENV LD_LIBRARY_PATH=/opt/zoom-sdk/lib:$LD_LIBRARY_PATH

# ── Python deps ───────────────────────────────────────────────────
# faster-whisper for streaming ASR on GPU (CTranslate2-backed).
# nats-py for the NATS client.
COPY deploy/docker/artemis-zoom-bot/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── bot entrypoint ────────────────────────────────────────────────
# TODO(phase3): actual entrypoint script.
# Responsibilities:
#   1. Parse env: session_id, meeting URL, credentials, NATS URL.
#   2. Start xvfb: `Xvfb :99 -screen 0 1024x768x24 &` ; export DISPLAY=:99
#   3. Launch Zoom SDK bot process; confirm "ARTEMIS" joins meeting.
#   4. Pipe per-participant audio frames into faster-whisper.
#   5. Publish each final transcript on NATS subject
#      `agi.rh.artemis.heard` with the schema in docs/ARTEMIS.md §4.
#   6. Subscribe to `agi.rh.artemis.say` filtered on session_id;
#      post each text reply into the meeting's chat via Zoom SDK.
#   7. Detect meeting-end → clean exit (Job completes).
COPY deploy/docker/artemis-zoom-bot/entrypoint.sh /app/entrypoint.sh
COPY deploy/docker/artemis-zoom-bot/bot.py        /app/bot.py
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
