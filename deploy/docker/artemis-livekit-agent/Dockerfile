# ARTEMIS LiveKit Agent — Phase 3 container
#
# Replaced the Zoom Meeting SDK approach on 2026-04-22. See
# docs/ARTEMIS.md §11 Phase 3 for the plan of record.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── OS deps ────────────────────────────────────────────────────────
# Much smaller than the Zoom variant: no xvfb, no display server, no
# C++ SDK. LiveKit agents runs as a plain Python process.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
      python3.11 python3.11-dev python3-pip \
      ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python

# ── Python deps ────────────────────────────────────────────────────
COPY deploy/docker/artemis-livekit-agent/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── agi-hpc source ────────────────────────────────────────────────
# Editable install keeps bind-mount development possible; production
# builds should pin to a released tag.
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
RUN pip install --no-cache-dir -e /app

COPY deploy/docker/artemis-livekit-agent/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
