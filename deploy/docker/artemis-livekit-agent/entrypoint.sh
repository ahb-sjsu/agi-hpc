#!/usr/bin/env bash
# ARTEMIS LiveKit Agent entrypoint.
#
# Runs the Python agent worker. All configuration is env-driven:
#   ARTEMIS_SESSION_ID   (required)  LiveKit room name == session id
#   LIVEKIT_URL          (required)  wss://... or ws://... for local
#   LIVEKIT_API_KEY      (required)
#   LIVEKIT_API_SECRET   (required)
#   NATS_URL             default nats://atlas-nats:4222 (K8s) or
#                                 nats://host.docker.internal:4222 (local)

set -euo pipefail

: "${ARTEMIS_SESSION_ID:?ARTEMIS_SESSION_ID is required}"
: "${LIVEKIT_URL:?LIVEKIT_URL is required}"
: "${LIVEKIT_API_KEY:?LIVEKIT_API_KEY is required}"
: "${LIVEKIT_API_SECRET:?LIVEKIT_API_SECRET is required}"

echo "ARTEMIS LiveKit agent starting"
echo "  session:  ${ARTEMIS_SESSION_ID}"
echo "  livekit:  ${LIVEKIT_URL}"
echo "  nats:     ${NATS_URL:-nats://atlas-nats:4222}"
echo "  whisper:  ${ARTEMIS_WHISPER_MODEL:-large-v3}"

exec python -m agi.primer.artemis.livekit_agent
