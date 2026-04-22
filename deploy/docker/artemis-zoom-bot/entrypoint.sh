#!/usr/bin/env bash
# ARTEMIS Zoom Bot — container entrypoint (Phase 3 scaffolding).
#
# Current behavior: print the env, fail loudly with a TODO.
# Phase 3 fills in xvfb + Zoom SDK boot + handoff to bot.py.

set -euo pipefail

echo "ARTEMIS Zoom Bot scaffolding (not yet implemented)."
echo "Session:    ${ARTEMIS_SESSION_ID:-<unset>}"
echo "NATS URL:   ${NATS_URL:-<unset>}"
echo "Display:    ${ARTEMIS_DISPLAY_NAME:-ARTEMIS}"
echo "Whisper:    ${ARTEMIS_WHISPER_MODEL:-large-v3}"
echo "Output:     ${ARTEMIS_OUTPUT_MODE:-text}"
echo
echo "TODO(phase3): boot xvfb, start Zoom SDK, exec bot.py."
echo "See docs/ARTEMIS.md §11 Phase 3 + deploy/docker/artemis-zoom-bot/README.md"
exit 2
