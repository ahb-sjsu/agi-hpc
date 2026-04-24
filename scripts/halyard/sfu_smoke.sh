#!/usr/bin/env bash
#
# ────────────────────────────────────────────────────────────────
# Halyard Table — LiveKit SFU smoke test
#
# Verifies that the SFU is reachable, mints a participant JWT,
# and confirms the server accepts it. Run this after bringing
# the SFU up for the first time and after any config change.
#
# Exits 0 on success, non-zero on failure. Suitable for CI.
#
# Usage:
#   scripts/halyard/sfu_smoke.sh [--host HOST] [--room ROOM]
#
# Defaults:
#   --host  halyard.atlas-sjsu.duckdns.org
#   --room  halyard-smoke-$(date +%s)
#
# Required env:
#   LIVEKIT_API_KEY       — API key (matches livekit-keys.yaml)
#   LIVEKIT_API_SECRET    — API secret
# ────────────────────────────────────────────────────────────────

set -euo pipefail

HOST="halyard.atlas-sjsu.duckdns.org"
ROOM="halyard-smoke-$(date +%s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)  HOST="$2"; shift 2 ;;
    --room)  ROOM="$2"; shift 2 ;;
    -h|--help)
      sed -n '3,20p' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

: "${LIVEKIT_API_KEY:?set LIVEKIT_API_KEY (see infra/local/livekit-sfu/livekit-keys.yaml)}"
: "${LIVEKIT_API_SECRET:?set LIVEKIT_API_SECRET}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "═══════════════════════════════════════════════════════════"
echo "  Halyard Table — SFU smoke test"
echo "  Host: ${HOST}"
echo "  Room: ${ROOM}"
echo "═══════════════════════════════════════════════════════════"

# ── 1. TLS reachability ─────────────────────────────────────────
echo "[1/4] TLS reachability check..."
if ! curl --silent --show-error --fail --max-time 5 \
    --output /dev/null \
    "https://${HOST}/rtc/validate?room=${ROOM}&identity=smoke-test"; then
  echo "  ✗ Cannot reach https://${HOST}/rtc/validate"
  echo "    Check: Caddy running, DNS, TLS cert, firewall :443"
  exit 1
fi
echo "  ✓ TLS reachable"

# ── 2. Mint a participant JWT ───────────────────────────────────
echo "[2/4] Minting participant JWT..."
TOKEN="$(
  python3 - <<PY
from agi.primer.artemis.livekit_agent.token import (
    GrantOptions,
    mint_participant_token,
)
tok = mint_participant_token(
    identity="smoke-test",
    room_name="${ROOM}",
    api_key="${LIVEKIT_API_KEY}",
    api_secret="${LIVEKIT_API_SECRET}",
    grants=GrantOptions(can_publish=True, can_subscribe=True, can_publish_data=True),
    name="smoke",
)
print(tok)
PY
)"
if [[ -z "${TOKEN}" ]]; then
  echo "  ✗ Token minting failed"
  exit 1
fi
echo "  ✓ Token minted (${#TOKEN} chars)"

# ── 3. Validate the token against the SFU ───────────────────────
echo "[3/4] Validating token against SFU..."
VALIDATE_URL="https://${HOST}/rtc/validate?access_token=${TOKEN}"
HTTP_CODE="$(curl --silent --output /dev/null --write-out '%{http_code}' \
    --max-time 5 "${VALIDATE_URL}")"
if [[ "${HTTP_CODE}" != "200" ]]; then
  echo "  ✗ Token validation failed (HTTP ${HTTP_CODE})"
  echo "    Check that the API key in \$LIVEKIT_API_KEY matches"
  echo "    the key in infra/local/livekit-sfu/livekit-keys.yaml"
  exit 1
fi
echo "  ✓ Token validated"

# ── 4. Health check on TURN port ────────────────────────────────
echo "[4/4] TURN reachability..."
if command -v nc >/dev/null 2>&1; then
  if nc -zu -w 3 "${HOST}" 3478 2>/dev/null; then
    echo "  ✓ TURN UDP :3478 reachable"
  else
    # Not a hard failure — many home routers block UDP probes.
    echo "  ⚠ TURN UDP :3478 did not respond (may be router filtering)"
  fi
else
  echo "  ⚠ netcat (nc) not installed — skipped TURN check"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  ✓ ALL CHECKS PASSED"
echo "  SFU is ready for the Halyard Table."
echo "═══════════════════════════════════════════════════════════"
