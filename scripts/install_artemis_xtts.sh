#!/usr/bin/env bash
# Install Coqui TTS (XTTS-v2) into the artemis-avatar venv on Atlas.
#
# Usage (on Atlas):
#   bash /home/claude/agi-hpc/scripts/install_artemis_xtts.sh [venv_path]
#
# Writes nothing outside the chosen venv. Downloads the ~2 GB XTTS-v2
# model on first use (cached under ~/.local/share/tts).
#
# Also creates the voice-reference directory under the caller's home
# so the user can drop an `artemis_ref.wav` (6–30 s of clean speech)
# to seed the voice clone.

set -euo pipefail

VENV="${1:-/home/claude/env}"
REF_DIR="${HOME}/artemis-voice-refs"

if [[ ! -x "$VENV/bin/python3" ]]; then
  echo "venv not found at $VENV — aborting" >&2
  exit 1
fi

echo "[1/3] Installing coqui-tts into $VENV"
"$VENV/bin/pip" install --upgrade pip
# coqui-tts is the community-maintained fork of the original Coqui
# TTS package (the original company dissolved; IdiapML kept it alive).
"$VENV/bin/pip" install "coqui-tts>=0.24"

echo "[2/3] Priming the XTTS-v2 model cache (this downloads ~2 GB)"
"$VENV/bin/python3" - <<'PY'
import os
os.environ.setdefault("COQUI_TOS_AGREED", "1")  # non-interactive acceptance
from TTS.api import TTS  # noqa: E402

TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
print("[xtts] model loaded OK")
PY

echo "[3/3] Preparing voice-reference directory: $REF_DIR"
mkdir -p "$REF_DIR"
cat > "$REF_DIR/README.txt" <<'EOF'
Drop a 6–30 s WAV of clean speech here as `artemis_ref.wav` to seed
XTTS-v2's voice clone. Matching room tone + mic quality is more
important than duration — a dry 10 s sample will outperform a noisy
30 s one. Mono, 22050 or 24000 Hz preferred.

You can record in-browser: https://www.onlinemictest.com/ → save WAV.
EOF

echo
echo "Done. Next:"
echo "  - Drop a reference WAV at: $REF_DIR/artemis_ref.wav"
echo "  - Set ARTEMIS_TTS_BACKEND=xtts (or leave unset for auto)"
echo "  - For burst offload: run \`python -m agi.primer.artemis.livekit_agent.tts.worker\`"
echo "    inside any GPU pod you already have scheduled."
