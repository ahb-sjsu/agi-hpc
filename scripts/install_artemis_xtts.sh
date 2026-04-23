#!/usr/bin/env bash
# Install Coqui TTS (XTTS-v2) into an ISOLATED venv on Atlas.
#
# Usage (on Atlas):
#   bash /home/claude/agi-hpc/scripts/install_artemis_xtts.sh [venv_path]
#
# Isolated by design: coqui-tts 0.24.x pulls transformers<4.46, which
# breaks the main /home/claude/env used by Primer / unsloth / trl /
# ARTEMIS handler. Never install into the main venv.
#
# Downloads the ~2 GB XTTS-v2 model on first use (cached under
# ~/.local/share/tts). Also creates the voice-reference directory so
# you can drop an `artemis_ref.wav` (6–30 s of clean speech) to seed
# the voice clone.

set -euo pipefail

VENV="${1:-/home/claude/env-xtts}"
REF_DIR="${HOME}/artemis-voice-refs"

if [[ ! -x "$VENV/bin/python3" ]]; then
  echo "creating venv at $VENV"
  python3 -m venv "$VENV"
fi

echo "[1/3] Installing coqui-tts + matched torch into $VENV"
"$VENV/bin/pip" install --upgrade pip
# Pin coqui-tts to 0.24.3 — the published 0.25+ and 0.27+ package
# metadata claims transformers>=4.57 but the code imports the pre-4.46
# `isin_mps_friendly` symbol. 0.24.3 is the last known-good with
# transformers 4.45.x.
"$VENV/bin/pip" install "coqui-tts==0.24.3" "transformers<4.46"

# Atlas GPUs are Quadro GV100 (Volta, sm_70). Recent torch wheels
# (2.11+cu128) dropped sm_70. Pin to 2.6.0+cu124 which still ships
# sm_50..sm_90 kernels and is compatible with coqui-tts 0.24.3.
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchaudio==2.6.0"

# Worker needs NATS client to subscribe to the burst queue.
"$VENV/bin/pip" install "nats-py>=2.6"

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
