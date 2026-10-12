# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Smoke-test CLI for the 3D avatar renderer.

Usage::

    # Render 30 frames (1 s at 30 fps) into ./avatar-smoke/
    python -m agi.primer.artemis.livekit_agent.avatar3d \
        --out ./avatar-smoke --frames 30

    # Use a custom model URL (RPM-style GLB, or any glTF)
    python -m agi.primer.artemis.livekit_agent.avatar3d \
        --out /tmp/avatar --model https://models.readyplayer.me/<id>.glb

Phase B of the 3D avatar sprint — runs on Atlas once Playwright +
Chromium are installed. Produces per-frame PNGs the operator can
inspect before Phase C pipes them into the LiveKit video track.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .renderer import smoke
from .scene import DEFAULT_MODEL_URL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(Path("./avatar-smoke").resolve()),
        help="output directory for per-frame PNGs (default: ./avatar-smoke)",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=30,
        help="number of frames to capture (default: 30 = 1 s at 30 fps)",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL_URL,
        help=f"glTF model URL to load (default: {DEFAULT_MODEL_URL})",
    )
    args = ap.parse_args()

    # The smoke() helper uses SceneConfig defaults; model-URL override
    # is applied via the renderer's scene builder. Keep CLI thin —
    # anyone needing more knobs can import AvatarRenderer directly.
    if args.model != DEFAULT_MODEL_URL:
        # Rebind the default so the helper picks it up.
        from . import scene as _scene

        _scene.DEFAULT_MODEL_URL = args.model  # type: ignore[assignment]

    out = smoke(out_dir=args.out, count=args.frames)
    print(f"wrote {args.frames} frame(s) into {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
