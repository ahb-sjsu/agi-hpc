# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""ARTEMIS 3D avatar — browser-side Three.js scene, rendered headlessly
on Atlas and published as the avatar's LiveKit video track.

This is Sprint 3 (phase B of the original roadmap). The renderer runs
headless Chromium via Playwright, loads a self-contained HTML page
that hosts a Three.js scene + a glTF humanoid avatar, and captures
the canvas at a fixed frame rate. Frames are emitted as numpy RGBA
arrays ready for :mod:`agi.primer.artemis.livekit_agent.avatar_hud`'s
video publisher.

Phases (mirrors the top-level conversation plan):

  Phase A — in-repo scaffolding (this commit): scene.py + renderer.py
    with tests. No Atlas install, no service change.
  Phase B — Atlas install + smoke: Chromium + playwright, produce an
    MP4 for review.
  Phase C — swap the HUD video track for the 3D capture, gated on
    ``ARTEMIS_AVATAR_MODE=3d``.
  Phase D — viseme lip-sync, idle animations, expression cues.

Entry points:

  - :func:`.scene.build_scene_html` — build the HTML page a given model
    URL will be rendered in.
  - :class:`.renderer.AvatarRenderer` — headless-Chromium capture loop.
"""

from __future__ import annotations

from .scene import SceneConfig, build_scene_html

__all__ = ["SceneConfig", "build_scene_html"]
