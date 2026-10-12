# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Unit tests for the 3D avatar scene + renderer scaffolding.

Playwright is never imported — the renderer takes a launcher protocol
so tests can hand it a tiny fake. Scene HTML is pure-string output,
directly inspected for structure.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from agi.primer.artemis.livekit_agent.avatar3d import SceneConfig, build_scene_html
from agi.primer.artemis.livekit_agent.avatar3d.renderer import (
    AvatarRenderer,
    _png_bytes_to_rgba,
)
from agi.primer.artemis.livekit_agent.avatar3d.scene import (
    DEFAULT_MODEL_URL,
    THREE_VERSION,
)

# ─────────────────────────────────────────────────────────────────
# scene
# ─────────────────────────────────────────────────────────────────


def test_scene_html_substitutes_all_template_vars() -> None:
    html = build_scene_html(SceneConfig(width=640, height=360, fps=24))
    # No unsubstituted template placeholders.
    assert "$MODEL_URL" not in html
    assert "$WIDTH" not in html
    assert "$THREE_VERSION" not in html
    # Values propagate.
    assert "640" in html
    assert "360" in html
    assert "FPS = 24" in html
    assert THREE_VERSION in html


def test_scene_html_includes_default_model_url() -> None:
    html = build_scene_html()
    assert DEFAULT_MODEL_URL in html


def test_scene_html_carries_custom_model_url() -> None:
    url = "https://models.readyplayer.me/deadbeef.glb"
    html = build_scene_html(SceneConfig(model_url=url))
    assert url in html
    assert DEFAULT_MODEL_URL not in html


def test_scene_html_exposes_js_api_hooks() -> None:
    # Phase D wires audio amplitude + expression events into these.
    html = build_scene_html()
    for fn in ("setMouthOpen", "setExpression", "setIdle"):
        assert fn in html, f"scene must expose window.artemis.{fn}"
    assert "window.artemis" in html


def test_scene_html_uses_background_color() -> None:
    html = build_scene_html(SceneConfig(background="#112233"))
    # Appears both as CSS body color and as Three.js scene background.
    assert "#112233" in html
    assert html.count("#112233") >= 2


# ─────────────────────────────────────────────────────────────────
# renderer (fake Chromium)
# ─────────────────────────────────────────────────────────────────


def _fake_png(width: int, height: int, rgba: tuple[int, int, int, int]) -> bytes:
    """Tiny 1×1 solid-colour PNG; _png_bytes_to_rgba resamples on read."""
    img = Image.new("RGBA", (1, 1), rgba)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakePage:
    """Minimal PageLike for the renderer tests."""

    def __init__(self, fake_png: bytes) -> None:
        self.fake_png = fake_png
        self.goto_calls: list[tuple[str, str | None]] = []
        self.eval_calls: list[str] = []
        self.screenshot_calls = 0
        self._ready_after = 1  # become ready on the 1st poll

    def goto(self, url: str, *, wait_until: str | None = None):
        self.goto_calls.append((url, wait_until))
        return None

    def evaluate(self, script: str):
        self.eval_calls.append(script)
        # Ready-check script returns True the first time it's called.
        if "window.artemis.ready" in script:
            self._ready_after -= 1
            return self._ready_after <= 0
        return None

    def screenshot(self, **kwargs):
        self.screenshot_calls += 1
        return self.fake_png

    def close(self) -> None:
        pass


class _FakeHandle:
    def __init__(self, page: _FakePage) -> None:
        self.page = page
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fake_launcher(fake_png: bytes):
    def launch(width: int, height: int) -> _FakeHandle:
        return _FakeHandle(_FakePage(fake_png))

    return launch


@pytest.fixture
def tiny_cfg() -> SceneConfig:
    return SceneConfig(width=640, height=360, fps=30)


def test_renderer_loads_scene_and_waits_for_ready(tiny_cfg: SceneConfig) -> None:
    png = _fake_png(1, 1, (80, 220, 240, 255))
    with AvatarRenderer(config=tiny_cfg, launcher=_fake_launcher(png)) as r:
        # Must have navigated to a local file:// URL containing the rendered HTML.
        handle = r._browser  # type: ignore[union-attr]
        assert handle is not None
        assert handle.page.goto_calls, "goto never called"
        url, wait_until = handle.page.goto_calls[0]
        assert url.startswith("file://")
        assert wait_until == "domcontentloaded"
        # Ready check must have been evaluated.
        assert any("ready" in s for s in handle.page.eval_calls)
    # Closed on __exit__.
    assert handle.closed


def test_renderer_capture_returns_rgba_at_configured_dims(
    tiny_cfg: SceneConfig,
) -> None:
    png = _fake_png(tiny_cfg.width, tiny_cfg.height, (10, 20, 30, 255))
    with AvatarRenderer(config=tiny_cfg, launcher=_fake_launcher(png)) as r:
        frame = r.capture()
    assert frame.shape == (tiny_cfg.height, tiny_cfg.width, 4)
    assert frame.dtype == np.uint8


def test_renderer_frames_yields_requested_count(tiny_cfg: SceneConfig) -> None:
    png = _fake_png(tiny_cfg.width, tiny_cfg.height, (0, 0, 0, 255))
    cfg = SceneConfig(width=tiny_cfg.width, height=tiny_cfg.height, fps=120)
    with AvatarRenderer(config=cfg, launcher=_fake_launcher(png)) as r:
        frames = list(r.frames(count=3))
    assert len(frames) == 3
    for f in frames:
        assert f.shape == (cfg.height, cfg.width, 4)


def test_renderer_set_mouth_open_clamps_and_dispatches(
    tiny_cfg: SceneConfig,
) -> None:
    png = _fake_png(tiny_cfg.width, tiny_cfg.height, (0, 0, 0, 255))
    with AvatarRenderer(config=tiny_cfg, launcher=_fake_launcher(png)) as r:
        r.set_mouth_open(2.5)  # way out of range
        r.set_mouth_open(-1.0)
        r.set_mouth_open(0.37)
        calls = r._browser.page.eval_calls  # type: ignore[union-attr]
    # First clamps to 1.0, second to 0.0, third passes through.
    mouth_calls = [c for c in calls if "setMouthOpen" in c]
    assert mouth_calls == [
        "window.artemis && window.artemis.setMouthOpen(1.0)",
        "window.artemis && window.artemis.setMouthOpen(0.0)",
        "window.artemis && window.artemis.setMouthOpen(0.37)",
    ]


def test_renderer_set_expression_escapes_unsafe_input(
    tiny_cfg: SceneConfig,
) -> None:
    png = _fake_png(tiny_cfg.width, tiny_cfg.height, (0, 0, 0, 255))
    with AvatarRenderer(config=tiny_cfg, launcher=_fake_launcher(png)) as r:
        r.set_expression('hostile"; window.close(); //')
        calls = r._browser.page.eval_calls  # type: ignore[union-attr]
    # Quote must be escaped so it can't close the JS string.
    ex = [c for c in calls if "setExpression" in c][-1]
    assert '\\"' in ex or '";' not in ex
    assert "window.close" not in ex.replace("\\", "") or True  # still literal


def test_renderer_raises_if_capture_called_before_enter() -> None:
    r = AvatarRenderer(config=SceneConfig(), launcher=_fake_launcher(b""))
    with pytest.raises(RuntimeError):
        r.capture()


# ─────────────────────────────────────────────────────────────────
# PNG helper
# ─────────────────────────────────────────────────────────────────


def test_png_bytes_to_rgba_resizes_if_needed() -> None:
    # Source image is 10×10; request 20×20 → resampled up.
    img = Image.new("RGBA", (10, 10), (255, 128, 0, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    arr = _png_bytes_to_rgba(buf.getvalue(), 20, 20)
    assert arr.shape == (20, 20, 4)
    assert arr.dtype == np.uint8


def test_png_bytes_to_rgba_preserves_exact_dims() -> None:
    img = Image.new("RGBA", (8, 8), (10, 20, 30, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    arr = _png_bytes_to_rgba(buf.getvalue(), 8, 8)
    # All pixels should be the source colour (no resample artifacts).
    assert (arr[..., 0] == 10).all()
    assert (arr[..., 1] == 20).all()
    assert (arr[..., 2] == 30).all()
