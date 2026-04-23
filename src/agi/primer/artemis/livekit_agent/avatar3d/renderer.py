# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""Headless-Chromium capture loop for the ARTEMIS 3D avatar.

The renderer owns a Chromium instance (via Playwright), loads the
Three.js scene HTML, waits for the model to finish loading, and
captures the canvas at a fixed rate. Each call to :meth:`capture`
returns a raw RGBA numpy frame ready for LiveKit.

Playwright is a heavy dependency — we keep the import lazy and inject
a fake launcher in tests so none of the CI steps need Chromium.
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import numpy as np

from .scene import SceneConfig, build_scene_html

log = logging.getLogger("artemis.avatar3d.renderer")


class PageLike(Protocol):
    """Minimal surface of ``playwright.sync_api.Page`` we touch.

    Declared as a Protocol so the tests can stand in with tiny fakes.
    """

    def goto(self, url: str, *, wait_until: str | None = ...) -> Any: ...
    def evaluate(self, script: str) -> Any: ...
    def screenshot(self, **kwargs: Any) -> bytes: ...
    def close(self) -> None: ...


class BrowserLauncher(Protocol):
    """Factory for :class:`PageLike` instances.

    Production uses a Playwright wrapper; tests use a fake that
    returns a page stub. We hide Playwright's async/sync duality and
    browser context lifecycle behind this so the renderer doesn't
    have to care.
    """

    def launch(self, width: int, height: int) -> "_BrowserHandle": ...


class _BrowserHandle(Protocol):
    page: PageLike

    def close(self) -> None: ...


# ──────────────────────────────────────────────────────────────
# Main renderer
# ──────────────────────────────────────────────────────────────


class AvatarRenderer:
    """Headless 3D avatar capture, frame-by-frame.

    Usage::

        with AvatarRenderer(config=SceneConfig(), launcher=pw_launcher) as r:
            for frame in r.frames(count=30):
                do_something_with(frame)  # frame is np.ndarray[H, W, 4]
    """

    def __init__(
        self,
        *,
        config: SceneConfig | None = None,
        launcher: BrowserLauncher | None = None,
        scene_builder: Callable[[SceneConfig], str] = build_scene_html,
    ) -> None:
        self.config = config or SceneConfig()
        self.launcher = launcher or _default_playwright_launcher
        self.scene_builder = scene_builder
        self._browser: _BrowserHandle | None = None
        self._tempdir: tempfile.TemporaryDirectory | None = None

    # ── context management ──────────────────────────────────────

    def __enter__(self) -> "AvatarRenderer":
        self._tempdir = tempfile.TemporaryDirectory(prefix="artemis-avatar3d-")
        html_path = Path(self._tempdir.name) / "scene.html"
        html_path.write_text(self.scene_builder(self.config), encoding="utf-8")

        launcher = (
            self.launcher
            if not callable(self.launcher) or isinstance(self.launcher, type)
            else self.launcher
        )
        if callable(self.launcher):
            self._browser = self.launcher(self.config.width, self.config.height)
        else:  # pragma: no cover — never hit in practice
            self._browser = launcher.launch(self.config.width, self.config.height)

        url = html_path.as_uri()
        self._browser.page.goto(url, wait_until="domcontentloaded")
        self._wait_for_ready()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            self._browser = None
            if self._tempdir is not None:
                self._tempdir.cleanup()
                self._tempdir = None

    # ── capture ─────────────────────────────────────────────────

    def capture(self) -> np.ndarray:
        """Grab a single frame as an RGBA numpy array."""
        if self._browser is None:
            raise RuntimeError("renderer not entered; use `with AvatarRenderer(...)`")
        png = self._browser.page.screenshot(type="png", omit_background=False)
        return _png_bytes_to_rgba(png, self.config.width, self.config.height)

    def frames(self, *, count: int) -> Iterator[np.ndarray]:
        """Yield ``count`` frames spaced by the configured frame rate.

        The caller typically hands each frame to the LiveKit video
        publisher; keeping this a generator makes backpressure
        explicit and trivial to test.
        """
        frame_dt = 1.0 / max(1, self.config.fps)
        next_t = time.monotonic()
        for _ in range(count):
            yield self.capture()
            next_t += frame_dt
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.monotonic()

    # ── JS bridge ──────────────────────────────────────────────

    def set_mouth_open(self, level: float) -> None:
        """Drive the avatar's mouth-open blend shape, 0..1."""
        if self._browser is None:
            return
        clamped = max(0.0, min(1.0, float(level)))
        self._browser.page.evaluate(
            f"window.artemis && window.artemis.setMouthOpen({clamped})"
        )

    def set_expression(self, name: str) -> None:
        if self._browser is None:
            return
        safe = str(name).replace('"', '\\"').replace("\n", "")[:40]
        self._browser.page.evaluate(
            f'window.artemis && window.artemis.setExpression("{safe}")'
        )

    def set_emotion(self, name: str, value: float) -> None:
        """Blend a single emotion 0..1 (happy/sad/angry/surprised/relaxed)."""
        if self._browser is None:
            return
        safe = str(name).replace('"', '\\"').replace("\n", "")[:40]
        v = max(0.0, min(1.0, float(value)))
        self._browser.page.evaluate(
            f'window.artemis && window.artemis.setEmotion("{safe}", {v})'
        )

    def set_viseme(self, name: str, value: float) -> None:
        """Set a single viseme shape 0..1 (aa/ih/ou/ee/oh)."""
        if self._browser is None:
            return
        safe = str(name).replace('"', '\\"').replace("\n", "")[:4]
        v = max(0.0, min(1.0, float(value)))
        self._browser.page.evaluate(
            f'window.artemis && window.artemis.setViseme("{safe}", {v})'
        )

    def set_look_at(self, x: float, y: float) -> None:
        """Look direction in normalized eye-space, each axis -1..+1."""
        if self._browser is None:
            return
        cx = max(-1.0, min(1.0, float(x)))
        cy = max(-1.0, min(1.0, float(y)))
        self._browser.page.evaluate(
            f"window.artemis && window.artemis.setLookAt({cx}, {cy})"
        )

    def set_pose(self, index: int | None) -> None:
        """Pin a specific pose (0..N-1) or ``None`` to resume auto-cycle."""
        if self._browser is None:
            return
        arg = "null" if index is None else str(int(index))
        self._browser.page.evaluate(f"window.artemis && window.artemis.setPose({arg})")

    # ── internal ────────────────────────────────────────────────

    def _wait_for_ready(self, timeout_s: float = 20.0) -> None:
        assert self._browser is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                ok = self._browser.page.evaluate(
                    "window.artemis && window.artemis.ready === true"
                )
                if ok:
                    return
            except Exception as e:  # noqa: BLE001 — page may still be loading
                log.debug("ready-check retry: %s", e)
            time.sleep(0.2)
        log.warning("avatar3d: model did not ready in %.1fs", timeout_s)


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────


def _png_bytes_to_rgba(png_bytes: bytes, width: int, height: int) -> np.ndarray:
    """Decode PNG bytes → RGBA ndarray shaped (H, W, 4).

    We use Pillow here because it's already a runtime dep of the
    avatar process (for the HUD path). Keeps the renderer itself free
    of heavier decode libs.
    """
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


@contextmanager
def _default_playwright_launcher_ctx(width: int, height: int):  # pragma: no cover
    """Real Playwright launcher. Only used outside of tests.

    Returns a handle exposing ``.page`` and ``.close()``; closing the
    handle tears down the browser + context.
    """
    from playwright.sync_api import sync_playwright

    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-gl=swiftshader",
        ],
    )
    context = browser.new_context(
        viewport={"width": width, "height": height},
        device_scale_factor=1.0,
    )
    page = context.new_page()

    class _Handle:
        def __init__(self) -> None:
            self.page = page

        def close(self) -> None:
            for step in (
                context.close,
                browser.close,
                pw.stop,
            ):
                try:
                    step()
                except Exception as e:  # noqa: BLE001
                    log.warning("launcher close step failed: %s", e)

    try:
        yield _Handle()
    finally:
        pass


def _default_playwright_launcher(width: int, height: int) -> _BrowserHandle:
    """Non-context-manager wrapper around the real launcher.

    The renderer's ``__exit__`` calls ``.close()`` explicitly so we
    don't need the caller to use a ``with`` block here.
    """
    # Call the context manager manually so we can return a bare handle
    # whose ``.close()`` performs the teardown.
    cm = _default_playwright_launcher_ctx(width, height)
    handle = cm.__enter__()

    original_close = handle.close

    def close_and_exit() -> None:
        original_close()
        try:
            cm.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            log.warning("launcher context exit error: %s", e)

    handle.close = close_and_exit  # type: ignore[assignment]
    return handle


# Exposed for tests that want to hand-craft a tiny PNG without pulling
# in the real decoder path.
def _rgba_to_png_bytes(arr: np.ndarray) -> bytes:  # pragma: no cover — test helper
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


__all__ = [
    "AvatarRenderer",
    "PageLike",
    "BrowserLauncher",
    "_default_playwright_launcher",  # re-exported for the CLI
]


# ──────────────────────────────────────────────────────────────
# Smoke-test CLI — referenced by avatar3d.__main__
# ──────────────────────────────────────────────────────────────


def smoke(out_dir: str | Path, count: int = 30) -> Path:
    """Render ``count`` frames to ``out_dir/frame_NNN.png``.

    Returns the output directory. Intended for Phase B Atlas smoke
    tests once Playwright + Chromium are installed; safe to run
    locally too.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = SceneConfig()
    with AvatarRenderer(config=cfg) as r:
        for i, frame in enumerate(r.frames(count=count)):
            _save_rgba_png(frame, out / f"frame_{i:03d}.png")
    return out


def _save_rgba_png(arr: np.ndarray, path: Path) -> None:  # pragma: no cover — I/O only
    from PIL import Image

    Image.fromarray(arr, mode="RGBA").save(str(path), format="PNG")
