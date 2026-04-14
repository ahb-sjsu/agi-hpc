#!/usr/bin/env python3
"""Generate PNG + favicon.ico exports from the brand SVG masters.

Usage::

    pip install cairosvg pillow
    python docs/brand/export.py

Outputs under ``docs/brand/exports/``. Regenerate rather than committing
these to keep the repo light.

Copyright (c) 2026 Andrew H. Bond. All rights reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import cairosvg
    from PIL import Image
except ImportError:
    sys.stderr.write(
        "This script requires cairosvg and pillow.\n"
        "Install with: pip install cairosvg pillow\n"
    )
    sys.exit(1)


HERE = Path(__file__).resolve().parent
OUT = HERE / "exports"
OUT.mkdir(exist_ok=True)


# (source SVG, basename, list of PNG widths to render)
PNG_TARGETS = [
    ("atlas_mark.svg", "atlas_mark_dark", [1024, 512, 256, 128, 64]),
    ("atlas_mark_light.svg", "atlas_mark_light", [1024, 512, 256, 128, 64]),
    ("atlas_sphere.svg", "atlas_sphere_dark", [1024, 512, 256, 128, 64, 32, 16]),
    ("atlas_sphere_light.svg", "atlas_sphere_light", [1024, 512, 256, 128, 64, 32, 16]),
    ("erisml_apple.svg", "erisml_apple_dark", [512, 256, 128, 64, 32]),
    ("erisml_apple_light.svg", "erisml_apple_light", [512, 256, 128, 64, 32]),
]


def render_png(svg_path: Path, out_path: Path, width: int) -> None:
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(out_path),
        output_width=width,
        output_height=width,
    )
    print(f"  wrote {out_path.relative_to(HERE)}")


def build_favicon() -> None:
    """Build favicon.ico from atlas_sphere (multi-size)."""
    sizes = [16, 32, 48]
    sources = []
    for size in sizes:
        tmp = OUT / f"_favicon_tmp_{size}.png"
        render_png(HERE / "atlas_sphere.svg", tmp, size)
        sources.append(Image.open(tmp))
    favicon_path = OUT / "favicon.ico"
    sources[0].save(
        favicon_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=sources[1:],
    )
    print(f"  wrote {favicon_path.relative_to(HERE)}")
    for size in sizes:
        (OUT / f"_favicon_tmp_{size}.png").unlink(missing_ok=True)


def main() -> int:
    print("Rendering PNGs...")
    for src, name, widths in PNG_TARGETS:
        svg_path = HERE / src
        if not svg_path.exists():
            print(f"  skip: {src} missing")
            continue
        for w in widths:
            render_png(svg_path, OUT / f"{name}_{w}.png", w)

    print("\nBuilding favicon.ico...")
    build_favicon()

    print(f"\nDone. Outputs in {OUT.relative_to(HERE.parent.parent)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
