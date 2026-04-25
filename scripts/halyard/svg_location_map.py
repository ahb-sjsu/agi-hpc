#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SVG location-map generator for the Halyard wiki.

Produces top-down schematic maps of cities, stations, and other
locations. The output is a clean-line SVG with labeled regions,
suitable for inline-markdown use.

The generator is region-based: a spec defines a set of regions
(rectangles, polygons, or circles) with labels. The generator
arranges them on a canvas with a title strip and a small legend.

Spec format (YAML/JSON)::

    title: "Hellas City — schematic"
    subtitle: "Mars; capital of the Republic"
    width: 1000          # optional; default 1000
    height: 700          # optional; default 700
    background:
      kind: "lake"       # optional decorative background; lake|panel|none
    regions:
      - kind: rect       # rect | circle | polygon
        x: 100
        y: 200
        w: 300
        h: 200
        label: "Founders' Crescent"
        accent: true     # optional; highlights the region
      - kind: circle
        cx: 500
        cy: 350
        r: 40
        label: "Capitol"
        accent: true
      - kind: polygon
        points: "100,400 200,500 300,400 200,300"
        label: "Mid-Crescent"
    legend:
      - { color: accent, label: "Government" }
      - { color: panel,  label: "Residential" }
      - { color: dim,    label: "Industrial" }

The map is *schematic*, not topographic. It communicates
relative position and named features, not precise geography.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BG = "#0a0e17"
PANEL = "#131a2b"
PANEL2 = "#1a2240"
LINE = "#7a8ba8"
ACCENT = "#4a9eff"
ACCENT_DIM = "rgba(74,158,255,0.25)"
TEXT = "#e0e6f0"
TEXT_DIM = "#7a8ba8"
TEXT_MUTED = "#4a5568"


COLOR_BY_NAME = {
    "accent": ACCENT,
    "panel": PANEL2,
    "panel2": PANEL2,
    "dim": LINE,
    "muted": TEXT_MUTED,
    "lake": "#15294a",
    "border": LINE,
    "text": TEXT,
}


def _resolve_color(name: str | None, default: str = LINE) -> str:
    if not name:
        return default
    if name.startswith("#") or name.startswith("rgb"):
        return name
    return COLOR_BY_NAME.get(name, default)


def _bg(width: int, height: int, kind: str | None) -> str:
    if kind == "lake":
        return f'<rect width="{width}" height="{height}" fill="{BG}"/>\n  <rect x="20" y="60" width="{width - 40}" height="{height - 80}" fill="#15294a" rx="6"/>'
    if kind == "panel":
        return f'<rect width="{width}" height="{height}" fill="{BG}"/>\n  <rect x="20" y="60" width="{width - 40}" height="{height - 80}" fill="{PANEL}" rx="6"/>'
    return f'<rect width="{width}" height="{height}" fill="{BG}"/>'


def _region_svg(r: dict) -> str:
    kind = r.get("kind", "rect")
    label = r.get("label", "")
    accent = bool(r.get("accent", False))
    fill = _resolve_color(r.get("fill"), ACCENT_DIM if accent else PANEL2)
    stroke = ACCENT if accent else LINE
    stroke_w = "2" if accent else "1.2"

    if kind == "rect":
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cx = x + w // 2
        cy = y + h // 2
        shape = (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}" rx="2"/>'
        )
    elif kind == "circle":
        cx = r["cx"]
        cy = r["cy"]
        rad = r["r"]
        shape = (
            f'<circle cx="{cx}" cy="{cy}" r="{rad}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        )
    elif kind == "polygon":
        points = r["points"]
        # crude centroid: average of all points
        coords = [c.strip().split(",") for c in points.split()]
        xs = [float(c[0]) for c in coords]
        ys = [float(c[1]) for c in coords]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        shape = (
            f'<polygon points="{points}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
        )
    else:
        return f"<!-- unknown region kind: {kind} -->"

    text_color = ACCENT if accent else TEXT_DIM
    label_svg = ""
    if label:
        label_svg = (
            f'\n  <text x="{cx}" y="{cy + 4}" fill="{text_color}" '
            f'font-size="11" text-anchor="middle">{label}</text>'
        )
    return f"  {shape}{label_svg}"


def _legend_svg(legend: list, width: int, y: int) -> str:
    if not legend:
        return ""
    parts = [
        f'<rect x="20" y="{y - 18}" width="{width - 40}" height="32" '
        f'fill="{PANEL}" stroke="none" rx="3"/>'
    ]
    x = 40
    for item in legend:
        color = _resolve_color(item.get("color"), LINE)
        label = item.get("label", "")
        parts.append(
            f'<rect x="{x}" y="{y - 9}" width="14" height="14" '
            f'fill="{color}" stroke="{LINE}" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{x + 22}" y="{y + 2}" fill="{TEXT_DIM}" font-size="10">{label}</text>'
        )
        x += 22 + 8 * len(label) + 32
    return "\n  ".join(parts)


def render(spec: dict) -> str:
    width = spec.get("width", 1000)
    height = spec.get("height", 700)
    title = spec.get("title", "Location")
    subtitle = spec.get("subtitle", "")
    bg_kind = (spec.get("background") or {}).get("kind") if isinstance(spec.get("background"), dict) else spec.get("background")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Consolas, monospace">',
        f"  {_bg(width, height, bg_kind)}",
        f'  <rect x="0" y="0" width="{width}" height="40" fill="{PANEL}"/>',
        f'  <rect x="0" y="40" width="{width}" height="2" fill="{ACCENT}"/>',
        f'  <text x="16" y="26" fill="{ACCENT}" font-size="16" font-weight="bold" letter-spacing="1">{title}</text>',
        f'  <text x="{width - 16}" y="26" fill="{TEXT_DIM}" font-size="11" text-anchor="end">{subtitle}</text>',
    ]

    for r in spec.get("regions", []):
        parts.append(_region_svg(r))

    legend = spec.get("legend", [])
    if legend:
        parts.append("  " + _legend_svg(legend, width, height - 14))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            print("PyYAML required.", file=sys.stderr)
            sys.exit(2)
        return yaml.safe_load(text)
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    spec = _load(Path(args.spec))
    svg = render(spec)
    Path(args.out).write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
