#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SVG comparison-chart generator for the Halyard wiki.

Renders side-by-side bar comparisons across multiple subjects
and metrics. Useful for:

- Weapon comparisons (range, ROF, damage)
- Faction-tonnage comparisons (UNN vs MCRN vs OPA)
- City-population comparisons
- Spaceship-class comparisons

Spec format (YAML/JSON)::

    title:    "Naval tonnage, 2347"
    subtitle: "Disclosed; capital + cruisers + frigates + corvettes"
    metric:   "Tonnage (millions)"
    subjects:
      - { name: "UNN",  value: 14.2, color: accent }
      - { name: "MCRN", value: 9.8,  color: panel }
      - { name: "Belt", value: 4.4,  color: dim }
    notes: "Source: 2347 UN Defense Disclosure; MCR White Paper"

Or *grouped* form for multi-metric comparison::

    title: "Service rifles compared"
    metrics:
      - "Range (m)"
      - "Rate of fire (rpm)"
      - "Mass (kg)"
    subjects:
      - { name: "M-340 (UNN)",  values: [600, 700, 3.4] }
      - { name: "RR-12 (MCRN)", values: [620, 720, 3.6] }
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
TEXT = "#e0e6f0"
TEXT_DIM = "#7a8ba8"
TEXT_MUTED = "#4a5568"

COLOR_BY_NAME = {
    "accent": ACCENT,
    "panel": PANEL2,
    "dim": LINE,
    "muted": TEXT_MUTED,
    "ok": "#4ade80",
    "warn": "#f59e0b",
    "err": "#f87171",
    "orange": "#fb923c",
}


def _color(name: str | None, default: str = ACCENT) -> str:
    if not name:
        return default
    if name.startswith("#") or name.startswith("rgb"):
        return name
    return COLOR_BY_NAME.get(name, default)


def _render_single(spec: dict, width: int, height: int) -> str:
    """Single-metric horizontal bar comparison."""
    title = spec.get("title", "Comparison")
    subtitle = spec.get("subtitle", "")
    metric = spec.get("metric", "")
    subjects = spec.get("subjects", [])
    notes = spec.get("notes", "")

    if not subjects:
        raise ValueError("spec must include 'subjects'")

    max_val = max(float(s["value"]) for s in subjects)
    bar_max_w = width - 280
    row_h = 36
    top = 80
    label_x = 30
    bar_x = 200

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Consolas, monospace">',
        f'  <rect width="{width}" height="{height}" fill="{BG}"/>',
        f'  <rect x="0" y="0" width="{width}" height="40" fill="{PANEL}"/>',
        f'  <rect x="0" y="40" width="{width}" height="2" fill="{ACCENT}"/>',
        f'  <text x="16" y="26" fill="{ACCENT}" font-size="15" font-weight="bold" letter-spacing="1">{title}</text>',
        f'  <text x="{width - 16}" y="26" fill="{TEXT_DIM}" font-size="11" text-anchor="end">{subtitle}</text>',
    ]
    if metric:
        parts.append(
            f'  <text x="{bar_x}" y="62" fill="{TEXT_DIM}" font-size="10" letter-spacing="1">{metric.upper()}</text>'
        )

    for i, s in enumerate(subjects):
        y = top + i * row_h
        name = s.get("name", "")
        val = float(s["value"])
        bar_w = int((val / max_val) * bar_max_w) if max_val > 0 else 0
        color = _color(s.get("color"), ACCENT)
        # name
        parts.append(
            f'  <text x="{label_x}" y="{y + 18}" fill="{TEXT}" font-size="12">{name}</text>'
        )
        # bar background
        parts.append(
            f'  <rect x="{bar_x}" y="{y + 6}" width="{bar_max_w}" height="20" '
            f'fill="{PANEL}" stroke="{TEXT_MUTED}" stroke-width="0.6"/>'
        )
        # bar fill
        parts.append(
            f'  <rect x="{bar_x}" y="{y + 6}" width="{bar_w}" height="20" '
            f'fill="{color}"/>'
        )
        # value text right of bar
        parts.append(
            f'  <text x="{bar_x + bar_w + 8}" y="{y + 21}" fill="{TEXT_DIM}" font-size="11">{val}</text>'
        )

    if notes:
        parts.append(
            f'  <text x="16" y="{height - 14}" fill="{TEXT_MUTED}" font-size="9" font-style="italic">{notes}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _render_grouped(spec: dict, width: int, height: int) -> str:
    """Multi-metric grouped bar comparison.

    Each subject occupies a row; each metric is a small bar within
    the row. Suitable for ~3-4 metrics across 2-5 subjects.
    """
    title = spec.get("title", "Comparison")
    subtitle = spec.get("subtitle", "")
    metrics = spec.get("metrics", [])
    subjects = spec.get("subjects", [])
    notes = spec.get("notes", "")

    if not metrics or not subjects:
        raise ValueError("grouped form requires 'metrics' and 'subjects'")

    n_metrics = len(metrics)
    # Per-metric max for normalization
    maxes = [
        max(float(s["values"][m]) for s in subjects) for m in range(n_metrics)
    ]

    row_h = 60
    top = 90
    label_x = 30
    chart_x = 220
    chart_w = width - chart_x - 40
    metric_h = 14

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Consolas, monospace">',
        f'  <rect width="{width}" height="{height}" fill="{BG}"/>',
        f'  <rect x="0" y="0" width="{width}" height="40" fill="{PANEL}"/>',
        f'  <rect x="0" y="40" width="{width}" height="2" fill="{ACCENT}"/>',
        f'  <text x="16" y="26" fill="{ACCENT}" font-size="15" font-weight="bold" letter-spacing="1">{title}</text>',
        f'  <text x="{width - 16}" y="26" fill="{TEXT_DIM}" font-size="11" text-anchor="end">{subtitle}</text>',
    ]

    # Metric labels on top
    for j, m in enumerate(metrics):
        col_x = chart_x
        col_y = 60 + j * 12
        parts.append(
            f'  <text x="{col_x}" y="{col_y}" fill="{TEXT_DIM}" font-size="9" letter-spacing="1">{m.upper()}</text>'
        )

    for i, s in enumerate(subjects):
        y = top + i * row_h
        name = s.get("name", "")
        parts.append(
            f'  <text x="{label_x}" y="{y + 22}" fill="{TEXT}" font-size="12">{name}</text>'
        )
        for j in range(n_metrics):
            val = float(s["values"][j])
            mx = maxes[j] or 1
            bar_w = int((val / mx) * chart_w)
            by = y + j * (metric_h + 2)
            color = _color(s.get("color"), ACCENT) if j == 0 else (
                _color(s.get("color"), ACCENT) if j == 0 else "#3a4666"
            )
            # Per-metric bars use a varying-saturation pattern for visual distinction
            metric_colors = [ACCENT, "#7aa6d8", "#a8c4e3", "#c8d8eb"]
            color = metric_colors[j % len(metric_colors)]
            parts.append(
                f'  <rect x="{chart_x}" y="{by}" width="{chart_w}" height="{metric_h}" '
                f'fill="{PANEL}" stroke="{TEXT_MUTED}" stroke-width="0.6"/>'
            )
            parts.append(
                f'  <rect x="{chart_x}" y="{by}" width="{bar_w}" height="{metric_h}" fill="{color}"/>'
            )
            parts.append(
                f'  <text x="{chart_x + bar_w + 6}" y="{by + 11}" fill="{TEXT_DIM}" font-size="9">{val}</text>'
            )

    if notes:
        parts.append(
            f'  <text x="16" y="{height - 14}" fill="{TEXT_MUTED}" font-size="9" font-style="italic">{notes}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render(spec: dict, width: int = 900, height: int | None = None) -> str:
    if "metrics" in spec:
        h = height or 120 + len(spec["subjects"]) * 60
        return _render_grouped(spec, width=width, height=h)
    h = height or 100 + len(spec["subjects"]) * 36 + 40
    return _render_single(spec, width=width, height=h)


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
    parser.add_argument("--width", type=int, default=900)
    args = parser.parse_args()
    spec = _load(Path(args.spec))
    svg = render(spec, width=args.width)
    Path(args.out).write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
