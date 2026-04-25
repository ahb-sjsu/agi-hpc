#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SVG organizational-chart generator for the Halyard wiki.

Renders hierarchical org charts (factions, agencies, command
structures) as clean line-art SVGs sized for inline markdown
use.

Spec format (YAML/JSON)::

    title:    "United Nations Navy — structure"
    subtitle: "UNN, 2348"
    nodes:
      - id:    sec
        label: "Secretary-General"
        kind:  exec
      - id:    cnav
        label: "Chief of Naval Operations"
        kind:  exec
        parent: sec
      - id:    isc
        label: "Inner System Command"
        kind:  branch
        parent: cnav
      ...

``kind`` values:
- exec    — top-level / executive (filled with accent)
- branch  — major subdivision
- unit    — operational unit
- stub    — small / placeholder

Layout: a simple top-down tree. Children of each parent are
spaced evenly horizontally. Depth >= 4 not recommended for
inline-markdown sizing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Palette matches svg_weapon_schematic.py
BG = "#0a0e17"
PANEL = "#131a2b"
LINE = "#7a8ba8"
ACCENT = "#4a9eff"
TEXT = "#e0e6f0"
TEXT_DIM = "#7a8ba8"

KIND_FILL = {
    "exec": ACCENT,
    "branch": "#1a2240",
    "unit": "#131a2b",
    "stub": "#0f1422",
}
KIND_STROKE = {
    "exec": ACCENT,
    "branch": LINE,
    "unit": LINE,
    "stub": "#4a5568",
}
KIND_TEXT = {
    "exec": "#0a0e17",
    "branch": TEXT,
    "unit": TEXT,
    "stub": TEXT_DIM,
}


def _build_tree(nodes: list[dict]) -> tuple[dict, list]:
    """Return (by_id, roots). Roots are nodes with no parent."""
    by_id = {n["id"]: dict(n, children=[]) for n in nodes}
    roots = []
    for n in nodes:
        parent = n.get("parent")
        if parent and parent in by_id:
            by_id[parent]["children"].append(by_id[n["id"]])
        else:
            roots.append(by_id[n["id"]])
    return by_id, roots


def _layout(roots: list, width: int, top: int, level_h: int) -> None:
    """Recursively assign x,y coordinates. Walks the tree breadth-first
    by depth and spreads siblings evenly within their parent's range."""

    def measure(node) -> int:
        """Return the leaf-count under this node (subtree width)."""
        if not node["children"]:
            return 1
        return sum(measure(c) for c in node["children"])

    def assign(node, x_start: int, x_end: int, y: int) -> None:
        node["x"] = (x_start + x_end) // 2
        node["y"] = y
        if not node["children"]:
            return
        leaf_total = sum(measure(c) for c in node["children"])
        cursor = x_start
        slot = (x_end - x_start) / max(leaf_total, 1)
        for c in node["children"]:
            cw = measure(c) * slot
            assign(c, int(cursor), int(cursor + cw), y + level_h)
            cursor += cw

    if not roots:
        return
    leaf_total = sum(measure(r) for r in roots)
    margin = 24
    available = width - margin * 2
    cursor = margin
    slot = available / max(leaf_total, 1)
    for r in roots:
        rw = measure(r) * slot
        assign(r, int(cursor), int(cursor + rw), top)
        cursor += rw


def _node_svg(n: dict) -> str:
    kind = n.get("kind", "unit")
    fill = KIND_FILL.get(kind, KIND_FILL["unit"])
    stroke = KIND_STROKE.get(kind, KIND_STROKE["unit"])
    text_color = KIND_TEXT.get(kind, KIND_TEXT["unit"])
    label = n.get("label", n["id"])
    # Node box centered on (x, y)
    box_w = 150
    box_h = 36
    bx = n["x"] - box_w // 2
    by = n["y"] - box_h // 2
    # Wrap label if long
    if len(label) > 22:
        # split on space nearest middle
        mid = len(label) // 2
        for offset in range(0, mid):
            for i in (mid - offset, mid + offset):
                if 0 < i < len(label) and label[i] == " ":
                    line1 = label[:i]
                    line2 = label[i + 1 :]
                    break
            else:
                continue
            break
        else:
            line1 = label[:22]
            line2 = label[22:]
        text_lines = (
            f'<text x="{n["x"]}" y="{n["y"] - 2}" fill="{text_color}" font-size="10" text-anchor="middle">{line1}</text>'
            f'<text x="{n["x"]}" y="{n["y"] + 12}" fill="{text_color}" font-size="10" text-anchor="middle">{line2}</text>'
        )
    else:
        text_lines = (
            f'<text x="{n["x"]}" y="{n["y"] + 4}" fill="{text_color}" font-size="11" text-anchor="middle">{label}</text>'
        )
    return (
        f'<rect x="{bx}" y="{by}" width="{box_w}" height="{box_h}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" rx="3"/>'
        f"{text_lines}"
    )


def _edge_svg(parent: dict, child: dict) -> str:
    px, py = parent["x"], parent["y"] + 18
    cx, cy = child["x"], child["y"] - 18
    midy = (py + cy) // 2
    return (
        f'<path d="M {px} {py} L {px} {midy} L {cx} {midy} L {cx} {cy}" '
        f'fill="none" stroke="{LINE}" stroke-width="1.2"/>'
    )


def render(spec: dict, width: int = 1100, height: int = 600) -> str:
    title = spec.get("title", "Organization")
    subtitle = spec.get("subtitle", "")
    nodes_in = spec.get("nodes", [])
    if not nodes_in:
        raise ValueError("spec must include 'nodes'")

    by_id, roots = _build_tree(nodes_in)
    _layout(roots, width=width, top=86, level_h=92)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Consolas, monospace">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        f'<rect x="0" y="0" width="{width}" height="38" fill="{PANEL}"/>',
        f'<rect x="0" y="38" width="{width}" height="2" fill="{ACCENT}"/>',
        f'<text x="16" y="24" fill="{ACCENT}" font-size="15" font-weight="bold" letter-spacing="1">{title}</text>',
        f'<text x="{width - 16}" y="24" fill="{TEXT_DIM}" font-size="10" text-anchor="end">{subtitle}</text>',
    ]
    # edges first so nodes render on top
    for n in by_id.values():
        for c in n["children"]:
            parts.append(_edge_svg(n, c))
    for n in by_id.values():
        parts.append(_node_svg(n))
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
    parser.add_argument("--spec", required=True, help="YAML/JSON spec file.")
    parser.add_argument("--out", required=True, help="Output SVG path.")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    spec = _load(Path(args.spec))
    svg = render(spec, width=args.width, height=args.height)
    Path(args.out).write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
