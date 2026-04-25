#!/usr/bin/env python3
# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""SVG weapon-schematic generator for the Halyard wiki.

Produces clean line-art schematic illustrations of weapons,
sized for inline use in markdown (default 800x300 px). The
output is a self-contained SVG with the weapon's silhouette,
labeled callouts for major features, and a small spec table.

The generator is rule-based; given a weapon's spec dict, it
builds the SVG by composing primitive shape components
(barrel, receiver, grip, stock, magazine, sight). This is
deliberately rough — these are *campaign reference
illustrations*, not catalog-grade gun-porn art. The goal is
to give players a quick visual handle on each weapon class.

Usage::

    # From a spec file (YAML or JSON list)
    python svg_weapon_schematic.py --specs weapons.yaml --outdir art/

    # Inline single weapon (for testing)
    python svg_weapon_schematic.py --kind rifle --name M-340 \
        --caliber "5.56mm" --out test.svg

Spec dict shape::

    name:    "M-340"
    kind:    "rifle"     # rifle | pistol | shotgun | smg
    caliber: "5.56mm"
    role:    "UNN service rifle"
    capacity: 30
    features: ["smart-sight", "burst"]   # optional
    notes:   "..."                       # optional, footer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Color palette — matches the Halyard table aesthetic
# ─────────────────────────────────────────────────────────────────

BG = "#0a0e17"
PANEL = "#131a2b"
LINE = "#7a8ba8"
ACCENT = "#4a9eff"
TEXT = "#e0e6f0"
TEXT_DIM = "#7a8ba8"
TEXT_MUTED = "#4a5568"


def _header(title: str, subtitle: str, width: int, height: int) -> str:
    """SVG header with background and title strip."""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="Consolas, monospace">
  <rect width="{width}" height="{height}" fill="{BG}"/>
  <rect x="0" y="0" width="{width}" height="34" fill="{PANEL}"/>
  <rect x="0" y="34" width="{width}" height="2" fill="{ACCENT}"/>
  <text x="14" y="22" fill="{ACCENT}" font-size="14" font-weight="bold" letter-spacing="1">{title}</text>
  <text x="{width - 14}" y="22" fill="{TEXT_DIM}" font-size="10" text-anchor="end">{subtitle}</text>
"""


def _footer() -> str:
    return "</svg>\n"


def _footnote(text: str, width: int, y: int) -> str:
    return f'  <text x="14" y="{y}" fill="{TEXT_MUTED}" font-size="9" font-style="italic">{text}</text>\n'


def _spec_row(label: str, value: str, x: int, y: int) -> str:
    return (
        f'  <text x="{x}" y="{y}" fill="{TEXT_DIM}" font-size="9" letter-spacing="1">{label.upper()}</text>\n'
        f'  <text x="{x + 80}" y="{y}" fill="{TEXT}" font-size="11">{value}</text>\n'
    )


def _callout(label: str, x: int, y: int, anchor: str = "start") -> str:
    return f'  <text x="{x}" y="{y}" fill="{ACCENT}" font-size="9" letter-spacing="1" text-anchor="{anchor}">{label.upper()}</text>\n'


# ─────────────────────────────────────────────────────────────────
# Weapon silhouettes — each is a composition of primitive shapes
# ─────────────────────────────────────────────────────────────────


def _silhouette_rifle(cx: int, cy: int) -> tuple[str, dict[str, tuple[int, int]]]:
    """Silhouette of a generic service rifle. Returns (svg, label_anchors)."""
    # Anchors are coordinates we'll attach callouts to.
    parts = []
    # barrel
    parts.append(f'<rect x="{cx + 60}" y="{cy - 4}" width="200" height="8" fill="{LINE}"/>')
    # muzzle device
    parts.append(f'<rect x="{cx + 260}" y="{cy - 7}" width="14" height="14" fill="{LINE}"/>')
    # receiver
    parts.append(f'<rect x="{cx - 30}" y="{cy - 14}" width="90" height="28" fill="{LINE}" rx="2"/>')
    # sight rail
    parts.append(f'<rect x="{cx - 20}" y="{cy - 22}" width="200" height="6" fill="{LINE}"/>')
    # smart sight
    parts.append(f'<rect x="{cx + 40}" y="{cy - 38}" width="50" height="22" fill="none" stroke="{ACCENT}" stroke-width="1.5" rx="2"/>')
    parts.append(f'<line x1="{cx + 65}" y1="{cy - 38}" x2="{cx + 65}" y2="{cy - 22}" stroke="{ACCENT}" stroke-width="1"/>')
    parts.append(f'<line x1="{cx + 40}" y1="{cy - 30}" x2="{cx + 90}" y2="{cy - 30}" stroke="{ACCENT}" stroke-width="1"/>')
    # magazine
    parts.append(f'<polygon points="{cx - 20},{cy + 14} {cx + 20},{cy + 14} {cx + 14},{cy + 60} {cx - 14},{cy + 60}" fill="{LINE}"/>')
    # grip
    parts.append(f'<polygon points="{cx - 50},{cy + 14} {cx - 30},{cy + 14} {cx - 36},{cy + 56} {cx - 56},{cy + 56}" fill="{LINE}"/>')
    # stock
    parts.append(f'<polygon points="{cx - 30},{cy - 14} {cx - 110},{cy - 8} {cx - 120},{cy + 12} {cx - 60},{cy + 14}" fill="{LINE}"/>')
    # cheek-rest line
    parts.append(f'<line x1="{cx - 30}" y1="{cy - 4}" x2="{cx - 100}" y2="{cy + 2}" stroke="{BG}" stroke-width="1"/>')

    anchors = {
        "barrel": (cx + 160, cy - 4),
        "sight": (cx + 65, cy - 42),
        "receiver": (cx + 15, cy - 14),
        "magazine": (cx, cy + 60),
        "grip": (cx - 46, cy + 60),
        "stock": (cx - 90, cy + 14),
    }
    return "\n".join("  " + p for p in parts), anchors


def _silhouette_pistol(cx: int, cy: int) -> tuple[str, dict[str, tuple[int, int]]]:
    parts = []
    # slide
    parts.append(f'<rect x="{cx - 60}" y="{cy - 18}" width="130" height="20" fill="{LINE}" rx="2"/>')
    # barrel tip
    parts.append(f'<rect x="{cx + 70}" y="{cy - 14}" width="14" height="12" fill="{LINE}"/>')
    # frame / trigger guard
    parts.append(f'<polygon points="{cx - 40},{cy + 2} {cx + 50},{cy + 2} {cx + 30},{cy + 22} {cx - 18},{cy + 22} {cx - 28},{cy + 18}" fill="{LINE}"/>')
    # grip
    parts.append(f'<polygon points="{cx - 28},{cy + 18} {cx + 12},{cy + 18} {cx + 4},{cy + 78} {cx - 36},{cy + 78}" fill="{LINE}"/>')
    # sights
    parts.append(f'<rect x="{cx - 56}" y="{cy - 24}" width="6" height="6" fill="{LINE}"/>')
    parts.append(f'<rect x="{cx + 60}" y="{cy - 24}" width="6" height="6" fill="{LINE}"/>')
    # magazine base
    parts.append(f'<rect x="{cx - 38}" y="{cy + 76}" width="44" height="6" fill="{LINE}"/>')

    anchors = {
        "slide": (cx + 5, cy - 18),
        "barrel": (cx + 75, cy - 18),
        "trigger": (cx + 6, cy + 22),
        "grip": (cx - 16, cy + 78),
        "sights": (cx + 63, cy - 28),
    }
    return "\n".join("  " + p for p in parts), anchors


def _silhouette_shotgun(cx: int, cy: int) -> tuple[str, dict[str, tuple[int, int]]]:
    parts = []
    # barrel — long, fatter
    parts.append(f'<rect x="{cx + 30}" y="{cy - 6}" width="240" height="12" fill="{LINE}"/>')
    # bead sight
    parts.append(f'<rect x="{cx + 264}" y="{cy - 11}" width="6" height="5" fill="{LINE}"/>')
    # receiver
    parts.append(f'<rect x="{cx - 40}" y="{cy - 16}" width="70" height="32" fill="{LINE}" rx="2"/>')
    # pump fore-end
    parts.append(f'<rect x="{cx + 30}" y="{cy + 4}" width="80" height="14" fill="{LINE}" rx="2"/>')
    # grip + stock (one piece, traditional shotgun)
    parts.append(f'<polygon points="{cx - 40},{cy + 16} {cx - 28},{cy + 16} {cx - 110},{cy + 50} {cx - 130},{cy + 36} {cx - 130},{cy + 22} {cx - 60},{cy - 12} {cx - 40},{cy - 16}" fill="{LINE}"/>')

    anchors = {
        "barrel": (cx + 150, cy - 6),
        "receiver": (cx + 15, cy - 16),
        "pump": (cx + 70, cy + 22),
        "stock": (cx - 80, cy + 50),
    }
    return "\n".join("  " + p for p in parts), anchors


SILHOUETTES = {
    "rifle": _silhouette_rifle,
    "pistol": _silhouette_pistol,
    "shotgun": _silhouette_shotgun,
    "smg": _silhouette_rifle,  # close enough for now
}


# ─────────────────────────────────────────────────────────────────
# Compose a full schematic
# ─────────────────────────────────────────────────────────────────


def render(spec: dict, width: int = 800, height: int = 320) -> str:
    """Generate the full schematic SVG."""
    name = str(spec.get("name", "weapon")).strip()
    kind = str(spec.get("kind", "rifle")).lower().strip()
    role = str(spec.get("role", "")).strip()

    if kind not in SILHOUETTES:
        kind = "rifle"

    title = name
    subtitle = role.upper() if role else kind.upper()

    out = [_header(title, subtitle, width, height)]

    # Silhouette in upper portion
    cx = width // 2
    cy = 130
    sil, anchors = SILHOUETTES[kind](cx, cy)
    out.append(sil)

    # Callouts — connect anchors to label text via short lines
    callout_y = 70
    # Build anchor → label map from features list
    features = spec.get("features", [])
    callout_specs = []
    for label, (ax, ay) in anchors.items():
        callout_specs.append((label, ax, ay))

    # Render callouts above and below the silhouette as needed
    for i, (label, ax, ay) in enumerate(callout_specs):
        if ay < cy:  # above
            ly = callout_y - 4
            out.append(
                f'  <line x1="{ax}" y1="{ay}" x2="{ax}" y2="{ly + 2}" stroke="{ACCENT}" stroke-width="0.7"/>\n'
            )
            out.append(_callout(label, ax, ly, anchor="middle"))
        else:  # below — leader line points down
            pass  # below-the-silhouette callouts not used in v1

    # Spec table at bottom
    spec_y = height - 90
    out.append(
        f'  <rect x="0" y="{spec_y - 18}" width="{width}" height="2" fill="{ACCENT}" opacity="0.4"/>\n'
    )
    rows = [
        ("name", name),
        ("kind", kind.upper()),
        ("caliber", str(spec.get("caliber", "n/a"))),
        ("capacity", str(spec.get("capacity", "n/a"))),
    ]
    if role:
        rows.append(("role", role))
    if features:
        rows.append(("features", ", ".join(features)))

    col_x = [14, width // 2 + 20]
    for i, (label, value) in enumerate(rows):
        x = col_x[i % 2]
        y = spec_y + (i // 2) * 18
        out.append(_spec_row(label, value, x, y))

    if spec.get("notes"):
        out.append(_footnote(spec["notes"], width, height - 14))

    out.append(_footer())
    return "".join(out)


def _load_specs(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            print("PyYAML not installed; install or use JSON.", file=sys.stderr)
            sys.exit(2)
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, list):
        print("Specs file must be a list.", file=sys.stderr)
        sys.exit(2)
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", help="YAML/JSON list of weapon specs.")
    parser.add_argument(
        "--outdir",
        default="wiki/halyard/art/weapons",
        help="Output directory for batch SVG files.",
    )
    # Single-render shortcuts (for ad-hoc testing)
    parser.add_argument("--kind")
    parser.add_argument("--name")
    parser.add_argument("--caliber")
    parser.add_argument("--role")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.specs:
        specs = _load_specs(Path(args.specs))
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            slug = str(spec.get("name", "weapon")).lower().replace(" ", "-").replace("/", "-")
            out_path = outdir / f"{slug}.svg"
            out_path.write_text(render(spec), encoding="utf-8")
            print(f"wrote {out_path}", flush=True)
        return 0

    if not args.name or not args.out:
        parser.error("Need --specs OR --name + --out (and --kind, --caliber).")
    spec = {
        "name": args.name,
        "kind": args.kind or "rifle",
        "caliber": args.caliber or "n/a",
        "role": args.role or "",
    }
    Path(args.out).write_text(render(spec), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
