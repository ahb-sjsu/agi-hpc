"""
Shared style + drawing helpers for the PyTorch Conf NA 2026 poster diagram assets.

Pure matplotlib (no graphviz / no external deps). Every diagram is emitted as
vector SVG *and* PDF so it prints crisp at 48 inches and stays editable in
Illustrator / Inkscape / Figma.

Palette is the matched-pair system from the LAYOUT specs:
    Poster #1 (Applications)    accent = Atlas teal   #0E7C86
    Poster #2 (Responsible AI)  accent = Eris violet  #6A4C93
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------------- palette
INK        = "#1A1A1A"
PAPER      = "#FFFFFF"
TEAL       = "#0E7C86"   # Poster #1 accent
TEAL_FILL  = "#F4F7F8"
TEAL_RULE  = "#C9D6D8"
VIOLET     = "#6A4C93"   # Poster #2 accent
VIOLET_FILL= "#F5F3F8"
VIOLET_RULE= "#D4CCE0"
GOOD       = "#1B7F4B"   # pass / allow / prefer
BAD        = "#B3261E"   # fail / block / forbid
WARN       = "#C77800"   # warn / neutral
MUTE       = "#6B6B6B"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",     # keep text as text in the SVG (editable, selectable)
    "pdf.fonttype": 42,         # TrueType in PDF
    "figure.dpi": 150,
})


def canvas(w_in, h_in):
    """Return (fig, ax) with a clean coordinate space of 0..100 x, 0..(100*h/w) y."""
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / w_in)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def box(ax, x, y, w, h, text, *, fc=PAPER, ec=INK, tc=INK, lw=2.2,
        fontsize=13, weight="normal", style="round", pad=0.02, dashed=False,
        align="center", round_size=0.6):
    """Rounded box centered at (x, y) with width w / height h. Returns (x, y)."""
    boxstyle = f"round,pad={pad},rounding_size={round_size}" if style == "round" else "square,pad=%s" % pad
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=boxstyle, fc=fc, ec=ec, lw=lw,
        linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(p)
    ha = {"center": "center", "left": "left", "right": "right"}[align]
    tx = x if align == "center" else (x - w / 2 + 1.4 if align == "left" else x + w / 2 - 1.4)
    ax.text(tx, y, text, ha=ha, va="center", fontsize=fontsize, color=tc,
            weight=weight, zorder=3, linespacing=1.25)
    return (x, y)


def arrow(ax, p1, p2, *, color=INK, lw=2.4, style="-|>", mut=18, ls="-", connect=None):
    """Arrow from point p1 to p2."""
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=mut, color=color, lw=lw,
        linestyle=ls, shrinkA=2, shrinkB=2, zorder=1,
        connectionstyle=connect or "arc3,rad=0",
    )
    ax.add_patch(a)


def label(ax, x, y, text, *, fontsize=12, color=INK, weight="normal",
          ha="center", va="center", style="normal", rotation=0):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            weight=weight, style=style, rotation=rotation, zorder=4, linespacing=1.25)


def chip(ax, x, y, text, *, fc, ec=None, tc=PAPER, fontsize=11, w=None, h=4.2):
    ec = ec or fc
    w = w or (len(text) * 0.95 + 4)
    box(ax, x, y, w, h, text, fc=fc, ec=ec, tc=tc, lw=1.6, fontsize=fontsize,
        weight="bold", round_size=1.2)


def title_strip(ax, x, y, text, accent, *, fontsize=15):
    """A small accent header bar for a panel."""
    label(ax, x, y, text, fontsize=fontsize, color=accent, weight="bold", ha="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    svg = os.path.join(OUT, name + ".svg")
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(svg, transparent=True)
    fig.savefig(pdf, transparent=True)
    fig.savefig(png, dpi=150, facecolor=PAPER)   # opaque preview
    plt.close(fig)
    print(f"  wrote {name}.svg + .pdf + .png")
    return svg
