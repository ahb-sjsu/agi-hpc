"""
Full 48x36 in landscape poster board for POSTER #2 (Responsible AI) —
"Moral Tensors and DecisionProofs: Compiling Language into an Auditable,
Grounded Safety Layer" — PyTorch Conference North America 2026.

Composes the six panel diagrams from build_poster2.py (drawn live as vectors,
not embedded images), plus title bar, hero band, footer strip and QR codes,
into a single print-ready PDF per pytorch-poster-2-LAYOUT.md.

Output: out/p2_board_48x36.pdf (vector, fonts embedded as TrueType)
        out/p2_board_preview.png (small raster preview)

Requires: matplotlib, numpy, qrcode.
"""
from __future__ import annotations
import qrcode
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from poster_common import (INK, PAPER, VIOLET, VIOLET_FILL, VIOLET_RULE,
                           GOOD, BAD, WARN, MUTE, OUT, label)
from build_poster2 import (draw_compiler_pipeline, draw_pluralism, draw_hohfeld_v4,
                           draw_gateway, draw_pytorch_lens, draw_nazi_attic)
import os

BOARD_W, BOARD_H = 48.0, 36.0
MARGIN = 1.5


# --------------------------------------------------------------- panel 1 (new)
def draw_thesis(ax):
    """Panel 1 — the thesis: don't collapse to a scalar."""
    label(ax, 50, 51, "Don't collapse to a scalar.", fontsize=17, color=VIOLET, weight="bold")

    # the moral tensor: stakeholders x dimensions grid
    rows = ["speaker", "village", "refugees", "nazis"]
    cols = ["harm", "rights", "consent", "risk", "fair"]
    cell_colors = [
        ["#B3261E", "#C77800", "#6A4C93", "#C77800", "#B3261E"],
        ["#B3261E", "#B3261E", "#C77800", "#B3261E", "#C77800"],
        ["#1B7F4B", "#1B7F4B", "#1B7F4B", "#C77800", "#1B7F4B"],
        ["#C77800", "#6A4C93", "#C77800", "#C77800", "#6A4C93"],
    ]
    gx, gy, cw, ch = 9, 36, 5.4, 5.4
    for j, c in enumerate(cols):
        label(ax, gx + j * cw + cw / 2, gy + 3.6, c, fontsize=9, color=MUTE)
    for i, rname in enumerate(rows):
        y = gy - i * ch
        label(ax, gx - 1.5, y, rname, fontsize=10, color=INK, ha="right")
        for j in range(len(cols)):
            ax.add_patch(Rectangle((gx + j * cw, y - ch / 2), cw - 0.5, ch - 0.5,
                                   fc=cell_colors[i][j], ec=PAPER, lw=0.8, alpha=0.75, zorder=2))
    label(ax, gx + 13.5, gy - 4 * ch - 1.4, "the moral tensor (who × what is owed)",
          fontsize=11, color=VIOLET, weight="bold")

    # collapse arrow -> crossed-out scalar
    ax.add_patch(FancyArrowPatch((38, 25), (48, 25), arrowstyle="-|>", mutation_scale=20,
                                 color=BAD, lw=2.6, zorder=3))
    label(ax, 55, 25, "★ 0.42", fontsize=20, color=BAD, weight="bold")
    ax.plot([50.5, 59.5], [21.5, 28.5], color=BAD, lw=3, zorder=4)
    label(ax, 55, 18, "one scalar —\nstructure lost", fontsize=11, color=BAD, weight="bold")

    # the kept-structure alternative
    label(ax, 79, 38, "✓ keep the structure", fontsize=14, color=GOOD, weight="bold")
    label(ax, 79, 27.5,
          "who is affected · what is owed\nwho consented · who bears imposed risk",
          fontsize=11, color=INK)
    label(ax, 79, 15.5,
          "contract to a decision only at the end,\nand log the contraction:\nweights · who lost · what residue remains",
          fontsize=11, color=VIOLET, weight="bold")


draw_thesis.native = (13, 7.4)


# ------------------------------------------------------------------- utilities
def scale_points(ax, s):
    """Scale all point-sized elements (fonts, linewidths, arrowheads) by s so a
    diagram placed at s x its native size keeps its designed proportions."""
    for t in ax.texts:
        t.set_fontsize(t.get_fontsize() * s)
    for p in ax.patches:
        p.set_linewidth(p.get_linewidth() * s)
        if isinstance(p, FancyArrowPatch):
            p.set_mutation_scale(p.get_mutation_scale() * s)
    for ln in ax.lines:
        ln.set_linewidth(ln.get_linewidth() * s)


def place_diagram(fig, draw_fn, box_x, box_y, box_w, box_h, pad=0.30):
    """Fit draw_fn's native canvas into the given box (inches), centered."""
    nw, nh = draw_fn.native
    avail_w, avail_h = box_w - 2 * pad, box_h - 2 * pad
    s = min(avail_w / nw, avail_h / nh)
    w, h = nw * s, nh * s
    x = box_x + (box_w - w) / 2
    y = box_y + (box_h - h) / 2
    ax = fig.add_axes([x / BOARD_W, y / BOARD_H, w / BOARD_W, h / BOARD_H])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * nh / nw)
    ax.set_aspect("equal")
    ax.axis("off")
    draw_fn(ax)
    scale_points(ax, s)
    return s


def panel_bg(board, x, y, w, h):
    board.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                   fc=VIOLET_FILL, ec=VIOLET_RULE, lw=2.0, zorder=1))


def draw_qr(board, cx, cy, size, url, caption):
    qr = qrcode.QRCode(border=1, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    m = qr.get_matrix()
    n = len(m)
    cell = size / n
    x0, y0 = cx - size / 2, cy - size / 2
    board.add_patch(Rectangle((x0 - 0.06, y0 - 0.06), size + 0.12, size + 0.12,
                              fc=PAPER, ec=VIOLET_RULE, lw=1.2, zorder=3))
    for i, row in enumerate(m):          # row 0 at top
        for j, v in enumerate(row):
            if v:
                board.add_patch(Rectangle((x0 + j * cell, y0 + (n - 1 - i) * cell),
                                          cell, cell, fc=INK, ec="none", zorder=4))
    board.text(cx, y0 - 0.22, caption, ha="center", va="top", fontsize=13,
               color=INK, weight="bold", zorder=4)


# ------------------------------------------------------------------- the board
def build_board():
    fig = plt.figure(figsize=(BOARD_W, BOARD_H))
    board = fig.add_axes([0, 0, 1, 1])
    board.set_xlim(0, BOARD_W)
    board.set_ylim(0, BOARD_H)
    board.set_aspect("equal")
    board.axis("off")
    board.add_patch(Rectangle((0, 0), BOARD_W, BOARD_H, fc=PAPER, ec="none", zorder=0))

    X0, X1 = MARGIN, BOARD_W - MARGIN            # 1.5 .. 46.5
    Y0, Y1 = MARGIN, BOARD_H - MARGIN            # 1.5 .. 34.5

    # ---------------------------------------------------------------- title bar
    ty = Y1 - 6.0                                 # 28.5
    board.text(X0, Y1 - 1.55, "Moral Tensors and DecisionProofs",
               fontsize=84, color=VIOLET, weight="bold", va="center", zorder=5)
    board.text(X0, Y1 - 3.05, "Compiling Language into an Auditable, Grounded Safety Layer",
               fontsize=46, color=INK, weight="bold", va="center", zorder=5)
    board.text(X0, Y1 - 4.35, "Structure-preserving representation before decision contraction.",
               fontsize=28, color=VIOLET, style="italic", va="center", zorder=5)
    board.text(X0, Y1 - 5.45,
               "pip install erisml-compiler   ·   github.com/ahb-sjsu/erisml-compiler · erisml-lib · agi-hpc",
               fontsize=21, color=MUTE, va="center", zorder=5, family="DejaVu Sans Mono")
    # author block, right 30%
    ax_r = X1
    board.text(ax_r, Y1 - 1.55, "Andrew H. Bond", fontsize=36, color=INK, weight="bold",
               ha="right", va="center", zorder=5)
    board.text(ax_r, Y1 - 2.75, "San José State University  ·  IEEE Senior Member",
               fontsize=21, color=INK, ha="right", va="center", zorder=5)
    board.text(ax_r, Y1 - 3.65, "ORCID 0009-0003-2599-6158",
               fontsize=19, color=MUTE, ha="right", va="center", zorder=5)
    board.text(ax_r, Y1 - 4.75, "PyTorch Conference North America 2026 — Responsible AI poster",
               fontsize=21, color=VIOLET, weight="bold", ha="right", va="center", zorder=5)
    board.plot([X0, X1], [ty, ty], color=VIOLET, lw=4, zorder=5)

    # ------------------------------------------------------------- panel grid
    GUTTER = 0.5
    col_w = (X1 - X0 - 2 * GUTTER) / 3            # 14.667
    col_x = [X0, X0 + col_w + GUTTER, X0 + 2 * (col_w + GUTTER)]
    hero_top = 10.5                               # hero band: 3.7 .. 10.5
    grid_bot = hero_top + 0.4                     # 10.9
    grid_top = ty - 0.4                           # 28.1
    p_h = (grid_top - grid_bot - 0.4) / 2         # ~8.4
    row_y = [grid_bot + p_h + 0.4, grid_bot]      # top row, bottom row

    panels = [
        (draw_thesis,            col_x[0], row_y[0]),
        (draw_compiler_pipeline, col_x[0], row_y[1]),
        (draw_pluralism,         col_x[1], row_y[0]),
        (draw_hohfeld_v4,        col_x[1], row_y[1]),
        (draw_gateway,           col_x[2], row_y[0]),
        (draw_pytorch_lens,      col_x[2], row_y[1]),
    ]
    for i, (fn, px, py) in enumerate(panels, start=1):
        panel_bg(board, px, py, col_w, p_h)
        s = place_diagram(fig, fn, px, py, col_w, p_h)
        board.text(px + 0.25, py + p_h - 0.18, str(i), fontsize=20, color=VIOLET_RULE,
                   weight="bold", va="top", zorder=6)
        print(f"  panel {i}: {fn.__name__} at scale {s:.2f}")

    # -------------------------------------------------------------- hero band
    hero_y = 3.7
    panel_bg(board, X0, hero_y, X1 - X0, hero_top - hero_y)
    # diagram on the left 2/3, caption block on the right
    s = place_diagram(fig, draw_nazi_attic, X0, hero_y, 26.0, hero_top - hero_y, pad=0.25)
    board.text(X0 + 0.25, hero_y + (hero_top - hero_y) - 0.18, "7", fontsize=20,
               color=VIOLET_RULE, weight="bold", va="top", zorder=6)
    print(f"  panel 7: draw_nazi_attic at scale {s:.2f}")
    cxr = X0 + 26.0 + (X1 - X0 - 26.0) / 2
    board.text(cxr, hero_y + 5.5, "One command. Real numbers.\nA hash you can verify.",
               fontsize=30, color=VIOLET, weight="bold", ha="center", va="center", zorder=5)
    board.text(cxr, hero_y + 3.4, "$ erisml-compile examples/nazi_attic",
               fontsize=20, color=INK, ha="center", va="center", zorder=5,
               family="DejaVu Sans Mono")
    board.text(cxr, hero_y + 1.9,
               "per-party verdicts · Gini 0.43 · exact Shapley\nDecisionProof chained to the IR hash — replay the judgment",
               fontsize=17, color=INK, ha="center", va="center", zorder=5)

    # ---------------------------------------------------------------- footer
    fy0, fy1 = Y0, 3.5
    board.plot([X0, X1], [fy1, fy1], color=VIOLET_RULE, lw=2.5, zorder=5)
    takeaways = [
        "1.  Structure before contraction — keep the tensor, log the collapse + the residue.",
        "2.  Pluralism is the responsible move — four lenses, defer to a human on conflict.",
        "3.  Auditable by construction; fails safe, never open.",
    ]
    for k, t in enumerate(takeaways):
        board.text(X0, fy1 - 0.42 - k * 0.58, t, fontsize=22, color=INK, va="center", zorder=5)
    board.text(X0 + 25.6, fy1 - 1.0,
               "alpha v0.9.0 · text path solid · activation lens early / uncalibrated\n"
               "ranks 1–3 real, higher partial · V4 measured, D4 posited (Lean + Mathlib verified)\n"
               "embodied = design target · conversational agents = live",
               fontsize=15, color=MUTE, style="italic", va="center", zorder=5)
    qy = (fy0 + fy1) / 2 + 0.22
    draw_qr(board, X1 - 8.6, qy, 1.45, "https://pypi.org/project/erisml-compiler/", "erisml-compiler")
    draw_qr(board, X1 - 5.1, qy, 1.45, "https://github.com/ahb-sjsu/erisml-lib", "erisml-lib")
    draw_qr(board, X1 - 1.6, qy, 1.45, "https://github.com/ahb-sjsu/agi-hpc", "agi-hpc")

    # ------------------------------------------------------------------ save
    os.makedirs(OUT, exist_ok=True)
    pdf = os.path.join(OUT, "p2_board_48x36.pdf")
    png = os.path.join(OUT, "p2_board_preview.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=40, facecolor=PAPER)
    plt.close(fig)
    print(f"  wrote {os.path.basename(pdf)} (48x36 in, vector) + preview.png")


if __name__ == "__main__":
    print("POSTER #2 board:")
    build_board()
