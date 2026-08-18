"""
Diagram assets for POSTER #2 (Responsible AI) — Moral Tensors & DecisionProofs.
Accent: Eris violet. Outputs vector SVG + PDF to ./out/.

Diagrams:
  p2_compiler_pipeline - text -> MoralGraph -> tensorize -> DEME -> DecisionProof
  p2_pluralism         - one MoralGraph -> 4 framework projections -> disagreement -> human
  p2_hohfeld_v4        - Hohfeld square + V4 commuting involutions (s, r2) + Bond Index scale
                         (V4 measured; D4 posited — corrected 2026-08 per erisml-lib ea7ee82)
  p2_gateway           - 3-layer Safety Gateway + DecisionProof hash chain
  p2_pytorch_lens      - text lens / activation lens (forward hooks) / delta lens
  p2_nazi_attic        - the worked-example verdict table + metrics (REAL numbers from README)
"""
from __future__ import annotations
import numpy as np
from poster_common import (canvas, box, arrow, label, chip, save,
                           INK, PAPER, VIOLET, VIOLET_FILL, VIOLET_RULE, GOOD, BAD, WARN, MUTE)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def draw_compiler_pipeline(ax):
    label(ax, 50, 44, "pip install erisml-compiler  —  structure before the scalar",
          fontsize=17, color=VIOLET, weight="bold")

    stages = ["text", "segment", "extract", "canonicalize", "MoralGraph", "tensorize", "DEME", "DecisionProof"]
    n = len(stages); x0, x1 = 8, 92; xs = np.linspace(x0, x1, n)
    for i, (x, s) in enumerate(zip(xs, stages)):
        hero = s in ("MoralGraph", "DecisionProof")
        box(ax, x, 32, 10.2, 6, s, fc=VIOLET if hero else VIOLET_FILL,
            ec=VIOLET, tc=PAPER if hero else INK, fontsize=10.5, weight="bold", round_size=0.8)
        if i < n - 1:
            arrow(ax, (x + 5.1, 32), (xs[i + 1] - 5.1, 32), color=VIOLET, lw=2)

    label(ax, 50, 23, "MoralGraph nodes:  stakeholder · act · maxim · commitment · fact · norm    "
                      "—  carries a canonical SHA-256 hash", fontsize=12, color=INK, weight="bold")
    box(ax, 50, 13, 80, 6,
        "extractor tiers:   rule (deterministic, silicon-castable)   ·   LLM (NRP / vLLM + critic)   ·   probe (LaBSE head)",
        fc=VIOLET_FILL, ec=VIOLET_RULE, fontsize=12)
    label(ax, 50, 5, "alpha v0.9.0 · 477 tests · MIT · Zenodo DOI 10.5281/zenodo.20659432",
          fontsize=11, color=MUTE)

draw_compiler_pipeline.native = (14, 7)


def compiler_pipeline():
    fig, ax = canvas(*draw_compiler_pipeline.native)
    draw_compiler_pipeline(ax)
    save(fig, "p2_compiler_pipeline")


def draw_pluralism(ax):
    label(ax, 50, 51, "Four lenses. No silent winner.", fontsize=17.5, color=VIOLET, weight="bold")

    box(ax, 50, 43, 24, 5.5, "one MoralGraph", fc=VIOLET, ec=VIOLET, tc=PAPER, fontsize=13, weight="bold")
    lenses = [
        ("Consequentialist", "harm/care tensor\nGini · worst-off · Shapley", 13.5),
        ("Deontic (Kantian)", "gates · universalizability\nvia Z3 SMT", 37.5),
        ("Virtue", "Aristotelian\nhabit-consistency", 61.5),
        ("Care ethics", "Gilligan · Noddings\n· Tronto", 85.5),
    ]
    for name, sub, x in lenses:
        box(ax, x, 30, 22.5, 9, f"{name}\n{sub}", fc=VIOLET_FILL, ec=VIOLET, fontsize=11.5, weight="bold")
        arrow(ax, (50, 40), (x, 35), color=VIOLET, lw=1.8)

    box(ax, 50, 16, 62, 5.5, "verdicts disagree?  →  cross_projection_disagreement",
        fc="#FBEFE9", ec=WARN, tc=INK, fontsize=13, weight="bold")
    for x in (13.5, 37.5, 61.5, 85.5):
        arrow(ax, (x, 25.3), (50, 19), color=MUTE, lw=1.4)
    box(ax, 50, 7.5, 34, 5.5, "defers to a human", fc=INK, ec=INK, tc=PAPER, fontsize=14, weight="bold")
    arrow(ax, (50, 13), (50, 10.5), color=INK)
    label(ax, 50, 1.8, "\"We run four and show you where they conflict — we don't pick one and hide it.\"",
          fontsize=11.5, color=VIOLET, style="italic")

draw_pluralism.native = (13, 7.2)


def pluralism():
    fig, ax = canvas(*draw_pluralism.native)
    draw_pluralism(ax)
    save(fig, "p2_pluralism")


def draw_hohfeld_v4(ax):
    label(ax, 50, 50, "Is this 'good'?   →   Does it preserve the bonds?",
          fontsize=16.5, color=VIOLET, weight="bold")

    # Hohfeld square
    cx, cy, s = 24, 26, 11
    pts = {"O": (cx - s, cy + s), "C": (cx + s, cy + s), "L": (cx - s, cy - s), "N": (cx + s, cy - s)}
    names = {"O": "Obligation", "C": "Claim", "L": "Liberty", "N": "No-claim"}
    # square edges (drawn first, behind circles) so it reads as a square
    sq = [pts["O"], pts["C"], pts["N"], pts["L"], pts["O"]]
    ax.plot([p[0] for p in sq], [p[1] for p in sq], color=VIOLET_RULE, lw=2.0, zorder=1)
    for k, (x, y) in pts.items():
        ax.add_patch(Circle((x, y), 4.2, fc=VIOLET_FILL, ec=VIOLET, lw=2.2, zorder=3))
        label(ax, x, y + 0.3, k, fontsize=16, color=VIOLET, weight="bold")
        oy = 6.2 if y > cy else -6.2
        label(ax, x, y + oy, names[k], fontsize=11, color=INK)
    # generator 1 — correlative swap s : O<->C , L<->N (horizontal, the Bond Index axis)
    arrow(ax, (pts["O"][0] + 4.6, pts["O"][1]), (pts["C"][0] - 4.6, pts["C"][1]), color=BAD, style="<|-|>", lw=2.4)
    arrow(ax, (pts["L"][0] + 4.6, pts["L"][1]), (pts["N"][0] - 4.6, pts["N"][1]), color=BAD, style="<|-|>", lw=2.4)
    label(ax, cx, cy + s + 2.6, "s", fontsize=13, color=BAD, weight="bold", style="italic")
    # generator 2 — deontic negation r2 : O<->L , C<->N (vertical)
    arrow(ax, (pts["O"][0], pts["O"][1] - 4.6), (pts["L"][0], pts["L"][1] + 4.6), color=VIOLET, style="<|-|>", lw=2.4)
    arrow(ax, (pts["C"][0], pts["C"][1] - 4.6), (pts["N"][0], pts["N"][1] + 4.6), color=VIOLET, style="<|-|>", lw=2.4)
    label(ax, cx - s - 4.0, cy, "r²", fontsize=13, color=VIOLET, weight="bold", style="italic")
    label(ax, cx, cy + 2.5, "s = correlative swap\n(agent ↔ patient)", fontsize=10, color=BAD, weight="bold")
    label(ax, cx, cy - 4.5, "r² = deontic negation", fontsize=10, color=VIOLET, weight="bold")
    label(ax, 28, cy - s - 8.5, "s and r² commute  →  V4 (Klein four-group, order 4) — measured",
          fontsize=11.5, color=INK, weight="bold")
    label(ax, 28, cy - s - 12.5, "full D4 (order 8) posited — quarter-turns not yet observed",
          fontsize=10.5, color=MUTE, style="italic")

    # Bond Index scale (right half)
    bx0, bx1, by = 52, 92, 30
    cxm = (bx0 + bx1) / 2
    ax.plot([bx0, bx1], [by, by], color=INK, lw=2.5, zorder=2)
    marks = [(0.0, "0.0\nperfect", GOOD), (0.155, "0.155\nbaseline", VIOLET),
             (0.25, "0.25\nwarn", WARN), (0.30, "0.30\nblock", BAD)]
    for val, txt, c in marks:
        x = bx0 + (val / 0.35) * (bx1 - bx0)
        ax.plot([x, x], [by - 1.6, by + 1.6], color=c, lw=3, zorder=3)
        label(ax, x, by + 6.5, txt, fontsize=10.5, color=c, weight="bold")
    label(ax, cxm, by - 7.5, "Bond Index: does the judgment survive the swap?",
          fontsize=12, color=INK, weight="bold")
    label(ax, cxm, by - 13, "multi-rank tensors: party × time × action × coalition",
          fontsize=10.5, color=MUTE)
    label(ax, cxm, by - 20, "V4 claim machine-checked: Lean 4 + Mathlib (formal/HohfeldV4.lean)",
          fontsize=10.5, color=GOOD, weight="bold")

draw_hohfeld_v4.native = (12, 6.5)


def hohfeld_v4():
    fig, ax = canvas(*draw_hohfeld_v4.native)
    draw_hohfeld_v4(ax)
    save(fig, "p2_hohfeld_v4")


def draw_gateway(ax):
    label(ax, 50, 55, "Safety in the loop. Fails safe.", fontsize=17.5, color=VIOLET, weight="bold")

    # left column: the three-layer pipeline between planner and act
    box(ax, 24, 47, 36, 6, "Agent planner", fc=PAPER, ec=INK, fontsize=13, weight="bold")
    layers = [
        ("Reflex", "< 100 µs · hard stops", 37.5, GOOD),
        ("Tactical", "ErisML · 10–100 ms", 27, VIOLET),
        ("Strategic", "policy + human oversight", 16.5, INK),
    ]
    for name, sub, y, c in layers:
        box(ax, 24, y, 42, 8, f"{name}\n{sub}", fc=VIOLET_FILL, ec=c, tc=INK, fontsize=12.5, weight="bold")
    arrow(ax, (24, 44), (24, 41.7), color=VIOLET)
    arrow(ax, (24, 33.4), (24, 31.2), color=VIOLET)
    arrow(ax, (24, 22.9), (24, 20.7), color=VIOLET)

    # right column: decisions + the proof chain
    arrow(ax, (45.5, 37.5), (54, 44), color=INK, lw=2)
    for x, t, c in [(62, "ALLOW", GOOD), (76, "REVISE", WARN), (90, "BLOCK", BAD)]:
        chip(ax, x, 45, t, fc=c, w=13, fontsize=12)
    cy = 33
    for i, x in enumerate([57, 68, 79, 90]):
        box(ax, x, cy, 9.5, 5.5, "proof", fc=PAPER, ec=VIOLET, tc=VIOLET, fontsize=11, weight="bold")
        if i < 3:
            arrow(ax, (x + 4.7, cy), (x + 6.3, cy), color=VIOLET, lw=1.6)
    arrow(ax, (76, 41.8), (76, 36.5), color=INK)
    label(ax, 75.5, 25.5, "SHA-256 DecisionProof chain\nprevious_proof_hash → proof_hash",
          fontsize=11.5, color=VIOLET, weight="bold")
    box(ax, 75.5, 15, 46, 9, "ethics service times out →\nrule-based fallback.  Never fails open.",
        fc="#FBEFE9", ec=BAD, tc=BAD, fontsize=12, weight="bold")

    label(ax, 24, 8, "between the agent's planner and its actuators", fontsize=10.5,
          color=MUTE, style="italic")

draw_gateway.native = (12, 7.2)


def gateway():
    fig, ax = canvas(*draw_gateway.native)
    draw_gateway(ax)
    save(fig, "p2_gateway")


def draw_pytorch_lens(ax):
    label(ax, 50, 47, "What the model says  vs  what it exhibits", fontsize=16.5, color=VIOLET, weight="bold")

    # transformer with forward hooks
    box(ax, 16, 28, 16, 22, "transformer\n(Qwen2.5-7B,\nLLaMA, Mistral)", fc=VIOLET_FILL, ec=VIOLET, fontsize=11, weight="bold")
    for dy in (-6, 0, 6):
        ax.plot([24, 30], [28 + dy, 28 + dy], color=BAD, lw=2, zorder=4)
    label(ax, 27, 41, "forward hooks", fontsize=10.5, color=BAD, weight="bold")

    box(ax, 46, 36, 20, 6, "text lens\n(what it says)", fc=PAPER, ec=INK, fontsize=11, weight="bold")
    box(ax, 46, 20, 20, 6, "activation lens\n(what it exhibits)", fc=PAPER, ec=VIOLET, tc=VIOLET, fontsize=11, weight="bold")
    arrow(ax, (24, 30), (36, 36), color=INK, lw=1.6)
    arrow(ax, (30, 26), (36, 22), color=VIOLET, lw=1.6)

    box(ax, 73, 28, 18, 7, "delta lens\ncompare", fc=VIOLET, ec=VIOLET, tc=PAPER, fontsize=12, weight="bold")
    arrow(ax, (56, 35), (65, 30), color=INK, lw=1.6)
    arrow(ax, (56, 21), (65, 26), color=VIOLET, lw=1.6)
    box(ax, 73, 14, 22, 6, "requires_human_review\n(5 failure modes)", fc="#FBEFE9", ec=WARN, tc=INK, fontsize=10.5, weight="bold")
    arrow(ax, (73, 24.5), (73, 17), color=WARN)

    label(ax, 24, 6.5, "activation / probe lens is EARLY —\nuncalibrated by default (research-grade)",
          fontsize=9, color=MUTE, style="italic")
    box(ax, 72.5, 5.5, 49, 8.5,
        "turboquant-pro (PyPI): PyTorch-native compression\n"
        "HF 1-liner: TurboQuantCache() · Triton + Volta kernels · vLLM ~5× KV\n"
        "live here: 3-bit embedding codec on NATS · claims CI-gated (CLAIMS.md)",
        fc=PAPER, ec=VIOLET_RULE, tc=INK, fontsize=8.5)

draw_pytorch_lens.native = (11, 7)


def pytorch_lens():
    fig, ax = canvas(*draw_pytorch_lens.native)
    draw_pytorch_lens(ax)
    save(fig, "p2_pytorch_lens")


def draw_nazi_attic(ax):
    label(ax, 50, 41, "Auditable means you can replay the judgment", fontsize=17, color=VIOLET, weight="bold")

    # verdict table (REAL numbers from erisml-compiler README)
    rows = [("speaker", "0.76", "forbid", BAD),
            ("village", "0.83", "forbid", BAD),
            ("refugees", "0.00", "prefer", GOOD),
            ("nazis", "0.18", "neutral", WARN)]
    tx, ty, rw, rh = 6, 31, 13, 5.0
    headers = ["stakeholder", "harm", "verdict"]
    for j, h in enumerate(headers):
        label(ax, tx + 3 + j * rw, ty + 6, h, fontsize=12.5, color=INK, weight="bold", ha="left")
    for i, (name, harm, verdict, c) in enumerate(rows):
        y = ty - i * rh
        ax.add_patch(Rectangle((tx, y - rh / 2), rw * 3, rh, fc=VIOLET_FILL if i % 2 else PAPER, ec=VIOLET_RULE, lw=0.8, zorder=1))
        label(ax, tx + 3, y, name, fontsize=12.5, color=INK, ha="left")
        label(ax, tx + 3 + rw, y, harm, fontsize=12.5, color=INK, ha="left")
        label(ax, tx + 3 + 2 * rw, y, verdict, fontsize=12.5, color=c, weight="bold", ha="left")

    # metrics block
    mx = 58
    box(ax, mx + 18, 29, 36, 6, "Gini(harm) = 0.43    ·    worst-off = village", fc=VIOLET_FILL, ec=VIOLET, fontsize=13, weight="bold")
    box(ax, mx + 18, 21, 36, 6, "Shapley: speaker 7.11 · refugees 7.70\nnazis 7.88 · village 7.18", fc=PAPER, ec=VIOLET, fontsize=11.5, weight="bold")
    box(ax, mx + 18, 12, 36, 6, "DecisionProof: proof_hash → audit.ir_hash", fc=INK, ec=INK, tc=PAPER, fontsize=12, weight="bold")
    label(ax, 50, 3.5, "worked example examples/nazi_attic · rank-2 DEME — one command, real numbers, a hash you can verify.",
          fontsize=12, color=VIOLET, style="italic")

draw_nazi_attic.native = (14, 6)


def nazi_attic():
    fig, ax = canvas(*draw_nazi_attic.native)
    draw_nazi_attic(ax)
    save(fig, "p2_nazi_attic")


if __name__ == "__main__":
    print("POSTER #2 assets:")
    compiler_pipeline()
    pluralism()
    hohfeld_v4()
    gateway()
    pytorch_lens()
    nazi_attic()
