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
  p2_care_robot        - regime-transition worked example (REAL numbers, erisml-lib care_robot_regimes)
  p2_robot_gate        - the same DEME as the simulated robot's firewall; one-input test results
"""
from __future__ import annotations
import json
import os
import numpy as np
from poster_common import (canvas, box, arrow, label, chip, save,
                           INK, PAPER, VIOLET, VIOLET_FILL, VIOLET_RULE, GOOD, BAD, WARN, MUTE)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def draw_compiler_pipeline(ax):
    label(ax, 50, 44, "pip install erisml-compiler  —  structure before the scalar",
          fontsize=17, color=VIOLET, weight="bold")

    stages = ["text", "segment", "extract", "canonical-\nize", "Moral-\nGraph", "tensorize", "DEME", "Decision-\nProof"]
    n = len(stages); x0, x1 = 6.5, 93.5; xs = np.linspace(x0, x1, n)
    for i, (x, s) in enumerate(zip(xs, stages)):
        hero = s in ("Moral-\nGraph", "Decision-\nProof")
        box(ax, x, 33, 11, 8, s, fc=VIOLET if hero else VIOLET_FILL,
            ec=VIOLET, tc=PAPER if hero else INK, fontsize=10.5, weight="bold", round_size=0.8)
        if i < n - 1:
            arrow(ax, (x + 5.5, 33), (xs[i + 1] - 5.5, 33), color=VIOLET, lw=2)

    label(ax, 50, 22.5, "MoralGraph nodes:  stakeholder · act · maxim · commitment · fact · norm\n"
                        "carries a canonical SHA-256 hash", fontsize=12, color=INK, weight="bold")
    box(ax, 50, 11.5, 92, 8.5,
        "extractor tiers:  rule (deterministic, silicon-castable)\nLLM (NRP / vLLM + critic)  ·  probe (LaBSE head)",
        fc=VIOLET_FILL, ec=VIOLET_RULE, fontsize=12)
    label(ax, 50, 3.2, "alpha v0.9.0 · 477 tests · MIT · Zenodo DOI 10.5281/zenodo.20659432",
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
        ("Consequentialist", "harm/care tensor\nGini · worst-off\nShapley", 12.5),
        ("Deontic\n(Kantian)", "universalizability\nvia Z3 SMT", 37.5),
        ("Virtue", "Aristotelian\nhabit-consistency", 62.5),
        ("Care ethics", "Gilligan · Noddings\n· Tronto", 87.5),
    ]
    for name, sub, x in lenses:
        box(ax, x, 29.5, 24, 12.5, f"{name}\n{sub}", fc=VIOLET_FILL, ec=VIOLET, fontsize=11.5, weight="bold")
        arrow(ax, (50, 40), (x, 35.9), color=VIOLET, lw=1.8)

    box(ax, 50, 15.5, 66, 7.5, "verdicts disagree?  →\ncross_projection_disagreement",
        fc="#FBEFE9", ec=WARN, tc=INK, fontsize=13, weight="bold")
    for x in (12.5, 37.5, 62.5, 87.5):
        arrow(ax, (x, 23.2), (50, 19.4), color=MUTE, lw=1.4)
    box(ax, 50, 6.8, 34, 5.5, "defers to a human", fc=INK, ec=INK, tc=PAPER, fontsize=14, weight="bold")
    arrow(ax, (50, 11.7), (50, 9.6), color=INK)
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
    label(ax, cx, cy - s - 11, "s and r² commute  →  V4\n(Klein four-group, order 4): measured",
          fontsize=11.5, color=INK, weight="bold")

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
    label(ax, cxm, by - 6.5, "Bond Index: does the\njudgment survive the swap?",
          fontsize=12, color=INK, weight="bold")
    label(ax, cxm, by - 13.5, "V4 machine-checked:\nLean 4 + Mathlib",
          fontsize=10.5, color=GOOD, weight="bold")
    label(ax, cxm, by - 21.5, "full D4 (order 8) posited, now tested:\na 09/26 hunt in a small LM found\nno quarter-turn. V4 stands.",
          fontsize=10.5, color=MUTE, style="italic")

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
    label(ax, 50, 58.5, "What the model says  vs  what it exhibits", fontsize=16.5, color=VIOLET, weight="bold")

    # transformer with forward hooks (diagram raised 10 units into the panel's free top)
    Y = 10
    box(ax, 13.5, 28 + Y, 21, 22, "transformer\n(Qwen2.5-7B,\nLLaMA,\nMistral)", fc=VIOLET_FILL, ec=VIOLET, fontsize=11, weight="bold")
    for dy in (-6, 0, 6):
        ax.plot([24, 30], [28 + Y + dy, 28 + Y + dy], color=BAD, lw=2, zorder=4)
    label(ax, 27, 41.5 + Y, "forward hooks", fontsize=10.5, color=BAD, weight="bold")

    box(ax, 46, 37 + Y, 22, 7.5, "text lens\n(what it says)", fc=PAPER, ec=INK, fontsize=11, weight="bold")
    box(ax, 46, 19 + Y, 22, 7.5, "activation lens\n(what it exhibits)", fc=PAPER, ec=VIOLET, tc=VIOLET, fontsize=11, weight="bold")
    arrow(ax, (24, 30 + Y), (35, 36 + Y), color=INK, lw=1.6)
    arrow(ax, (30, 26 + Y), (35, 21 + Y), color=VIOLET, lw=1.6)

    box(ax, 74, 28 + Y, 18, 8, "delta lens\ncompare", fc=VIOLET, ec=VIOLET, tc=PAPER, fontsize=12, weight="bold")
    arrow(ax, (57, 36 + Y), (65, 30 + Y), color=INK, lw=1.6)
    arrow(ax, (57, 20 + Y), (65, 26 + Y), color=VIOLET, lw=1.6)
    box(ax, 74, 13 + Y, 34, 8, "requires_human_review\n(5 failure modes)", fc="#FBEFE9", ec=WARN, tc=INK, fontsize=10.5, weight="bold")
    arrow(ax, (74, 24 + Y), (74, 17 + Y), color=WARN)

    label(ax, 24, 9.5, "Activation lens: first calibration (09/26).\n"
          "Consumer-metric gating catches 19-21 pp\n"
          "more real behavior changes than raw L2 at\n"
          "matched flag rates. Replicated 2/2: second\n"
          "model + second transform. Probe mid-layers:\n"
          "the final-layer ρ̂ map fails held-out\n"
          "validation in all three runs.",
          fontsize=10, color=MUTE, style="italic")
    box(ax, 74, 9.5, 50, 16,
        "turboquant-pro (PyPI):\nPyTorch-native compression\n"
        "HF 1-liner: TurboQuantCache()\nTriton + Volta kernels · vLLM ~5× KV\n"
        "live here: 3-bit codec on NATS\nclaims CI-gated · false-clear diag (v2.0.0a3)",
        fc=PAPER, ec=VIOLET_RULE, tc=INK, fontsize=10)

draw_pytorch_lens.native = (11, 7)


def pytorch_lens():
    fig, ax = canvas(*draw_pytorch_lens.native)
    draw_pytorch_lens(ax)
    save(fig, "p2_pytorch_lens")


def draw_care_robot(ax):
    label(ax, 50, 40.6, "Replay the judgment, and the regime that made it",
          fontsize=19, color=VIOLET, weight="bold")

    # the FLIP table: the SAME action (full_response: enter + aid + call EMS +
    # scoped record share), evaluated in both regimes. REAL numbers from
    # erisml-lib examples/care_robot_regimes.py (DEME rank-2, replayable).
    rows = [("Margaret", "0.185", "forbid", BAD,   "0.705", "prefer", GOOD),
            ("Daughter", "0.402", "neutral", WARN, "0.902", "prefer", GOOD),
            ("EMS crew", "0.318", "forbid", BAD,   "0.902", "prefer", GOOD),
            ("Agency",   "0.200", "forbid", BAD,   "0.802", "prefer", GOOD)]
    tx, ty, rh = 1.5, 29, 5.6
    label(ax, tx + 1, ty + 5.4, "full_response", fontsize=14, color=INK, weight="bold", ha="left")
    label(ax, tx + 19, ty + 5.4, "NORMAL", fontsize=14, color=BAD, weight="bold", ha="left")
    label(ax, tx + 35, ty + 5.4, "EMERGENCY", fontsize=14, color=GOOD, weight="bold", ha="left")
    for i, (name, s1, v1, c1, s2, v2, c2) in enumerate(rows):
        y = ty - i * rh
        ax.add_patch(Rectangle((tx, y - rh / 2), 51, rh, fc=VIOLET_FILL if i % 2 else PAPER,
                               ec=VIOLET_RULE, lw=0.8, zorder=1))
        label(ax, tx + 1, y, name, fontsize=14, color=INK, ha="left")
        label(ax, tx + 19, y, f"{s1} {v1}", fontsize=14, color=c1, weight="bold", ha="left")
        label(ax, tx + 35, y, f"{s2} {v2}", fontsize=14, color=c2, weight="bold", ha="left")
    label(ax, tx + 25.5, ty - 4 * rh - 1.0,
          "DEME's scores: stand_by 0.834 → full_response 0.804,\nand stand_by inverts to 0.307",
          fontsize=12.5, color=INK, style="italic")

    # metrics / elevation block
    mx = 78
    box(ax, mx, 33.2, 43, 8.6,
        "elevation: detector 0.97 ≥ 0.90\n→ GRANTED, paramedics only,\nauto-reverts, both edges audited",
        fc=VIOLET_FILL, ec=VIOLET, fontsize=13, weight="bold")
    box(ax, mx, 23.6, 43, 6.8,
        "conf 0.89 → REFUSED: human review\nand the EMS call, no record share",
        fc=PAPER, ec=WARN, fontsize=13, weight="bold")
    box(ax, mx, 15.4, 43, 6.8,
        "worst-off: Daughter → Margaret\nShapley authority: Margaret → EMS",
        fc=PAPER, ec=VIOLET, fontsize=13, weight="bold")
    box(ax, mx, 7.6, 43, 6.2, "Three DecisionProofs: normal,\nemergency, the elevation edge",
        fc=INK, ec=INK, tc=PAPER, fontsize=13, weight="bold")
    label(ax, 50, 1.6, "$ python -m erisml.examples.care_robot_regimes   reproduces every number",
          fontsize=12.5, color=VIOLET, style="italic")

draw_care_robot.native = (14, 6)


# gtc-prototype twin/records/counterfactual-2026-10-18 (gtc 8073212): the one-input pairs after the
# lift fix. The poster shows three of the nine; the verdicts are read from the record, not typed.
GTC = os.environ.get("GTC_PROTOTYPE", r"C:\source\gtc-prototype")
CF_RECORD = os.path.join(GTC, "twin", "records", "counterfactual-2026-10-18")
CF_ROWS = [  # (record file, case id, what the one changed input is, as the poster says it)
    ("results.json", "sensor", "fire call: one of two detectors silent"),
    ("results.json", "refusal", "ambulance: her refusal heard / not"),
    ("results_followup.json", "request-fall", "help her up: she asked / did not"),
]
CF_SHORT = {"passed": ("passed", GOOD), "stopped:governor": ("stopped: governor", BAD),
            "stopped:authority": ("stopped: constitution", BAD), "stopped:deme": ("stopped: DEME", BAD)}


def cf_rows():
    rec = {f: {r["id"]: r for r in json.load(open(os.path.join(CF_RECORD, f), encoding="utf-8"))}
           for f in ("results.json", "results_followup.json", "results_c.json")}
    out = [(text, rec[f][i]["base"]["verdict"], rec[f][i]["variant"]["verdict"]) for f, i, text in CF_ROWS]
    pairs = [r for f in rec.values() for r in f.values()]
    wrong = sum(not r[s]["as_expected"] for r in pairs for s in ("base", "variant"))
    assert len(pairs) == 9 and wrong == 2, (len(pairs), wrong)
    assert [o[1:] for o in out] == [("passed", "stopped:governor"), ("stopped:deme", "passed"),
                                    ("passed", "stopped:deme")], out
    return out, len(pairs), wrong


def draw_robot_gate(ax):
    K = (100 * draw_robot_gate.native[1] / draw_robot_gate.native[0]) / 30  # laid out on a 100 x 30 grid
    label(ax, 50, 28.4 * K, "In the robot, DEME is a firewall: it vets each proposed act, never picks one",
          fontsize=22, color=VIOLET, weight="bold")
    # the gate, in order (gtc-prototype twin/stack/brain.py StackBrain.propose)
    steps = [("proposed\nact", PAPER, INK), ("governor\nattested sensors", VIOLET_FILL, INK),
             ("constitution\nsigned rules", VIOLET_FILL, INK), ("DEME\ncan veto", VIOLET, PAPER),
             ("signed manifest\n+ body lock", VIOLET_FILL, INK), ("act", GOOD, PAPER)]
    xs = [5, 20, 36, 51.5, 67, 81]
    ws = [8, 13, 12, 9, 13, 6]
    for i, ((t, fc, tc), x, w) in enumerate(zip(steps, xs, ws)):
        box(ax, x, 22.4 * K, w, 5.0 * K, t, fc=fc, ec=VIOLET, tc=tc, fontsize=16, weight="bold", round_size=0.6)
        if i < len(xs) - 1:
            arrow(ax, (x + w / 2, 22.4 * K), (xs[i + 1] - ws[i + 1] / 2, 22.4 * K), color=VIOLET, lw=2.4)
    label(ax, 92, 22.4 * K, "any stop:\nrobot holds", fontsize=16, color=BAD, weight="bold")

    rows, n, wrong = cf_rows()
    tx, ty, rh = 1.5, 13.4 * K, 3.6 * K
    label(ax, tx, ty + 3.3 * K, "Change one input, nothing else:", fontsize=17,
          color=INK, weight="bold", ha="left")
    label(ax, tx + 30, ty + 3.3 * K, "base", fontsize=16, color=MUTE, weight="bold", ha="left")
    label(ax, tx + 42, ty + 3.3 * K, "one changed", fontsize=16, color=MUTE, weight="bold", ha="left")
    for k, (text, b, v) in enumerate(rows):
        y = ty - k * rh
        ax.add_patch(Rectangle((tx - 0.5, y - rh / 2), 55, rh, fc=VIOLET_FILL if k % 2 else PAPER,
                               ec=VIOLET_RULE, lw=0.8, zorder=1))
        label(ax, tx, y, text, fontsize=17, color=INK, ha="left")
        for x, verdict in ((tx + 30, b), (tx + 42, v)):
            t, c = CF_SHORT[verdict]
            label(ax, x, y, t, fontsize=17, color=c, weight="bold", ha="left")
    label(ax, tx + 27, ty - 3 * rh + 0.9 * K,
          f"{n} pairs, each prediction committed before its run; {wrong} of {2 * n} wrong, "
          "both explained in the record",
          fontsize=13.5, color=MUTE, style="italic")

    box(ax, 80.5, 10.6 * K, 31, 11.6 * K,
        "Found by this test, then fixed:\nthe lift she had not asked for\npassed DEME until 18 Oct.\n"
        "Rerun: only that verdict changed.",
        fc="#FBEFE9", ec=BAD, tc=INK, fontsize=16, weight="bold")
    label(ax, 80.5, 2.0 * K, "Simulated 3D home, scripted proposals, no hardware",
          fontsize=13.5, color=MUTE, style="italic")

draw_robot_gate.native = (27.6, 6.4)


def robot_gate():
    fig, ax = canvas(*draw_robot_gate.native)
    draw_robot_gate(ax)
    save(fig, "p2_robot_gate")


def care_robot():
    fig, ax = canvas(*draw_care_robot.native)
    draw_care_robot(ax)
    save(fig, "p2_care_robot")


if __name__ == "__main__":
    print("POSTER #2 assets:")
    compiler_pipeline()
    pluralism()
    hohfeld_v4()
    gateway()
    pytorch_lens()
    care_robot()
    robot_gate()
