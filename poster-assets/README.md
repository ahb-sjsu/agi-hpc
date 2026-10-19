# Poster diagram assets — PyTorch Conf NA 2026

Vector diagram assets for the two poster submissions. Built with pure matplotlib
(no graphviz / no external binaries). Every diagram is emitted as **SVG + PDF**
(vector, prints crisp at 48″, editable in Illustrator / Inkscape / Figma) plus a
**PNG** preview.

## Regenerate

```bash
cd poster-assets
python build_poster1.py        # Applications — Atlas teal accent
python build_poster2.py        # Responsible AI — Eris violet accent
python build_poster2_board.py  # FULL 48x36 print-ready board for Poster #2
```

Output lands in `out/`. Requires `matplotlib` + `numpy` (+ `qrcode` for the board).

`out/p2_board_48x36.pdf` is the assembled Poster #2 board (vector, landscape
48"x36", per `pytorch-poster-2-LAYOUT.md`): title bar, six panels, the
hero band (care-robot regime transition + robot firewall), footer takeaways +
honesty strip + four QR codes (companion paper, erisml-compiler, erisml-lib,
agi-hpc). Panels are drawn live from `build_poster2.py`'s `draw_*` functions
(fonts/arrows rescaled with each panel, per-panel `TEXT_BOOST`), so editing a
panel updates both the standalone asset and the board. `check_text_fit` runs
before save and reports any text that overlaps or spills out of its box. Proof
a letter-size tile at 100% before sending to print.

The companion-paper QR code points at the PDF under the tag
`pytorch-conf-2026`, so the link outlives the branch. Move the tag (force-push)
whenever the paper changes, and check that the served PDF matches the local one.

## Companion paper, brochure, talk — `paper/`

| File | What it is | Build |
|---|---|---|
| `pytorch2026_poster2_paper.tex/.pdf` | 4-page companion paper (rev. 4); the poster's QR target | `pdflatex` ×2 |
| `pytorch2026_poster2_brochure.tex/.pdf` | one letter sheet, double-sided (flip on long edge), two columns, 11 pt, five QR codes | `pdflatex` ×2 |
| `pytorch2026_poster2_talk_script.md` | talk script: 30 s opener, panel-by-panel walkthrough, short versions, likely questions | — |
| `pytorch2026_poster2_cuecard.tex/.pdf` | one-page 12 pt cue card condensed from the script | `pdflatex` |

Every number in the brochure, script and cue card comes from the companion
paper; change the paper first, then these.

## What maps where

Match these to the panel numbers in the `*-LAYOUT.md` specs.

### Poster #1 (Applications) — `build_poster1.py`
| Asset | Panel | Notes |
|---|---|---|
| `p1_architecture` | 2 | Ego/Superego/Id/Council + NATS bus + NRP burst |
| `p1_verified_loop` | 3 | Primer verify-before-publish; task056 callout |
| `p1_thermal` | 4 | batch-probe worker→temp. **See data note below.** |
| `p1_turboquant` | 4 | VRAM fit; FP16 weights vs 3–4 bit. Illustrative est. |
| `p1_bursting` | 5 | nats-bursting leaf node into NRP namespace |

### Poster #2 (Responsible AI) — `build_poster2.py`
| Asset | Panel | Notes |
|---|---|---|
| `p2_compiler_pipeline` | 2 | text → MoralGraph → DEME → DecisionProof |
| `p2_pluralism` | 3 | 4 framework lenses → disagreement → human |
| `p2_hohfeld_v4` | 4 | Hohfeld square + V4 (s, r² commuting involutions) + Bond Index scale. Replaces `p2_hohfeld_d4` — V4 measured, D4 posited (2026-08 correction) |
| `p2_gateway` | 5 | 3-layer Safety Gateway + DecisionProof chain |
| `p2_pytorch_lens` | 6 | forward hooks: text vs activation vs delta lens |
| `p2_care_robot` | 7 (hero, left) | regime transition worked example — **real numbers** from `erisml-lib` `care_robot_regimes` |
| `p2_robot_gate` | 7 (hero, right) | the same DEME as a firewall in the simulated robot; one-input test results |

`out/p2_nazi_attic.*` is a stale output from the earlier hero; `build_poster2.py`
no longer draws it (the case stays bundled with the compiler).

## Honesty notes (don't break these)

- **`p1_thermal` is a TEMPLATE** until real data exists. It looks for a CSV
  (`thermal_sweep_results.csv`, columns `workers,cpu_c,gpu_c`) next to the script
  or one level up. If found, it plots the **measured** curve. If not, it draws a
  watermarked "ILLUSTRATIVE TEMPLATE" so no fabricated numbers are ever mistaken
  for measurements. After `thermal_sweep.py` runs on Atlas, drop the CSV in and
  rebuild — the watermark disappears automatically.
- **`p1_turboquant`** numbers are a weights-based arithmetic estimate (32B params),
  labeled as illustrative on the figure — not a measured benchmark.
- **`p2_care_robot`** numbers ARE real — they come from
  `python -m erisml.examples.care_robot_regimes` in `erisml-lib` (per-party
  scores, DEME ranking, elevation confidence 0.97 vs 0.90, refusal at 0.89).
  Safe to present as a worked result.
- **`p2_robot_gate`** describes a simulated 3D home with scripted proposals and
  no hardware; say so whenever it is presented.

## Palette (matched pair)

| | accent | fill | rule |
|---|---|---|---|
| Poster #1 | Atlas teal `#0E7C86` | `#F4F7F8` | `#C9D6D8` |
| Poster #2 | Eris violet `#6A4C93` | `#F5F3F8` | `#D4CCE0` |

Shared: ink `#1A1A1A`, good `#1B7F4B`, bad `#B3261E`, warn `#C77800`.
Edit `poster_common.py` to retune globally.
