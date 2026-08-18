# Poster diagram assets — PyTorch Conf NA 2026

Vector diagram assets for the two poster submissions. Built with pure matplotlib
(no graphviz / no external binaries). Every diagram is emitted as **SVG + PDF**
(vector, prints crisp at 48″, editable in Illustrator / Inkscape / Figma) plus a
**PNG** preview.

## Regenerate

```bash
cd poster-assets
python build_poster1.py     # Applications — Atlas teal accent
python build_poster2.py     # Responsible AI — Eris violet accent
```

Output lands in `out/`. Requires only `matplotlib` + `numpy`.

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
| `p2_nazi_attic` | 7 (hero) | worked example — **real numbers** from the compiler README |

## Honesty notes (don't break these)

- **`p1_thermal` is a TEMPLATE** until real data exists. It looks for a CSV
  (`thermal_sweep_results.csv`, columns `workers,cpu_c,gpu_c`) next to the script
  or one level up. If found, it plots the **measured** curve. If not, it draws a
  watermarked "ILLUSTRATIVE TEMPLATE" so no fabricated numbers are ever mistaken
  for measurements. After `thermal_sweep.py` runs on Atlas, drop the CSV in and
  rebuild — the watermark disappears automatically.
- **`p1_turboquant`** numbers are a weights-based arithmetic estimate (32B params),
  labeled as illustrative on the figure — not a measured benchmark.
- **`p2_nazi_attic`** numbers ARE real — they come from the `erisml-compiler`
  README's end-to-end `nazi_attic` run (per-party verdicts, Gini 0.43, Shapley,
  DecisionProof). Safe to present as a worked result.

## Palette (matched pair)

| | accent | fill | rule |
|---|---|---|---|
| Poster #1 | Atlas teal `#0E7C86` | `#F4F7F8` | `#C9D6D8` |
| Poster #2 | Eris violet `#6A4C93` | `#F5F3F8` | `#D4CCE0` |

Shared: ink `#1A1A1A`, good `#1B7F4B`, bad `#B3261E`, warn `#C77800`.
Edit `poster_common.py` to retune globally.
