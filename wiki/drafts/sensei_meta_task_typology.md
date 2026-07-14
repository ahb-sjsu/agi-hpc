---
type: sensei_note
tags: [observe, typology, shape, meta, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Meta — Output Shape Tells You the Task Class (Before Any Hypothesis)

Erebus's reasoning-error budget is dominated by applying a strategy
from the wrong class. The fix is cheap and mechanical: **the first
line of every OBSERVE note is the output-shape class**. It
eliminates most of the wrong-family hypotheses before they burn
attempts.

## The four classes

| Output shape                     | Class              | Strategy vocabulary               |
|----------------------------------|--------------------|-----------------------------------|
| `1×1`                            | CLASSIFICATION     | count, predicate, color-answer    |
| small fixed (e.g. `3×3`, `5×5`)  | EXTRACTION         | pick object, synthesize pattern   |
| `output.shape == input.shape`    | TRANSFORMATION     | pixel-map, object-mutation        |
| `output.shape > input.shape`     | EXPANSION          | tile, outline, upscale, fractal   |

Check the class on every training example. If it's consistent
across all examples, the class is locked. If it varies (rare), the
task is parameterized and you need to figure out what determines
the output shape.

## Why this matters

- Task 048 output is 1×1 → CLASSIFICATION. Asking "what color bridges
  the 2×2 blocks?" is the right shape of question. Asking "how do I
  transform the grid?" is the wrong shape of question. That's
  ~30 wasted attempts right there.
- Task 056 output is 1×1 → CLASSIFICATION (`[[1]]` vs `[[2]]`). Same
  thing — it's a predicate on the input, not a transformation.
- Task 175 output.shape == input.shape → TRANSFORMATION. Specifically
  a COMPLETION operator (see symmetry-completion family).

## Symmetry-completion family

A sub-family of TRANSFORMATION tasks where:

1. Input contains sparse 0s (holes) in an otherwise periodic/symmetric
   motif.
2. Non-zero cells form a repeating pattern (stripe, tile, mirror,
   rotational).
3. Output fills the holes using the surrounding period.

Members: `complete_by_horizontal_stripe`, `complete_by_vertical_stripe`,
`complete_by_diagonal`, `complete_by_tile`, `complete_by_mirror`,
`complete_by_rotation`. Recognize by the input shape fingerprint —
sparse 0s + periodic non-zero motif.

Task 175 is the diagonal member. Task 030 is not (centering is not
completion). Learn the family, don't re-derive it each task.

## Practical change

In `run_cycle`, before strategy selection, compute `output_class`
from the first training example's output shape. Log it. When
primitives/examples are formatted, prepend: `"This is a
{output_class} task."` LLMs respond much better to a framed
question than an open one.
