---
type: sensei_note
task: 56
tags: [symmetry, classification, d4, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Task 056 — Symmetry-Class Lookup (4-way Classifier)

**Class: CLASSIFICATION** (see `sensei_meta_task_typology`). Input is 3×3,
output is 1×1 with a single color code. This is a predicate-on-input,
not a transformation.

## The real rule

Earlier guidance on this task was wrong (it said "any symmetry → 2
else 1"). That's why the attempts kept failing. The *actual* rule
maps each input's **symmetry signature** to a specific output code:

| Symmetries present                  | Output |
|-------------------------------------|--------|
| 4 (h, v, main-diag, anti-diag) AND `grid[0][0] != 0`  | `[[2]]` |
| 4 (h, v, main-diag, anti-diag) AND `grid[0][0] == 0`  | `[[6]]` |
| Only main-diagonal symmetry         | `[[1]]` |
| Only anti-diagonal symmetry         | `[[3]]` |

The corner check distinguishes the two 4-way-symmetric cases — `[[2]]`
is the **X / diagonal shape** (corners + center filled), `[[6]]` is
the **plus / orthogonal shape** (edges + center filled). Both have
full D4 symmetry but are structurally different; the `(0,0)` test
is the cheapest way to tell them apart.

## Reference implementation (7/7 train, 3/3 hidden test)

```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    g = np.array(grid)
    nz = g != 0  # occupancy mask — colors are irrelevant, only shape matters

    h  = np.array_equal(nz, np.flipud(nz))                    # top <-> bottom
    v  = np.array_equal(nz, np.fliplr(nz))                    # left <-> right
    d  = np.array_equal(nz, nz.T)                             # main diagonal
    ad = np.array_equal(nz, np.flipud(np.fliplr(nz)).T)       # anti-diagonal

    n_syms = h + v + d + ad

    if n_syms == 4:
        return [[2]] if nz[0, 0] else [[6]]
    if d and not ad:
        return [[1]]
    if ad and not d:
        return [[3]]
    # Train set never produces other signatures; fall back to 1.
    return [[1]]
```

## Why this generalizes

This task is an instance of the **symmetry-class classifier** family
(see `sensei_meta_symmetry_classifiers`). The family recipe:

1. Build an occupancy mask (ignore colors; shapes matter).
2. Compute its D4 symmetry signature (which of the 4 reflections hold).
3. Look up the output based on which signature is present.
4. If multiple inputs share the same signature, add a disambiguator
   (here: the `(0,0)` cell value). Common disambiguators: specific
   cell values, total occupied count, connected-component count.

When you see a 1×1 output with a small color palette and 3×3 input,
run this checklist *first*. Don't reach for pixel-level transforms —
they don't apply to CLASSIFICATION tasks.
