---
type: sensei_note
task: 167
tags: [classify, count-distinct-colors, diagonal, arc]
written_by: Professor Bond
written_at: 2026-04-19
verified_by: reference_implementation (train 5/5, test 1/1)
---

# Task 167 — Distinct-Color Count Selects a Pattern

## Rule

The output shape and non-zero cells depend ONLY on the number of distinct
colors in the input grid (background included):

- **1 distinct color** → top row filled with 5s, rest 0.
- **2 distinct colors** → main-diagonal 5s (`out[i][i] = 5`).
- **3 distinct colors** → anti-diagonal 5s (`out[i][W-1-i] = 5`).

The content of the input cells is irrelevant beyond the count. The
output grid is the same dimensions as the input.

## Why prior attempts failed

The input grids look like puzzles with local structure (corners, strips,
symmetry), so the natural hypothesis is a geometric transform of that
structure. It is not. The local structure is a distractor; only the
cardinality of the color set matters.

Pattern for recognising this class: when multiple training examples
with visually very different local structure produce identical outputs
(both ex2 and ex3 give `[5,5,5]/[0,0,0]/[0,0,0]`), the rule is almost
certainly reading a summary statistic of the input, not a geometric
transform.

## Reference implementation

```python
def transform(grid):
    H = len(grid)
    W = len(grid[0])
    colors = {v for row in grid for v in row}
    n = len(colors)
    out = [[0] * W for _ in range(H)]
    if n == 1:
        for c in range(W):
            out[0][c] = 5
    elif n == 2:
        for i in range(min(H, W)):
            out[i][i] = 5
    elif n == 3:
        for i in range(min(H, W)):
            out[i][W - 1 - i] = 5
    return out
```

Verified against `train[0..4]` and `test[0]` — all pass exactly.
