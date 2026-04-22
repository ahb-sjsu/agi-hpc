---
type: sensei_note
task: 167
tags: [transformation, count-distinct-colors, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

## The rule

Count the number of distinct colors present anywhere in the input grid. The output pattern depends ONLY on this count:

- **1 distinct color** -> fill the entire top row with 5s, all other cells 0
- **2 distinct colors** -> draw the main diagonal with 5s (`out[i][i] = 5`), all other cells 0
- **3 distinct colors** -> draw the anti-diagonal with 5s (`out[i][W-1-i] = 5`), all other cells 0

The output grid has the same dimensions as the input. The actual color values and their spatial arrangement in the input are irrelevant—only the cardinality of the color set matters.

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

## Why this generalizes

This task belongs to the **count-distinct-colors** primitive family. The key insight is that visually different inputs can produce identical outputs when they share the same color cardinality. For example, Train 3 (all 4s) and Train 4 (all 3s) both have 1 distinct color and both produce the top-row pattern. This is a strong signal that the rule operates on a summary statistic rather than geometric structure.

When you see multiple training examples with very different local patterns yielding the same output, suspect a classification rule based on a global property (color count, object count, symmetry type) rather than a transformation of local structure. The output pattern itself (top row vs. diagonal vs. anti-diagonal) encodes the classification result.
