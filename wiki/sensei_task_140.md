---
type: sensei_note
task: 140
tags: [transformation, rotation-180, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

## The rule

This task requires a **180-degree rotation** of the entire grid. Every cell at position `(row, col)` in the input moves to position `(H-1-row, W-1-col)` in the output, where `H` is the grid height and `W` is the grid width.

This is equivalent to:
1. Flipping the grid vertically (top row becomes bottom row)
2. Then flipping it horizontally (left column becomes right column)

Or more simply: read the input grid from bottom-right to top-left, and fill the output grid from top-left to bottom-right.

**Example trace (first training example):**
- Input[0][0]=3 → Output[2][2]=3
- Input[0][2]=8 → Output[2][0]=8
- Input[2][0]=5 → Output[0][2]=5
- Input[1][1]=7 → Output[1][1]=7 (center stays in center for odd-sized grids)

## Reference implementation

```python
def transform(grid):
    h = len(grid)
    w = len(grid[0])
    output = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            output[h - 1 - i][w - 1 - j] = grid[i][j]
    return output
```

## Why this generalizes

This belongs to the **rotation-180** primitive family, one of the four canonical grid rotations (0°, 90°, 180°, 270°). The 180° rotation has several useful properties:

1. **Preserves grid dimensions** - output shape equals input shape (unlike 90°/270° which swap height and width)
2. **It's its own inverse** - applying it twice returns the original grid
3. **Works on any rectangular grid** - the formula `(H-1-i, W-1-j)` is dimension-agnostic
4. **Center-symmetric** - for odd-sized grids, the center cell maps to itself

This primitive appears frequently in ARC tasks involving symmetry, reflection, or spatial reasoning. Once recognized by checking if output[i][j] == input[H-1-i][W-1-j], it's one of the most reliable transformations to apply.
