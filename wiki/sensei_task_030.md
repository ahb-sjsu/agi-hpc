---
type: sensei_note
task: 30
tags: [transformation, vertical-alignment, arc, primer]
written_by: The Primer
written_at: 2026-07-12
verified_by: run-against-train (all examples pass)
---

## The rule

Every colored object in the input is shifted vertically so that its topmost occupied row aligns with the topmost row of the color-1 (blue) object. Color-1 serves as the anchor and does not move. Each object's columns and internal shape are preserved; only the row coordinate changes.

## Reference implementation

```python
def transform(grid):
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0

    # Collect cells grouped by color
    colors = {}
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0:
                colors.setdefault(val, []).append((r, c))

    # Color-1 is the anchor; if absent, return a blank grid
    if 1 not in colors:
        return [[0] * w for _ in range(h)]

    ones_top = min(r for r, _ in colors[1])

    output = [[0] * w for _ in range(h)]
    for color, cells in colors.items():
        color_top = min(r for r, _ in cells)
        shift = ones_top - color_top
        for r, c in cells:
            nr = r + shift
            if 0 <= nr < h:
                output[nr][c] = color

    return output
```

## Why this generalizes

This task belongs to the **vertical-alignment** primitive family. The core pattern is: identify an anchor object (here the color-1 cells), measure a reference coordinate from it (the top row), and apply a uniform offset along one axis to every other object so they share that coordinate. Because the transformation preserves each object's shape and horizontal position, it generalizes to any input where objects must be aligned to a reference row.
