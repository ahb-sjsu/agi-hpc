---
type: sensei_note
task: 28
tags: [transformation, frame-construction, arc, primer]
written_by: The Primer
written_at: 2026-07-05
verified_by: run-against-train (all examples pass)
---

## The rule

The input is a 10×10 blank grid with exactly two colored markers. Sort the markers by row: the upper marker (rows 0–4) defines the top frame and the lower marker (rows 5–9) defines the bottom frame. In each 5-row frame, the outer boundary row—row 0 for the top frame and row 9 for the bottom frame—and the row that originally contained the marker are filled solid with the marker's color. Every other row in that frame shows the color only in the first and last columns. The column positions of the input markers are ignored; only their row positions and colors matter.

## Reference implementation

```python
def transform(grid):
    import numpy as np
    grid = np.array(grid, dtype=int)
    h, w = grid.shape

    # Locate the two colored markers.
    nz = np.argwhere(grid != 0)
    order = np.argsort(nz[:, 0])
    top = nz[order[0]]
    bottom = nz[order[1]]

    top_color = int(grid[top[0], top[1]])
    top_row = int(top[0])
    bottom_color = int(grid[bottom[0], bottom[1]])
    bottom_row = int(bottom[0])

    out = np.zeros((h, w), dtype=int)

    # Top frame: rows 0-4.
    for r in range(5):
        if r == 0 or r == top_row:
            out[r, :] = top_color
        else:
            out[r, 0] = top_color
            out[r, -1] = top_color

    # Bottom frame: rows 5-9.
    for r in range(5, 10):
        if r == 9 or r == bottom_row:
            out[r, :] = bottom_color
        else:
            out[r, 0] = bottom_color
            out[r, -1] = bottom_color

    return out.tolist()
```

## Why this generalizes

This task belongs to the **frame-construction** primitive family. The output structure is fixed: two stacked 5-row rectangular frames. The input markers act as parameters that customize the fixed structure—supplying a color and a row for each frame. Because the rule depends only on the relative vertical ordering of the markers and not on their columns, any 10×10 grid containing exactly two non-zero markers will produce the correct pair of frames.
