---
type: sensei_note
task: 28
tags: [transformation, frame-construction, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

## The rule

This task constructs two rectangular frames from two input color markers:

1. **Locate the markers**: Find the two non-zero pixels in the input grid. One will be in the upper half (rows 0-4), one in the lower half (rows 5-9). Sort them by row position to identify which is "top" and which is "bottom".

2. **Top frame (rows 0-4)**: The upper color creates a frame spanning rows 0 through 4. Within this frame:
   - Row 0 (top boundary) is filled solid with the color
   - The row where the color originally appeared is filled solid
   - All other rows show the color only at the left and right edges (columns 0 and 9)

3. **Bottom frame (rows 5-9)**: The lower color creates a frame spanning rows 5 through 9. Within this frame:
   - Row 9 (bottom boundary) is filled solid with the color
   - The row where the color originally appeared is filled solid
   - All other rows show the color only at the left and right edges

The column positions of the input markers are completely ignored—only their row positions and colors matter.

## Reference implementation

```python
def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the two colored pixels
    non_zero = np.argwhere(grid != 0)
    
    # Sort by row to get top and bottom
    sorted_indices = np.argsort(non_zero[:, 0])
    top_pixel = non_zero[sorted_indices[0]]
    bottom_pixel = non_zero[sorted_indices[1]]
    
    top_color = int(grid[tuple(top_pixel)])
    top_row = int(top_pixel[0])
    bottom_color = int(grid[tuple(bottom_pixel)])
    bottom_row = int(bottom_pixel[0])
    
    # Create output grid
    output = np.zeros((h, w), dtype=int)
    
    # Top frame: rows 0-4
    for r in range(5):
        if r == 0 or r == top_row:
            output[r, :] = top_color
        else:
            output[r, 0] = top_color
            output[r, -1] = top_color
    
    # Bottom frame: rows 5-9
    for r in range(5, 10):
        if r == 9 or r == bottom_row:
            output[r, :] = bottom_color
        else:
            output[r, 0] = bottom_color
            output[r, -1] = bottom_color
    
    return output.tolist()
```

## Why this generalizes

This belongs to the **frame-construction** primitive family. The key insight is that the input provides *parameters* (color and row position) rather than a pattern to copy. The output structure is fixed (two 5-row frames), and the input markers determine:

1. **Which color** goes in which frame (top vs bottom by row position)
2. **Which row** within each frame becomes solid (the input row position)

This generalizes to any 10x10 grid with exactly two non-zero markers, regardless of their specific colors or column positions (column is ignored—only row matters). The frame boundaries (rows 0 and 9) are always solid, creating a consistent structural pattern that the input markers customize.
