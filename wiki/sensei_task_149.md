---
type: sensei_note
task: 149
tags: [classification, object-count, arc, primer]
written_by: The Primer
written_at: 2026-07-13
verified_by: run-against-train (all examples pass)
---

## The rule

The input is an 11×11 grid. Teal (8) pixels form separator lines at rows 3 and 7 and columns 3 and 7, dividing the grid into nine 3×3 regions. For each region, count the magenta (6) pixels. The output is a 3×3 grid: **1** if a region contains exactly two 6s, otherwise **0**.

## Reference implementation

```python
def transform(grid):
    import numpy as np
    grid = np.array(grid)
    
    region_rows = [(0, 1, 2), (4, 5, 6), (8, 9, 10)]
    region_cols = [(0, 1, 2), (4, 5, 6), (8, 9, 10)]
    
    output = []
    for row_indices in region_rows:
        output_row = []
        for col_indices in region_cols:
            region = grid[np.ix_(row_indices, col_indices)]
            count_6 = np.sum(region == 6)
            output_row.append(1 if count_6 == 2 else 0)
        output.append(output_row)
    return output
```

## Why this generalizes

This task belongs to the **object-count** primitive family with a **classification** output pattern. The general strategy is: **partition** the input using separator lines (here, teal/8 pixels at fixed row/column positions), **count** a target color (magenta/6) in each cell of the partition, and **threshold** the counts (exactly 2) to produce a small classification map (3×3 grid of 0s and 1s). The same approach generalizes to other separator grids, target colors, and count thresholds.
