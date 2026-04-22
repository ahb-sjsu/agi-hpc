---
type: sensei_note
task: 149
tags: [classification, object-count, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

# Task 149: Region-based Magenta Count Classification

## The rule

The input is an 11x11 grid containing three colors:
- **0** (black/empty)
- **6** (magenta) - the objects to count
- **8** (teal) - separator lines

The teal (8) pixels form a grid pattern: horizontal lines at rows 3 and 7, and vertical lines at columns 3 and 7. This divides the grid into **9 regions** arranged in a 3x3 layout, where each region is 3x3 cells.

For each of the 9 regions, count how many magenta (6) pixels it contains. The output is a 3x3 grid where:
- **1** = the corresponding input region contains **exactly 2** magenta pixels
- **0** = the corresponding input region contains any other number of magenta pixels (0, 1, 3, etc.)

## Reference implementation

```python
def transform(grid):
    import numpy as np
    grid = np.array(grid)
    
    # The grid is 11x11 with separator lines (value 8) at rows 3, 7 and cols 3, 7
    # This creates 9 regions of 3x3 cells each
    
    # Region row indices (skip separator rows 3 and 7)
    region_rows = [(0, 1, 2), (4, 5, 6), (8, 9, 10)]
    # Region column indices (skip separator cols 3 and 7)
    region_cols = [(0, 1, 2), (4, 5, 6), (8, 9, 10)]
    
    output = []
    
    for row_indices in region_rows:
        output_row = []
        for col_indices in region_cols:
            # Extract the 3x3 region
            region = grid[np.ix_(row_indices, col_indices)]
            # Count the number of magenta (6) pixels
            count_6 = np.sum(region == 6)
            # Output 1 if exactly 2 sixes, else 0
            output_row.append(1 if count_6 == 2 else 0)
        output.append(output_row)
    
    return output
```

## Why this generalizes

This task belongs to the **object-count** primitive family. The key insight is recognizing the separator pattern (teal lines) that partitions the input into independent regions, then applying a threshold-based classification to each region based on object count.

This pattern generalizes to:
1. Any grid with separator lines creating regular regions
2. Any target color to count (not just magenta/6)
3. Any threshold value (not just 2)

The strategy is: **partition → count → threshold → classify**. This is a common ARC pattern where spatial structure defines independent subproblems.
