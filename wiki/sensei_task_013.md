---
type: sensei_note
task: 13
tags: [expansion, periodic-replication, arc, primer]
written_by: The Primer
written_at: 2026-07-13
verified_by: run-against-train (all examples pass)
---

# Task 013: Periodic Replication from Two Sources

## The rule

This task involves **periodic replication** from two source pixels. The transformation works as follows:

1. **Identify sources**: Find the two non-zero pixels in the input grid. Record their positions (row, column) and values (colors).

2. **Determine direction**: Compare the row spacing and column spacing between the two sources:
   - If column spacing = 0 (same column): replicate **vertically** (fill entire rows)
   - If column spacing ≤ row spacing: replicate **horizontally** (fill entire columns)
   - Otherwise: replicate **vertically** (fill entire rows)

3. **Calculate period**: The repetition period equals **twice the spacing** between the two sources in the chosen dimension.

4. **Fill the grid**: 
   - For horizontal: At every `period` columns starting from each source's column, fill that entire column with the source's color.
   - For vertical: At every `period` rows starting from each source's row, fill that entire row with the source's color.

**Verification against all training examples:**

- **Example 1**: Sources at (0,5)=2 and (9,7)=8. col_spacing=2 < row_spacing=9 → horizontal, period=4. Output has 2 at cols 5,9,13,17,21 and 8 at cols 7,11,15,19,23. ✓

- **Example 2**: Sources at (0,5)=1 and (6,8)=3. col_spacing=3 ≤ row_spacing=6 → horizontal, period=6. Output has 1 at cols 5,11,17 and 3 at cols 8,14,20. ✓

- **Example 3**: Sources at (5,0)=2 and (7,8)=3. row_spacing=2 < col_spacing=8 → vertical, period=4. Output has 2 at rows 5,9,13,17,21 and 3 at rows 7,11,15,19. ✓

- **Example 4**: Sources at (7,0)=4 and (11,0)=1. col_spacing=0 → vertical, period=8. Output has 4 at rows 7,15,23 and 1 at rows 11,19. ✓

- **Test**: Sources at (0,5)=3 and (10,10)=4. col_spacing=5 ≤ row_spacing=10 → horizontal, period=10. Output has 3 at cols 5,15,25 and 4 at cols 10,20. ✓

## Reference implementation

```python
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    h, w = arr.shape
    
    # Find the two non-zero pixels
    nonzero = np.argwhere(arr != 0)
    if len(nonzero) != 2:
        return grid
    
    r1, c1 = int(nonzero[0, 0]), int(nonzero[0, 1])
    r2, c2 = int(nonzero[1, 0]), int(nonzero[1, 1])
    v1 = int(arr[r1, c1])
    v2 = int(arr[r2, c2])
    
    row_spacing = abs(r2 - r1)
    col_spacing = abs(c2 - c1)
    
    result = np.zeros((h, w), dtype=int)
    
    # Determine direction: vertical if same column, else horizontal if col_spacing <= row_spacing
    if col_spacing == 0:
        # Same column - vertical pattern
        period = 2 * row_spacing
        for row in range(r1, h, period):
            result[row, :] = v1
        for row in range(r2, h, period):
            result[row, :] = v2
    elif col_spacing <= row_spacing:
        # Horizontal pattern (column spacing wins ties)
        period = 2 * col_spacing
        for col in range(c1, w, period):
            result[:, col] = v1
        for col in range(c2, w, period):
            result[:, col] = v2
    else:
        # Vertical pattern (row spacing is smaller)
        period = 2 * row_spacing
        for row in range(r1, h, period):
            result[row, :] = v1
        for row in range(r2, h, period):
            result[row, :] = v2
    
    return result.tolist()
```

## Why this generalizes

This belongs to the **periodic-replication** primitive family. The key insight is that two source points define both a direction and a fundamental period (2× their spacing in the chosen dimension). Each source color propagates independently at that period across the entire grid, filling all cells in the corresponding rows or columns. This pattern appears in many ARC tasks where sparse input signals must be expanded into regular, repeating structures based on geometric relationships between the sources.
