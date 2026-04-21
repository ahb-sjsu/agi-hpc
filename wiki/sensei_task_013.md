---
type: sensei_note
task: 13
tags: [transformation, periodic-replication, arc, primer]
written_by: The Primer
written_at: 2026-04-21
verified_by: run-against-train (all examples pass)
---

## The rule

This task involves **periodic replication** from two source pixels. The transformation works as follows:

1. **Identify sources**: Find the two non-zero pixels in the input grid. Record their positions (row, column) and values (colors).

2. **Determine direction**: 
   - If both sources are in the **same row**, replicate **horizontally** (fill columns across all rows).
   - If both sources are in the **same column**, replicate **vertically** (fill rows across all columns).
   - If sources differ in both row and column, use the dimension with the **smaller spacing** between them.

3. **Calculate period**: The repetition period equals **twice the spacing** between the two sources in the chosen dimension.

4. **Fill the grid**: 
   - For horizontal: At every `period` columns starting from each source's column, fill that entire column with the source's color.
   - For vertical: At every `period` rows starting from each source's row, fill that entire row with the source's color.

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
    
    # Order by position (top-left first)
    if r1 > r2 or (r1 == r2 and c1 > c2):
        r1, c1, v1, r2, c2, v2 = r2, c2, v2, r1, c1, v1
    
    row_spacing = abs(r2 - r1)
    col_spacing = abs(c2 - c1)
    
    result = np.zeros((h, w), dtype=int)
    
    if row_spacing == 0:
        # Same row - horizontal pattern
        period = 2 * col_spacing
        for col in range(c1, w, period):
            result[:, col] = v1
        for col in range(c2, w, period):
            result[:, col] = v2
    elif col_spacing == 0:
        # Same column - vertical pattern
        period = 2 * row_spacing
        for row in range(r1, h, period):
            result[row, :] = v1
        for row in range(r2, h, period):
            result[row, :] = v2
    else:
        # Different row and column - use smaller spacing
        if col_spacing < row_spacing:
            # Horizontal pattern
            period = 2 * col_spacing
            for col in range(c1, w, period):
                result[:, col] = v1
            for col in range(c2, w, period):
                result[:, col] = v2
        else:
            # Vertical pattern
            period = 2 * row_spacing
            for row in range(r1, h, period):
                result[row, :] = v1
            for row in range(r2, h, period):
                result[row, :] = v2
    
    return result.tolist()
```

## Why this generalizes

This belongs to the **periodic-replication** primitive family. The key insight is that two source points define a fundamental period (2× their spacing), and the pattern alternates between the two source colors at that period. The direction selection rule (same row → horizontal, same column → vertical, otherwise smaller spacing) handles all observed cases consistently. This pattern appears in tasks where sparse markers should propagate across the entire grid with regular spacing.

**Verification against all train examples:**
- Example 1: Sources at (0,5)=2 and (9,7)=8, col_spacing=2 < row_spacing=9 → horizontal, period=4 ✓
- Example 2: Sources at (0,5)=1 and (6,8)=3, col_spacing=3 < row_spacing=6 → horizontal, period=6 ✓
- Example 3: Sources at (5,0)=2 and (7,8)=3, row_spacing=2 < col_spacing=8 → vertical, period=4 ✓
- Example 4: Sources at (7,0)=4 and (11,0)=1, same column → vertical, period=8 ✓
