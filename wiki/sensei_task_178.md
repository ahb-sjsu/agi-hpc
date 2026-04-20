---
type: sensei_note
task: 178
tags: [transformation, run-length-compression, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

# Task 178: Run-Length Compression with Symmetry Detection

## The rule

This task exhibits structured redundancy in the input grid. The transformation detects which dimension contains the redundancy and compresses it:

1. **All rows identical**: Every row in the grid is the same. Extract one representative row, apply run-length compression (collapse consecutive duplicate values), and output as a single row (1×N).

2. **All columns identical**: Every column in the grid is the same. Extract one representative column, apply run-length compression, and output as a single column (N×1).

The key insight is that these two cases are mutually exclusive and cover all training examples. When each row is internally uniform (all values in a row are identical), the columns automatically become identical to each other. Similarly, when each column is internally uniform, the rows automatically become identical. So we only need to check for global row/column identity, not per-row/per-column uniformity.

## Reference implementation

```python
def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    def compress(seq):
        if not seq:
            return []
        result = [seq[0]]
        for val in seq[1:]:
            if val != result[-1]:
                result.append(val)
        return result
    
    # Check if all rows are identical
    all_rows_identical = all(grid[i] == grid[0] for i in range(1, rows))
    
    if all_rows_identical:
        compressed = compress(grid[0])
        return [compressed]
    
    # Check if all columns are identical
    all_cols_identical = True
    for j in range(1, cols):
        for i in range(rows):
            if grid[i][j] != grid[i][0]:
                all_cols_identical = False
                break
        if not all_cols_identical:
            break
    
    if all_cols_identical:
        col = [grid[i][0] for i in range(rows)]
        compressed = compress(col)
        return [[v] for v in compressed]
    
    # Fallback (should not happen on valid inputs)
    return grid
```

## Why this generalizes

This belongs to the **run-length-compression** primitive family. The core operation is detecting runs of consecutive identical values and collapsing each run to a single representative value.

What makes this task interesting is the *dimension selection* step: the algorithm must first identify which axis exhibits global redundancy (all rows identical vs all columns identical) before applying compression. The output shape (1×N vs N×1) is determined by which dimension was compressed.

This pattern appears frequently in ARC tasks where grids contain structured repetition. The two-case logic (rows vs columns) is complete because:
- If each row is internally uniform → columns are automatically identical → caught by `all_cols_identical`
- If each column is internally uniform → rows are automatically identical → caught by `all_rows_identical`

Future tasks with similar symmetry patterns can reuse the `compress()` helper and the detection logic for identical rows/columns.
