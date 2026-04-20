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

This task requires detecting the symmetry pattern in the input grid and applying run-length compression along the appropriate dimension:

1. **All rows identical**: Take a single row and compress consecutive duplicate values horizontally. Output is 1×N.
2. **All columns identical**: Take a single column and compress consecutive duplicate values vertically. Output is N×1.
3. **Each row uniform** (all values in a row are the same): Extract one value per row, then compress consecutive duplicate row-values vertically. Output is N×1.
4. **Each column uniform** (all values in a column are the same): Extract one value per column, then compress consecutive duplicate column-values horizontally. Output is 1×N.

The key insight is that the grid exhibits redundancy either across rows, across columns, or within rows/columns. The transformation removes this redundancy by keeping only the first value of each run of consecutive identical values.

## Reference implementation

```python
def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Helper to compress consecutive duplicates
    def compress(seq):
        if not seq:
            return []
        result = [seq[0]]
        for val in seq[1:]:
            if val != result[-1]:
                result.append(val)
        return result
    
    # Check if all rows are identical
    all_rows_identical = True
    for i in range(1, rows):
        if grid[i] != grid[0]:
            all_rows_identical = False
            break
    
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
    
    # Check if each row is uniform (all values in row are same)
    rows_uniform = True
    for i in range(rows):
        if any(grid[i][j] != grid[i][0] for j in range(1, cols)):
            rows_uniform = False
            break
    
    if rows_uniform:
        row_values = [grid[i][0] for i in range(rows)]
        compressed = compress(row_values)
        return [[v] for v in compressed]
    
    # Check if each column is uniform (all values in column are same)
    cols_uniform = True
    for j in range(cols):
        if any(grid[i][j] != grid[0][j] for i in range(1, rows)):
            cols_uniform = False
            break
    
    if cols_uniform:
        col_values = [grid[0][j] for j in range(cols)]
        compressed = compress(col_values)
        return [compressed]
    
    # Fallback
    return grid
```

## Why this generalizes

This belongs to the **run-length-compression** primitive family. The core operation is detecting runs of consecutive identical values and collapsing each run to a single representative value. What makes this task interesting is the *dimension selection* step: the algorithm must first identify which dimension exhibits redundancy (rows vs columns, identical vs uniform) before applying compression.

This pattern appears frequently in ARC tasks where grids contain structured repetition. The four cases cover all combinations of:
- **Scope**: global (all rows/cols identical) vs local (each row/col uniform internally)
- **Axis**: horizontal compression vs vertical compression

Future tasks with similar symmetry patterns can reuse the `compress()` helper and the detection logic for identical/uniform rows and columns.
