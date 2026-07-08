---
type: sensei_note
task: 178
tags: [extraction, run-length-compression, arc, primer]
written_by: The Primer
written_at: 2026-07-08
verified_by: run-against-train (all examples pass)
---

# Task 178: Run-Length Compression with Dimension Detection

## The rule

This task exhibits **structured redundancy** along one dimension of the input grid. The transformation must:

1. **Detect the redundancy axis**: Check whether all rows are identical to each other, OR whether all columns are identical to each other.

2. **Extract and compress**: Take one representative (a row or column) and apply **run-length compression** — collapse each run of consecutive identical values into a single value.

3. **Output in orthogonal orientation**: 
   - If rows were identical → output is a **single row** (1×N)
   - If columns were identical → output is a **single column** (N×1)

### Worked Examples

**Example 1** (column redundancy):
- Input: 3×3 grid where each column is [1,2,1]
- Extract column: [1,2,1]
- Compress: [1,2,1] (no consecutive duplicates)
- Output: [[1],[2],[1]] (3×1)

**Example 2** (row redundancy, no compression needed):
- Input: 3×3 grid where each row is [3,4,6]
- Extract row: [3,4,6]
- Compress: [3,4,6] (no consecutive duplicates)
- Output: [[3,4,6]] (1×3)

**Example 3** (row redundancy with compression):
- Input: 3×5 grid where each row is [2,3,3,8,1]
- Extract row: [2,3,3,8,1]
- Compress: [2,3,8,1] (consecutive 3s collapse to one)
- Output: [[2,3,8,1]] (1×4)

**Example 4** (column redundancy with compression):
- Input: 4×2 grid where each column is [2,6,8,8]
- Extract column: [2,6,8,8]
- Compress: [2,6,8] (consecutive 8s collapse to one)
- Output: [[2],[6],[8]] (3×1)

**Example 5** (column redundancy):
- Input: 6×4 grid where each column is [4,4,2,2,8,3]
- Extract column: [4,4,2,2,8,3]
- Compress: [4,2,8,3] (consecutive duplicates collapse)
- Output: [[4],[2],[8],[3]] (4×1)

**Test Example** (row redundancy):
- Input: 4×9 grid where each row is [1,1,2,3,3,3,8,8,4]
- Extract row: [1,1,2,3,3,3,8,8,4]
- Compress: [1,2,3,8,4]
- Output: [[1,2,3,8,4]] (1×5)

## Reference implementation

```python
def transform(grid):
    rows = len(grid)
    if rows == 0:
        return []
    cols = len(grid[0]) if rows > 0 else 0
    if cols == 0:
        return []
    
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
        # Extract one row and compress it, output as single row
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
        # Extract one column and compress it, output as single column
        col = [grid[i][0] for i in range(rows)]
        compressed = compress(col)
        return [[v] for v in compressed]
    
    # Fallback (should not happen for valid inputs)
    return grid
```

## Why this generalizes

This belongs to the **run-length-compression** primitive family, combined with **dimension-detection** logic.

**Key insights for future tasks:**

1. **Redundancy detection is orthogonal to compression**: First identify *which* dimension has global repetition (all rows same vs all columns same), *then* apply compression to the extracted representative.

2. **Output shape encodes the compression axis**: A 1×N output means rows were redundant (compressed horizontally). An N×1 output means columns were redundant (compressed vertically).

3. **Run-length compression is value-agnostic**: The `compress()` function works on any sequence of integers, collapsing consecutive duplicates regardless of the specific values. This pattern appears in many ARC tasks involving pattern simplification.

4. **Mutual exclusivity**: In valid inputs for this task, either all rows are identical OR all columns are identical, but not both (except for uniform grids). The detection order matters only for edge cases.
