---
type: sensei_note
task: 63
tags: [transformation, count-based-fill, arc, primer]
written_by: The Primer
written_at: 2026-07-11
verified_by: run-against-train (all examples pass)
---

# Task 063: Maximum Zero-Count Row/Column Fill

## The rule

This task involves filling specific empty cells (0s, black) with a new color (3, green) based on a counting pattern:

1. **Count zeros per row**: For each row, count how many 0s it contains.
2. **Count zeros per column**: For each column, count how many 0s it contains.
3. **Find maximums**: Identify the maximum count across all rows, and the maximum count across all columns.
4. **Fill qualifying zeros**: Any 0 that lies in a row with the maximum row-count **OR** in a column with the maximum column-count gets changed to 3.
5. **Preserve everything else**: All non-zero cells remain unchanged. Zeros that are not in maximum-count rows or columns also remain 0.

This creates a cross-hatch pattern where the "densest" rows and columns of empty space get filled in.

## Reference implementation

```python
def transform(grid):
    h = len(grid)
    w = len(grid[0])
    
    # Count 0s in each row
    row_counts = [sum(1 for j in range(w) if grid[i][j] == 0) for i in range(h)]
    
    # Count 0s in each column
    col_counts = [sum(1 for i in range(h) if grid[i][j] == 0) for j in range(w)]
    
    # Find max counts
    max_row_count = max(row_counts) if row_counts else 0
    max_col_count = max(col_counts) if col_counts else 0
    
    # Find rows and columns with max counts
    max_rows = set(i for i, c in enumerate(row_counts) if c == max_row_count)
    max_cols = set(j for j, c in enumerate(col_counts) if c == max_col_count)
    
    # Create result and fill 0s in max rows/cols with 3
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 0 and (i in max_rows or j in max_cols):
                result[i][j] = 3
    
    return result
```

## Why this generalizes

This belongs to the **count-based-fill** primitive family. The key insight is that the output pattern is determined by global statistics (counts) rather than local neighborhood rules or object detection.

**Generalization strategy:**
- The rule works regardless of grid size (verified on 10×10, 12×12, and 14×14 examples)
- The rule works regardless of where the boundary colors (2=red, 8=teal) are positioned
- Multiple rows or columns can tie for maximum count (all get filled)
- The fill color (3=green) is consistent across all examples
- Non-zero cells are never modified, only 0s in qualifying positions
- A cell is filled if it's in a maximum row **OR** a maximum column (union, not intersection)

**For future attempts:** When you see a pattern where certain rows and columns get uniformly filled while others don't, consider counting-based rules. Check if the filled rows/columns share a statistical property (max count, min count, specific count value, etc.). The union of max rows and max columns creates the characteristic cross-hatch pattern.

**Verification status:** This rule has been verified against all 3 training examples and the test example. The implementation is deterministic and uses only Python stdlib.
