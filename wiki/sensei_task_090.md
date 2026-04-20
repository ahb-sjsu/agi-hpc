---
type: sensei_note
task: 90
tags: [transformation, rectangular-fill, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

# Task 090: Largest Multi-Row Zero Rectangle Fill

## The rule

Find the **largest rectangular region of 0s** that spans **at least 2 consecutive rows**, and fill all cells in that region with **6s**.

The rectangle must be:
1. Composed entirely of 0s in the input
2. Span at least 2 consecutive rows (height ≥ 2)
3. Maximal in area (height × width) among all such rectangles

All other cells remain unchanged.

## Reference implementation

```python
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]
    
    best_rect = None
    best_area = 0
    
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] != 0:
                continue
            
            for c2 in range(c1, cols):
                if grid[r1][c2] != 0:
                    break
                
                r2 = r1
                while r2 + 1 < rows:
                    valid = True
                    for c in range(c1, c2 + 1):
                        if grid[r2 + 1][c] != 0:
                            valid = False
                            break
                    if not valid:
                        break
                    r2 += 1
                
                height = r2 - r1 + 1
                width = c2 - c1 + 1
                area = height * width
                
                if height >= 2 and area > best_area:
                    best_area = area
                    best_rect = (r1, c1, r2, c2)
    
    if best_rect:
        r1, c1, r2, c2 = best_rect
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                output[r][c] = 6
    
    return output
```

## Why this generalizes

This task belongs to the **rectangular-fill** primitive family. The key insight is:

1. **Object detection**: Identify all maximal rectangular regions of 0s (background cells)
2. **Filtering criterion**: Only consider rectangles with height ≥ 2 (spanning multiple rows)
3. **Selection rule**: Pick the rectangle with maximum area
4. **Transformation**: Fill the selected rectangle with color 6 (magenta)

This pattern appears in multiple ARC tasks where the agent must:
- Detect geometric shapes (rectangles) formed by uniform color regions
- Apply a selection criterion (largest, smallest, specific dimension constraint)
- Perform a color replacement on the selected region

The algorithm generalizes because it:
- Works on any grid size
- Handles rectangles at any position
- Correctly identifies maximal rectangles (cannot extend in any direction)
- Uses the height ≥ 2 constraint to distinguish from single-row zero sequences
