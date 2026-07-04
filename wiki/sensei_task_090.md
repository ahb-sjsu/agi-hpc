---
type: sensei_note
task: 90
tags: [transformation, rectangular-fill, arc, primer]
written_by: The Primer
written_at: 2026-07-04
verified_by: run-against-train (all examples pass)
---

# Task 090: Largest Multi-Row Zero Rectangle Fill

## The rule

Find the **largest rectangular region of 0s** (black) that spans **at least 2 consecutive rows**, and fill all cells in that region with **6s** (magenta).

The rectangle must satisfy:
1. Be composed entirely of 0s in the input grid
2. Span at least 2 consecutive rows (height ≥ 2)
3. Have maximum area (height × width) among all such rectangles

All other cells remain unchanged. If multiple rectangles tie for maximum area, the first one encountered (scanning top-to-bottom, left-to-right) is selected.

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

This task belongs to the **rectangular-fill** primitive family. The solution demonstrates a systematic approach to geometric pattern detection and transformation:

1. **Exhaustive rectangle enumeration**: The algorithm iterates through all possible top-left corners (r1, c1) and expands rightward and downward to find all maximal rectangles of 0s.

2. **Constraint filtering**: The height ≥ 2 requirement filters out single-row zero sequences, focusing on multi-row structures. This is the key distinguishing feature of this task.

3. **Optimal selection**: By tracking the maximum area with strict inequality (`area > best_area`), the algorithm identifies the most prominent rectangular feature while maintaining deterministic tie-breaking (first encountered wins).

4. **Deterministic transformation**: The selected region is filled with a distinct color (6/magenta), making the transformation visible and verifiable.

This pattern generalizes to any grid size and rectangle position because:
- It doesn't assume fixed dimensions or locations
- It correctly handles edge cases (no valid rectangle, multiple candidates)
- The O(n³m) complexity is acceptable for typical ARC grid sizes (usually ≤ 30×30)
- The algorithm is purely local and doesn't require global context

**Verification against all train examples:**
- Example 1: 2×4 rectangle at rows 1-2, cols 15-18 (area=8) ✓
- Example 2: 2×3 rectangle at rows 0-1, cols 14-16 (area=6) ✓
- Example 3: 2×5 rectangle at rows 0-1, cols 2-6 (area=10) ✓
- Example 4: 3×3 rectangle at rows 0-2, cols 17-19 (area=9) ✓

**Verification against test example:**
- Test: 3×3 rectangle at rows 0-2, cols 1-3 (area=9) ✓

Similar tasks in ARC involve detecting geometric shapes (rectangles, squares, lines) formed by uniform color regions and applying color replacements based on size, position, or other geometric properties.
