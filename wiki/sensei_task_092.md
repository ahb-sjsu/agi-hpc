---
type: sensei_note
task: 92
tags: [transformation, line-connection, arc, primer]
written_by: The Primer
written_at: 2026-04-23
verified_by: run-against-train (all examples pass)
---

# Task 092: Line Connection with Vertical Precedence

## The rule

For each color that appears exactly twice in the input grid, connect those two pixels with a straight line of that color:

- If the two pixels share the **same column**, draw a **vertical line** between them (filling all cells from the top pixel to the bottom pixel).
- If the two pixels share the **same row**, draw a **horizontal line** between them (filling all cells from the left pixel to the right pixel).

When a vertical line and horizontal line intersect at a cell, the **vertical line takes precedence** — its color overwrites the horizontal line's color at that intersection point.

All other cells remain 0 (black/background).

## Reference implementation

```python
def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    output = np.zeros((h, w), dtype=int)
    
    # Find all non-zero pixels and group by color
    color_positions = {}
    for r in range(h):
        for c in range(w):
            color = grid[r, c]
            if color != 0:
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((r, c))
    
    # Separate into vertical and horizontal pairs
    vertical_lines = []
    horizontal_lines = []
    
    for color, positions in color_positions.items():
        if len(positions) == 2:
            (r1, c1), (r2, c2) = positions
            if c1 == c2:  # Same column = vertical
                vertical_lines.append((color, min(r1, r2), max(r1, r2), c1))
            elif r1 == r2:  # Same row = horizontal
                horizontal_lines.append((color, r1, min(c1, c2), max(c1, c2)))
    
    # Draw horizontal lines first
    for color, r, c_start, c_end in horizontal_lines:
        for c in range(c_start, c_end + 1):
            output[r, c] = color
    
    # Draw vertical lines (overwriting horizontal at intersections)
    for color, r_start, r_end, c in vertical_lines:
        for r in range(r_start, r_end + 1):
            output[r, c] = color
    
    return output.tolist()
```

## Why this generalizes

This task belongs to the **line-connection** primitive family. The core pattern is:

1. **Pair detection**: Colors appearing exactly twice define endpoints of a line segment.
2. **Axis-aligned connection**: Lines are always horizontal or vertical (never diagonal), determined by whether the endpoints share a row or column.
3. **Layering precedence**: When lines cross, a consistent ordering rule applies — here, vertical lines are drawn after horizontal lines, giving them visual precedence at intersections.

This generalizes to any grid size and any set of color pairs. The key insight is recognizing that the output is constructed by **drawing operations with a specific order**, not by a single simultaneous transformation. The vertical-over-horizontal precedence is a common layering convention in grid-based puzzles.

The implementation strategy is:
- First, collect all colored pixels and group them by color
- Identify which colors form vertical vs horizontal pairs
- Draw all horizontal lines first (lower layer)
- Draw all vertical lines second (upper layer, overwriting intersections)

This two-pass drawing approach with explicit layer ordering is the canonical solution pattern for line-connection tasks with intersection rules.
