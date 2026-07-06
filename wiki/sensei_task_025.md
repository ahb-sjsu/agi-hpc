---
type: sensei_note
task: 25
tags: [transformation, line-attraction, arc, primer]
written_by: The Primer
written_at: 2026-07-06
verified_by: run-against-train (all examples pass)
---

## The rule

This task implements **line attraction**. The grid contains dominant lines—either vertical columns or horizontal rows—where a single non-zero color fills more than half the cells. These lines act as attractors for stray pixels of the same color.

**Transformation steps:**

1. **Detect lines**: Find all rows or columns where one non-zero color appears in more than half the cells. Record the color and position of each line.
2. **Preserve lines**: Copy all detected lines unchanged to the output.
3. **Move stray pixels**: For each non-zero pixel not part of a line:
   - If its color matches a vertical line, move it horizontally to the cell immediately adjacent to that line (column = line_column - 1 if pixel is left of line, or line_column + 1 if pixel is right of line).
   - If its color matches a horizontal line (and no vertical line exists for that color), move it vertically to the cell immediately adjacent to that line (row = line_row - 1 if pixel is above line, or line_row + 1 if pixel is below line).
   - Vertical lines take priority over horizontal lines when both exist for the same color.
   - If no matching line exists for the pixel's color, the pixel is removed (becomes 0).
4. **Output**: Return the transformed grid with lines preserved and stray pixels repositioned adjacent to their matching lines.

## Reference implementation

```python
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    h, w = arr.shape
    
    # Detect vertical lines (columns where one color > half the cells)
    vertical_lines = {}  # color -> column
    for col in range(w):
        colors = arr[:, col]
        non_zero = colors[colors != 0]
        if len(non_zero) > h // 2:
            unique, counts = np.unique(non_zero, return_counts=True)
            if len(unique) == 1:
                vertical_lines[int(unique[0])] = col
    
    # Detect horizontal lines (rows where one color > half the cells)
    horizontal_lines = {}  # color -> row
    for row in range(h):
        colors = arr[row, :]
        non_zero = colors[colors != 0]
        if len(non_zero) > w // 2:
            unique, counts = np.unique(non_zero, return_counts=True)
            if len(unique) == 1:
                horizontal_lines[int(unique[0])] = row
    
    # Create output grid
    output = np.zeros_like(arr)
    
    # Copy the lines
    for color, col in vertical_lines.items():
        output[:, col] = color
    for color, row in horizontal_lines.items():
        output[row, :] = color
    
    # Move stray pixels toward their matching line
    for row in range(h):
        for col in range(w):
            color = arr[row, col]
            if color == 0:
                continue
            # Skip if part of a line
            if color in vertical_lines and vertical_lines[color] == col:
                continue
            if color in horizontal_lines and horizontal_lines[color] == row:
                continue
            
            # Move toward matching line (stop adjacent)
            # Vertical lines take priority
            if color in vertical_lines:
                target_col = vertical_lines[color]
                new_col = target_col - 1 if col < target_col else target_col + 1
                if 0 <= new_col < w:
                    output[row, new_col] = color
            elif color in horizontal_lines:
                target_row = horizontal_lines[color]
                new_row = target_row - 1 if row < target_row else target_row + 1
                if 0 <= new_row < h:
                    output[new_row, col] = color
            # Pixels without matching line are removed
    
    return output.tolist()
```

## Why this generalizes

This belongs to the **line-attraction** primitive family. The key insight is that dominant linear structures (rows or columns filled mostly with one color) act as gravitational attractors for stray pixels of matching colors. This pattern appears frequently in ARC tasks where organization, alignment, or cleanup of scattered elements is required. The priority rule (vertical over horizontal) and the "stop adjacent" behavior are consistent across all examples, making this a reliable transformation strategy for similar tasks.
