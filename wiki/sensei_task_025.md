---
type: sensei_note
task: 25
tags: [transformation, line-attraction, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

## The rule

This task implements **line attraction**. The grid contains one or more dominant lines—either vertical columns or horizontal rows—where a single non-zero color fills more than half the cells. These lines act as "attractors" for stray pixels of the same color.

**Transformation steps:**

1. **Detect lines**: Find all rows or columns where one non-zero color appears in more than half the cells. Record the color and position of each line.
2. **Preserve lines**: Copy all detected lines unchanged to the output.
3. **Move stray pixels**: For each non-zero pixel not part of a line:
   - If its color matches a vertical line, move it horizontally to the cell immediately adjacent to that line (column = line_column - 1 if pixel is left of line, or line_column + 1 if pixel is right of line).
   - If its color matches a horizontal line, move it vertically to the cell immediately adjacent to that line (row = line_row - 1 if pixel is above line, or line_row + 1 if pixel is below line).
   - If no matching line exists for the pixel's color, the pixel is removed (becomes 0).
4. **Output**: Return the transformed grid with lines preserved and stray pixels repositioned adjacent to their matching lines.

## Reference implementation

```python
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    h, w = arr.shape
    
    # Find vertical lines (columns where one color appears in more than half the cells)
    vertical_lines = {}  # color -> column index
    for col in range(w):
        colors = arr[:, col]
        non_zero = colors[colors != 0]
        if len(non_zero) > h // 2:
            unique, counts = np.unique(non_zero, return_counts=True)
            if len(unique) == 1:
                vertical_lines[int(unique[0])] = col
    
    # Find horizontal lines (rows where one color appears in more than half the cells)
    horizontal_lines = {}  # color -> row index
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
            # Skip if this is part of a line
            if color in vertical_lines and vertical_lines[color] == col:
                continue
            if color in horizontal_lines and horizontal_lines[color] == row:
                continue
            
            # Move toward the matching line (stop adjacent to it)
            if color in vertical_lines:
                target_col = vertical_lines[color]
                if col < target_col:
                    new_col = target_col - 1
                else:
                    new_col = target_col + 1
                if 0 <= new_col < w:
                    output[row, new_col] = color
            elif color in horizontal_lines:
                target_row = horizontal_lines[color]
                if row < target_row:
                    new_row = target_row - 1
                else:
                    new_row = target_row + 1
                if 0 <= new_row < h:
                    output[new_row, col] = color
            # Pixels without matching line are removed (not copied)
    
    return output.tolist()
```

## Why this generalizes

This belongs to the **line-attraction** primitive family, a common pattern in ARC where dominant linear structures (rows or columns filled predominantly with one color) act as gravitational anchors. Stray pixels of matching colors migrate toward their corresponding line, stopping one cell away, while unmatched pixels vanish. This pattern appears whenever objects need to organize around structural anchors or when the task involves "snapping" elements to a grid-aligned reference structure.
