---
type: sensei_note
task: 30
tags: [transformation, vertical-alignment, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

## The rule

All colored objects shift vertically to align their top rows with the top row of the color-1 (blue) object. Color-1 serves as the anchor and doesn't move. Horizontal positions are preserved; only vertical positions change.

The transformation works as follows:
1. Identify all non-zero colored cells and group them by color
2. Find the minimum row index (top) of the color-1 object — this is the anchor position
3. For each color, calculate the vertical shift needed: `shift = ones_top - color_top`
4. Apply this shift to all cells of that color, keeping column positions unchanged
5. Place the shifted cells in the output grid

## Reference implementation

```python
def transform(grid):
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    
    # Collect cells by color
    colors = {}
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0:
                if val not in colors:
                    colors[val] = []
                colors[val].append((r, c))
    
    # If no 1s exist, return empty grid
    if 1 not in colors:
        return [[0] * w for _ in range(h)]
    
    # Find the top row of the 1s object
    ones_top = min(r for r, c in colors[1])
    
    # Create output grid
    output = [[0] * w for _ in range(h)]
    
    # For each color, shift vertically to align with 1s top
    for color, cells in colors.items():
        color_top = min(r for r, c in cells)
        shift = ones_top - color_top
        
        for r, c in cells:
            new_r = r + shift
            if 0 <= new_r < h:
                output[new_r][c] = color
    
    return output
```

## Why this generalizes

This belongs to the **vertical-alignment** primitive family. The pattern is:

- **Anchor identification**: One color (here, color-1) serves as the reference point.
- **Relative transformation**: All other objects transform relative to the anchor's position.
- **Preservation of shape**: Objects maintain their internal structure; only their absolute position changes.
- **Single-axis transformation**: Only vertical (row) coordinates change; horizontal (column) coordinates are preserved.

This generalizes to any task where objects need to align to a reference object along a single axis. The anchor color could vary by task, but the mechanism (find anchor position, compute shift, apply to all objects) remains the same.

## Verification

This implementation has been verified against all 3 training examples:

- **Example 1**: Color-1 top at row 1 → colors 2 and 4 shift to align (color-2 down 1, color-4 up 1)
- **Example 2**: Color-1 top at row 5 → colors 2 and 4 shift down to align (color-2 down 3, color-4 down 5)
- **Example 3**: Color-1 top at row 2 → colors 2 and 4 shift to align (color-2 down 1, color-4 up 1)

All training examples pass exactly.
