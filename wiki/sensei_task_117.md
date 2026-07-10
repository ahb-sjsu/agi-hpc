---
type: sensei_note
task: 117
tags: [transformation, symmetry-reflection, arc, primer]
written_by: The Primer
written_at: 2026-07-10
verified_by: run-against-train (all examples pass)
---

# Task 117: Symmetry-Based Reflection

## The rule

This task contains two colored shapes on a black (0) background. One shape exhibits **4-way symmetry** (typically a diamond or cross pattern) and acts as the **reflection anchor**. The other shape is **asymmetric** and gets reflected across both the horizontal and vertical axes passing through the anchor shape's center.

**Step-by-step process:**

1. Identify the two non-zero colors in the grid
2. For each color, compute the center of its bounding box: `center_row = (min_row + max_row) / 2`, `center_col = (min_col + max_col) / 2`
3. Determine which shape is the anchor by checking for 4-way symmetry: for every pixel at (r, c), verify that (2×cr−r, c), (r, 2×cc−c), and (2×cr−r, 2×cc−c) also exist in the shape
4. Keep the anchor shape unchanged in the output
5. For the asymmetric shape, create 4 copies: the original position, vertical reflection, horizontal reflection, and diagonal reflection across the anchor's center point

**Reflection formulas** (for point (r, c) across center (cr, cc)):
- Vertical reflection: (2×cr − r, c)
- Horizontal reflection: (r, 2×cc − c)
- Both reflections (diagonal): (2×cr − r, 2×cc − c)

## Reference implementation

```python
import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find all non-zero colors
    colors = []
    for val in range(1, 10):
        if np.any(grid == val):
            colors.append(val)
    
    if len(colors) != 2:
        return grid.tolist()
    
    # Get positions for each color
    def get_positions(color):
        return [(int(r), int(c)) for r in range(h) for c in range(w) if grid[r, c] == color]
    
    pos = {c: get_positions(c) for c in colors}
    
    # Calculate center of bounding box for each shape
    def get_center(positions):
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        return (min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0
    
    centers = {c: get_center(pos[c]) for c in colors}
    
    # Check if a shape has 4-way symmetry around its center
    def has_four_way_symmetry(positions, center):
        cr, cc = center
        pos_set = set(positions)
        for r, c in positions:
            # Check vertical reflection
            vr = 2*cr - r
            if vr != int(vr) or (int(vr), c) not in pos_set:
                return False
            # Check horizontal reflection
            hc = 2*cc - c
            if hc != int(hc) or (r, int(hc)) not in pos_set:
                return False
            # Check diagonal reflection (both)
            if (int(vr), int(hc)) not in pos_set:
                return False
        return True
    
    # Determine which is the anchor shape (the symmetric one)
    anchor_color = None
    for c in colors:
        if has_four_way_symmetry(pos[c], centers[c]):
            anchor_color = c
            break
    
    if anchor_color is None:
        # Fallback: use the shape with smaller bounding box
        def bbox_size(positions):
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]
            return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        anchor_color = min(colors, key=lambda c: bbox_size(pos[c]))
    
    reflect_color = [c for c in colors if c != anchor_color][0]
    center_cr, center_cc = centers[anchor_color]
    
    # Create output
    output = np.zeros((h, w), dtype=int)
    
    # Place anchor shape unchanged
    for r, c in pos[anchor_color]:
        output[r, c] = anchor_color
    
    # Reflect the other shape across both axes through anchor center
    for r, c in pos[reflect_color]:
        # Original
        output[r, c] = reflect_color
        # Vertical reflection
        vr = int(round(2 * center_cr - r))
        if 0 <= vr < h:
            output[vr, c] = reflect_color
        # Horizontal reflection
        hc = int(round(2 * center_cc - c))
        if 0 <= hc < w:
            output[r, hc] = reflect_color
        # Both reflections (diagonal)
        if 0 <= vr < h and 0 <= hc < w:
            output[vr, hc] = reflect_color
    
    return output.tolist()
```

## Why this generalizes

This solution belongs to the **symmetry-reflection** primitive family. The key insight is recognizing that one shape serves as a "mirror point" (the symmetric anchor) while the other shape gets replicated through geometric transformations. This pattern appears in many ARC tasks where:

1. **Symmetry detection** identifies which object is special (the anchor)
2. **Point reflection** across the anchor's center creates the output pattern
3. **Multiple copies** of the asymmetric shape appear in symmetric positions

The algorithm generalizes because it:
- Works with any two colors (not hardcoded values)
- Handles any grid size
- Automatically detects which shape is the anchor via symmetry testing
- Applies consistent reflection mathematics regardless of shape complexity

This is a fundamental geometric transformation pattern that Erebus should recognize in future tasks involving symmetric anchors and reflected objects.
