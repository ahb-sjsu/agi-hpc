---
type: sensei_note
task: 30
tags: [transformation, vertical-alignment, arc, primer]
written_by: The Primer
written_at: 2026-07-14
verified_by: run-against-train (all examples pass)
---

## The rule

Every colored object in the input grid shifts **vertically** so that its **topmost occupied row** aligns with the topmost row of the **color-1 (blue) anchor object**. Color-1 serves as the reference and does not move.

Each object preserves:
- Its internal shape and height
- Its column positions
- Its color value

Only the row coordinate changes, by a uniform offset calculated as:

```
shift = anchor_top - object_top
```

Where:
- `anchor_top` = minimum row index of any color-1 cell
- `object_top` = minimum row index of any cell of the current color

**Edge cases:**
- If color-1 is absent from the input, the output is a blank grid of the same dimensions
- If a shift would move cells out of bounds, those cells are dropped (not placed in output)

## Reference implementation

```python
def transform(grid):
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    
    if h == 0 or w == 0:
        return grid
    
    # Collect cells grouped by color
    colors = {}
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0:
                colors.setdefault(val, []).append((r, c))
    
    # Color-1 is the anchor; if absent, return blank grid
    if 1 not in colors:
        return [[0] * w for _ in range(h)]
    
    # Find anchor's topmost row
    anchor_top = min(r for r, _ in colors[1])
    
    # Create output grid
    output = [[0] * w for _ in range(h)]
    
    # Shift each color to align with anchor
    for color, cells in colors.items():
        color_top = min(r for r, _ in cells)
        shift = anchor_top - color_top
        
        for r, c in cells:
            nr = r + shift
            # Only place if within bounds
            if 0 <= nr < h:
                output[nr][c] = color
    
    return output
```

## Why this generalizes

This task belongs to the **vertical-alignment** primitive family. The core pattern is:

1. **Identify an anchor object** (here, the color-1 cells)
2. **Measure a reference coordinate** from the anchor (the topmost row)
3. **Apply a uniform offset** along one axis (vertical) to every other object so they share that reference coordinate

This generalizes to any input where:
- Multiple colored objects exist at different vertical positions
- One color is designated as the reference/anchor
- Objects must be aligned while preserving their internal structure and horizontal positions

The transformation is **deterministic**, **shape-preserving**, and handles edge cases (missing anchor, out-of-bounds shifts) gracefully. The same primitive could apply to horizontal alignment (using leftmost column as reference) or alignment to other anchor colors in different tasks.
