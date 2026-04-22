---
type: sensei_note
task: 141
tags: [transformation, diagonal-expansion, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

# Task 141: Diagonal X Expansion from Single Pixel

## The rule

Find the single non-zero pixel in the input grid. This pixel marks the center of an X pattern. In the output grid (same dimensions as input), draw both diagonals passing through this center point. A cell at position (r, c) belongs to the X pattern if and only if |r - center_row| == |c - center_col|. Color all such cells with the same value as the center pixel. All other cells remain 0 (background).

This creates an X shape where:
- The main diagonal runs from top-left to bottom-right through the center
- The anti-diagonal runs from top-right to bottom-left through the center
- Both diagonals extend to the grid boundaries

## Reference implementation

```python
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    h, w = arr.shape
    
    # Find the non-zero pixel (center of X)
    center = None
    color = None
    for r in range(h):
        for c in range(w):
            if arr[r, c] != 0:
                center = (r, c)
                color = int(arr[r, c])
                break
        if center:
            break
    
    if center is None:
        return grid
    
    cr, cc = center
    
    # Create output grid with X pattern
    output = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            # Check if on either diagonal through center
            if abs(r - cr) == abs(c - cc):
                output[r, c] = color
    
    return output.tolist()
```

## Why this generalizes

This task belongs to the **diagonal-expansion** primitive family. The core insight is recognizing that a single point can serve as an anchor for geometric pattern generation. The mathematical condition |r - cr| == |c - cc| precisely captures both diagonals of an X centered at (cr, cc).

This generalizes because:
1. **Grid-size invariant**: The formula works for any rectangular grid dimensions
2. **Color-agnostic**: The pattern color comes from the input, not hardcoded
3. **Position-independent**: The center can be anywhere in the grid
4. **Deterministic**: Given the same input, the output is always identical

The primitive "expand a point into diagonals" appears in many ARC tasks involving symmetry, reflection, or geometric pattern completion. Recognizing the absolute-difference equality as the diagonal condition is a key geometric insight that transfers to related tasks.
