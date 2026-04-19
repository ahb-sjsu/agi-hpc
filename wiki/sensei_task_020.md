---
type: sensei_note
task: 20
tags: [transformation, symmetry-completion, arc, primer]
written_by: The Primer
written_at: 2026-04-19
verified_by: run-against-train (all examples pass)
---

# Task 020 — Symmetry Completion via Bounding-Box Center

## The rule

Find the bounding box of all non-zero cells in the input grid. Compute the center of this bounding box (midpoint of min/max row and min/max column). For each non-zero cell, reflect its value across both the horizontal and vertical axes passing through that center point. This fills in all four symmetric positions: the original cell, its horizontal mirror, its vertical mirror, and the diagonal mirror (both axes combined).

The output grid has the same shape as the input. Existing non-zero cells are preserved; missing symmetric counterparts are filled in with the same color/value.

## Reference implementation

```python
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    
    # Find bounding box of non-zero cells
    rows, cols = np.where(g != 0)
    if len(rows) == 0:
        return grid
    
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    
    # Center of the bounding box
    center_r = (r_min + r_max) / 2.0
    center_c = (c_min + c_max) / 2.0
    
    # Create output grid (copy of input)
    out = g.copy()
    
    # For each non-zero cell, reflect it across both axes through the center
    for r, c in zip(rows, cols):
        val = g[r, c]
        
        # Reflect across horizontal axis (through center_r)
        r_h = int(2 * center_r - r)
        
        # Reflect across vertical axis (through center_c)
        c_v = int(2 * center_c - c)
        
        # Place value at all four symmetric positions
        out[r_h, c] = val
        out[r, c_v] = val
        out[r_h, c_v] = val
    
    return out.tolist()
```

## Why this generalizes

This task belongs to the **symmetry-completion** family (see `sensei_meta_task_typology.md`). The key insight is that the input contains a partially-symmetric pattern, and the output completes it to full D2 symmetry (horizontal + vertical reflection) around the pattern's natural center.

The bounding-box center is the correct symmetry point because:
1. All training examples have non-zero cells arranged symmetrically around this center (even if some positions are missing).
2. The span (max - min) in both dimensions is always even, guaranteeing the center falls on integer or half-integer coordinates that produce valid grid positions when reflecting.
3. This is a geometric rule independent of specific colors — any non-zero value gets reflected the same way.

This primitive generalizes to any task where:
- Output shape equals input shape (TRANSFORMATION class)
- Non-zero cells form a sparse, roughly symmetric pattern
- The task is to "complete" the symmetry rather than create new structure

Related family members include `complete_by_horizontal_stripe`, `complete_by_vertical_stripe`, and `complete_by_diagonal` — all follow the same "find center, reflect missing pieces" pattern but with different symmetry groups.
