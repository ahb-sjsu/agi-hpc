---
type: sensei_note
task: 139
tags: [transformation, bounding-box-fill, arc, primer]
written_by: The Primer
written_at: 2026-06-29
verified_by: run-against-train (all examples pass)
---

# Task 139: Bounding Box Fill

## The rule

For each connected cluster of yellow (4) cells in the input grid:
1. Find the rectangular bounding box that contains all cells in that cluster
2. Fill every black (0) cell inside that bounding box with orange (7)
3. Yellow (4) cells remain unchanged
4. Cells outside all bounding boxes remain unchanged

Connectivity is 4-directional (up, down, left, right). Each distinct cluster of 4s gets its own bounding box filled independently.

## Reference implementation

```python
import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    result = grid.copy()
    visited = np.zeros((h, w), dtype=bool)
    
    def get_component(start_r, start_c):
        """Get all cells in the connected component of 4s starting from (start_r, start_c)"""
        component = []
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if visited[r, c]:
                continue
            if grid[r, c] != 4:
                continue
            visited[r, c] = True
            component.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))
        return component
    
    def fill_bounding_box(component):
        """Fill 0s with 7s within the bounding box of a component"""
        if not component:
            return
        rows = [r for r, c in component]
        cols = [c for r, c in component]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if result[r, c] == 0:
                    result[r, c] = 7
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 4 and not visited[r, c]:
                component = get_component(r, c)
                fill_bounding_box(component)
    
    return result.tolist()
```

## Why this generalizes

This task belongs to the **bounding-box-fill** primitive family. The key insight is:

1. **Object detection**: Yellow cells form distinct connected objects (clusters)
2. **Spatial reasoning**: Each object defines a rectangular region (its bounding box)
3. **Fill operation**: Empty space within that region gets filled with a new color

This pattern appears in many ARC tasks where:
- Objects are defined by a specific color
- The task requires completing or filling regions defined by those objects
- The fill color is different from both the object color and background

The algorithm generalizes to any number of objects, any object shapes, and any grid size. It correctly handles cases where bounding boxes might overlap (each 0 gets filled once) and where objects are adjacent or separated.
