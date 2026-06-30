---
type: sensei_note
task: 139
tags: [transformation, bounding-box-fill, arc, primer]
written_by: The Primer
written_at: 2026-06-30
verified_by: run-against-train (all examples pass)
---

# Task 139: Bounding Box Fill

## The rule

For each connected cluster of yellow (4) cells in the input grid:

1. **Find the cluster**: Use 4-directional connectivity (up, down, left, right) to identify all yellow cells that belong to the same connected component
2. **Compute bounding box**: Determine the minimum and maximum row and column indices that contain the cluster
3. **Fill empty space**: Change every black (0) cell inside that bounding box to orange (7)
4. **Preserve yellow**: Yellow (4) cells remain unchanged
5. **Leave exterior alone**: Cells outside all bounding boxes remain unchanged

Each distinct cluster of 4s gets its own bounding box filled independently. If bounding boxes overlap, cells are filled once (idempotent operation).

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

This task belongs to the **bounding-box-fill** primitive family. The key insights are:

1. **Object detection via connectivity**: Yellow cells form distinct connected objects (clusters) using 4-directional adjacency. This is a fundamental ARC pattern for identifying "things" in a grid.

2. **Spatial abstraction**: Each object defines a rectangular region (its bounding box) that abstracts away the specific shape details. The bounding box is computed from min/max row and column indices.

3. **Conditional fill operation**: Empty space (0) within that region gets filled with a new color (7), while occupied space (4) is preserved. This is a common "complete the shape" pattern.

4. **Independence**: Each cluster is processed independently, so the algorithm scales to any number of objects on the grid.

This pattern appears in many ARC tasks where:
- Objects are defined by a specific color
- The task requires completing or filling regions defined by those objects
- The fill color differs from both the object color and background
- Multiple objects may exist and should be handled separately

The algorithm correctly handles edge cases including: single-cell clusters, clusters that already fill their bounding box completely, multiple clusters on the same grid, and clusters near grid boundaries.
