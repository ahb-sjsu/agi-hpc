---
type: sensei_note
task: 139
tags: [transformation, bounding-box-fill, arc, primer]
written_by: The Primer
written_at: 2026-07-07
verified_by: run-against-train (all examples pass)
---

# Task 139: Bounding Box Fill with 8-Directional Connectivity

## The rule

For each connected cluster of yellow (4) cells in the input grid:

1. **Find the cluster**: Use **8-directional connectivity** (up, down, left, right, AND all four diagonals) to identify all yellow cells that belong to the same connected component. Cells that touch only at corners are considered part of the same cluster.
2. **Compute bounding box**: Determine the minimum and maximum row and column indices that contain the cluster.
3. **Fill empty space**: Change every black (0) cell inside that bounding box to orange (7).
4. **Preserve yellow**: Yellow (4) cells remain unchanged.
5. **Leave exterior alone**: Cells outside all bounding boxes remain unchanged (stay 0).

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
        """Find all 4s connected via 8-directional connectivity."""
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
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    stack.append((r + dr, c + dc))
        return component
    
    def fill_bounding_box(component):
        """Fill all 0s in the bounding box with 7s."""
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

This task belongs to the **bounding-box-fill** primitive family with **8-directional connectivity** for object detection.

The key insights are:

1. **Object detection via 8-way connectivity**: Yellow cells form distinct connected objects using 8-directional adjacency (including diagonals). This is more permissive than 4-directional connectivity—cells that touch only at corners are considered part of the same cluster.

2. **Spatial abstraction**: Each object defines a rectangular region (its bounding box) computed from min/max row and column indices of all cells in that connected component.

3. **Conditional fill operation**: Empty space (0) within the bounding box gets filled with orange (7), while occupied space (4) is preserved. This creates a "complete the rectangle" effect.

4. **Independence**: Each cluster is processed independently, so the algorithm scales to any number of objects on the grid.

**Critical distinction**: Using 4-directional connectivity instead of 8-directional would incorrectly split some clusters that only connect diagonally. The 8-directional approach ensures all visually-connected yellow regions are treated as single objects.

**Verification**: This implementation has been verified against all training examples and correctly handles the test case structure.
