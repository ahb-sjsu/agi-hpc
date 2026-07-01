---
type: sensei_note
task: 48
tags: [classification, connectivity-classifier, arc, primer]
written_by: The Primer
written_at: 2026-07-01
verified_by: run-against-train (all examples pass)
---

# Task 048: Connectivity Classification via Teal Paths

## The rule

This is a **classification task** where the output is always a 1x1 grid containing either 0 or 8.

The rule:
1. **Identify terminals**: Find the two 2x2 blocks of red (2) cells in the input grid. Every valid example contains exactly two such blocks.
2. **Extract adjacent wires**: Find all teal (8) cells that are adjacent to each 2x2 block using **8-connectivity** (including diagonals). These are the "entry points" to the circuit.
3. **Check connectivity**: Determine if there exists a path of teal (8) cells connecting the two blocks. A path exists if you can travel from any 8 adjacent to the first block to any 8 adjacent to the second block by moving through adjacent 8s using **8-connectivity** (orthogonal and diagonal moves allowed).
4. **Classify**: Output [[8]] if such a path exists (circuit complete), otherwise output [[0]] (circuit broken).

Think of the 2x2 red blocks as "terminals" and the teal cells as "wires". The task asks: is the circuit complete?

**Critical distinction**: Both adjacency detection AND path traversal use **8-connectivity** (not 4-connectivity). This was a common source of errors in previous attempts.

## Reference implementation

```python
import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find all 2x2 blocks of 2s
    blocks = []
    for r in range(h - 1):
        for c in range(w - 1):
            if (grid[r:r+2, c:c+2] == 2).all():
                blocks.append((r, c))
    
    if len(blocks) != 2:
        return [[0]]
    
    # Get cells adjacent to each block (8-connectivity for adjacency)
    def get_adjacent_eights(block_r, block_c):
        adjacent = set()
        # Check the perimeter around the 2x2 block
        for dr in range(-1, 3):
            for dc in range(-1, 3):
                # Skip the 2x2 block itself
                if 0 <= dr <= 1 and 0 <= dc <= 1:
                    continue
                nr, nc = block_r + dr, block_c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 8:
                    adjacent.add((nr, nc))
        return adjacent
    
    adj1 = get_adjacent_eights(blocks[0][0], blocks[0][1])
    adj2 = get_adjacent_eights(blocks[1][0], blocks[1][1])
    
    if not adj1 or not adj2:
        return [[0]]
    
    # Find all 8s in the grid
    eights = set()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 8:
                eights.add((r, c))
    
    # BFS to check if any 8 in adj1 connects to any 8 in adj2 (8-connectivity)
    def bfs_connects(start_set, end_set):
        visited = set(start_set)
        queue = list(start_set)
        
        while queue:
            r, c = queue.pop(0)
            if (r, c) in end_set:
                return True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in eights and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False
    
    if bfs_connects(adj1, adj2):
        return [[8]]
    else:
        return [[0]]
```

## Why this generalizes

This task belongs to the **connectivity-classifier** primitive family. The core pattern is:

1. **Object detection**: Identify specific structured objects (2x2 blocks of a particular color)
2. **Adjacency extraction**: Find which connector cells (8s) touch each object using 8-connectivity
3. **Path existence**: Determine if a connected component of connector cells bridges the objects using 8-connectivity BFS
4. **Binary classification**: Map connectivity (yes/no) to output value (8/0)

This pattern appears in many ARC tasks where the goal is to determine whether two regions are connected through a specific medium. The key insight is that **8-connectivity must be used consistently** for both adjacency detection and path traversal—using 4-connectivity instead is a common failure mode.

The metaphor of "electrical circuits" is helpful: red 2x2 blocks are terminals, teal cells are wires, and the question is whether current can flow from one terminal to the other.
