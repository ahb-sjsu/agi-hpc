---
type: sensei_note
task: 381
tags: [transformation, gap-fill, arc, primer]
written_by: The Primer
written_at: 2026-07-13
verified_by: run-against-train (all examples pass)
---

# Task 381 — Fill Horizontal Gaps Between Rectangle Pairs With Color 9

## The rule

1. **Identify objects:** Find all connected components (4-connectivity) of color 2. Each component forms a solid rectangle; compute its bounding box `(r0, r1, c0, c1)` where r0/r1 are the min/max row indices and c0/c1 are the min/max column indices.

2. **Consider pairs:** For every unordered pair of rectangles `(A, B)`:
   - Compute their **row overlap**: `rs = max(A.r0, B.r0)`, `re = min(A.r1, B.r1)`. If `rs > re`, skip (no shared rows).
   - Compute their **horizontal gap**: If `A` is left of `B` (`A.c1 < B.c0`), the gap columns are `A.c1+1` through `B.c0-1`. Symmetrically if `B` is left of `A`. If they overlap or touch horizontally, skip.
   - **Check for blockers:** A third rectangle `C` blocks this pair if `C`'s row range intersects `[rs, re]` AND `C`'s column range intersects the gap columns (i.e., `C` occupies at least one cell in the gap within the shared rows). If any blocker exists, skip the entire pair.
   - **Fill the gap:** If unblocked, set all cells in rows `rs..re` and gap columns to color 9 (only if currently 0).

3. **Preserve everything else:** All non-gap pixels copy unchanged from input to output.

## Reference implementation

```python
import numpy as np
from collections import deque

def _components(mask):
    """Find all connected components (4-connectivity) of True values, return bounding boxes."""
    m = np.asarray(mask)
    H, W = m.shape
    seen = np.zeros_like(m, dtype=bool)
    rects = []
    
    for i in range(H):
        for j in range(W):
            if not m[i, j] or seen[i, j]:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            r0 = r1 = i
            c0 = c1 = j
            
            while q:
                r, c = q.popleft()
                r0 = min(r0, r)
                r1 = max(r1, r)
                c0 = min(c0, c)
                c1 = max(c1, c)
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            
            rects.append((r0, r1, c0, c1))
    
    return rects

def transform(grid):
    g = np.array(grid)
    out = g.copy()
    rects = _components(g == 2)
    n = len(rects)
    
    for i in range(n):
        for j in range(i + 1, n):
            r0a, r1a, c0a, c1a = rects[i]
            r0b, r1b, c0b, c1b = rects[j]
            
            # Compute row overlap
            rs, re = max(r0a, r0b), min(r1a, r1b)
            if rs > re:
                continue
            
            # Compute horizontal gap
            if c1a < c0b:
                L, R = c1a, c0b
            elif c1b < c0a:
                L, R = c1b, c0a
            else:
                continue  # Rectangles overlap or touch horizontally
            
            # Check for blockers
            blocked = False
            for k in range(n):
                if k in (i, j):
                    continue
                r0c, r1c, c0c, c1c = rects[k]
                # Check row intersection with shared rows
                if max(rs, r0c) > min(re, r1c):
                    continue
                # Check column intersection with gap (cols L+1 to R-1)
                if c1c < L + 1 or c0c > R - 1:
                    continue
                blocked = True
                break
            
            if not blocked:
                for r in range(rs, re + 1):
                    for c in range(L + 1, R):
                        if out[r, c] == 0:
                            out[r, c] = 9
    
    return out.tolist()
```

## Why this generalizes

This solution belongs to the **gap-fill** primitive family, which is a fundamental spatial reasoning pattern in ARC. The key insights that enable generalization:

1. **Object abstraction:** By extracting bounding boxes of connected components, the algorithm works regardless of rectangle size, position, or count. The logic operates on abstract geometric relationships, not pixel patterns.

2. **Relational reasoning:** The gap-fill decision depends only on relative positioning (row overlap, horizontal separation, blocker presence), not absolute coordinates. This makes the rule invariant to grid size and object placement.

3. **Compositional structure:** The algorithm composes three primitives—connected component extraction, pairwise spatial relationship computation, and conditional cell modification—each of which generalizes independently.

4. **Blocking semantics:** The blocker check ensures the rule handles complex multi-object scenes correctly, filling only "visible" gaps that aren't occluded by intervening objects.

This pattern appears in many ARC tasks where the goal is to complete or connect partial structures while respecting occlusion and spatial constraints.
