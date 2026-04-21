---
type: sensei_note
task: 381
tags: [transformation, gap-fill, arc, primer]
written_by: The Primer
written_at: 2026-04-21
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
    m = np.asarray(mask); H, W = m.shape
    seen = np.zeros_like(m, dtype=bool); rects = []
    for i in range(H):
        for j in range(W):
            if not m[i, j] or seen[i, j]: continue
            q = deque([(i, j)]); seen[i, j] = True
            r0 = r1 = i; c0 = c1 = j
            while q:
                r, c = q.popleft()
                r0 = min(r0, r); r1 = max(r1, r)
                c0 = min(c0, c); c1 = max(c1, c)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True; q.append((nr, nc))
            rects.append((r0, r1, c0, c1))
    return rects

def transform(grid):
    g = np.array(grid); out = g.copy()
    rects = _components(g == 2); n = len(rects)
    for i in range(n):
        for j in range(i+1, n):
            r0a, r1a, c0a, c1a = rects[i]
            r0b, r1b, c0b, c1b = rects[j]
            rs, re = max(r0a, r0b), min(r1a, r1b)
            if rs > re: continue
            if c1a < c0b: L, R = c1a, c0b
            elif c1b < c0a: L, R = c1b, c0a
            else: continue
            blocked = False
            for k in range(n):
                if k in (i, j): continue
                r0c, r1c, c0c, c1c = rects[k]
                if max(rs, r0c) > min(re, r1c): continue
                if c1c < L + 1 or c0c > R - 1: continue
                blocked = True; break
            if blocked: continue
            for r in range(rs, re + 1):
                for c in range(L + 1, R):
                    if out[r, c] == 0: out[r, c] = 9
    return out.tolist()
```

## Why this generalizes

This belongs to the **gap-fill** primitive family: identify discrete objects (connected components), find spatial relationships between pairs (here: horizontal separation with row overlap), and fill the intervening space with a new color. The **blocking check** is critical — it ensures fills only occur when the gap is truly empty, preventing incorrect fills when a third object interrupts the line of sight between the pair. This pattern appears in multiple ARC tasks where objects must "connect" or "bridge" across empty space, but only when unobstructed. The key insight is that the fill color (9) is distinct from the object color (2), signaling a "connection" or "bridge" rather than object growth.
