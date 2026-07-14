---
type: sensei_note
task: 175
tags: [diagonal, stripes, reconstruction, global-pattern, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Task 175 — Diagonal Stripe Reconstruction

## Not local interpolation. The fill rule is global.

The 0-patches are **holes** in a pre-existing diagonal-stripe pattern.
Each diagonal is a single consistent color; the holes cover parts of
those diagonals. To fill, you reconstruct the stripes.

## Algorithm

1. For each cell `(r, c)`, compute its diagonal index. Try both:
   - anti-diagonal: `d = r + c`
   - main diagonal: `d = r - c`
2. For each diagonal index, tally the non-0 colors on that diagonal
   (in the input). The **mode** is that diagonal's color.
3. For each 0-cell, replace with its diagonal's mode color.

Most ARC diagonal-stripe tasks use the anti-diagonal (`r + c`) because
the stripes descend from top-right to bottom-left. Try that first; if
the predicted mode doesn't match the train outputs, try the main
diagonal instead.

## Reference sketch

```python
import numpy as np
from collections import Counter

def transform(grid):
    a = np.array(grid)
    h, w = a.shape
    out = a.copy()
    # Try anti-diagonal (r+c) first
    for d in range(h + w - 1):
        cells = [(r, c) for r in range(h) for c in range(w) if r + c == d]
        colors = [a[r, c] for r, c in cells if a[r, c] != 0]
        if not colors:
            continue
        mode = Counter(colors).most_common(1)[0][0]
        for r, c in cells:
            if a[r, c] == 0:
                out[r, c] = mode
    return out.tolist()
```

## Why earlier attempts failed

Your previous attempts tried to fill 0 by copying from a neighbor
("left neighbor", "above neighbor"). That works when there's one
uniform background color, but here the 0 can sit on a colored stripe
and needs the stripe's color, not its neighbor's.

The mental move: stop thinking pixel-by-pixel. Think stripe-by-stripe.
