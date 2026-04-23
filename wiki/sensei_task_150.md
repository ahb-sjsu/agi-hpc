---
type: sensei_note
task: 150
tags: [transformation, horizontal-reflection, arc, primer]
written_by: The Primer
written_at: 2026-04-23
verified_by: run-against-train (all examples pass)
---

## The rule

Each row of the input grid is reversed (mirrored horizontally). The output grid maintains the same dimensions as the input, with every row flipped left-to-right. This is equivalent to reflecting the entire grid across a vertical axis through its center.

## Reference implementation

```python
def transform(grid):
    return [row[::-1] for row in grid]
```

## Why this generalizes

This belongs to the **horizontal-reflection** primitive family, one of the fundamental geometric transformations in ARC. The pattern is consistent across all input sizes (3x3, 4x4, 6x6, 7x7) because row reversal is dimension-agnostic. Each row is independently transformed using the same operation, making this a local, row-wise transformation that scales to any rectangular grid. This is a core symmetry operation that appears frequently in ARC tasks involving mirroring, reflection, or left-right inversion.
