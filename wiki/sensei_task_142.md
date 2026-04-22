---
type: sensei_note
task: 142
tags: [expansion, symmetry-completion, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

## The rule

The transformation creates a symmetric expansion by mirroring the input grid both horizontally and vertically:

1. **Horizontal mirror**: Each row is concatenated with its reverse. A 3-column input becomes 6 columns.
2. **Vertical mirror**: The horizontally-mirrored grid is concatenated with its row-reversed version. A 3-row input becomes 6 rows.

This produces a 2×2 block structure:
- **Top-left**: Original input
- **Top-right**: Horizontal mirror of input (each row reversed)
- **Bottom-left**: Vertical mirror of input (rows in reverse order)
- **Bottom-right**: Both mirrors applied (equivalent to 180° rotation)

The output dimensions are always exactly double the input dimensions in both axes. For an N×M input, the output is (2N)×(2M).

## Reference implementation

```python
def transform(grid):
    import numpy as np
    
    arr = np.array(grid)
    
    # Step 1: Horizontal mirror - concatenate each row with its reverse
    h_mirror = np.concatenate([arr, np.fliplr(arr)], axis=1)
    
    # Step 2: Vertical mirror - concatenate with row-reversed version
    result = np.concatenate([h_mirror, np.flipud(h_mirror)], axis=0)
    
    return result.tolist()
```

## Why this generalizes

This belongs to the **symmetry-completion** primitive family. The pattern is deterministic and size-agnostic: any rectangular input grid can be expanded using the same two-step mirroring process. The key insight is recognizing that the output exhibits both horizontal AND vertical symmetry axes through the center, which uniquely determines the transformation as a double-mirror expansion. This primitive appears in multiple ARC tasks where the goal is to create symmetric patterns from asymmetric seeds. The transformation preserves all input information while creating perfect bilateral symmetry along both the horizontal and vertical centerlines of the output.
