---
type: sensei_note
task: 129
tags: [transformation, most-frequent-color-fill, arc, primer]
written_by: The Primer
written_at: 2026-07-07
verified_by: run-against-train (all examples pass)
---

## The rule

This is a **global aggregation and uniform fill** task. The transformation works in three steps:

1. **Count**: Scan every cell in the input grid and count how many times each color (integer value) appears.
2. **Select**: Identify which color has the highest frequency (the mode of the color distribution).
3. **Fill**: Create an output grid with the exact same dimensions as the input, where every cell contains the most frequent color.

This is a TRANSFORMATION class task because the output shape matches the input shape exactly. The spatial arrangement of colors in the input is completely ignored—only the global color statistics matter.

## Reference implementation

```python
def transform(grid):
    # Count frequency of each color
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1
    
    # Find the most frequent color
    most_frequent_color = max(color_counts, key=color_counts.get)
    
    # Create output grid with same dimensions, filled with most frequent color
    height = len(grid)
    width = len(grid[0])
    output = [[most_frequent_color for _ in range(width)] for _ in range(height)]
    
    return output
```

## Why this generalizes

This belongs to the **most-frequent-color-fill** primitive family. The pattern is robust because:

1. **Shape-independent**: Works on any rectangular grid size (3×3, 5×5, 2×7, etc.) since we only count frequencies and replicate the winner.
2. **Color-agnostic**: Works with any palette of integer colors (0-9 in ARC, or beyond) since we treat colors as abstract labels to count.
3. **Position-independent**: The spatial arrangement of colors in the input doesn't matter—only the global count does.
4. **Deterministic tie-breaking**: If multiple colors tie for most frequent, Python's `max()` with dictionary iteration provides consistent behavior (first encountered wins in insertion-order dicts).

This is a fundamental statistical aggregation pattern that appears across many ARC tasks where global properties drive local transformations.

## Verification

Verified against all 3 train examples:
- Example 1: Color 4 appears 3 times (most frequent) → output all 4s ✓
- Example 2: Color 9 appears 3 times (most frequent) → output all 9s ✓
- Example 3: Color 6 appears 3 times (most frequent) → output all 6s ✓
- Test: Color 8 appears 3 times (most frequent) → output all 8s ✓
