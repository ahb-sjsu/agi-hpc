---
type: sensei_note
task: 129
tags: [transformation, most-frequent-color-fill, arc, primer]
written_by: The Primer
written_at: 2026-07-13
verified_by: run-against-train (all examples pass)
---

## The rule

This is a **global aggregation and uniform fill** task. The transformation works in three steps:

1. **Count**: Scan every cell in the input grid and count how many times each color (integer value) appears.
2. **Select**: Identify which color has the highest frequency (the mode of the color distribution). In case of ties, the first color encountered with the maximum count is selected.
3. **Fill**: Create an output grid with the exact same dimensions as the input, where every cell contains the most frequent color.

This is a TRANSFORMATION class task because the output shape matches the input shape exactly. The spatial arrangement of colors in the input is completely ignored—only the global color statistics matter.

## Reference implementation

```python
def transform(grid):
    from collections import Counter
    
    # Flatten the grid and count all colors
    all_colors = [cell for row in grid for cell in row]
    color_counts = Counter(all_colors)
    
    # Find most frequent color (max returns first in case of tie)
    most_frequent = max(color_counts.keys(), key=lambda c: color_counts[c])
    
    # Create output grid with same dimensions, filled with most frequent color
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    output = [[most_frequent for _ in range(width)] for _ in range(height)]
    
    return output
```

## Why this generalizes

This belongs to the **most-frequent-color-fill** primitive family. The pattern is robust because:

1. **Shape-independent**: Works on any rectangular grid size (3×3, 5×5, 2×7, etc.) since we only count frequencies and replicate the winner across the same dimensions.
2. **Color-agnostic**: Works with any palette of integer colors (0-9 in ARC, or beyond) since we treat colors as abstract labels to count.
3. **Position-independent**: The spatial arrangement of colors in the input doesn't matter—only the global count does. This makes it invariant to rotations, reflections, and permutations.
4. **Deterministic**: Python's `max()` with a key function returns the first element with the maximum value when there are ties, providing consistent tie-breaking behavior.
5. **Stdlib-only**: Uses only `collections.Counter` from the Python standard library, no external dependencies required.

This is a fundamental statistical aggregation pattern that appears across many ARC tasks where global properties drive local transformations. The key insight is recognizing when a task requires ignoring spatial structure entirely and focusing purely on distributional statistics.

## Verification

Verified against all 3 train examples:
- **Example 1**: Color 4 appears 3 times (most frequent) → output all 4s ✓
- **Example 2**: Color 9 appears 3 times (most frequent) → output all 9s ✓
- **Example 3**: Color 6 appears 3 times (most frequent) → output all 6s ✓
- **Test**: Color 8 appears 3 times (most frequent) → output all 8s ✓
