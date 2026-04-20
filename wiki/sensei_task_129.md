---
type: sensei_note
task: 129
tags: [transformation, most-frequent-color-fill, arc, primer]
written_by: The Primer
written_at: 2026-04-20
verified_by: run-against-train (all examples pass)
---

## The rule

Find the most frequently occurring color (integer value) in the input grid. Then create an output grid with the exact same dimensions as the input, where every cell is filled with that most frequent color.

In other words: count all colors in the input, identify which color appears most often, and flood-fill the entire output with that winner color.

## Reference implementation

```python
def transform(grid):
    from collections import Counter
    
    # Flatten the grid to count all colors
    flat = [cell for row in grid for cell in row]
    
    # Find the most frequent color
    color_counts = Counter(flat)
    most_frequent_color = color_counts.most_common(1)[0][0]
    
    # Create output grid with same dimensions, filled with most frequent color
    height = len(grid)
    width = len(grid[0])
    output = [[most_frequent_color for _ in range(width)] for _ in range(height)]
    
    return output
```

## Why this generalizes

This belongs to the **most-frequent-color-fill** primitive family. The pattern is:

1. **Global aggregation**: Scan the entire input grid to compute a statistic (color frequency)
2. **Winner selection**: Pick the color with maximum count
3. **Uniform expansion**: Replicate that single value across all output positions

This generalizes to any grid size and any color palette because it only depends on counting frequencies, not on spatial relationships or specific color values. The output shape always matches the input shape (TRANSFORMATION class), making this a shape-preserving fill operation driven by global color statistics.
