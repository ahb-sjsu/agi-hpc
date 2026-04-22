---
type: sensei_note
task: 129
tags: [transformation, most-frequent-color-fill, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

## The rule

Count the frequency of each color (integer value) in the input grid. Identify which color appears most often. Then create an output grid with the exact same dimensions as the input, where every single cell is filled with that most frequent color.

This is a global aggregation task: you must scan the entire input to compute color frequencies, select the winner, and then uniformly expand that single value across all output positions.

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

This belongs to the **most-frequent-color-fill** primitive family. The pattern consists of three steps:

1. **Global aggregation**: Scan the entire input grid to compute a statistic (color frequency distribution)
2. **Winner selection**: Pick the color with the maximum count
3. **Uniform expansion**: Replicate that single winning value across all output positions

This generalizes to any grid size and any color palette because it depends only on counting frequencies, not on spatial relationships, specific color values, or geometric patterns. The output shape always matches the input shape (TRANSFORMATION class), making this a shape-preserving fill operation driven entirely by global color statistics.
