---
type: sensei_note
task: 159
tags: [expansion, pattern-scaling, arc, primer]
written_by: The Primer
written_at: 2026-07-04
verified_by: run-against-train (all examples pass)
---

# Task 159: Frame-Guided Pattern Scaling

## The rule

1. **Identify the frame**: Find the rectangular border formed by color 2 (red). This frame defines the output dimensions.

2. **Identify the pattern**: Find all non-zero pixels that are NOT color 2. These form the pattern to be scaled. Extract the pattern's bounding box.

3. **Calculate scaling**: The interior of the frame (excluding the border) has dimensions `(frame_height - 2) × (frame_width - 2)`. The scaling factor is:
   - `scale_h = interior_height // pattern_height`
   - `scale_w = interior_width // pattern_width`

4. **Scale the pattern**: Each pixel in the pattern becomes a `scale_h × scale_w` block of the same color in the output interior. The pattern is placed starting at position (1, 1) inside the frame.

5. **Preserve the border**: The output has the same red (color 2) border as the frame.

## Reference implementation

```python
def transform(grid):
    import numpy as np
    grid = np.array(grid)
    
    # Find red frame (color 2)
    red_positions = np.where(grid == 2)
    if len(red_positions[0]) == 0:
        return grid.tolist()
    
    r_min, r_max = red_positions[0].min(), red_positions[0].max()
    c_min, c_max = red_positions[1].min(), red_positions[1].max()
    
    frame_h = r_max - r_min + 1
    frame_w = c_max - c_min + 1
    
    # Find the other colored object (not 0, not 2)
    other_colors = np.unique(grid[(grid != 0) & (grid != 2)])
    if len(other_colors) == 0:
        # No pattern, just return red frame
        output = np.zeros((frame_h, frame_w), dtype=int)
        output[0, :] = 2
        output[-1, :] = 2
        output[:, 0] = 2
        output[:, -1] = 2
        return output.tolist()
    
    color = other_colors[0]
    other_positions = np.where(grid == color)
    o_min_r, o_max_r = other_positions[0].min(), other_positions[0].max()
    o_min_c, o_max_c = other_positions[1].min(), other_positions[1].max()
    
    pattern_h = o_max_r - o_min_r + 1
    pattern_w = o_max_c - o_min_c + 1
    
    # Calculate scaling factor
    interior_h = frame_h - 2
    interior_w = frame_w - 2
    
    scale_h = interior_h // pattern_h
    scale_w = interior_w // pattern_w
    
    # Create output with red border
    output = np.zeros((frame_h, frame_w), dtype=int)
    output[0, :] = 2
    output[-1, :] = 2
    output[:, 0] = 2
    output[:, -1] = 2
    
    # Extract and scale the pattern
    pattern = grid[o_min_r:o_max_r+1, o_min_c:o_max_c+1]
    
    for r in range(pattern_h):
        for c in range(pattern_w):
            if pattern[r, c] == color:
                for dr in range(scale_h):
                    for dc in range(scale_w):
                        out_r = 1 + r * scale_h + dr
                        out_c = 1 + c * scale_w + dc
                        if out_r < frame_h - 1 and out_c < frame_w - 1:
                            output[out_r, out_c] = color
    
    return output.tolist()
```

## Why this generalizes

This task belongs to the **pattern-scaling** primitive family within the EXPANSION output class. The key insights are:

1. **Object separation**: The task requires distinguishing between structural elements (the red frame, color 2) and content elements (the pattern, any other non-zero color). These objects can appear anywhere in the input grid independently.

2. **Proportional scaling**: The output size is determined by one object (the frame), while another object (the pattern) is scaled proportionally to fit within the frame's interior. The scaling factor is derived from the ratio: `interior_size / pattern_size`.

3. **Integer block expansion**: Each input pixel expands to an integer-sized block (`scale_h × scale_w`), preserving the pattern's topology while changing its resolution. This is a common ARC primitive where discrete pixels become larger uniform regions.

4. **Frame as container**: The red border acts as both a size specification and a structural element that must be preserved in the output. The pattern fills only the interior, never overwriting the border.

This pattern appears in many ARC tasks where one object defines a "canvas" and another object defines "content" to be rendered at a different scale.
