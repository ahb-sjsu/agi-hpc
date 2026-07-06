---
type: sensei_note
task: 159
tags: [expansion, pattern-scaling, arc, primer]
written_by: The Primer
written_at: 2026-07-06
verified_by: run-against-train (all examples pass)
---

# Task 159: Frame-Guided Pattern Scaling

## The rule

1. **Identify the frame**: Find the rectangular border formed by color 2 (red). This frame defines the output dimensions.

2. **Identify the pattern**: Find ALL non-zero pixels that are NOT color 2, regardless of their color. These form the pattern to be scaled. Extract the pattern's bounding box (the smallest rectangle containing all pattern pixels).

3. **Calculate scaling**: The interior of the frame (excluding the border) has dimensions `(frame_height - 2) × (frame_width - 2)`. The scaling factor is:
   - `scale_h = interior_height // pattern_height`
   - `scale_w = interior_width // pattern_width`

4. **Scale the pattern**: Each non-zero pixel in the pattern becomes a `scale_h × scale_w` block of the SAME color in the output interior. The pattern is placed starting at position (1, 1) inside the frame.

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
    
    # Find all non-zero, non-red pixels (the pattern)
    pattern_mask = (grid != 0) & (grid != 2)
    pattern_positions = np.where(pattern_mask)
    
    if len(pattern_positions[0]) == 0:
        # No pattern, just return red frame
        output = np.zeros((frame_h, frame_w), dtype=int)
        output[0, :] = 2
        output[-1, :] = 2
        output[:, 0] = 2
        output[:, -1] = 2
        return output.tolist()
    
    p_min_r, p_max_r = pattern_positions[0].min(), pattern_positions[0].max()
    p_min_c, p_max_c = pattern_positions[1].min(), pattern_positions[1].max()
    
    pattern_h = p_max_r - p_min_r + 1
    pattern_w = p_max_c - p_min_c + 1
    
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
    pattern = grid[p_min_r:p_max_r+1, p_min_c:p_max_c+1]
    
    for r in range(pattern_h):
        for c in range(pattern_w):
            if pattern[r, c] != 0:
                for dr in range(scale_h):
                    for dc in range(scale_w):
                        out_r = 1 + r * scale_h + dr
                        out_c = 1 + c * scale_w + dc
                        if out_r < frame_h - 1 and out_c < frame_w - 1:
                            output[out_r, out_c] = pattern[r, c]
    
    return output.tolist()
```

## Why this generalizes

This task belongs to the **pattern-scaling** primitive family within the EXPANSION output class. The key insights are:

1. **Object separation by role, not just color**: The task requires distinguishing between structural elements (the red frame, color 2) and content elements (the pattern, ANY other non-zero color). Previous implementations failed because they only extracted pixels of a single color. The correct approach extracts ALL non-frame pixels regardless of color, preserving each pixel's original color during scaling.

2. **Proportional scaling**: The output size is determined by one object (the frame), while another object (the pattern) is scaled proportionally to fit within the frame's interior. The scaling factor is derived from the ratio: `interior_size / pattern_size`. This works for any integer scaling factor.

3. **Integer block expansion**: Each input pixel expands to an integer-sized block (`scale_h × scale_w`), preserving the pattern's topology and colors while changing its resolution. This is a common ARC primitive where discrete pixels become larger blocks.

4. **Spatial independence**: The frame and pattern can appear anywhere in the input grid independently. The transformation extracts their relative geometry and recomposes them in a canonical output where the frame becomes the border and the scaled pattern fills the interior.

**Critical fix from previous attempts**: The pattern extraction must use `(grid != 0) & (grid != 2)` to capture ALL pattern pixels, not just those of a single color. Each pixel retains its original color when scaled.
