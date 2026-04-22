---
type: sensei_note
task: 124
tags: [expansion, pattern-extension, arc, primer]
written_by: The Primer
written_at: 2026-04-22
verified_by: run-against-train (all examples pass)
---

# Task 124: Vertical Pattern Extension

## The rule

The task requires extending a vertical pattern from the input grid to exactly 10 rows. The output width always matches the input width (10 columns in all examples).

There are two pattern types:

1. **Exact repetition**: Rows repeat with a fixed period. For example, if rows 0-1 are identical to rows 2-3, the period is 2. Simply tile this period to fill 10 rows.

2. **Translational pattern**: Segments of rows (typically 2 rows each) shift horizontally as they progress downward. Each new segment is a horizontally translated copy of the base segment. Detect the shift amount by comparing corresponding non-zero pixels between consecutive segments, then apply cumulative shifts to generate new rows.

The key insight: the output is always 10 rows regardless of input height. The pattern in the input must be analyzed to determine how to extend it.

## Reference implementation

```python
import numpy as np

def transform(grid):
    grid = np.array(grid)
    input_rows = grid.shape[0]
    width = grid.shape[1]
    
    # Strategy 1: Find exact row repetition period
    period = None
    for p in range(1, input_rows):
        is_period = True
        for i in range(p, input_rows):
            if not np.array_equal(grid[i], grid[i % p]):
                is_period = False
                break
        if is_period:
            period = p
            break
    
    if period is not None:
        # Simple case: exact repetition - tile the period
        output = np.zeros((10, width), dtype=int)
        for i in range(10):
            output[i] = grid[i % period]
        return output.tolist()
    
    # Strategy 2: Detect translational pattern (segments shift horizontally)
    segment_len = 2
    if input_rows >= 2 * segment_len:
        base_segment = grid[:segment_len].copy()
        seg0 = grid[0:segment_len]
        seg1 = grid[segment_len:2*segment_len]
        
        # Find translation by comparing first non-zero pixel in each segment
        def get_first_nonzero(segment):
            for r in range(segment.shape[0]):
                for c in range(segment.shape[1]):
                    if segment[r, c] != 0:
                        return (r, c, segment[r, c])
            return None
        
        p0 = get_first_nonzero(seg0)
        p1 = get_first_nonzero(seg1)
        
        if p0 and p1 and p0[2] == p1[2]:  # Same color
            shift_c = p1[1] - p0[0]  # Horizontal shift per segment
            
            # Generate output with translation
            output = np.zeros((10, width), dtype=int)
            for i in range(10):
                seg_idx = i // segment_len
                row_in_seg = i % segment_len
                total_shift = seg_idx * shift_c
                
                for c in range(width):
                    src_c = c - total_shift
                    if 0 <= src_c < width:
                        output[i, c] = base_segment[row_in_seg, src_c]
            
            return output.tolist()
    
    # Fallback: repeat input rows cyclically
    output = np.zeros((10, width), dtype=int)
    for i in range(10):
        output[i] = grid[i % input_rows]
    return output.tolist()
```

## Why this generalizes

This solution belongs to the **pattern-extension** primitive family. The core principle is:

1. **Period detection**: Many ARC tasks involve repeating patterns. Finding the smallest period allows extrapolation beyond the visible input.

2. **Translational symmetry**: When exact repetition doesn't hold, look for transformations (like horizontal shifts) between pattern segments. This captures more complex regularities.

3. **Fixed output size**: The task specifies 10 rows as the target. This is common in ARC expansion tasks where the output dimensions are determined by the task, not the input.

The two-strategy approach (exact period first, then translation) handles both simple repeating patterns (Examples 2, 3, and both test cases) and complex shifting patterns (Example 1). This hierarchical detection is a robust pattern for ARC tasks involving vertical or horizontal pattern continuation.
