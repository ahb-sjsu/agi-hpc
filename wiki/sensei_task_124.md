---
type: sensei_note
task: 124
tags: [expansion, pattern-extension, arc, primer]
written_by: The Primer
written_at: 2026-07-11
verified_by: run-against-train (all examples pass)
---

# Task 124: Vertical Pattern Extension to 10 Rows

## The rule

The output is always exactly **10 rows** with the same width as the input. The input contains a vertical pattern that must be detected and extended downward to fill all 10 rows.

There are two pattern types to detect:

1. **Exact repetition (period detection)**: Rows repeat with a fixed period. For example, if rows 0-1 are identical to rows 2-3, the period is 2. Simply tile this period to fill 10 rows.

2. **Translational pattern**: Segments of rows (typically 2 rows each) shift horizontally as they progress downward. Each new segment is a horizontally translated copy of the base segment. Detect the shift amount by comparing consecutive segments, then apply cumulative shifts to generate new rows.

The key insight: the output is always 10 rows regardless of input height. Analyze the input pattern to determine how to extend it.

## Reference implementation

```python
import numpy as np

def transform(grid):
    """
    Task 124: Vertical Pattern Extension to 10 Rows
    
    Detects either exact row repetition (period) or translational patterns
    (horizontal shifts between segments), then extends to exactly 10 rows.
    """
    grid = np.array(grid)
    input_rows = grid.shape[0]
    width = grid.shape[1]
    
    # Strategy 1: Find exact row repetition period
    for period in range(1, input_rows):
        is_period = True
        for i in range(period, input_rows):
            if not np.array_equal(grid[i], grid[i % period]):
                is_period = False
                break
        if is_period:
            # Simple case: exact repetition - tile the period to 10 rows
            output = np.zeros((10, width), dtype=int)
            for i in range(10):
                output[i] = grid[i % period]
            return output.tolist()
    
    # Strategy 2: Detect translational pattern (segments shift horizontally)
    for segment_len in range(2, input_rows // 2 + 1):
        if input_rows >= 2 * segment_len:
            seg0 = grid[0:segment_len]
            seg1 = grid[segment_len:2*segment_len]
            
            # Check if seg1 is a horizontal shift of seg0
            for shift in range(-width + 1, width):
                if shift == 0:
                    continue
                
                # Create shifted version of seg0
                shifted_seg0 = np.zeros_like(seg0)
                for r in range(segment_len):
                    for c in range(width):
                        src_c = c - shift
                        if 0 <= src_c < width:
                            shifted_seg0[r, c] = seg0[r, src_c]
                
                if np.array_equal(shifted_seg0, seg1):
                    # Found translational pattern
                    output = np.zeros((10, width), dtype=int)
                    for i in range(10):
                        seg_idx = i // segment_len
                        row_in_seg = i % segment_len
                        total_shift = seg_idx * shift
                        
                        for c in range(width):
                            src_c = c - total_shift
                            if 0 <= src_c < width:
                                output[i, c] = seg0[row_in_seg, src_c]
                    
                    return output.tolist()
    
    # Fallback: repeat input rows cyclically
    output = np.zeros((10, width), dtype=int)
    for i in range(10):
        output[i] = grid[i % input_rows]
    return output.tolist()
```

## Why this generalizes

This solution belongs to the **pattern-extension** primitive family. The core principles are:

1. **Period detection**: Many ARC tasks involve repeating patterns. Finding the smallest period allows extrapolation beyond the visible input. This handles cases like vertical lines or alternating row patterns.

2. **Translational symmetry**: When patterns shift position as they repeat (like diagonal lines moving rightward), detecting the shift amount between consecutive segments enables correct extrapolation.

3. **Fixed output size**: The task specifies exactly 10 output rows regardless of input height. This is a common ARC pattern where the output dimensions are predetermined.

4. **Hierarchical strategy**: Try simpler patterns first (exact repetition), then more complex ones (translational), with a safe fallback. This mirrors how humans approach pattern recognition.

This approach generalizes to any task where:
- Output has a fixed size different from input
- Input shows regular vertical structure (repetition or translation)
- Pattern must be extrapolated beyond visible examples
