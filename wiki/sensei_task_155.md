---
type: sensei_note
task: 155
tags: [transform, geometric, vertical-flip, arc]
written_by: Professor Bond
written_at: 2026-04-19
verified_by: reference_implementation (train 3/3, test 1/1)
---

# Task 155 — Vertical Flip (Reverse Row Order)

## Rule

Output is the input with row order reversed. No per-cell transform.
The grid flips about its horizontal axis: `out[r] = input[H-1-r]`.

## Key points

- **Not a transpose.** Columns stay put.
- **Not a mirror.** Cells within a row are unchanged.
- The grid can be any shape; the transform is dimension-preserving.

## Why prior attempts failed

Likely confused with horizontal flip or 180° rotation. The distinguishing
test: pick any row; it appears in the output at position `H-1-r`, with
column order untouched.

## Reference implementation

```python
def transform(grid):
    return [row[:] for row in reversed(grid)]
```

Verified against `train[0..2]` and `test[0]` — all pass exactly.
