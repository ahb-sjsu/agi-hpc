---
type: sensei_note
task: 30
tags: [centering, placement, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Task 030 — Horizontal Centering, Not Left-Alignment

Objects in the output are placed **horizontally centered** in the grid,
not flush-left. The 27-attempt log shows the code always starts at
column 0.

## Rule

For each object to place, compute:

```python
object_width = max_c - min_c + 1
x_offset = (grid_width - object_width) // 2
```

Place the object starting at column `x_offset`.

Verify this is the rule on every training example. If it holds, use it.
