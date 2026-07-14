---
type: sensei_note
task: 18
tags: [color-pair, lookup, parameterized-rule, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Task 018 — Color-Pair-Indexed Lookup

The rule is **parameterized by the pair of marker colors present**.
Different training examples use different pairs: `(1, 3)`, `(1, 4)`,
etc. The transformation for each pair is different.

## Algorithm

1. For each training example:
   - Identify the two distinct non-background colors in the input.
     Call them `a` and `b`.
   - Observe what the output does with those two colors.
2. Build `dict[(a, b)] = action`. The action is typically a
   placement rule (put `a`s here, put `b`s there, based on positions).
3. For the test input:
   - Identify its color pair.
   - Look up the action.
   - Apply.

## Why earlier attempts failed

The code assumed the pair was always `(1, 2)` — it hard-coded direction
using 1 as "source" and 2 as "target". The train examples show that the
pair VARIES and each pair uniquely determines placement.

Stop hard-coding. Read the pair off the input, dispatch by pair.
