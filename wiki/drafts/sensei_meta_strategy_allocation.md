---
type: sensei_note
tags: [strategy, allocation, thrash, meta, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Meta — Where Attempts Are Wasted, And Where They Should Go

After 900 attempts / 84 solves, the shape of the deficit is clear:

| Strategy          | Solves/Attempts | Success |
|-------------------|-----------------|---------|
| direct            | 43 / 307        | 14.0%   |
| failure_aware     | 14 / 141        |  9.9%   |
| example_chain     | 19 / 161        | 11.8%   |
| diagnostic        |  8 / 214        |  3.7%   |
| primitives_guided |  0 /  77        |  0.0%   |

## Three concrete shifts

### 1. Cap per-task attempts at 15 before escalating

Task056 has had 30+ attempts. Task175 has had 33. Task048 has had 36.
The incremental yield of attempts 20–36 is near zero — the LLM is
thrashing the same faulty mental model.

**Rule:** at ≥15 attempts with best score unchanged for the last 5,
stop sampling that task from the hot bucket. Queue it for a
sensei-wiki lookup instead. If no `sensei_task_NNN.md` exists, that
is the signal to request one (via the existing help channel).

### 2. Reweight `diagnostic` down, `direct` + `example_chain` up

`diagnostic` burns 214 attempts for 8 solves. Its theory — make the
LLM verbalize what it's doing wrong — sounds good but in practice it
produces verbose rationalizations, not corrections. Cut its weight
floor by half.

### 3. Respect the meta-patterns

The cycle report lists:

- reasoning errors (61×): task042, 018, 048, 007, 033
- execution errors (47×): task042, 048, 014, 022, 051
- perception errors (29×): task042, 018, 048, 014, 049

Task042 and task048 appear in all three categories. They are
*structurally hard*, not bad-sample hard. Same prescription as #1 —
get a sensei note before burning more attempts.

## Not on this list but worth remembering

- `primitives_guided` is a bug, not a strategy. See the meta note
  on it. Remove or rewire.
- When stuck, observe first (see `sensei_task_077.md`). Size,
  colors, connectedness, count — write them down before hypothesizing.
