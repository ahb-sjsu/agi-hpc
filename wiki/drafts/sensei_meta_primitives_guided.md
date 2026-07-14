---
type: sensei_note
tags: [strategy, primitives, meta, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Meta — The `primitives_guided` Strategy Is Broken, Not Just Weak

0/77 across every attempted task (avg score 0.00). This is not a
low-signal strategy — it is a bug. Killing the branch outright is
more honest than letting the UCB gate "learn" to avoid it.

## Why it fails

The prompt in `arc_scientist.py` dumps `PRIMITIVE_CATALOG` (a text
description of available functions) and then says:

    You can call any primitive above inside your function.

But the generated `def transform(grid)` runs in a sandbox that does
**not** import those primitives. The LLM has three bad options:

1. Call `find_components(grid)` → `NameError` at execution.
2. Re-implement the primitive inline → defeats the point; worse than
   `direct` because the prompt is now bloated with catalog text.
3. Skip the primitive idea entirely and produce no-code → which is
   the observed failure mode.

## Fix paths (pick one)

**Option A — remove.** Delete the `r_strategy < 0.55 and n_prior >= 2`
branch. Free up those attempts for `direct` / `failure_aware`.

**Option B — rewire.** Prepend `from agi.autonomous.primitives
import *` inside the executed code string, and narrow the catalog
to 4–6 high-value primitives per prompt (not the full dump). Include
ONE worked example showing the exact call syntax.

**Option C — defer.** Gate the branch behind an env flag
`EREBUS_ENABLE_PRIMITIVES_GUIDED=1`. Default off until a worked
example shows non-zero success on a held-out task.

Pick Option A for now. Primitive composition is a real direction,
but the current implementation is not moving the needle and it is
burning attempts that could go toward `direct`.
