---
type: sensei_note
tags: [classification, symmetry, d4, primitive-family, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Meta — Symmetry-Class Classifier Family

A common sub-class of CLASSIFICATION tasks (output 1×1) on small grids.
The answer is a function of the input's D4 symmetry signature.

## Recognize the family

All three must hold:

1. **Output shape is `1×1`** (or a small fixed shape like `3×1`) — see
   `sensei_meta_task_typology` for why that's the gate question.
2. **Input is a small square grid**, usually 3×3 or 4×4.
3. **Output colors are a small discrete set** (typically 2–4 values).

If all three hold, the answer is almost certainly a symmetry-class
lookup. Stop reaching for pixel-level transforms.

## The four D4 reflections

Given an occupancy mask `nz` (non-zero cells of the input):

```python
h  = np.array_equal(nz, np.flipud(nz))                # horizontal axis
v  = np.array_equal(nz, np.fliplr(nz))                # vertical axis
d  = np.array_equal(nz, nz.T)                         # main diagonal
ad = np.array_equal(nz, np.flipud(np.fliplr(nz)).T)   # anti-diagonal
```

There are 16 possible subsets of these 4, but typically only 4–6 appear
in a given task: `{none}`, `{d}`, `{ad}`, `{h,v,d,ad}`, and
occasionally `{h,v}` or `{d,ad}`.

## The recipe

1. **Classify each training input by its signature.**
2. **Map signature → output.** If all inputs with the same signature
   produce the same output, you have the full rule.
3. **If two inputs share a signature but produce different outputs,
   add a disambiguator.** Try in order:
   - `nz[0,0]` — is the corner filled? (Separates X-shape from
     plus-shape at full D4 symmetry.)
   - Total occupied count: `nz.sum()`.
   - Number of connected components (4-connectivity).
   - Center value: `g[h//2, w//2]`.
4. **Verify on all training examples before writing code.**

## Why colors don't matter

For this family, the task is purely geometric. The palette is a
distractor — two inputs with identical shape but different colors
always produce the same output. Build the occupancy mask first;
operate on the mask, not the colored grid.

## Tasks in this family

- `task056`: output ∈ {1, 2, 3, 6}; disambiguator is `nz[0,0]` for full-D4.
- Likely candidates (not yet verified): any task where `train[i].output`
  is 1×1 and multiple `train[i].input` share color palette sizes. Check
  these with the recipe before burning attempts.

## Common failure modes

- Treating **X-shape** and **plus-shape** as "same symmetry, same
  output" — they're not; both have full D4 but the disambiguator
  `nz[0,0]` separates them.
- Using `n_syms >= 1` as a binary predicate (any-symmetry vs
  no-symmetry) — this collapses 4 output classes into 2 and
  misclassifies half the training set.
- Confusing main-diagonal symmetry with anti-diagonal symmetry in
  the code — they're different operations. `nz.T` gives main-diag;
  anti-diag needs a 180-rotate before transposing.
