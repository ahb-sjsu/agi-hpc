---
type: sensei_note
task: 78
tags: [spatial, translation, holes, global-transform, arc]
written_by: Professor Bond
written_at: 2026-04-19
---

# Task 078 — Translate Shape Until It Fills Holes

## This is a GLOBAL rigid-body translation, not a local fill.

The whole 2-shape moves as one piece. Do not fill the 1-region's holes
with scattered 2s from local rules.

## Algorithm

1. Find connected components. Identify:
   - the 1-region (bounding box, plus its internal 0-holes)
   - the 2-shape (connected cluster of 2s, located below the 1-region)
2. Translate the 2-shape **upward** so that its cells exactly cover the
   0-holes inside the 1-region. The relative shape of the moving 2-cluster
   must match the shape of the hole pattern at the aligned offset.
3. After the move:
   - Original positions of the 2-shape become 0 (they vacated).
   - The 2s appear at their new positions inside the 1-region.
   - A trail of 2s remains in a vertical column from the lowest new 2
     down to the highest original 2 position (see train[1] and train[2]:
     the 2s stretch from the hole-fill zone down to where the shape used
     to end).

## Missing primitive

```python
def translate_shape_until_fits(source_cells, mask_cells, axis='up'):
    """Slide source along `axis` by k steps until, when overlaid, its
    cells coincide with mask_cells. Returns the translation vector."""
```

The word you've been reaching for: this is "gravitation" — the 2-shape
falls/rises until it lodges into the 1-region's holes.

## Why earlier attempts failed

The code tried to compute hole-fill from the perspective of the
1-region (look at each hole, ask 'what color goes here?'). That's
local. The correct perspective is from the 2-shape: 'where can I
translate to so that my silhouette lines up with the available
holes?' That's global.
