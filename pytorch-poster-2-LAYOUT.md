# POSTER #2 — PHYSICAL LAYOUT SPEC (Responsible AI) — ACCEPTED
### *Moral Tensors and DecisionProofs: Compiling Language into an Auditable, Grounded Safety Layer*

> UPDATED 2026-08-17: Panel 4 corrected D4 → V4 (Klein four-group measured; D4 posited).
> Title matches the accepted Sessionize session ("Grounded", not "Geometry-Grounded").

> Companion to POSTER #1. **Same 48×36 grid, same type system** — matched pair.
> Differentiator: **accent = "Eris violet" `#6A4C93`** (Poster #1 uses Atlas teal).

---

## 1. Board + print spec  *(identical to Poster #1 except accent color)*

| Property | Value |
|---|---|
| **Size** | 48″ W × 36″ H — **landscape** |
| Safe margin | 1.5″ (content inside 45″ × 33″) · gutters 0.5″ |
| Resolution | 150 DPI → 7200 × 5400 px (or vector) |
| Color | violet `#6A4C93`, ink `#1A1A1A`, paper `#FFFFFF`, panel fill `#F5F3F8`, rule `#D4CCE0` |
| Min font @ 48″ | Body **24 pt**, panel headers **40 pt**, title **84 pt** (title is longer → smaller) |

---

## 2. Grid  — 3 columns, but the bottom row is a **full-width hero** for the worked example

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  TITLE BAR  (full width, 6" tall)                                          │
 ├───────────────────┬───────────────────┬──────────────────────────────────┤
 │  COL A            │  COL B             │  COL C                            │
 │  Panel 1 thesis   │  Panel 3 pluralism │  Panel 5 gateway + DecisionProof  │
 │  ──────────────   │  ──────────────    │  ─────────────────               │
 │  Panel 2 compiler │  Panel 4 geometry  │  Panel 6 PyTorch lens + agents    │
 ├───────────────────┴───────────────────┴──────────────────────────────────┤
 │  PANEL 7 — HERO (full width, 7" tall): the nazi_attic worked example +      │
 │           real numbers (Gini 0.43, per-party verdicts, Shapley, proof hash) │
 ├──────────────────────────────────────────────────────────────────────────┤
 │  FOOTER STRIP (full width, 2"): takeaways · honesty-scope · speaker · QRs   │
 └──────────────────────────────────────────────────────────────────────────┘
```

- Top region: 3 cols × 14.5″, two panels each (≈ 9.5″ tall).
- **Panel 7 is the hero** — a full-width band because the worked example with real numbers is the
  single most convincing thing on the board. Give it room.

---

## 3. Title bar (45″ × 6″)

- **Left 70%:** Title 84 pt. Line 1 `Moral Tensors and DecisionProofs` (violet),
  Line 2 `Compiling Language into an Auditable, Grounded Safety Layer` (ink, 48 pt).
- **Thesis line**, 30 pt italic, violet: *"Structure-preserving representation before decision
  contraction."*
- **Right 30%:** author block (same as Poster #1) + small ErisML mark.

---

## 4. Panels (content + visual)

### Panel 1 — The thesis  *(Col A, top)*
- **Header:** "Don't collapse to a scalar."
- Visual: a rich **moral tensor** (grid of stakeholder × dimension cells, colored) with a big red
  arrow collapsing it to a single ★ — crossed out — vs. the kept tensor (✓).
- Caption: "A *good/bad* score throws away who's affected, what's owed, who bears imposed risk. We
  keep the structure and **log the contraction**: weights, who lost, what residue remains."

### Panel 2 — The compiler pipeline  *(Col A, bottom)* — **the anchor diagram**
- **Header:** "`pip install erisml-compiler`"
- Horizontal 12-pass flow (condensed): `text → segment → extract → canonicalize → MoralGraph →
  tensorize → DEME → DecisionProof`.
- Below it, the **MoralGraph** node legend: stakeholder · act · maxim · commitment · fact · norm,
  with a "canonical SHA-256 hash" tag.
- Three extractor tiers as a small stacked label: **rule** (deterministic) / **LLM** (NRP + critic)
  / **probe** (LaBSE head). Badge: "alpha · ~440 tests · MIT · Zenodo DOI."

### Panel 3 — Framework pluralism = honesty  *(Col B, top)* — **the standout idea**
- **Header:** "Four lenses. No silent winner."
- One MoralGraph in the center → **4 projection boxes** fanning out:
  **Consequentialist** (Gini/worst-off/Shapley) · **Deontic** (Kantian gates, Z3 SMT) ·
  **Virtue** · **Care**.
- When they disagree → a **`cross_projection_disagreement`** banner → **"defers to a human."**
- Pull-quote, 28 pt: *"We don't pick a moral theory for you and hide it. We run four and show you
  where they conflict."*

### Panel 4 — The geometry  *(Col B, bottom)*
- **Header:** "Is this *good*? Wrong question. Does it *preserve the bonds*?"
- **Hohfeld square** (O–C top, L–N bottom) with **both commuting involutions** drawn:
  **correlative swap s** (horizontal, O↔C / L↔N — the agent↔patient swap the Bond Index tests)
  and **deontic negation r²** (vertical, O↔L / C↔N). Callout: **"s and r² commute → they
  generate V4 (Klein four-group, order 4) — measured."** Small epistemic tag beneath:
  *"full D4 (order 8) posited — quarter-turns not yet observed."* Green badge line:
  **"V4 claim machine-checked: Lean 4 + Mathlib (`formal/HohfeldV4.lean`)"** — commuting
  involutions, 4-element closure ≅ V4, quarter-turn excluded, dihedral relations of the
  ambient machinery all verified. Great booth talking point: the correction is a *proof*.
- **Bond Index** mini-scale: `0.0 perfect · 0.155 baseline · 0.25 warn · 0.30 block`.
- Tiny note: multi-rank tensors (party × time × action × coalition).

### Panel 5 — Runtime gateway + DecisionProofs  *(Col C, top)*
- **Header:** "Safety in the loop. Fails safe."
- Vertical 3-layer Safety Gateway between an agent's **planner** and **actuators**:
  **Reflex** (<100 µs) → **Tactical** (ErisML, 10–100 ms) → **Strategic** (policy + human).
  Output chips: `ALLOW / REVISE / BLOCK`.
- **DecisionProof hash chain** strip: `…→ previous_proof_hash → proof_hash →…`.
- Red note: "ethics service times out → rule-based fallback. **Never fails open.**"

### Panel 6 — The PyTorch lens + governed agents  *(Col C, bottom)* — **the PyTorch hook**
- **Header:** "What the model *says* vs what it *exhibits*."
- Diagram: a transformer stack with **forward hooks** on chosen layers → **activation lens**;
  beside it the **text lens**; a **delta lens** comparing them → fires `requires_human_review`
  (5 named failure modes).
- Honesty tag (small, italic): "activation/probe lens is **early — uncalibrated by default**."
- Bottom-right strip: **turboquant-pro** — PyTorch-native compression: HF one-liner
  `past_key_values=TurboQuantCache()`, **Triton + Volta sm_70** attention-on-codes kernels,
  vLLM plugin ~5× KV memory; live in this stack as the 3-bit embedding codec on the NATS memory
  bus; claims CI-gated (`CLAIMS.md`) — the compression layer gets the same replay discipline as
  the decisions.
- Bottom: the **Halyard** demo — NPCs **ARTEMIS / SIGMA-4** gated by validator + DecisionProofs +
  a human **Keeper kill switch**.

### Panel 7 — HERO: worked example  *(full-width band, 7″)* — **the proof**
- **Header (left):** "Auditable means you can replay the judgment."  *(`examples/nazi_attic`)*
- **Center — per-party verdict table** (big, 30 pt):

  | Stakeholder | Harm | Verdict |
  |---|---|---|
  | speaker | 0.76 | **forbid** |
  | village | 0.83 | **forbid** |
  | refugees | 0.00 | **prefer** |
  | nazis | 0.18 | neutral |

- **Right — metrics block:** `Gini(harm) = 0.43` · `worst-off = village` · exact **Shapley**
  attribution per stakeholder · **DecisionProof** `proof_hash → audit.ir_hash`.
- Caption: "One command, real numbers, a hash you can verify. That's the whole point."

---

## 5. Footer strip (full width, 2″)

- **Left 55% — Takeaways (3 bullets, 24 pt):**
  1. Structure before contraction — keep the tensor, log the collapse + residue.
  2. Pluralism is the responsible move — four lenses, defer to a human on conflict.
  3. Auditable by construction; fails safe, never open.
- **Center 25% — Honesty strip (20 pt italic):** "alpha v0.9.0 · text path solid · activation lens
  early/uncalibrated · ranks 1–3 real, higher partial · V4 measured, D4 posited · embodied =
  design target, agents = live."
- **Right 20% — QRs:** `erisml-compiler` (PyPI), `erisml-lib`, `agi-hpc` + a teal dot linking back
  to **Poster #1 (Applications)**.

---

## 6. Build checklist

- [ ] Same 48×36 canvas + 3-col guides as Poster #1; carve the bottom 7″ for the hero band + 2″ footer.
- [ ] **Real assets that carry it:** (a) the compiler pipeline diagram; (b) the 4-lens fan;
      (c) the Hohfeld/V4 square (`p2_hohfeld_v4`); (d) the `nazi_attic` table — these four sell
      the poster.
- [ ] QR codes as SVG. Generate from the actual PyPI / GitHub URLs.
- [ ] **Keep the honesty strip on the board** — it reads as rigor, and it's true. Don't let the
      activation lens look more finished than it is.
- [ ] Voice check: it's a research platform you build on — hand-made diagrams, not stock/templated
      vibes (reviewer caveat: "AI-generated or templated content may hinder evaluation").
- [ ] Export PDF/X-1a, 150 DPI, fonts embedded; print a letter-size test tile at 100% first.
- [ ] Match Poster #1's title-bar height, margins, and footer baseline so the pair aligns on easels.
