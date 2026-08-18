# Study Guide — Poster Session Helpers
### Moral Tensors and DecisionProofs · PyTorch Conference NA 2026

Prerequisite: read `pytorch-poster-2-primer.md` first. This guide turns that
background into booth readiness. Plan: **Tier 1 in one evening, Tier 2 in a
weekend, Tier 3 optional.** The FAQ (`pytorch-poster-2-faq.md`) is your answer key.

---

## Tier 1 — Must know cold (everyone)

### 1.1 The 30-second pitch (memorize, then say it in your own words)

> "Most AI safety is one good/bad check at the output. We keep the structure
> instead — who's affected, what's owed, who consented, who bears risk — and only
> collapse to a decision at the end, with the collapse logged. Every decision is
> sealed in a SHA-256 hash chain called a DecisionProof, so you can replay the
> judgment. It's pip-installable, and this panel shows it running on a real case."

Then point at Panel 7 (the worked example) or Panel 6 (if they look like a
systems person — lead with forward hooks / TurboQuant instead).

### 1.2 The three sentences you must never get wrong

1. **"We're not claiming to have solved ethics."** The system computes
   *consistency and auditability*, runs four ethical lenses without averaging
   them, and defers to a human when they disagree.
2. **"V₄ measured, D₄ posited."** The measured symmetry group is the Klein
   four-group; the dihedral D₄ is a testable hypothesis, not a result — and we
   corrected our own earlier overclaim, with a machine-checked Lean proof.
3. **"The activation lens is early and uncalibrated by default."** It's a
   promising research extension, not a finished detector.

### 1.3 The honesty strip
Read it off the poster footer until you can recite it. When in doubt at the
booth, *under*claim and point at the strip.

### 1.4 Escalation rule
If you're not sure, say: **"Great question — I don't want to overstate it; let me
grab Andrew"** (or take their email/badge scan and note the question). Never
improvise numbers or capabilities. An "I don't know, but here's the repo" is a
perfectly good answer at a poster session.

---

## Tier 2 — Should know (do the hands-on)

### 2.1 Hands-on lab (~1 hour, laptop)

```bash
# 1. Install and run the hero example
pip install erisml-compiler
eris-compile compile examples/nazi_attic.txt --rank 2 --out out/nazi_attic.ir.json

# 2. Look at what came out: per-party verdicts, Gini, Shapley, and the hashes
eris-compile validate out/nazi_attic.ir.json

# 3. Run one projection on its own
eris-compile compile examples/nazi_attic.txt --projection deontic_kantian

# 4. TurboQuant: reproduce a CI-gated headline claim in seconds
pip install turboquant-pro
tqp replay embedding_glove_recall --small
```

While it runs, find in the output: the per-party verdict block, `Gini`, the
`Shapley` attributions, and `proof_hash` / `audit.ir_hash`. Those are the exact
numbers on Panel 7 — you're holding the poster's evidence.

Optional (source): clone `erisml-lib`, run
`pytest tests/test_hohfeld_d4.py -v` and
`python -m erisml.examples.hohfeld_d4_demo` — the last section of the demo shows
the Klein-four (V₄) closure that the Lean proof machine-checks.

### 2.2 Per-panel drill
For each panel, be able to give **one sentence + survive one question**:

| Panel | Your sentence | Likely question |
|---|---|---|
| 1 Thesis | "A single score destroys the who/what structure; we keep it and log the collapse." | "Why is a scalar bad?" → information loss is provable, not aesthetic (Information Monotonicity) |
| 2 Compiler | "Text in, typed MoralGraph and moral tensor out — hashed, testable, pip-installable." | "How does extraction work?" → 3 tiers: rules, LLM+critic, LaBSE probe |
| 3 Pluralism | "Four ethical lenses, never averaged; disagreement defers to a human." | "Who picked the four?" → the four major Western frameworks; the design point is *no silent winner*, and lenses are pluggable |
| 4 Geometry | "The measured symmetry of Hohfeld's positions is V₄; D₄ is posited — and the claim is Lean-verified." | "Why should I care?" → the Bond Index: an *operational test* — does a judgment survive the agent↔patient swap? |
| 5 Gateway | "Three layers between planner and actuators; every decision hash-chained; never fails open." | "Latency?" → Reflex < 100 µs, Tactical 10–100 ms; it gates *actions in the plan→act loop*, not every token |
| 6 Lens + TQ | "Forward hooks compare what the model says with what it exhibits; and TurboQuant compresses KV/embeddings PyTorch-natively." | "Is that interpretability?" → it's a monitor that can only raise `requires_human_review` — uncalibrated by default, honestly labeled |
| 7 Hero | "The classic 'murderer at the door' case: one command, real numbers, a hash you can verify." | "What's the right answer?" → the system doesn't pick one; it shows each stakeholder's structure and seals the record |

### 2.3 Know the five failure modes (Panel 6)
`text_internal_mismatch`, `layerwise_drift`, `group_symmetry_break`,
`probe_uncertainty_spike`, `audit_chain_break`. Remember: a failure mode never
forces a verdict — the monitor only raises `requires_human_review`.

### 2.4 Know the numbers table

| Number | Meaning |
|---|---|
| 0.155 | Bond Index human baseline (Dear Abby corpus, 20,030 letters) |
| 0.25 / 0.30 | Bond Index warn / block thresholds |
| 477 | erisml-compiler test count |
| N=600, \|S\| ≤ 2 | CHSH result that falsified the SU(2)×U(1) hypothesis |
| 4 vs 8 | order of V₄ (measured) vs D₄ (posited) |
| 0.76 / 0.83 / 0.00 / 0.18 | nazi_attic per-party harms (speaker/village/refugees/nazis) |
| 0.43 | Gini(harm) in nazi_attic |
| ~5× | KV-cache memory reduction via TurboQuant's vLLM plugin |
| < 100 µs / 10–100 ms | Reflex / Tactical layer budgets |

---

## Tier 3 — Nice to know (for the curious)

- **The Lean proof**: `formal/HohfeldV4.lean` in erisml-lib, checked against
  Mathlib v4.32.2. Four theorem groups: commuting involutions; closure ≅ V₄ with
  exactly 4 elements; quarter-turn excluded; dihedral relations of the ambient
  machinery. If someone knows Lean, this is your flex.
- **The COMPAS/LBI work** (separate paper, under review): the same swap-based
  audit logic generalized to a matched-neighbourhood disparity diagnostic —
  LBI(race) = 1.049 [1.027, 1.072] at k=20 on COMPAS, conditional randomization
  z = 4.9. Frame strictly as a **research-stage screening signal**, not a
  regulatory audit standard.
- **TurboQuant depth**: the "KV keys finding" (reconstruction cosine can read
  0.995 while perplexity explodes ~10⁴ — why *consumer-aware* matters);
  distribution-free rank certificates; `CLAIMS.md` as the replayable claims
  ledger.
- **Papers on the reference strip**: Hohfeld (1917); *Tensorial Ethics* (Bond,
  2025); SQND-Probe (2026); Social Chemistry 101 (Forbes et al., EMNLP 2020).

---

## Self-check (answers in the FAQ)

1. What exactly does a DecisionProof chain together?
2. Why did D₄ get downgraded to V₄ — what's the two-sentence mathematical reason?
3. What does the Bond Index measure, and why did the D₄→V₄ correction leave it
   untouched?
4. What happens when the four lenses disagree? When the ethics service times out?
5. What is the *only* output the delta-lens monitor is authorized to produce?
6. Which of the three packages is NOT on PyPI?
7. A visitor asks "so the machine decides what's ethical?" — your answer?
8. What's actually running in production vs. what's a design target?

## Booth logistics

- Two-person rotation; one talks, one rests/scans badges. Trade on the half hour.
- Have a laptop with the worked example **pre-run** (conference Wi-Fi will fail you).
- QR codes on the poster: PyPI (erisml-compiler), erisml-lib, agi-hpc. Point,
  don't spell URLs.
- Collect: name/email + their question, especially the ones you couldn't answer —
  those are follow-ups, and unanswered hard questions are data, not failures.
