# Primer — Moral Tensors and DecisionProofs
### Background reading for session helpers · PyTorch Conference NA 2026 (Responsible AI poster)

Read this first, in one sitting (~45 min). The study guide tells you how to practice;
the FAQ gives you booth-ready answers. You don't need a philosophy background — every
concept you need is here.

---

## 1. The one idea everything hangs on

Most AI safety is a **scalar at the output**: the system produces an action, one
good/bad check runs, the action ships or doesn't. A single score throws away exactly
the information an audit needs — *who* is affected, *what* is owed to each person,
*who* consented, *who* bears imposed risk.

ErisML keeps that structure as a typed object all the way through evaluation, and
collapses ("contracts") to a decision **only at the end** — logging what the collapse
weighed, who lost, and what *moral residue* remains. The claim is **not** "we know
what is good." The claim is: **every collapse from structure to verdict is recorded
and replayable.** Auditability, not solved ethics. If you remember one sentence for
the booth, it's that one.

## 2. The three code pieces

| Piece | What it is | Status |
|---|---|---|
| **erisml-compiler** | Turns natural-language moral material into typed structure and evaluates it | alpha v0.9.x, **on PyPI**, MIT, 477 tests, Zenodo DOI 10.5281/zenodo.20659432 |
| **erisml-lib** | The DEME v3 runtime: three-layer gateway + DecisionProofs + the Hohfeld/V₄ module | v3.0.0, **source install only** (not on PyPI) |
| **turboquant-pro** | Consumer-aware compression (embeddings, LLM KV caches) with a native PyTorch surface | **on PyPI**, MIT, Zenodo DOI 10.5281/zenodo.20660087 |

## 3. The compiler pipeline (Panel 2)

`text → segment → extract → canonicalize → MoralGraph → tensorize → DEME → DecisionProof`

- The **MoralGraph** is typed — stakeholder, act, maxim, commitment, fact, norm nodes —
  and carries a canonical SHA-256 hash.
- **Three extractor tiers**: rules (deterministic), LLM (vLLM + a critic pass), and a
  probe (LaBSE classifier head).
- The **moral tensor** has ranks 1…6 (axes like party × time × action × coalition).
  Honesty: **ranks 1–3 are fully real; higher ranks are partial** (the action axis is
  a stub).

## 4. Four lenses, no silent winner (Panel 3)

One MoralGraph is read through four ethical projections **simultaneously**:

- **Consequentialist** — harm/care tensor; Gini, worst-off, exact Shapley attribution.
- **Deontic (Kantian)** — universalizability gates formulated in **Z3 SMT** (an UNSAT
  result witnesses a "contradiction in conception"). *The Z3 gate lives in the
  compiler; erisml-lib's deontic gate is a deliberately dependency-free rule table.*
- **Virtue** — Aristotelian habit-consistency.
- **Care** — Gilligan / Noddings / Tronto.

They are **never averaged**. When they disagree, `cross_projection_disagreement`
fires and the decision **defers to a human**. This is the poster's ethical stance:
we don't pick a moral theory for you and hide the choice.

## 5. The runtime gateway + DecisionProofs (Panel 5)

Between an agent's planner and its actuators sit three layers:
**Reflex** (< 100 µs, hard stops) → **Tactical** (full ErisML evaluation, 10–100 ms)
→ **Strategic** (policy recording + human oversight), emitting
`ALLOW / REVISE / BLOCK`.

Every decision is sealed in a **DecisionProof**: a SHA-256 `proof_hash` chained to
the previous proof and to the IR hash of the compiled material. That's what
"auditable" means concretely — you can *replay* a judgment and verify the chain.
If the ethics service is unavailable, the gateway falls back to rule-based checks.
**It never fails open.**

## 6. The geometry (Panel 4) — read this section twice

Wesley Hohfeld (1917) showed legal/moral relations reduce to four positions:
**Obligation (O), Claim (C), Liberty (L), No-claim (N)**, arranged on a square.
Two operations act on them:

- **Correlative swap `s`** — the agent↔patient perspective swap: O↔C, L↔N.
  ("My obligation to you is your claim on me.")
- **Deontic negation `r²`** — O↔L, C↔N.

**The key fact (and the story behind it):** `s` and `r²` **commute** — applying them
in either order gives the same result. Two commuting involutions generate the
**Klein four-group V₄** = {e, s, r², sr²} — abelian, order 4. That is the **measured**
structure. The full dihedral group **D₄** (order 8, non-abelian — the complete
symmetry group of a square) is **posited**: it would require "quarter-turn"
operations to be demonstrated empirically as normative operations, which has not
been done.

Earlier materials — including the originally submitted version of this very poster —
claimed D₄ outright. That was an overclaim, corrected in July 2026. The correction is
**machine-checked**: a Lean 4 + Mathlib proof (`formal/HohfeldV4.lean` in erisml-lib)
verifies that s and r² are commuting involutions, that their closure has exactly 4
elements ≅ V₄, that the quarter-turn lies *outside* it, and that the ambient D₄
machinery satisfies the dihedral relations. There was an even earlier correction:
an SU(2)×U(1) "quantum" gauge hypothesis was falsified by CHSH tests (N=600, all
|S| ≤ 2 — purely classical correlations). **Tell this as a strength**: the framework
makes claims sharp enough to be wrong, and twice it was wrong and said so.

**The Bond Index (Bd)** measures the correlative-symmetry defect: does a judgment
survive the agent↔patient swap? Bd = 0 is perfect symmetry; the empirical human
baseline is **0.155** (from the Dear Abby corpus: 20,030 advice-column letters,
1985–2017); the runtime warns at 0.25 and blocks at 0.30. The Bond Index tests only
the `s` operation — the fully confirmed one — so it was untouched by the D₄→V₄
correction.

## 7. The PyTorch lens (Panel 6)

The PyTorch hook is literal: **forward hooks** on chosen transformer layers feed an
**activation lens** (what the model internally *exhibits*), compared against a
**text lens** (what it *says*). A **delta lens** compares them and can raise
`requires_human_review` across **five named failure modes** (from the code,
`delta/failure_modes.py`):

1. `text_internal_mismatch` — text vs activation lens disagree in direction
2. `layerwise_drift` — per-layer probes drift as if a representation is being suppressed
3. `group_symmetry_break` — an equivariance test fails where the probe should be invariant
4. `probe_uncertainty_spike` — a dimension's uncertainty exceeds a hard ceiling
5. `audit_chain_break` — the trace's audit hash doesn't match the chain

Two honesty points, always: (a) a failure mode **never forces a verdict** — the
monitor's only authorized output is `requires_human_review`; verdicts remain DEME's
job; (b) the activation probe is **uncalibrated by default** — research-grade, a
promising extension, not a finished detector.

## 8. The worked example (Panel 7, the hero)

`nazi_attic` is the classic Kantian **"murderer at the door"** case (you're hiding
refugees; someone dangerous asks directly). One command:

```
pip install erisml-compiler
eris-compile compile examples/nazi_attic.txt --rank 2
```

Real outputs: per-party harm/verdicts (speaker 0.76 *forbid*, village 0.83 *forbid*,
refugees 0.00 *prefer*, nazis 0.18 neutral), Gini(harm) = 0.43, worst-off = village,
exact Shapley attribution, and a DecisionProof whose `proof_hash` chains to
`audit.ir_hash`. The pitch line: *"One command, real numbers, a hash you can verify."*
Note what the example demonstrates: the evaluation **keeps the per-stakeholder
structure** instead of averaging it away.

## 9. TurboQuant Pro (Panel 6 strip)

The second PyTorch-native piece — consumer-aware compression, meaning each vector is
compressed by the metric its downstream consumer actually uses, never reconstruction
cosine alone:

- **One-line HuggingFace drop-in**:
  `model.generate(**inputs, past_key_values=TurboQuantCache(hot_window=512))`
- **Triton** fused compute-on-codes kernels + **Volta sm_70** kernels (the deployment
  GPUs) — attention scores computed from compressed codes without materializing fp16
  keys
- **vLLM plugin** — ~5× KV-cache memory reduction
- torch as the portability plane: one optional dependency covers CUDA, ROCm, Apple
  MPS, Intel XPU
- **Live in this stack today** as the 3-bit embedding codec on the NATS memory bus
- Every headline number is a CI-gated, replayable row in `CLAIMS.md`
  (`tqp replay <claim>`) — the compression-layer analogue of a DecisionProof

## 10. Deployment reality

The layer gates a live multi-agent research stack (Qwen3/Gemma models on dual-GPU
hardware) and two governed AI NPCs — **ARTEMIS** and **SIGMA-4** in the Halyard RPG —
each behind the validator, DecisionProofs, and a human Keeper kill switch.
**Embodied robotics is the design target; conversational agents are the live
deployment.** Never imply a physical robot is in the loop.

## 11. The honesty strip (memorize it)

> alpha v0.9.x · text path solid · activation lens early/uncalibrated ·
> ranks 1–3 real, higher partial · **V4 measured, D4 posited (Lean + Mathlib
> verified)** · embodied = design target, conversational agents = live ·
> erisml-lib not yet on PyPI

This is printed on the poster on purpose. It reads as rigor because it is.
