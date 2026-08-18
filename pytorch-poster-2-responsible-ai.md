# PyTorch Conference NA 2026 — POSTER #2 (Responsible AI track) — ACCEPTED

> Sessionize: https://sessionize.com/pytorch-conference-north-america-2026-posters/
> ACCEPTED — participation confirmed 17 Aug 2026. Self-printed poster.
> UPDATED 2026-08-17: the D4 claim is corrected to V4 throughout (DEME keystone
> correction, erisml-lib commit ea7ee82): correlative swap and deontic negation are
> COMMUTING involutions, so the measured group is the Klein four-group V4 (order 4);
> D4 (order 8, non-abelian) is posited, licensed only if quarter-turn operations are
> demonstrated. Also refreshed: 477 tests (was ~440).
> NOTE: reviewers flag "AI-generated or templated content." This draft is grounded in
> the real ErisML / erisml-compiler codebase; **edit it into your own builder's voice.**
> Companion to POSTER #1 (Applications): this is the governance layer that gates those agents.

---

## Session title (as accepted)
**Moral Tensors and DecisionProofs: Compiling Language into an Auditable, Grounded Safety Layer**

## Track
Responsible AI

## Audience level
Intermediate

## Session description (Sessionize field — 1198 chars, cap 1200 — UPDATED, V4 correction)

Most AI safety is a scalar at the output: one good/bad check before an action ships. We keep the
structure instead — who is affected, what is owed, who consented, who bears imposed risk — and
contract to a decision only at the end, auditably.

Two open-source pieces realize it. erisml-compiler (alpha, on PyPI) turns natural-language moral
material into a typed MoralGraph and a rank-1…6 moral tensor, read through four lenses at once:
consequentialist, deontic (Kantian gates via a Z3 solver), virtue, and care. When they disagree,
it refuses to silently aggregate and defers to a human. At runtime, ErisML's three-layer gateway
drops the same evaluation into an agent's plan→act loop, sealing each decision in a SHA-256
hash-chained DecisionProof, failing safe, never open.

The geometry pays off: correlative swap and deontic negation commute on Hohfeld's normative
positions, generating the measured Klein four-group V4 (D4: testable, unproven); a Bond Index
tests whether a judgment survives the agent↔patient swap. The PyTorch hook is literal — forward
hooks on transformer layers compare what a model says with what it exhibits. Running today on a
Qwen/Gemma stack and two governed AI NPCs.

> Exact 1198-char string (with `—`/`…`/`→`/`↔` counted as 1 char each) is saved as
> `pytorch-poster-2-description-1198.txt` — paste that file verbatim into Sessionize.

> Glyph note: `—`, `…`, `→`, `↔`, and the en-dash in `rank-1…6` each count as 1 char in
> Sessionize. For an ASCII-only version swap to `-`/`...`/`->`/`<->` (adds ~6 chars, still under).

## What the poster shows (panels)

1. **The thesis** — *structure before contraction.* A scalar verdict throws away the tensor; we
   keep it. The contraction step is logged: which dimensions were weighted, which stakeholders lost,
   what **moral residue** remains.
2. **The compiler pipeline** — `text → segment → extract → canonicalize → MoralGraph → tensorize →
   DEME → DecisionProof` (12 passes). The typed MoralGraph (stakeholder / act / maxim / commitment /
   fact / norm nodes) with a canonical SHA-256 hash. Three extractor tiers: **rule** (deterministic),
   **LLM** (NRP / vLLM + a critic pass), **probe** (LaBSE classifier head).
3. **Framework pluralism = honesty** — four projections (consequentialist · deontic · virtue · care)
   reading one graph. *We don't pick a moral theory for you and hide it; we run four and show you
   where they conflict.* `cross_projection_disagreement` defers the metaethical call to a human.
4. **The geometry** — Hohfeld's square (O–C / L–N) under its two **commuting involutions**:
   the **correlative swap s** (agent↔patient, the one the **Bond Index** tests) and the **deontic
   negation r²**. Together they generate the **Klein four-group V4** (order 4, abelian) — the
   *measured* structure. The full **D4** (order 8, non-abelian) is *posited*: licensed only if
   quarter-turn operations are demonstrated empirically. *"Not 'is this good?' (a scalar) — 'does
   this judgment preserve the bonds?' (a geometry)."* Multi-rank moral tensors (party × time ×
   action × coalition).
5. **Runtime gateway + DecisionProofs** — Reflex → Tactical → Strategic; `ALLOW / REVISE / BLOCK`;
   the hash chain (`previous_proof_hash → proof_hash`). **Graceful degradation:** ethics service
   times out → fall back to rule-based checks. Fails *safe*.
6. **Worked example (real numbers)** — the bundled `nazi_attic` case: per-party harm splits cleanly
   (speaker 0.76 *forbid*, village 0.83 *forbid*, refugees 0.0 *prefer*), **Gini 0.43**, worst-off =
   village, exact **Shapley** attribution, DecisionProof chained to the IR hash. *This is what
   "auditable" means — you can replay the judgment.*
7. **The PyTorch lens + governed agents + QR** — the **activation lens** (forward hooks on
   transformer layers) vs the **text lens**, with a **delta lens** + five named failure modes →
   `requires_human_review`. Demo: the **Halyard** AI NPCs (ARTEMIS / SIGMA-4) gated by the validator
   + DecisionProofs + a human Keeper kill switch. Repos: `erisml-compiler`, `erisml-lib`, `agi-hpc`.

## Key takeaways

- **Structure before contraction.** Keep the moral tensor (stakeholder / time / action / coalition);
  collapse to a decision only at the end, and log the collapse + the residue.
- **`pip install erisml-compiler`** — turn text into a MoralGraph + moral tensor + hash-chained
  DecisionProof. Alpha, 477 tests, MIT, Zenodo DOI. You can run the worked example on a laptop.
- **Framework pluralism is the responsible move.** Four ethical lenses, no silent aggregation; when
  they disagree the compiler surfaces it and **defers to a human** instead of faking a consensus.
- **Ethics as geometry.** The measured V4 symmetry + the Bond Index test *internal consistency
  under perspective swaps* — violations come with step-and-party receipts, not a vibe. (We
  corrected our own overclaim: D4 is now the posited extension, not the established result —
  and the poster says so.)
- **Auditable by construction; fails safe.** SHA-256 DecisionProof chains; the runtime gateway falls
  back to rule-based checks (never open) when the ethics engine is unavailable.
- **A real PyTorch hook:** an activation lens reads transformer internals to catch *say-vs-exhibit*
  mismatches — demonstrated on a Qwen/Gemma stack and two governed AI NPCs.

## Speaker

**Andrew H. Bond** — Lead consultant at AT&T (networks); 20+ years across AT&T, Cisco, and
Fujitsu Network Communications. IEEE Senior Member (since 1989). BSEE, NC State; MSSE, San
José State. Teaches computer engineering / data science and advises graduate theses at San
José State and Sonoma State. Independent researcher in AGI safety and cognitive-systems
benchmarking. ORCID 0009-0003-2599-6158.

## Notes
- Companion to **POSTER #1 (Applications)** — *Atlas + Erebus*, the platform whose agents this layer
  governs (`pytorch-poster-1-applications.md`).
- **Honesty / scope (per SCOPE.md — keep this on the poster's edge, don't oversell):**
  - **Symmetry epistemic status (the July 2026 keystone correction):** the two implemented
    Hohfeldian operations (correlative swap `s`, negation `r²`) commute, so the *measured* group
    is **V4 = {e, s, r², sr²}** (abelian, order 4). **D4 is posited**, testable, not established —
    it needs quarter-turn operations (`r, r³, sr, sr³`) observed as normative operations, which has
    not happened. Say "V4 measured, D4 posited." Source: `erisml-lib/src/erisml/ethics/hohfeld.py`
    docstring + `tests/test_hohfeld_d4.py::test_hohfeldian_operations_generate_v4`.
  - **Where things live:** the Hohfeld/V4 module + Bond Index are in **erisml-lib**; the compiler
    has **no Hohfeld module yet** (designed, roadmap). The **Z3 SMT** universalizability gate is
    **compiler-side** (`delta/universalizability_smt.py`); erisml-lib's deontic gate is a
    deliberate dependency-free lookup table. Don't cross-attribute if asked.
  - `erisml-compiler` is **alpha (v0.9.0)**. The **rule/LLM text path is solid and tested**; the
    **activation/probe lens is early** — the `ActivationProbe` is *uncalibrated by default*, so its
    numeric output is research-grade, not load-bearing yet. Frame the activation lens as "promising
    extension," not a finished detector.
  - The **action axis** on ranks 4–6 is a **stub** (parametric length, replicated values); the
    coalition axis is real. Say "ranks 1–3 real, higher ranks partial."
  - **Silicon emit** is Vitis HLS **C++ only** (hardware-emulation verified); no on-FPGA bring-up.
    Mention only if asked.
  - `erisml-lib` (DEME V3 runtime) is **not yet on PyPI** — install from source.
  - The **embodied-robotics** framing is the design target; the **live** deployment is conversational
    agents (Atlas stack + Halyard NPCs). Do not imply a physical robot is in the loop.
- **DEME** referenced as the tiered ethics profile/pipeline; avoid committing to an acronym expansion.
- Cite on the references strip: *Tensorial Ethics* (Bond, 2025); SQND-Probe (2026); Hohfeld (1917);
  Social Chem 101 (Forbes et al., EMNLP 2020) for the ethos profiles; Zenodo DOI 10.5281/zenodo.20659432.
