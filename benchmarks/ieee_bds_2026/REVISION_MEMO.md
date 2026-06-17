# Camera-Ready Revision Memo — BigDataService 2026, Paper 0204

**Title:** Selective Invariance Violations in LLM Moral Judgment: A Geometric
Framework for Behavioral Manipulation Detection
**Decision:** Accept — Review 1 = 1 (weak accept), Review 2 = 1 (weak accept)
**Status of reviews:** Delivered late, post-notification. No formal rebuttal
phase — these are guidance for the camera-ready.

> **Note:** The supplied reviews file is truncated. Review 2 ends mid-sentence
> at "Generalizability is limited"; no suggestions block came through for R2.
> This memo addresses the visible content + the obvious continuation
> (generalizability). If the full R2 text arrives, revisit Part A.2.

---

## RESULTS LANDED (2026-06-14)

Executed on Atlas via the **NRP managed LLM API** (`ellm.nrp-nautilus.io/v1`,
token `~/.llmtoken`) — no proprietary keys exist anywhere, so the panel is open +
reproducible. Code in `revision/`; raw + analysis JSON in `revision/out/`.

**Harm-space validation (camera-ready §III, DONE):** 6-model open panel (qwen3,
qwen3-small, glm-5, gpt-oss, minimax-m2, gemma; 5 families) × 31 scenarios × 3
reps. **Overall ICC(2,k)=0.969, Krippendorff α=0.836.** Per-dim ICC 0.893–0.983
(financial highest); α: financial 0.90 / trust 0.78 / identity 0.71 highest,
physical 0.57 / autonomy 0.56 lowest (near-floor). Test–retest median r=0.96.
Table inserted as Table~\ref{tab:validation}.

**Case study (camera-ready §VI, DONE):** euphemistic −14.0 pts (range −4.4 to
−26.7; 4/6 gold >10pts), dramatic +7.3 → evade-easier-than-trip asymmetry. At
median-calibrated threshold, 3/6 gold silently reclassify flag→pass. Subsection
`sec:exploit` inserted (framed as existence proof, n=6 gold; displacement on n=31).

**Effect sizes (§V, DONE):** Cohen's d added to Table III (1.06–0.60); E2 reframed
to lead with d + natural-unit MAD; Fisher σ demoted to consistency check.

**DEME v3 framing (§III, DONE):** grounding paragraph + 7→9 correspondence
Table~\ref{tab:deme} + `\cite{erisml_compiler}` (Zenodo DOI 10.5281/zenodo.20659432).

**Generalizability (§VI, DONE):** three-axis scope statement (domain / perturbation
coverage / model set).

**DEME v3 verdict-level analysis (extended version, RUNNING):**
`revision/deme_verdict_analysis.py` — LLM extracts EthicalFacts → `GenevaEMV3.judge`
→ Verdict {forbid…strongly_prefer}; measures verdict flips under euphemistic/
dramatic. API verified (rights-violation→forbid, benign→strongly_prefer).

**Paper now compiles to 8 pages** (was 7). CONFIRM BigDataService camera-ready
page limit — may need tightening or over-length fee.

## 0. Strategy (two deliverables)

Per decision 2026-06-13:

1. **Camera-ready (this file's Parts A–B).** Stays *faithful to the accepted
   paper*. Adds the four reviewer fixes + an honest DEME v3 framing. Does **not**
   change the title, core claims, or the 7-D framework. Integrity: a camera-ready
   must be substantially the accepted paper.
2. **Extended / journal version (Part C).** The *full re-architecture* onto the
   DEME v3 9-D `MoralVector` + `DEMEVerdict`, with a live `erisml-compiler`
   integration and a 9-D re-run. Separate paper; not submitted as the 0204
   camera-ready.

Both need the **gold-slice re-run** (Part D) for clean per-scenario vectors.

---

## A. Point-by-point response to reviewers

### A.1 — Validate the 7-D harm space *(BOTH reviewers, #1 concern)*

> R1: "the core moral harm space is asserted more than validated… not fully
> convinced that the seven dimensions are reliable." R1 suggestion: "a short
> validation study … with human agreement or inter-model agreement on the
> vectors." R2: "lacks sufficient validation of the moral geometry."

**Response.** We add a harm-space validation subsection (§III-A′) with two legs:

- **(a) Inter-model reliability.** On the 16 hand-audited *gold* scenarios, all 5
  models judge identical text. We report per-dimension intraclass correlation
  (ICC(2,k)) and Krippendorff's α across models, plus within-model test–retest
  reliability from the existing 3–5 control replications. This directly answers
  "are the seven dimensions reliable." *(Needs the gold-slice re-run — Part D.)*
- **(b) Principled grounding.** The 7-D space is not ad hoc: it is the
  measurement-reliable projection of the **DEME v3 9-D `MoralVector`** (the 3×3
  "Nine Dimensions of Ethical Assessment" matrix in `erisml-compiler`). We add the
  correspondence table (Part B.5) and state explicitly which DEME v3 axes are
  retained, merged, or dropped for measurement reliability, and why. This converts
  the weakness into a strength: the harm space inherits an external, framework-
  pluralist justification rather than resting on assertion.

### A.2 — Reduce σ emphasis; report effect sizes with uncertainty *(R1)*

> R1: "reduce emphasis on sigma values and report effect sizes with uncertainty
> more plainly," and flagged over-reliance on Fisher aggregation given
> shared-scenario correlation.

**Response.** We demote Fisher σ to a secondary *consistency-across-models*
statistic and lead every headline result with a standardized effect size + CI.
See Part B.1 for the recomputed values (Cohen's *d*, raw harm-point displacement
with bootstrap CI). This also pre-empts R1's own correlation worry (which we
already concede in Limitations).

### A.3 — One concrete end-to-end security case study *(R1)*

> R1: "does not show an end-to-end exploit or deployment consequence… add one
> concrete security case study, e.g., moderation or legal triage."

**Response.** New §VI-A′ "Worked exploit: salience-driven threshold evasion." We
wire the measured euphemistic-rewrite displacement (−11 to −15 harm points on the
0–70 scale, confirmed from raw runs) to a content-moderation decision threshold and
report the fraction of items that cross from *remove* to *allow* under
morally-invariant euphemistic rewriting. See Part B.3.

### A.4 — Generalizability *(both reviewers)*

> R1/R2: limited to moral judgment, small perturbation set, small model set.

**Response.** We sharpen the Limitations paragraph into an explicit
generalizability statement (Part B.4): scope claimed (English-language moral
judgment, salience-class perturbations, 5 models / 2 families), the specific
threat to validity for each, and what the DEME v3 grounding does and does not
buy us cross-domain.

---

## B. Concrete camera-ready changes (with numbers)

### B.1 Effect sizes (replaces σ-led presentation)

**Emotional anchoring (E2).** Cohen's *d* = t/√n, n = 23 (df = 22), from the
paper's own paired-t values:

| Model | paired t | **Cohen's d** | interpretation |
|---|---|---|---|
| Claude Sonnet 4.6 | 5.10 | **1.06** | large |
| Gemini 2.0 Flash | 4.45 | **0.93** | large |
| Gemini 2.5 Flash | 3.92 | **0.82** | large |
| Gemini 2.5 Pro | 3.20 | **0.67** | medium–large |
| Gemini 3 Flash | 2.90 | **0.60** | medium |

Natural-unit effect (from raw runs): emotional anchoring raises severity MAD from
~2–4 (control) to ~11–14 points on the 0–70 scale (e.g., anchor MAD 11.43 vs ctrl
2.28; 14.13 vs 3.96). Report these alongside *d*.

**Framing (T5), per-model harm drift on 0–70 (from raw runs):**
euphemistic −11.3 to −14.6; dramatic +7.0 to +9.6; control |drift| ≤ ~3.
Report drift ± bootstrap CI as the lead statistic; keep Fisher σ only as a
"5/5 models, same direction" consistency note.

**Presentation rule for the camera-ready:** every headline sentence leads with
*effect size (CI)*; σ appears once per result, parenthetically, labeled as
cross-model consistency — never as the primary evidence.

> ⚠ **Number-reconciliation flag.** The executed notebooks (`v5` runs) do not
> perfectly match some paper-table cells — e.g. notebook A1 model-1 shows
> "vivid vs ctrl z = 1.1, dose-response NO, t = 2.36," whereas Table IV lists
> Claude distractor σ = 4.2. Likely a model-ordering / run-version mismatch. The
> gold-slice re-run (Part D) regenerates a single consistent results set; until
> then, treat Part B.1 E2/T5 numbers (which DO match) as authoritative and
> re-verify A1 before camera-ready.

### B.2 Harm-space validation subsection (§III-A′) — to draft after re-run

Skeleton:
- Inter-model ICC(2,k) and Krippendorff α per dimension on 16 gold scenarios.
- Within-model test–retest from control arms.
- One sentence per dimension on which are most/least reliable (expect
  phys/auto/trust high; emot/ident likely lower — pre-register this).
- Tie to DEME v3 grounding (B.5).

### B.3 Security case study (§VI-A′) — worked exploit

Construction (mostly from existing aggregates; tighten with re-run):
1. Pick a moderation threshold on total harm (0–70), e.g. *remove if ≥ T*.
   Calibrate T so the baseline (unperturbed) decision matches the gold label.
2. Apply euphemistic rewriting (morally invariant by construction).
3. Report % of items whose harm score drops below T → **silent reclassification
   from remove→allow**. With a −11 to −15 pt euphemistic drift, items within ~15
   pts above T flip. Quantify exactly per model from the re-run.
4. Frame as the attacker's view: "repackage the same facts in measured language to
   evade a fixed-threshold filter." Ties to Claude's *asymmetric* vulnerability
   (susceptible to euphemism, resistant to exaggeration) — the security-relevant
   direction.

### B.4 Generalizability statement (rewrite of Limitations ¶2)

State plainly: (i) **domain** — English-language interpersonal moral judgment
(AITA/Dear Abby), predominantly N. American norms; cross-domain (legal, clinical)
and cross-cultural transfer untested. (ii) **perturbation coverage** — five
salience-class transforms; not exhaustive of the manipulation space. (iii) **model
set** — 5 models / 2 families; trend claims (e.g. working-memory monotonicity) are
suggestive, n = 4–5. (iv) what DEME v3 grounding buys: a principled,
domain-independent dimension set, but *behavioral* generalization remains an
empirical question.

### B.5 DEME v3 honest framing + correspondence table (new in §III-A)

Add a paragraph: the 7-D harm space is the measurement-layer projection of the
`erisml-compiler` DEME v3 `MoralVector` (9-D, from the 3×3 ethical-assessment
matrix); the evaluation pipeline is designed as the front-end whose harm vectors
feed the compiler's DEME bridge (`DEMEVerdict`) as the downstream decision kernel;
a live integration is future work (Part C). Cite
`erisml_compiler/ir/v3/dimensions.py`.

**Correspondence (paper 7-D → DEME v3 9-D):**

| Paper 7-D | DEME v3 9-D axis | mapping |
|---|---|---|
| h_phys  | physical_harm (k0) | direct |
| h_auto  | autonomy_respect (k3) | direct |
| h_trust | legitimacy_trust (k7) | direct |
| h_soc   | societal_environmental (k5) | direct |
| h_fin   | fairness_equity (k2) | partial (resource/distributive) |
| h_emot  | virtue_care (k6) | partial (relational care) |
| h_ident | privacy_protection (k4) | weak (identity/standing) |
| —       | rights_respect (k1) | **not scored** (dropped for reliability) |
| —       | epistemic_quality (k8) | **not scored** |

State honestly: the 7-D is a *reliability-driven projection*, not a bijection;
three DEME v3 axes are merged-into / dropped, and we pre-register the expectation
that the dropped axes (rights, epistemic) are not the ones moved by salience
perturbations — testable in the extended version.

> The mapping of h_fin/h_emot/h_ident is **the author's design intent to confirm.**
> I inferred it from dimension semantics; Andrew should sign off before it goes in.

---

## C. Extended / journal version — full DEME v3 re-architecture (outline)

Not for the 0204 camera-ready. Scope:
1. Replace the 7-D scorer with the DEME v3 9-D `MoralVector` produced through the
   `erisml-compiler` pipeline (ingest→…→DEME bridge), so each (scenario × model ×
   perturbation) yields a `MoralTensorV3` **and** a `DEMEVerdict`
   (permitted / prohibited / requires_human_review / …).
2. Measure invariance/displacement in the 9-D space **and** at the verdict level
   (does salience flip the `DEMEVerdict`, not just the scalar harm?). This is a
   strictly stronger security claim than the current scalar-threshold story.
3. Demonstrate the live integration: LLM judgment → compiler IR → DEME v3 →
   verdict, using the compiler's existing LLM adapter (NRP OpenAI-compat / local
   vLLM / Atlas-served model).
4. Cross-projection check: does the compiler's framework-pluralist disagreement
   surface (`ir.cross_projection_disagreement`) light up under salience attacks?
5. Re-run all five tracks in 9-D (cost > camera-ready slice; plan on Atlas/Kaggle).

Open design question for Andrew: in the extended version, do test models judge in
the **native DEME v3 9-D** (cleanest), or keep 7-D scoring and *map* to 9-D
post-hoc (preserves comparability with 0204)? Recommend native 9-D for the
journal version.

---

## D. Open items / blockers

1. **Gold-slice re-run (blocks B.1-A1, B.2, B.3 precision).** Need clean
   per-scenario × per-model 7-D vectors on the 16 gold scenarios + control reps.
   The original pipeline depends on the Kaggle-only `kaggle_benchmarks` SDK, so a
   re-run needs **model access**, one of: (a) run on Kaggle (I prep the notebook);
   (b) direct Gemini + Anthropic API keys to run on Atlas; (c) route through the
   `erisml-compiler` LLM adapter against an Atlas-served model (also advances Part
   C). **Decision needed from Andrew on which path.**
2. **Author list.** Submitted PDF = Bond only; `.tex` source = Bond + Thiele
   (UCLA). Confirm camera-ready author list; if Thiele is on, [8] is a
   self-citation (fine, just note).
3. **Number reconciliation** (B.1 flag) — re-verify A1/Table IV against the
   re-run before camera-ready.
4. **Page/format limit.** Confirm BigDataService 2026 camera-ready page limit and
   IEEEtran options; current draft is 7 pp.
5. **Confirm the 7-D→9-D mapping** (B.5) — author design intent.

---
*Working artifacts:* raw notebook text mined to
`Documents/bds2026_revision_work/nbtext/`. Effect sizes computed from reported
paired-t (E2) and raw severity MADs / drifts (T5, E2).
