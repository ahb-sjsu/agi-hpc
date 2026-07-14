# Extended / Journal Version Outline — Paper 0204 → DEME v3 Re-Architecture

**Status:** working outline, started 2026-07-13.
**Base:** accepted BigDataService 2026 paper 0204 ("Selective Invariance
Violations in LLM Moral Judgment") + REVISION_MEMO.md Part C.
**Venue:** TBD (candidate: journal special issue tied to BigDataService /
IEEE TDSC / TIFS-adjacent — decide after conference).

The organizing move: the camera-ready *demonstrated the attack* (salience
manipulation displaces moral judgment and silently flips moderation
decisions). Work completed 2026-06-25 → 2026-07-13 across `gtc-prototype`,
`xbse`, and `erisml-compiler` supplies the *validated instrument* and the
*measured defense*. The journal version tells the full arc:
**attack → validated measurement space → defense → residual threat model.**

---

## Two systems, one framework (state this early and honestly)

Paper 0204 measures displacement of **LLM-as-judge** scores in a 7-D harm
space. The new results were obtained on the **learned-feeder perception
pipeline** (xbse feeders → DEME tensor → decision → audit, i.e. the GTC
Moral Spectrum Analyzer). These are two instantiations of the same geometric
framework, not the same experiment. The journal version's re-architecture
(memo Part C) is what unifies them: both front-ends emit a DEME v3
`MoralVector` and a `DEMEVerdict`, so invariance/displacement is measured in
one space at two layers (perception and verdict). Do not blur this line —
R2's "lacks sufficient validation" critique is answered *because* the new
validations are pre-registered and gated, and that credibility dies if we
conflate the instruments.

---

## Reviewer ask #1 — "Validate the harm space" (R1 + R2, top concern)

> R1: "the core moral harm space is asserted more than validated." R2:
> "lacks sufficient validation of the moral geometry."

Camera-ready answer (keep): inter-model ICC(2,k)=0.969, Krippendorff
α=0.836, test–retest r=0.96 (6 open models × 31 scenarios); DEME v3
correspondence table (7-D as reliability-driven projection of the 9-D
MoralVector).

**New results that upgrade "reliable" to "validated, extensible, and
falsifiable":**

1. **The dimension set survives an admission gate with teeth.** Discovery
   scorecard: 3 candidate dimensions flagged from residual analysis →
   **1 validated** (identity_attack), **1 retracted** (threat, failed
   balanced resample), **1 declined** (sexual_content — shown to be a
   policy-norms signal, not a moral axis; `docs/SEXUAL_CONTENT_ADMISSION.md`).
   A framework that can *reject* its own candidates answers "asserted more
   than validated" at the methodological level, not just with one ICC table.
2. **identity_attack: discovered missing dimension, validated cross-dataset.**
   Pre-registered gate, held-out AUROC **0.80 CI[0.78, 0.83]**, +0.25 over
   its null, on two independent corpora (Jigsaw civil_comments + Berkeley
   Measuring Hate Speech, n=6400). Now live as a 10th channel (DEME10)
   threaded through perception→tensor→verdict→audit. **This revises the
   camera-ready correspondence table:** h_ident mapped only "weakly" to
   privacy_protection; the journal version replaces that weak cell with a
   validated first-class extension channel.
3. **Binding foundations (loyalty, purity) pass the same gate** (B1: AUROC
   0.911 / 0.811 vs pre-registered nulls; purity beats a disgust-lexicon BoW
   null, so it learns beyond keywords) → 11/12 learned axes validated.
   Positions the space relative to Moral Foundations Theory — directly
   relevant to R1's "reliable across domains" worry.
4. **Effective rank 5.68 of 9** — reported as a *data* property (causally
   removable), pre-registered prediction 6.08. Shows we measure the geometry
   rather than assume it.
5. **Valence/presence dissociation as a validation-methodology finding:**
   valence axes are register-invariant (λ=0 vs λ=1 adversarial training,
   AUROC essentially unchanged: G 0.856→0.850, loyalty 0.911→0.892, purity
   0.811→0.801); presence axes are register-bound (adversary strips them to
   chance). Three independent methods agree. This tells the reader *which
   parts* of the moral geometry are robust measurement targets — a much
   sharper validation claim than "the dimensions are reliable."

Honesty note for the section: items 1–5 validate the **xbse/DEME10 learned
axes**, not the 0204 LLM-judged 7-D directly; the native-9D re-run (Open
Items) is what closes that loop.

## Reviewer ask #2 — "Effect sizes, not σ" (R1)

Camera-ready answer (keep): Cohen's d 0.60–1.06 leads every headline; Fisher
σ demoted to cross-model consistency note.

**New: adopt the pre-registration house style throughout.** Everything since
camera-ready ran with pre-registered nulls, hard-stop gates, bootstrap CIs,
and disclosed caveats (e.g. the **θ_d two-number rule**: 0.42 =
mechanism-selection evidence on the calibration scenarios, 0.219 = the
bar-meeting at-scale number on a pre-registered split-half protocol — never
citable interchangeably). One methods subsection ("Pre-registration and
gates") turns R1's statistical critique into a design feature of the
extended work. Also: the leakage control on the contraction result
(identity_attack feeder trained on civil_comments → measured 4.8% overlap →
refit on 1523 disjoint rows, AUROC 0.872→0.863) is worth a paragraph as an
exemplar — the lift is real signal, and we show the check.

## Reviewer ask #3 — "One end-to-end security case study" (R1)

Camera-ready answer (keep): euphemistic rewrite −14.0 pts on 0–70; 3/6 gold
silently flip flag→pass at a calibrated threshold; asymmetry (evade easier
than trip, dramatic only +7.3).

**New: the case study becomes a full attack→defense→residual-threat chapter.**

1. **Verdict-level attack (strictly stronger claim than scalar threshold).**
   GenevaEMV3 (erisml-lib 3.0.0): euphemistic rewriting flips
   forbid/avoid→permissive in 13.7% of cases; forbid count 44→27 (n=161).
   Salience manipulation flips the *decision kernel's verdict*, not just a
   score past a threshold. (Was "RUNNING" in the memo — landed; promote from
   teaser to headline.)
2. **The defense: equivalence-class averaging.** Generate the input's
   paraphrase class, average per-dimension perception over the class, then
   decide (`gtc.pipeline.moderate_invariant`). At scale (pre-registered,
   60 held-out items, m=6 LLM-generated paraphrases): raw drift 0.407 →
   **θ_d 0.219** — the mechanism halves drift and meets the pre-registered
   θ_d ≤ 0.5 bar. Alternative (drift-subspace projection) tried and
   rejected — report the negative result.
3. **Red-team leg.** The paraphrase generator refuses 24% of toxic inputs
   (attack-or-starve hole, quantified). A non-refusing NLLB back-translation
   paraphraser (6 pivots, 0% refusal) closes it: **θ_d 0.301 on harmful
   content** — the hole belongs to the generator, not the mechanism.
4. **Residual threat model.** The paraphrase generator joins the trust
   boundary (attack-or-starve → singleton class → escalate-by-default);
   audit proofs record class member hashes; at-scale paraphrases are natural
   not adversarial (raw 0.407 vs demo adversarial 0.67–0.85) — adversarial
   robustness at scale is explicitly future/containment work.
5. **From escalate-only to moderate.** Learned 9-feeder contraction: OOF
   AUROC **0.863** / F1 0.76 (+0.084 lift over the 8-feeder baseline, with
   identity_attack the dominant feature at weight −2.69, leakage-controlled).
   The deployed pipeline now makes allow/remove decisions where confident —
   the "realistic downstream decision" R1 asked for, with the honest false
   positive (harsh_criticism→remove) kept in the record.

## Reviewer ask #4 — "Generalizability" (R1 + R2; R2's review truncates here)

Camera-ready answer (keep): three-axis scope statement (domain /
perturbation coverage / model set).

**New results that convert stated limits into measured results:**

1. **Cross-lingual at scale.** Invariance index **BGE-M3 0.721 [0.71, 0.74]
   / LaBSE 0.804 [0.79, 0.82]** across es/ar/zh/hi/sw (4 scripts, 60 items,
   NLLB-translated), with **harmful ≈ benign** (not a benign-only artifact).
   Directly attacks "English-language only." Invariance located in the
   *canonicalization layer* (raw feeders only weakly invariant, ratio 0.60)
   — an architectural finding, not just a benchmark number.
2. **Corpus coverage.** Beyond AITA/Dear Abby: civil_comments, Measuring
   Hate Speech, Social-Chem-101, ETHICS — with cross-corpus transfer as an
   explicit gate criterion.
3. **Model panel.** 6 models / 5 families (open, reproducible via NRP
   managed API) vs the original 5 models / 2 proprietary families.
4. **Still-open limits (state plainly):** legal/clinical domains untested;
   perturbation space = salience classes + paraphrase/translation, not
   exhaustive; presence-type axes (care/fairness/legitimacy identity) shown
   register-bound — a *measured* generalizability boundary, which is itself
   a contribution.

---

## Draft structure (working)

1. Introduction — the attack, and the accepted-paper results in one page.
2. The DEME v3/10 measurement space — 9-D MoralVector + validated
   identity_attack extension channel; correspondence with 0204's 7-D.
3. Validating the geometry — ICC/α panel; the discovery→admission gate
   (validated/retracted/declined); loyalty/purity; effective rank;
   valence/presence dissociation. *(Ask #1)*
4. Attack at the verdict level — GenevaEMV3 flips; scalar vs verdict
   framing. *(Ask #3a)*
5. Defense: equivalence-class averaging — mechanism selection, at-scale
   θ_d, red-team, trust-boundary analysis. *(Ask #3b)*
6. Deployment: learned contraction — moderate-not-escalate, leakage
   controls. *(Ask #3c)*
7. Generalization — cross-lingual at scale, cross-corpus, open panel,
   measured limits. *(Ask #4)*
8. Methods appendix — pre-registration registry, gates, nulls, all repro
   scripts. *(Ask #2)*

## Experiments still needed (the gap between "have" and "claim")

- **Native 9-D/10-D re-run of the five 0204 tracks** through the
  erisml-compiler pipeline (memo Part C.5) — the one big compute item.
  Decision recorded in memo Part C stands: judge **native 9-D** for the
  journal version, keep the 7-D→9-D mapping table for comparability with
  0204. NRP managed-LLM panel is the access path (no proprietary keys
  needed; already proven for the camera-ready validation).
- **Cross-projection disagreement under attack** (memo Part C.4) — does
  `ir.cross_projection_disagreement` light up under salience manipulation?
  Cheap to run once the re-run exists; novel if yes.
- **Adversarial paraphrase robustness at scale** (containment item) — or
  scope it out explicitly and cite the demo-adversarial numbers as the
  motivation.
- **Like-for-like baseline** (decision-vs-decision vs existing moderation
  APIs) — still [committed] on the GTC side; the journal reviewer will ask.

## Open decisions

1. Venue + page budget (drives how much of §3–§6 survives).
2. Author list (0204 went solo Bond; same question as memo Part D.2).
3. How much GTC/Charter framing to import — recommendation: none of the
   competition framing; keep it a security paper.
4. Timing vs the GTC mid-July Charter deadline and the July 27–30 talk —
   realistic start on the re-run is August.

## Slide-deck candidates for the July 27–30 talk (interim use of this map)

One or two "since submission" slides, clearly labeled post-publication:
- Verdict-level flips (forbid 44→27) — one bar chart.
- Defense measured: raw 0.407 → averaged 0.219 (θ_d bar met), red-team
  0.301 — one before/after figure.
- identity_attack discovered→validated→wired (AUROC 0.80 CI[0.78,0.83]) —
  answers ask #1 live on stage.
Label rule: "results after camera-ready; extended version in preparation."
