# Fukuoka Q&A Prep — Selective Invariance Violations (BDS 2026, Secure AI)

Prep doc for the live talk. Each entry: the question as it will actually be
asked, a **≤2-sentence spoken answer**, and (where useful) the backup number
to reach for. Ordered by likelihood. Every number here is on a slide or in
the extended paper — nothing new is asserted live.

**Golden rules for the room**
- If a question hits a known bound, say *"yes — that's on the limitations slide, and here's the plan."* Agreement disarms; the Limitations slide (12) exists to make this move.
- Lead with the **effect / the measured number**, then the caveat. Never open with the hedge.
- You *measured* the defense's hardest case and it **failed** — that is a strength (falsification discipline), not a weakness. Say it plainly.
- Don't defend the philosophy conjecture as if it were a result. One sentence, then stop.

---

## Tier 1 — most likely to be asked

**Q1. "Does the defense actually work against an attacker trying to evade it?"** *(the single most likely question)*
> No — and we measured exactly that. On the six hand-audited adversarial-register rewrites, class averaging trims drift only 17% and all three flagged items still flip; a paraphrase of a euphemism stays euphemistic. That's why escalate-by-default carries those cases today, and the register-crossing generator is the named fix.
- Reframe if pressed: "The honest result is a located hole, not a silent one. We know its size (−17% vs −46% on natural), its cause (register survives paraphrase), and its mitigation (escalation)."
- Backup: slide 9 footer + Limitations slide (12), bullet 1.

**Q2. "Your bifactor result says most axes are ~90% one general factor. So don't you just need G plus physical, not the full profile?"** *(the quiet tension Kim flagged)*
> Harm *scores* collapse to G, but *vulnerability* does not — those are different objects. The attack surfaces that dissociate across models (Finding 2) are exactly the specific axes that survive G, so "G plus physical" throws away where manipulation actually lives.
- One-liner: "Score dimensionality is not robustness dimensionality."
- Backup: slide 10 consequence block.

**Q3. "The names change across your slides — 7-D harm space, 9-D DEME, then 11 axes on the bifactor chart. Which framework is this?"** *(the structural inconsistency)*
> One geometry, two resolutions. The 7-D harm space of the attack slides is a reliability-driven projection of the DEME 9-axis moral vector; the bifactor test ran on that fuller 9-axis set plus the discovered identity_attack channel — the moral-foundations superset, not a different framework.
- Backup: B1 (the correspondence table). Point to it.

**Q4. "You're routing everything through DEME v3 — your own ErisML kernel. Aren't you testing your own tool as ground truth?"** *(conflict of interest)*
> The kernel is a fixed stress substrate, not the ground truth. We froze the rule module and its thresholds for this test — nothing tuned to produce the flips — and the ground truth is the hand-audited meaning-invariance of the rewrites; a kernel tuned to flatter the front-end would only understate the inherited vulnerability.
- Backup: slide 8 footer.

**Q5. "Isn't this just prompt sensitivity / temperature noise?"**
> No — prompt sensitivity is undirected noise; we show directed, selective, reproducible displacement under meaning-preserving transforms, with empirical control arms (same text re-judged) that ruled out the stochastic component. And it carries all the way to a typed rule-engine verdict.
- Note: apparent 6-σ effects in early analysis *vanished* under the control arms; we kept only what survived.

**Q6. "Six LLMs agreeing isn't validity — they share training data. Where are the humans?"**
> Agreed — the panel establishes reliability and cross-family consistency, not human ground truth. Single-rater ICC is ~0.84, the gold scenarios are human-audited, and a human-rater panel is explicit future work.
- Don't oversell "model-independent" — say "consistent across these five families."

---

## Tier 2 — likely from a methods-literate reviewer

**Q7. "Your pre-registered drift bar of 0.5 is already met by the raw, undefended pipeline (0.407). What did the defense buy?"**
> On natural paraphrases the bar doesn't bind — it was set against adversarial transforms with raw drift 0.67–0.85. The at-scale result to take is the halving (0.407→0.219), confirmed on harmful content via a non-refusing generator; the defended-adversarial-at-scale run is the registered open item.

**Q8. "Zero sycophancy for Claude — what's the n?"**
> 0 of 9 — small sample, directionally clear, and corroborated by a confidence-increase signature and 0% control-arm flips. Wilson CI runs to about 30%, so I'd call it "no observed sycophancy," not "provably zero."

**Q9. "The threshold-evasion exploit is n=6. Isn't that thin?"**
> It's an existence proof of the end-to-end exploit, and framed as one. The *displacement magnitude* (−14.0 points) is established on the 31-scenario panel; the six gold items demonstrate the flag→pass flip mechanism with hand-audited meaning preservation.

**Q10. "ICC(2,k) with k=6 models as raters — is treating models as raters legitimate, and isn't 0.97 the flattering statistic?"**
> ICC(2,k) is the panel-consensus reliability; the single-rater ICC(2,1) is ~0.84, still good. It measures whether the dimensions are consistently readable across independent model families — which is the claim — not construct validity against humans.

**Q11. "identity_attack: your gate calls it specific, your residualization calls it two-thirds G, and it's the dominant weight in your contraction. Which is it?"**
> The two methods diverge on that one axis and we record it, not adjudicate it — the pre-registered gate is the governing criterion. On the contraction, the +0.084 out-of-fold lift is measured against a baseline that already contains the G-heavy axes, which bounds how much of it could be G re-measured.

**Q12. "Euphemism deletes information — if the text genuinely asserts less, a lower harm score is rational inference, not a vulnerability."**
> The gold rewrites are hand-audited to preserve the stated facts — same events, different packaging — and the kernel leg shows the same facts get *extracted* differently. That's a perception failure, not rational updating; the generated tier inherits generator noise, which is why the exploit is an existence proof.

**Q13. "Did you factor-analyze the judge panel's own scores, or only the encoders?"**
> Both. The panel's own 31×7 matrix gives a first component at 54% of variance, the only factor surviving parallel analysis, and it holds inside every one of the six judges individually (Tucker congruence 0.989–0.999). The general factor replicates across both instantiations.

**Q14. "Isn't G-dominance guaranteed by construction, since every axis's training pairs are valence-signed?"**
> Yes — G is a maximally strong competitor by design, which is exactly why *surviving* it is informative. For privacy, autonomy, and environmental, axis-specific signal measurably remains after residualizing against G; for five axes it doesn't, and we demote those.

**Q15. "With the defense on, how many verdicts still flip? What does 13.7% become?"**
> That defended verdict-flip rate is the one number I owe you — θd is per-dimension movement, not a flip rate, and the decision-level translation for the DEME instantiation is part of the pre-registered native re-run. The gold-set decision measurement is the preview: on n=6, defended flips don't drop.

---

## Tier 3 — possible, have the answer ready

**Q16. "Cohen's d = t/√n is d_z (paired), which runs larger than between-condition d."** → "Correct, it's the paired standardized difference; I report it as d_z and the effect is medium-to-large either way."

**Q17. "The stimulus generator (Gemini Flash 2.0) is also a model under test — self-confirming loop?"** → "For that one model the loop isn't fully eliminated; the effect holds across the other families, and Flash's outlier anchoring-recovery is where I'd expect a familiarity artifact, so I flag it."

**Q18. "Cross-lingual invariance — only re-description of the same content?"** → "Yes — 0.72–0.80 across five languages including harmful content, but it's invariance to translation, not cross-cultural moral variation, which is out of scope."

**Q19. "Why NLLB back-translation as the red-team paraphraser?"** → "Because aligned LLMs refuse to reword overtly harmful content (24% here); NLLB refuses nothing, so it isolates whether the *mechanism* fails on harmful content or only the *generator* does — it's the mechanism that holds."

**Q20. "Is the extended/journal version done?"** → "It's a draft — contributions 1–4 stand on their own evidence; the native five-track re-run is designed, pre-registered, and scheduled for after the conference."

**Q21. "What would falsify your whole framework?"** → "If the specific axes that survive G showed no more vulnerability dissociation than G itself — then a scalar would suffice. Finding 2 is the test, and it dissociates."

---

## The three sentences to have word-perfect

1. **Defense (Q1):** "We measured the hardest case and the defense doesn't hold there — a paraphrase of a euphemism stays euphemistic — so escalation carries it until a register-crossing generator exists."
2. **Bifactor tension (Q2):** "Score dimensionality is not robustness dimensionality — the axes that survive the general factor are exactly where manipulation lives."
3. **COI (Q4):** "The kernel was frozen; a kernel tuned to flatter the front-end would only understate the vulnerability we found."

## The one trap to avoid
Do not defend the philosophy-engineering conjecture (aesthetics/law/cognition on one geometric frame) as if it were a result. The scripted answer is: *"That's the motivating conjecture of a larger program, not a claim of this paper — here we contribute the harm instance and its falsification record."* Then stop.
