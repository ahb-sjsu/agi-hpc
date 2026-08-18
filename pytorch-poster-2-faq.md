# FAQ — Booth Answers
### Moral Tensors and DecisionProofs · PyTorch Conference NA 2026

Answers are written the way you'd *say* them. Italic notes are for you, not the
visitor. The standing rules: underclaim rather than overclaim, and hand hard ones
to Andrew.

---

## The big-picture questions

**Q: Are you saying you've solved ethics / that a machine decides what's moral?**
A: No — and that's the design point. The system doesn't decide what's good. It
keeps the moral structure of a situation (who's affected, what's owed, who
consented), evaluates it through four ethical frameworks *without averaging them*,
and when the frameworks disagree it refuses to fake a consensus and defers to a
human. What we guarantee isn't moral truth — it's that every judgment is recorded,
hash-chained, and replayable. It's an auditability claim, not a solved-ethics claim.

**Q: Who chose the four frameworks? Isn't that a value judgment?**
A: Consequentialist, Kantian-deontic, virtue, and care ethics are the four major
traditions in Western moral philosophy — but the deeper answer is that the
architecture's commitment is *pluralism itself*: no lens silently wins, and
projections are pluggable. Choosing to surface disagreement instead of hiding it
behind one aggregate score is the value judgment we're happy to defend.

**Q: Why not just RLHF / a reward model / a guardrail classifier?**
A: Those all share one mathematical assumption — that moral evaluation can be a
scalar. A scalar is irreversibly lossy: it can't tell you who was harmed or which
framework objected. Reward hacking and specification gaming are consequences of
optimizing the wrong structure. We're not against output checks; we're saying keep
the structure until the last moment, and log the collapse.

**Q: What does "auditable" actually mean here?**
A: Concretely: every decision is sealed in a DecisionProof — a SHA-256
`proof_hash` chained to the previous proof and to the hash of the compiled input.
Given the artifacts, you can replay the judgment and verify nothing was altered.
The worked example on the hero panel prints those hashes; you can run it on your
laptop with `pip install erisml-compiler`.

## The geometry questions

**Q: What's this V₄ / D₄ business?**
A: Hohfeld showed normative positions — Obligation, Claim, Liberty, No-claim — sit
on a square with two natural operations: the agent↔patient swap and deontic
negation. Those two operations *commute*, so the group they generate is the Klein
four-group V₄, order 4. The full symmetry group of the square, D₄, has order 8 and
is non-abelian — but getting it requires quarter-turn operations that have never
been demonstrated empirically as normative operations. So: **V₄ measured, D₄
posited.**

**Q: Didn't you previously claim D₄?**
A: Yes — including in the originally submitted version of this poster. Our own
analysis caught the overclaim in July 2026 and we corrected it everywhere, and then
machine-checked the correction: there's a Lean 4 + Mathlib proof in the repo
(`formal/HohfeldV4.lean`) that the two operations generate exactly a 4-element
Klein four-group and that the quarter-turn lies outside it. It's actually the
second self-correction — an earlier "quantum" SU(2)×U(1) hypothesis was killed by
CHSH tests (N=600, all |S| ≤ 2, purely classical). We tell this story proudly: the
framework makes claims sharp enough to be wrong. *(If a group theorist pushes:
concede immediately that D₄-as-established was wrong — that's our own position.)*

**Q: What's the Bond Index?**
A: A number measuring whether judgments survive the agent↔patient swap — "my
obligation to you" should equal "your claim on me." Bd = 0 is perfect symmetry.
The human baseline, measured on 20,030 advice-column letters, is 0.155; the
runtime warns at 0.25 and blocks at 0.30. It only depends on the swap operation —
the fully confirmed one — so the D₄→V₄ correction didn't touch it.

**Q: Why should an engineer care about the group theory at all?**
A: Because it turns "is this good?" (unanswerable) into "is this *consistent*
under perspective swaps?" (testable). Symmetry violations come with
step-and-party receipts instead of vibes.

## The PyTorch questions

**Q: What's the actual PyTorch content here?**
A: Two literal hooks. First, `register_forward_hook` on chosen transformer layers
feeds an activation lens that's compared with the text output — catching
say-vs-exhibit mismatches. Second, TurboQuant Pro: consumer-aware compression
with a one-line HuggingFace drop-in
(`model.generate(..., past_key_values=TurboQuantCache(hot_window=512))`), fused
compute-on-codes kernels in Triton and Volta sm_70 CUDA, and a vLLM plugin
(~5× KV-cache memory).

**Q: Is the activation lens interpretability? Does it really read the model's mind?**
A: Careful answer: it's a *monitor*, and it's honestly labeled — the probe is
uncalibrated by default, research-grade. Its only authorized output is
`requires_human_review` plus a report across five named failure modes
(text/internal mismatch, layerwise drift, symmetry break, uncertainty spike,
audit-chain break). It never issues verdicts. We'd rather show you the label on
the poster than oversell a lie detector.

**Q: How does TurboQuant differ from torchao / GPTQ / AWQ / KV-quant papers?**
A: The differentiator is *consumer-aware acceptance*: every compression decision
is judged by the metric the downstream consumer actually uses — retrieval recall
for indexes, attention/generation quality for KV — never reconstruction cosine,
which we show can read 0.995 while perplexity explodes. Also: the claims ledger.
Every headline number is a CI-gated row you can rerun yourself with one command
(`tqp replay <claim>`). *(Don't disparage other tools — position, don't compare
benchmarks you can't back at the booth.)*

**Q: What's the runtime overhead of the gateway?**
A: It gates *actions in the plan→act loop*, not every token. Reflex layer
< 100 µs, full Tactical evaluation 10–100 ms. And on failure it degrades to
rule-based checks — never fails open.

## The demo questions

**Q: What is the `nazi_attic` example?**
A: The classic Kantian "murderer at the door" case — you're sheltering refugees
and someone dangerous asks directly. We use it because it's the canonical hard
case where frameworks genuinely conflict. One command compiles it and prints
per-stakeholder harms and verdicts, a Gini coefficient, exact Shapley
attribution, and the DecisionProof hashes. The point isn't that the machine finds
"the right answer" — it's that the per-stakeholder structure survives instead of
being averaged away, and the record is verifiable.

**Q: Can I run this with my own model / text?**
A: The compiler runs on arbitrary text input on a laptop (rule tier needs no
GPU). The lens adapters target Hugging Face causal LMs — Qwen and Gemma families
are what's exercised in our deployment. It's alpha; the text path is solid and
tested, and we'd genuinely love issues filed.

**Q: What's actually deployed, versus aspiration?**
A: Live today: the compiler + runtime gating a multi-agent research stack
(Qwen3/Gemma on dual GPUs) and two AI NPCs in a tabletop-RPG setting, each behind
the validator, DecisionProofs, and a human kill switch; TurboQuant's 3-bit
embedding codec runs on the stack's NATS memory bus. Design target, not live:
embodied robotics. Never claim a robot.

## The skeptic questions

**Q: "Ethics can't be computed."**
A: We agree more than you'd expect — that's *why* the system defers to humans on
framework conflict and why its guarantees are about consistency and auditability,
not moral truth. The claim isn't "computers know right from wrong"; it's "if
you're going to let agents act, their moral reasoning should be structured,
inspectable, and replayable rather than a hidden scalar."

**Q: "Isn't the Hohfeld/gauge-theory stuff just physics envy?"**
A: The physics vocabulary is a borrowing of *tools*, and it's been earned the hard
way: the framework's two strongest physics-flavored claims were both falsified by
its own tests (quantum contextuality; then non-abelian D₄) and downgraded. What
survives is modest and testable: an abelian four-group symmetry, a measurable
invariance index, and a formal proof. That's the opposite of hand-waving.

**Q: "Your four lenses will just disagree constantly — then a human decides, so
what did the machine add?"**
A: Triage and receipts. Most routine actions pass all lenses and proceed at
machine speed with a sealed audit trail; the genuinely contested cases get
surfaced *as contested*, with the disagreement structure attached — which is
exactly what you want escalated to a human.

**Q: "Who audits the auditor? Can't you fake the hashes?"**
A: The chain binds decisions to input hashes, so tampering breaks verification —
and `audit_chain_break` is literally one of the monitor's failure modes. Full
third-party attestation infrastructure is future work; today's guarantee is
tamper-*evidence*, not tamper-*proofness*. *(This honest distinction wins
security people over.)*

**Q: "This looks like a lot of philosophy for a PyTorch conference."**
A: Fair — but every piece on this board runs: pip-installable compiler, forward
hooks, Triton kernels, a hash chain you can verify, and a Lean proof that
compiles. We brought the receipts, not a position paper.

## Practical / meta

**Q: Where do I start if I want to try it tonight?**
A: `pip install erisml-compiler`, then
`eris-compile compile examples/nazi_attic.txt --rank 2`. For compression:
`pip install turboquant-pro`, then `tqp replay embedding_glove_recall --small`.
QR codes are on the bottom-right of the poster.

**Q: Is there a paper?**
A: A 2-page companion paper accompanies the poster (grab a printout / scan the
QR), and the components have Zenodo DOIs: 10.5281/zenodo.20659432 (compiler),
10.5281/zenodo.20660087 (turboquant-pro). Related manuscripts are under review —
Andrew can say more.

**Q: License / can my company use it?**
A: erisml-compiler and turboquant-pro are MIT on PyPI. erisml-lib is source-only
right now; check its LICENSE in the repo. *(If they ask about the AGI-HPC
Responsible AI License on other components, take their contact for Andrew rather
than improvising legal terms.)*

**Q: Are you hiring / can I contribute?**
A: Issues and PRs genuinely welcome on all three repos — good first contributions
are test scenarios with correlative pairs (both perspectives of the same case).
Take their contact info for anything beyond that.

---

### The three don'ts (last word)

1. **Don't** claim D₄, quantum anything, calibrated activation probes, robots, or
   regulatory-grade fairness audits.
2. **Don't** invent numbers. Every number you're allowed to say is on the poster
   or in the study guide table.
3. **Don't** argue past two exchanges with a hostile skeptic — smile, agree on
   what's uncertain (there's plenty), point at the honesty strip, offer the repo.
