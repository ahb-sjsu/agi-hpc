# Poster talk script: Moral Tensors and DecisionProofs

PyTorch Conference NA 2026, Responsible AI poster. Panels are numbered on the board:
top row 1, 3, 5; bottom row 2, 4, 6; the wide band along the bottom is 7.

---

## The 30-second opener (say this to everyone who stops)

> "Most AI safety comes down to one number at the output: good or bad, ship it or don't.
> We keep the structure instead: who's affected, what's owed to each of them, who consented,
> who carries the risk. We only collapse that to a decision at the very end, and we record
> how we collapsed it. Every decision gets sealed in a hash-chained proof you can replay.
> It's open source, it's on PyPI, and it runs today in a live agent stack and in a simulated
> home-care robot."

Then ask: **"Are you more on the model side or the agent and deployment side?"**
Model people: go to panel 6 (the PyTorch lens). Agent people: go to panel 5, then 7.

---

## The 3-minute walkthrough

**Panel 1, the thesis (top left).**
> "The idea is structure before the scalar. A single score throws away exactly what an
> auditor needs. So we keep a typed object through the whole evaluation, and the collapse
> to a verdict happens last, with the weights, who lost, and what's left over all logged.
> We aren't claiming the system knows what's good. We're claiming every collapse is
> recorded and replayable."

**Panel 2, the compiler (bottom left).**
> "erisml-compiler takes plain-language moral material and runs it through a 12-pass
> pipeline: segment, extract, canonicalize, build a typed MoralGraph, turn it into a tensor,
> and end in a DecisionProof. Every stage is hashed. Extraction has three tiers:
> deterministic rules, an LLM with a critic pass, and a LaBSE classifier probe."

**Panel 3, pluralism (top middle).**
> "One graph, four ethical readings at once: consequences, duties, virtue, care. The duty
> check is done in Z3, so a contradiction comes with a proof. We don't average the four.
> If they disagree, the compiler refuses to pick and sends the case to a human. We think
> that's the responsible default. The tool shouldn't quietly pick a moral theory for you."

**Panel 4, the geometry (bottom middle).**
> "This is Hohfeld's square of legal positions: obligation, claim, liberty, no-claim. There
> are two natural operations on it, swapping agent and patient, and negation. They commute,
> so together they make the Klein four-group. That's what's measured, and it's proved in
> Lean with Mathlib. We used to claim the bigger dihedral group, D4. That was an overclaim,
> and we corrected it. A test in September looked for the extra quarter-turn inside a small
> language model and didn't find it."
>
> "The Bond Index comes out of the swap. It's the fraction of cases where your verdict
> changes when you swap who acts and who's acted on. The human baseline is 0.155, measured
> on about 20,000 advice-column letters. The runtime warns at 0.25 and blocks at 0.30."

**Panel 5, the runtime gateway (top right).**
> "erisml-lib sits between an agent's planner and its actuators, in three layers. Reflex
> does fast hard stops. Tactical does the full reasoning in 10 to 100 milliseconds.
> Strategic records policy and brings in a human. The output is allow, revise or block.
> Every decision is sealed in a SHA-256 proof that chains to the one before it. If the
> ethics service goes down, it falls back to rules. It fails safe, never open."

**Panel 6, the PyTorch lens (bottom right). The PyTorch hook.**
> "Here's the PyTorch part, and it's literal. We put forward hooks on transformer layers.
> That gives an activation lens, what the model is doing inside, and we compare it with a
> text lens, what the model says. If they diverge, it raises requires-human-review. The
> first calibration came in September. Reading mid-layer activations the way the next layer
> reads them catches 19 to 21 points more real behavior changes than raw L2 norm at the same
> flag rate. That was on Qwen2.5-0.5B, and it replicated on TinyLlama and on a second kind of
> transform. The final layer failed validation, so we probe mid-layers. It's early. It's a
> measurement that has replicated, not a finished detector."

**Panel 7, the hero band (bottom). The worked example and the robot.**
> "Left side: a home-care robot finds Margaret collapsed in her bedroom. Normally, going
> into her bedroom and sharing her records is forbidden. In an emergency the same action
> is preferred by all four parties, and the ranking flips from stand-by to full response.
> Switching into emergency mode is treated like privilege escalation. It has to be
> authenticated, here at confidence 0.97 against a 0.90 threshold. It's scoped to the
> paramedics only, it can be reversed, and both edges are audited. At 0.89 it's refused:
> the robot still calls EMS, but it doesn't share her records. A false alarm unlocks
> nothing, and aid is never blocked. It's one command, and the hashes check out."
>
> "Right side: the same evaluator inside the simulated robot, and there it chooses nothing.
> Every proposed act has to pass the sensor governor, the signed constitution, the
> evaluator, and the signed manifest. Any one of them can stop it, and a stop leaves the
> robot holding still. We test it by changing one input and nothing else, and we write the
> prediction down before each run. In nine pairs a silent or forged detector stopped the
> fire call, and whether her refusal was heard decided the ambulance. The test also caught
> a bug. A lift she hadn't asked for got through. We fixed it, and on the rerun only the
> lift's verdict changed."

**Close.**
> "Three things to take away: keep the structure and log the collapse; when ethical
> theories disagree, defer to a human; and fail safe, never open. The QR codes go to the
> paper and the code. The brochure has everything on one sheet."

---

## Short versions

**10 seconds (someone walking past):**
> "It's a safety layer that keeps the full moral structure and seals every decision in a
> replayable hash-chained proof. The PyTorch part compares what a model says with what its
> activations show."

**For a PyTorch engineer:** opener, then panel 6, then the turboquant-pro line:
> "We also ship turboquant-pro on PyPI. It's a one-line Hugging Face KV cache drop-in with
> Triton kernels and a vLLM plugin, about 5x less KV memory."

---

## Likely questions, honest answers

**"Who decides the ethics? Isn't this just your values?"**
> "We don't claim the system knows what's good. It keeps four readings side by side,
> refuses to average them, and defers to a human when they conflict. What we claim is that
> it's auditable, not that it's right."

**"Is it on a real robot?"**
> "No. It's a simulated 3D home with scripted proposals and no hardware. The agent stack
> and the two game NPCs are live."

**"How fast is it?"**
> "The tactical layer takes 10 to 100 milliseconds. Reflex stops are faster."

**"What happens if the ethics service crashes?"**
> "It falls back to rule-based checks and fails closed. It never fails open."

**"Does the activation lens actually work?"**
> "It's early. The mid-layer result replicated on a second model and a second transform.
> The final layer failed held-out validation, so we don't use it. It isn't a finished
> detector."

**"Why the group theory?"**
> "It gives a symmetry test you can actually run. Swapping agent and patient shouldn't
> change a fair verdict. The Bond Index measures how often it does. We claim only the
> four-element group we can measure. We corrected our own earlier D4 claim, and before that
> we falsified a gauge-theory model with CHSH-style tests on human moral judgments, N = 600."

**"What's the COMPAS number?"**
> "The swap logic applied to matched neighbours gives a Legal Bond Index of 1.049, with a
> confidence interval of 1.027 to 1.072, and z = 4.9. It drops to about 1.00 to 1.02 under
> matching-parity specifications. It's a research screening signal, not a regulatory audit."

**"Can I use it?"**
> "Yes. pip install erisml-compiler. The compiler is alpha, v0.9.0, MIT. erisml-lib is
> v3.1.0 on PyPI, but the worked example needs the source tree."

**"What's not done?"**
> "Tensor ranks 1 to 3 are real and the higher ranks are partial. The action axis is a stub.
> The Hohfeld module and the Bond Index live in erisml-lib, not yet in the compiler. And D4
> is posited, not measured."

---

## Tips
- Point at the panel number as you talk. People track the board, not you.
- Don't read the whole walkthrough to everyone. Opener, one question, then the panel that
  matches their answer.
- Hand a brochure to anyone who stays longer than a minute.
