# Teaching an Autonomous Agent: Why Verification Matters More Than the Teacher

*What I learned building The Primer, an always-on mentor for an autonomous ARC solver*

---

## The problem I didn't expect

We run an autonomous puzzle solver on our SJSU research cluster called Erebus. It chews through Kaggle's NeuroGolf task set on its own, generating Python programs, running them against training examples, scoring itself, updating a memory file, and retrying with different strategies. The design goal was self-direction.

A week into running it, Erebus had accumulated more than 50 failed attempts on several tasks. It wasn't stuck on hard ones. It was stuck on the same *wrong hypothesis* on the same tasks, over and over. It had no way to ask anyone for help.

I gave it a help channel. Within a day, it was posting messages like:

> task381: I have tried 57 times (best: 2/3). Error types: reasoning, execution, perception. I need guidance: is this transformation local or global? Am I missing a spatial primitive?

Nobody was reading the file.

That's how The Primer got built. It's a small always-on daemon whose only job is to read Erebus's help queue and write useful replies into a shared wiki.

## The naive version actively made things worse

The obvious implementation is a ten-line loop: poll the help queue, pick a stuck task, send it to the smartest LLM you have, publish the answer. I ran that for about three hours.

The LLM returned a confident rule for task 381. The rule was wrong. It sounded plausible enough that our pipeline committed it to the wiki. Erebus picked it up on its next tick, started applying the rule, and because the rule *looked* consistent with the training examples, Erebus's internal sanity check passed each new attempt as a legitimate failure.

By the time I noticed, Erebus had 102 failed attempts on that one task, most of them faithful variations of a wrong rule the wiki told it was correct.

The lesson: a wrong teacher is worse than no teacher. A confidently-stated wrong hypothesis doesn't just fail to help — it displaces the investigation the agent would have done on its own.

## What The Primer actually does

When The Primer picks up a stuck task, it does not publish the LLM's answer. It consults three frontier models (Kimi, GLM-4.7, Qwen3) via our NRP cluster and asks each for a candidate `transform(grid) -> grid` function. Each candidate gets handed to a validator.

```
tick():
  stuck_tasks = read help queue, apply cooldown filter
  for task in stuck_tasks[:3]:
      for expert in vmoe.experts:
          candidate = expert.propose(task)
          if validator.verify(candidate, task):
              publish_sensei_note(task, candidate)
              break
      else:
          set_cooldown(task, 6h)
```

The validator runs each candidate in an isolated subprocess with a ten-second timeout. It iterates over every training example and the test example for the task, executes the candidate, compares the output byte-for-byte with the expected output. Only if all comparisons match does the candidate make it into the wiki, with the verified reference implementation included as part of the note.

About sixty lines of code. It is the piece that makes everything else trustworthy.

In other words: the LLM proposes, a deterministic oracle disposes. The bottleneck is the oracle, not the LLM.

## Why the ensemble matters less than you'd think

We wrap the three LLMs in what we call a vMOE, a virtual mixture of experts. It supports four policies: route by task features, cascade cheap-to-expensive, ensemble with quorum voting, or first-verified. Production runs first-verified because it gives you the fastest useful answer without burning the ensemble budget on tasks the cheapest expert can already solve.

Here's the counterintuitive part: with verification in the loop, the choice of LLM matters less than we expected. Any of the three will eventually produce a candidate that passes. A slower expert that produces valid candidates is worth more than a fast expert that produces plausible-looking wrong ones. Verification turns "how smart is the teacher" into "how fast is the teacher at getting to a verified answer," which is a much friendlier optimization target.

## A concrete story: task 381

Earlier this week I ran the existing wiki note for task 381 through the validator. It failed on all three training examples. The note had been written months ago by hand, before The Primer existed. It said something like "identify pairs of rectangles where widths match AND aligned vertically, OR heights match AND aligned horizontally, then fill the gap between them with the marker color." That was not the rule.

The actual rule: for any two rectangles of 2s whose row ranges overlap and which are horizontally separated, fill the gap with color 9 (not the marker color) — *unless* a third rectangle intersects both the overlap rows and the gap columns, in which case the entire pair is cancelled.

That cancellation clause is what makes the task interesting. The presence of a third unrelated object erases the relationship between the first two. It's a geometric primitive worth teaching deliberately.

I wrote a reference implementation, verified it against the full training + test set, and replaced the sensei note. Erebus's next attempt solved it. Then I realized the failure mode that had allowed this: our verify-before-publish rule applied to The Primer's writes but not to human-authored notes in the same directory. So I'm adding a pre-commit hook that refuses any wiki note without a passing reference implementation. Same invariant, enforced at the commit boundary.

## Things I'd do earlier next time

Building The Primer took about two days. If I were starting over:

**Build the validator before the proposer.** The verification oracle should exist before any component that could produce unverified output. I built the validator as an afterthought the first afternoon, and paid for it with three hours of wrong sensei notes.

**Structured logs from day one.** Events like `primer.tick_start`, `primer.candidate_generated`, `primer.validation_passed`, `primer.note_published` give you a timeline per task. The wrong-note bug would have been visible in an hour instead of a day.

**Atomic state writes everywhere.** Every stateful service should use a tempfile + fsync + atomic-rename helper from the first commit. Retrofitting is easy; the hard part is noticing the silent corruption. We found out our cooldown file had been silently corrupting on crashes for about a week.

## Where this fits in the larger picture

The Primer is a single node in a broader AGI safety research program at SJSU. Erebus is one agent. The Primer is one mentor. Our dreaming service consolidates episodic memory into wiki articles. The DEME safety gateway evaluates every proposed action through the ErisML ethical reasoning framework. All of them coordinate via a NATS event fabric and persist through PostgreSQL with pgvector.

The unifying idea across all of them is the same as the one in this piece: the useful invariants are not "what the LLM believes" but "what survives verification." Agents that can be fooled by their own plausible hypotheses need oracles, not smarter priors. Giving an autonomous agent a teacher is easy. Giving it a *verifiable* teacher is the whole point.

---

**Open source.** The Primer and the rest of the stack live at [github.com/ahb-sjsu/agi-hpc](https://github.com/ahb-sjsu/agi-hpc) under a responsible-AI license. The relevant files are `src/agi/primer/service.py` (the daemon, ~600 lines), `src/agi/primer/validator.py` (the oracle, ~60 lines), and `docs/THE_PRIMER.md` (operations reference).

**Comments welcome.** Especially from folks building autonomous research agents — the verify-before-publish invariant generalizes well beyond ARC puzzles. I'm curious what equivalents people have landed on in other domains.
