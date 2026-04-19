# A Young Agent's Illustrated Primer

*On building a verifiable teacher for an autonomous research agent — with apologies and gratitude to Neal Stephenson*

---

In Neal Stephenson's 1995 novel *The Diamond Age*, a street kid named Nell gets her hands on an artifact: **A Young Lady's Illustrated Primer**. It is a book, but a strange one. It tells her stories, fairy tales where the princess happens to be named Nell. It teaches her to read, to think, to fight, to rule. It adapts minute-by-minute to what she needs next. And critically, it never tells her the wrong thing.

We named our mentor daemon after that book. It runs on a workstation in our lab at San Jose State and teaches an autonomous ARC puzzle solver we call Erebus. I want to talk about why the homage to Nell's Primer was not just a cute nod. It was a design constraint.

## Erebus, alone

Erebus is an autonomous program-synthesis agent. It works through Kaggle's NeuroGolf task set without supervision, generating candidate Python programs, running them against training pairs, scoring itself, updating a memory file, retrying with different strategies. No human in the loop. It was designed for self-direction.

Self-direction turns out not to be the same thing as self-improvement. A week into running it, Erebus had over 50 failed attempts on several tasks. Same tasks. Same wrong hypothesis each time. It was, in effect, a very energetic child who had been left in a room with puzzles and no one to tell it when it was on the wrong track.

I gave it a help channel. Within a day it was surfacing messages like:

> task381: I have tried 57 times (best: 2/3). Error types: reasoning, execution, perception. I need guidance: is this transformation local or global? Am I missing a spatial primitive?

Nobody was reading the file.

## The temptation to hire a dumb teacher

The obvious fix: poll that help queue, hand each stuck task to the smartest LLM we have, publish the answer into a shared wiki Erebus reads. I had this running in under an hour.

In about three hours it nearly broke the project.

The LLM returned a confident rule for task 381. The rule was wrong in two distinct ways, but it *sounded* plausible. It got committed to the wiki. Erebus picked it up, applied it, and because the rule was superficially consistent with the training examples, Erebus's internal sanity checks passed each new attempt as a real failure rather than flagging "wait, my teacher might be wrong."

By the time I caught it, Erebus had 102 failed attempts on that one task, most of them careful variations of a rule the wiki had told it was correct.

A wrong teacher is worse than no teacher. A confidently-stated wrong hypothesis does more than fail to help. It actively displaces the investigation the student would have done on their own. Nell's Primer, in Stephenson's novel, is careful about exactly this. It rarely just hands Nell the answer. When it does teach her something, it is because the Primer has already verified, through her own interaction with a story, that she is in a state to learn it.

## What we actually built

Our Primer does not publish what the LLM says. It consults three frontier models (Kimi, GLM-4.7, Qwen3, all hosted on the NRP research cluster) and asks each for a candidate `transform(grid) -> grid` function: a program that claims to be the rule for the stuck task.

Each candidate goes to a validator.

The validator is about sixty lines of Python. It runs the candidate in an isolated subprocess with a ten-second timeout, iterates over every training example and the test example, executes the candidate, and compares the output byte-for-byte with the expected output. Only if every comparison matches does the candidate make it into the wiki. The verified reference implementation gets embedded in the note alongside the prose explanation.

In other words: the LLM proposes, a deterministic oracle disposes. The bottleneck is the oracle, not the LLM.

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

## The surprising consequence

Once the verifier is in the loop, *which* LLM you use stops being the interesting question. Any of the three will eventually propose something that passes. A slow expert that produces valid candidates is worth more than a fast expert that produces plausible-looking wrong ones. Verification turns "how smart is the teacher" into "how fast does this teacher reach a verified answer," which is a much kinder optimization target.

Nell's Primer, in Stephenson's novel, has a human performer (a "ractor," short for remote actor) behind the scenes, whispering the character voices. The Primer itself is a shell around them. Our vMOE ensemble is the same structural move: the wrapper doesn't need to be brilliant, it needs to be correct about when to speak.

## Task 381, the ghost story

Here is how I found the 102-failure bug.

I pulled the existing wiki note for task 381 down and ran it through the validator. It failed on all three training examples. The note had been written months ago, by hand, before the Primer existed. It had never been verified. It said (paraphrasing): "identify pairs of rectangles where widths match AND aligned vertically, OR heights match AND aligned horizontally, then fill the gap between them with the marker color." That is not the rule for this task.

The real rule: for any two rectangles of 2s whose row ranges overlap and which are horizontally separated, fill the gap with color 9 (not the marker color), *unless* a third rectangle intersects both the overlap rows and the gap columns — in which case the entire pair is cancelled.

That cancellation clause is what makes task 381 philosophically interesting. An unrelated third object erases the relationship between the first two. It is a geometric primitive worth teaching deliberately — and exactly the kind of thing Stephenson's Primer would have smuggled into a fable about Princess Nell finding that a drawbridge she and her companion are crossing becomes impassable only when a dragon perches on the opposite tower.

I wrote a verified reference implementation. Replaced the sensei note. Erebus's next attempt on task 381 solved it.

Then I realized the failure mode: our verify-before-publish rule applied to the Primer's writes, but not to old human-authored notes in the same directory. The verifier was the moat. The moat had a door. So we are adding a pre-commit hook that refuses to check in any wiki note without an attached reference implementation that passes the training fixtures. Same invariant. Different boundary.

## What I'd do earlier next time

Build the verifier before the proposer. The oracle should exist before any component that could emit unverified output.

Log every decision, from day one. Events like `primer.tick_start`, `primer.candidate_generated`, `primer.validation_passed`, `primer.note_published` turn a "something is off" feeling into a fifteen-minute investigation instead of a two-day one.

Write every state file atomically. Every one. We had silent corruption of the Primer's cooldown file for roughly a week because `path.write_text(...)` is two syscalls and a crash between them leaves the file empty. Atomic rename via tempfile + fsync is three lines of code and prevents a whole class of bug that you otherwise only discover from the confused behavior downstream.

## The bigger picture

The Primer is one node of a larger cognitive-safety research program at SJSU. Erebus is one agent. The DEME safety gateway runs every proposed action through an ethical-reasoning pipeline. The dreaming service consolidates episodic memory into wiki articles on a schedule. They all coordinate via a NATS event fabric and persist through Postgres with pgvector.

The unifying move across all of them is the one I've just described: the useful invariants are not what the LLM *believes*, but what *survives verification*. Agents that can be fooled by their own plausible hypotheses need oracles, not smarter priors. And mentors, whether for a street kid in the Leased Territories or an autonomous program-synthesis agent in a university lab, need to be cautious about what they teach, because a confidently-stated falsehood does more harm than silence.

Nell's Primer got that right in fiction. We are trying to get it right in code.

---

**Open source.** The Primer lives at [github.com/ahb-sjsu/agi-hpc](https://github.com/ahb-sjsu/agi-hpc) under a responsible-AI license. The core files: `src/agi/primer/service.py` (the daemon, around 600 lines), `src/agi/primer/validator.py` (the oracle, around 60 lines), and `docs/THE_PRIMER.md` (operations reference).

**If you haven't read Stephenson.** *The Diamond Age* is a 1995 novel about post-scarcity nanotechnology, caste, and the mechanics of teaching. If you have any stake in AI, it will ruin your ability to think about pedagogy the same way again. I cannot recommend it highly enough.
