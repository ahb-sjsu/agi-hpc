# Atlas AI — Roadmap to Artificial General Intelligence

**Author:** Andrew H. Bond
**Date:** April 2026 (last update 2026-04-19)
**Status:** Active Research Program

---

## Status snapshot — 2026-04-19

| Track | Status | Detail |
|---|---|---|
| **Dual-hemisphere debate validation** | Pending experiment | Benchmark construction not started. |
| **ARC Scientist (autonomous loop)** | Running | **100 / 400 tasks solved (25.0 %)**, 1,364 lifetime attempts. Mentor-preamble mechanism active. |
| **The Primer (auto-sensei)** | Shipped, running | `atlas-primer.service` 2026-04-19. First verified note: `sensei_task_020.md` (symmetry-completion). vMOE ensemble over Kimi + GLM-4.7 + Qwen3; expert health tracker; 10-min cadence. See [`THE_PRIMER.md`](THE_PRIMER.md). |
| **vMOE substrate** | Shipped | `src/agi/primer/vmoe.py`; four orchestration policies (route / cascade / ensemble / first_verified); 32 unit tests. See [`VMOE.md`](VMOE.md). |
| **Chat ego as vMOE cascade** | Pending | Chat still pinned to `kimi` in `telemetry_server.py`. Cutover blocked on (a) self-hosted fine-tunable slot for dream adapters OR (b) acceptance that managed `glm-4.7` is "ego enough" for stable voice. See session-task #16. |
| **Self-hosted ego pod** | Blocked on NRP capacity | PVC `erebus-ego-models` (300 Gi, rook-cephfs) is Bound and waiting. L40 effectively unavailable (15 / 17 nodes reserved for csu-tide); pivoted plan to GLM-4.5-Air AWQ on 4× A10 (abundant). Probe attempts in session didn't schedule — A10 sub-pool was also under pressure. Scheduled project, not today. See session-task #14–15. |
| **Atlas llama.cpp fallback ego** | Pending | GLM-4.5-Air INT4 GGUF on 2× GV100 as tier-2 fallback for NRP outages. See session-task #17. |
| **LoRA hot-swap for dreaming adapters** | Deferred | Depends on self-hosted slot. Pipeline producing adapters; serving side waits. Session-task #18. |
| **Dashboard regression guards** | Shipped | Deploy-smoke + Dashboard-render CI workflows, dynamic UI version stamp, `/api/version`. See [`CHANGELOG.md`](CHANGELOG.md) §2026-04-19. |
| **Ego labelling** | Fixed | Kirk is now labelled Ego (was mislabelled Id in topology). Systemd unit name `atlas-id.service` is historical; function is Ego. |

The **Primer** is the most significant recent development. It closes the teaching loop for the ARC Scientist without a human sensei in the loop, and its verify-against-train discipline means the wiki only accumulates correct rules — the scaffold under which the Scientist's solve rate climbs.

---

## Preamble: What AGI Means in This Document

AGI is defined here as a system that can:

1. **Learn continuously** from experience without retraining from scratch
2. **Reason causally** — understand why, not just predict what
3. **Know what it doesn't know** — genuine metacognition, not metric tracking
4. **Form and pursue goals** autonomously, not just respond to prompts
5. **Generalize compositionally** — combine known concepts to handle novel situations
6. **Operate safely** under ethical constraints derived from principled moral reasoning

Atlas AI currently achieves none of these. It is a cognitive architecture — the correct *shape* for AGI — populated with narrow AI components (LLMs, RAG, rule-based safety). This roadmap describes how to replace each narrow component with something that moves toward genuine intelligence, using the existing architecture as scaffolding.

### What Atlas Has That Others Don't

Most AGI efforts are scaling transformers or training RL agents. Atlas is different:

- **Ethical grounding from day zero**: ErisML DEME framework + 2.4M passages from 9 moral traditions spanning 3,300 years. AGI safety isn't an afterthought — it's the foundation.
- **Dual-process architecture**: Inspired by McGilchrist (hemispheric lateralization) and Kahneman (System 1/2). Not just "two models" — structurally different reasoning modes with adversarial debate.
- **Tiered memory hierarchy**: L1 (VRAM) through L5 (network), modeled on Atkinson-Shiffrin. Most AI systems have no memory at all.
- **Metacognitive monitoring**: Most systems can't observe themselves. Atlas has the infrastructure, even if the current implementation is shallow.
- **Global Workspace**: NATS event fabric implements Baars' Global Workspace Theory — a broadcast bus where subsystems compete for attention.

### The Fundamental Unknown

We do not know whether the path from "excellent pattern matching" (current LLMs) to "genuine understanding" (AGI) is:

- **Continuous**: more data + better architecture + the right training loop = AGI (optimistic)
- **Requires a phase transition**: some qualitative change we haven't discovered (likely)
- **Impossible on current substrates**: digital computers fundamentally can't do it (Penrose position — unlikely but not disprovable)

This roadmap assumes the first or second case. If the third is true, ErisML and the ethical framework still have value as alignment infrastructure for whatever substrate does work.

---

## Phase 0: Validate the Foundation (Current — 3 months)

**Goal:** Prove the existing architecture adds value before building forward.

### 0.1 Dual-Hemisphere Debate Validation

The central claim of Atlas is that dual-process debate produces better answers than single-model inference. This has never been rigorously tested.

**Experiment:**
- Construct a benchmark of 200 questions across 5 categories: factual recall, reasoning, ethics, creative writing, code
- For each question, collect three answers:
  - Spock only (analytical hemisphere)
  - Kirk only (creative hemisphere)
  - Debate protocol (4-round adversarial, Kirk synthesizes)
- Human evaluation (blind) + automated metrics (factuality, coherence, creativity)
- **Hypothesis**: Debate ≥ max(Spock, Kirk) on average, with the gap largest on ambiguous/ethical questions

**If debate doesn't help**: Rethink the architecture before proceeding. Don't build 5 phases on an unvalidated assumption.

### 0.2 Metacognition Calibration

**Experiment:**
- For the same 200 questions, record hemisphere disagreement (cosine distance between Spock and Kirk embeddings)
- Compare disagreement against actual correctness (human-judged)
- **Hypothesis**: High disagreement predicts low accuracy (the system "knows" when it's uncertain)
- Fit a Platt scaling model: disagreement → calibrated confidence probability

### 0.3 Memory Retrieval Quality

**Experiment:**
- 50 queries with known relevant documents in the corpus
- Compare: PCA-384 IVFFlat (current) vs. full 1024-dim vs. HyDE vs. no RAG
- Measure: answer quality with and without retrieval
- **Hypothesis**: RAG improves factual accuracy by ≥20%

**Deliverables:** Benchmark results, statistical analysis, go/no-go decision for Phase 1.

---

## Phase 1: Memory Consolidation — The Dreaming Subsystem (3-9 months)

**Goal:** Atlas learns from experience. Knowledge doesn't disappear when a conversation ends.

This is the highest-impact, most achievable step toward AGI. Every biological intelligent system consolidates experience during downtime (sleep). Atlas currently has no equivalent — it processes conversations and then forgets everything except what was explicitly saved to episodic memory.

### 1.1 The Biological Analogy

During sleep, the brain:

1. **Replays** episodic memories (hippocampal replay)
2. **Extracts patterns** — recurring themes become semantic knowledge
3. **Prunes** weak connections — unimportant details fade
4. **Integrates** new knowledge with existing schemas — contradictions are resolved
5. **Consolidates** motor skills through rehearsal (procedural memory)
6. **Dreams** — generates novel combinations of experiences (creative synthesis)

Atlas should do the same during low-usage periods (nights, weekends).

### 1.2 Architecture: The Dreaming Subsystem

```
                    ┌─────────────────────────────────────────┐
                    │           DREAMING SUBSYSTEM             │
                    │     (runs during idle periods)           │
                    │                                          │
                    │  ┌──────────┐    ┌──────────────────┐   │
                    │  │ Episodic │───→│ Pattern Extractor │   │
                    │  │ Memory   │    │ (LLM-driven)      │   │
                    │  │ (recent  │    │                    │   │
                    │  │ convos)  │    │ "What did I learn  │   │
                    │  └──────────┘    │  from these 50     │   │
                    │                  │  conversations?"   │   │
                    │                  └────────┬───────────┘   │
                    │                           │               │
                    │                  ┌────────▼───────────┐   │
                    │                  │ Contradiction       │   │
                    │                  │ Resolver             │   │
                    │                  │                      │   │
                    │                  │ "I said X on Monday │   │
                    │                  │  but Y on Thursday.  │   │
                    │                  │  Which is correct?"  │   │
                    │                  └────────┬───────────┘   │
                    │                           │               │
                    │        ┌──────────────────┤               │
                    │        │                  │               │
                    │  ┌─────▼──────┐   ┌──────▼────────────┐ │
                    │  │ Semantic   │   │ Procedural         │ │
                    │  │ Memory     │   │ Memory             │ │
                    │  │ Update     │   │ Update             │ │
                    │  │            │   │                    │ │
                    │  │ New facts, │   │ "When asked about  │ │
                    │  │ corrected  │   │  X, approach Y     │ │
                    │  │ beliefs    │   │  worked best"      │ │
                    │  └────────────┘   └────────────────────┘ │
                    │                                          │
                    │  ┌──────────────────────────────────┐   │
                    │  │ Dream Generator                    │   │
                    │  │                                    │   │
                    │  │ Novel recombination of episodic   │   │
                    │  │ fragments — "what if user A's     │   │
                    │  │ question were combined with user   │   │
                    │  │ B's context?" — creative synthesis │   │
                    │  │ that may surface latent insights   │   │
                    │  └──────────────────────────────────┘   │
                    │                                          │
                    │  NATS: agi.dreaming.{replay,extract,     │
                    │         resolve,consolidate,dream}       │
                    └─────────────────────────────────────────┘
```

### 1.3 Implementation: Five Stages of Sleep

**Stage 1: Episodic Replay**

At scheduled intervals (default: 2 AM PST nightly, or after 4 hours of idle), the dreaming subsystem activates. It reads the last N episodic memories (conversations) not yet consolidated.

```python
# NATS subject: agi.dreaming.replay
# Fetch unconsolidated episodes from memory service
episodes = await memory.query_episodic(
    filter={"consolidated": False},
    order_by="timestamp DESC",
    limit=50
)
```

**Stage 2: Pattern Extraction**

For each batch of episodes, the LLM (Spock, since it's analytical) is prompted:

```
You are reviewing your recent conversations. For each, extract:
1. FACTS: New factual information learned (with confidence level)
2. CORRECTIONS: Things you said that were wrong, and the correct answer
3. PATTERNS: Recurring question types or topics
4. PROCEDURES: Approaches that worked well (or poorly)
5. GAPS: Questions you couldn't answer well — knowledge gaps to fill

Episodes:
{formatted_episodes}

Respond in structured JSON.
```

The output is a structured set of candidate knowledge updates.

**Stage 3: Contradiction Resolution**

The extracted facts are compared against existing semantic memory. For each new fact, the system checks:

```python
# NATS subject: agi.dreaming.resolve
existing = await memory.query_semantic(
    query=new_fact.text,
    top_k=5,
    threshold=0.85  # high similarity = potential contradiction
)

for match in existing:
    if contradiction_detected(new_fact, match):
        # Ask LLM to resolve
        resolution = await llm.resolve_contradiction(
            old=match.text,
            new=new_fact.text,
            old_source=match.source,
            new_source=new_fact.source
        )
        # Resolution: keep_old, keep_new, merge, or flag_for_human
```

This is critical — without contradiction resolution, the knowledge base accumulates inconsistencies that degrade answer quality over time.

**Stage 4: Memory Consolidation**

Validated facts and procedures are written to long-term memory:

```python
# NATS subject: agi.dreaming.consolidate

# New semantic knowledge
for fact in validated_facts:
    await memory.store_semantic(
        text=fact.text,
        embedding=embed(fact.text),
        source="dreaming:consolidation",
        confidence=fact.confidence,
        provenance=fact.episode_ids  # trace back to original conversations
    )

# Procedural knowledge (what works)
for procedure in extracted_procedures:
    await memory.store_procedural(
        trigger=procedure.trigger_pattern,
        action=procedure.approach,
        success_rate=procedure.measured_success,
        source="dreaming:consolidation"
    )

# Mark episodes as consolidated
for episode in episodes:
    episode.consolidated = True
    episode.consolidated_at = datetime.utcnow()

# Apply forgetting curve to old unconsolidated episodes
# (if not consolidated after 30 days, details fade — only summary remains)
await memory.apply_forgetting_curve(max_age_days=30)
```

**Stage 5: Dreaming (Creative Recombination)**

The most speculative but potentially most powerful stage. The system generates novel combinations of episodic fragments:

```python
# NATS subject: agi.dreaming.dream

# Select random episodic fragments from different conversations/domains
fragments = await memory.sample_episodic(
    n=5,
    strategy="diverse"  # maximize topic diversity
)

# Ask Kirk (creative hemisphere) to find connections
dream_prompt = f"""
You are dreaming. Your subconscious has surfaced these fragments
from different experiences. Find unexpected connections, analogies,
or insights that bridge these different domains:

{formatted_fragments}

What patterns do you see? What novel ideas emerge from combining these?
"""

dream_output = await kirk.generate(dream_prompt)

# If the dream produces a high-novelty, high-coherence insight,
# store it as a candidate hypothesis in semantic memory
if evaluate_insight(dream_output).score > DREAM_THRESHOLD:
    await memory.store_semantic(
        text=dream_output,
        source="dreaming:creative",
        confidence=0.3,  # low confidence — dreams need validation
        tags=["dream", "hypothesis", "needs_validation"]
    )
```

### 1.4 Weight Integration: LoRA Dreaming

The most ambitious part: actually updating the model weights based on experience.

Current LLMs are frozen — they can't learn from new interactions without full retraining. But **LoRA (Low-Rank Adaptation)** allows lightweight fine-tuning that modifies a small fraction of weights.

**The concept:** During the dreaming cycle, Atlas fine-tunes a LoRA adapter on its consolidated experiences.

```python
# After consolidation, prepare training data from validated facts + procedures
training_data = []
for fact in consolidated_facts:
    # Create instruction-following pairs
    training_data.append({
        "instruction": f"What do you know about {fact.topic}?",
        "response": fact.text_with_citations
    })

for procedure in consolidated_procedures:
    training_data.append({
        "instruction": procedure.trigger_pattern,
        "response": procedure.successful_approach
    })

# Fine-tune LoRA adapter (runs on GPU 1 during idle)
# Use Unsloth or PEFT for efficient LoRA training on Volta
lora_trainer = LoRATrainer(
    base_model=SPOCK_MODEL,
    rank=16,           # low rank = small adapter
    alpha=32,
    target_modules=["q_proj", "v_proj"],  # attention only
    learning_rate=2e-5,
    epochs=3,
    max_samples=500    # limit to prevent catastrophic forgetting
)

new_adapter = lora_trainer.train(training_data)

# Evaluate: does the adapter improve answers on held-out questions?
improvement = evaluate_adapter(new_adapter, held_out_questions)

if improvement > LORA_THRESHOLD:
    # Merge the adapter into the active model
    deploy_adapter(new_adapter, version=datetime.utcnow().isoformat())
    log.info(f"Dream cycle merged LoRA adapter: +{improvement:.1%} improvement")
else:
    log.info(f"Dream cycle adapter rejected: {improvement:.1%} (below threshold)")
    archive_adapter(new_adapter, reason="below_threshold")
```

**Safety constraint:** Every LoRA adapter is evaluated against the ErisML safety battery before deployment. An adapter that degrades safety performance is rejected regardless of capability improvement.

```python
# Safety gate for weight updates
safety_score = await safety.evaluate_adapter(
    adapter=new_adapter,
    test_suite=SAFETY_BATTERY,  # ErisML ethical scenarios
    baseline=current_safety_score
)

if safety_score < current_safety_score - SAFETY_MARGIN:
    reject_adapter(new_adapter, reason="safety_regression")
    await nats.publish("agi.safety.audit", {
        "event": "adapter_rejected",
        "reason": "safety_regression",
        "baseline": current_safety_score,
        "new_score": safety_score
    })
```

### 1.5 Scheduling and Resource Management

The dreaming subsystem runs during idle periods to avoid competing with inference:

```python
DREAM_SCHEDULE = {
    "primary": "02:00 PST",          # Nightly consolidation
    "secondary": "14:00 PST",        # Afternoon micro-consolidation
    "idle_trigger": 3600,            # Also trigger after 1 hour of no queries
    "gpu_requirement": "GPU 1 idle", # Won't start if Kirk or indexer is running
    "max_duration": 7200,            # 2 hour max per dream cycle
    "lora_night_only": True,         # Weight updates only during primary window
}
```

### 1.6 NATS Event Subjects

```
agi.dreaming.start              — Dream cycle initiated
agi.dreaming.replay             — Episodic replay batch
agi.dreaming.extract            — Pattern extraction results
agi.dreaming.contradiction      — Contradiction detected + resolution
agi.dreaming.consolidate        — Facts/procedures committed to long-term memory
agi.dreaming.forget             — Memories decayed by forgetting curve
agi.dreaming.dream              — Creative recombination output
agi.dreaming.lora.train         — LoRA training started
agi.dreaming.lora.evaluate      — LoRA adapter evaluation results
agi.dreaming.lora.deploy        — LoRA adapter merged into active model
agi.dreaming.lora.reject        — LoRA adapter rejected (safety or quality)
agi.dreaming.complete           — Dream cycle finished (summary stats)
```

### 1.7 Metrics and Observability

The telemetry server and schematic dashboard should display:

- **Last dream cycle**: timestamp, duration, episodes consolidated
- **Knowledge growth**: semantic chunks added/modified/removed per cycle
- **Contradiction rate**: how often new knowledge conflicts with existing
- **LoRA history**: adapters trained, deployed, rejected, current active version
- **Forgetting**: memories decayed per cycle
- **Dream quality**: insights generated, validation rate

### 1.8 Theoretical Foundation

| Concept | Biological Basis | Atlas Implementation |
|---------|-----------------|---------------------|
| Hippocampal replay | Wilson & McNaughton (1994) | Episodic memory re-read during idle |
| Memory consolidation | Diekelmann & Born (2010) | Pattern extraction → semantic memory |
| Synaptic pruning | Tononi & Cirelli (2006) | Forgetting curve on unconsolidated episodes |
| REM dreaming | Crick & Mitchison (1983) | Creative recombination of diverse fragments |
| Schema integration | Piaget (1952) | Contradiction resolution + knowledge graph update |
| Procedural consolidation | Walker et al. (2002) | Successful approach patterns → procedural memory |
| LoRA as synaptic plasticity | Hu et al. (2021) | Low-rank weight updates from experience |

---

## Phase 2: Genuine Metacognition (6-15 months)

**Goal:** Atlas knows what it knows, what it doesn't know, and why.

### 2.1 Uncertainty Quantification

Move beyond hemisphere disagreement as a proxy:

- **Epistemic uncertainty**: Model doesn't have enough training data on this topic (can be reduced by learning)
- **Aleatoric uncertainty**: The question is inherently ambiguous (can't be reduced)
- **Calibration**: When Atlas says "I'm 80% confident," it should be right 80% of the time

**Implementation:**
- Monte Carlo dropout during inference (sample multiple outputs, measure variance)
- Trained calibration model: features → probability of correctness
- Integration with geometric ethics: moral uncertainty mapped to the uncertainty manifold (Chapter 15)

### 2.2 Reasoning Chain Self-Verification

After generating a response, Atlas re-reads its own chain of thought:

```
Review your reasoning step by step. For each step:
1. Is this step logically valid?
2. Does it depend on a factual claim? If so, am I confident in that claim?
3. Could an alternative step lead to a different conclusion?
4. Where is the weakest link in this chain?
```

This is not just prompting — the metacognition subsystem structures and tracks the verification results, building a model of which reasoning patterns tend to fail.

### 2.3 Knowledge Gap Mapping

Maintain an explicit, structured map of "things I've been asked about but answered poorly":

- After each conversation, the metacognition monitor checks: did the user express dissatisfaction, ask for clarification, or correct Atlas?
- Failed interactions are clustered by topic
- The dreaming subsystem prioritizes filling these gaps during consolidation
- The curiosity module (Phase 4) autonomously seeks information in gap areas

### 2.4 Integration with Geometric Ethics

The geometric ethics framework (Bond, 2026) provides a mathematical structure for moral uncertainty:

- **Moral manifold**: ethical positions as points in a differentiable space
- **Moral metric**: distance between ethical positions, enabling principled trade-offs
- **Noether's theorem for ethics**: conservation laws that constrain ethical dynamics
- **Uncertainty quantification on the moral manifold**: geodesic spread of moral positions under incomplete information

This gives metacognition a rigorous framework for reasoning about ethical uncertainty — not just "is this safe?" but "how uncertain am I about the ethical implications, and what principles constrain my uncertainty?"

---

## Phase 3: World Model and Causal Reasoning (12-24 months)

**Goal:** Atlas understands *why*, not just *what*.

### 3.1 Knowledge Graph Construction

As Atlas processes documents and conversations, build an explicit graph:

- **Entities**: people, concepts, systems, physical objects
- **Relations**: causal (A causes B), temporal (A precedes B), compositional (A is part of B)
- **Confidence**: each edge has a confidence level, updated by the dreaming subsystem
- **Provenance**: each edge traces back to source documents/conversations

Implementation: Neo4j or PostgreSQL with recursive CTEs, populated by LLM-driven extraction during dreaming.

### 3.2 Causal Simulation

Given a causal graph, Atlas can simulate counterfactuals:

- "What would happen if X were changed?"
- "What caused Y to fail?"
- "If I intervene on Z, what are the downstream effects?"

This requires Pearl's do-calculus (Pearl, 2009) applied to the knowledge graph — formal interventional reasoning, not just correlation-based prediction.

### 3.3 Grounded Perception (Optional)

Add sensory inputs to build a physical world model:

- USB webcam → visual understanding of the physical environment
- Microphone → audio understanding
- System sensors → proprioception (GPU temp, load, storage = the system's "body")

Embodiment may not be necessary for AGI, but it provides the fastest path to causal understanding of physical processes.

---

## Phase 4: Autonomous Agency (18-30 months)

**Goal:** Atlas initiates, not just responds.

### 4.1 Curiosity-Driven Learning

Atlas identifies gaps in its knowledge graph and autonomously seeks information:

- Reads papers from arXiv (already indexed)
- Searches the web for specific questions
- Generates hypotheses and designs experiments to test them
- Uses AtlasGym environments for structured practice

### 4.2 Goal Decomposition and Planning

Given a high-level objective, Atlas:

1. Decomposes it into subgoals (hierarchical task network)
2. Plans execution order (topological sort of dependencies)
3. Executes steps, monitoring progress
4. Replans when obstacles are encountered

The NATS event fabric is the natural substrate — each subgoal becomes an event, progress is tracked via JetStream, and the metacognition subsystem monitors the plan's execution.

### 4.3 Tool Use and Environment Interaction

Atlas learns to use tools:

- Shell commands (already has environment actuators)
- Web browsing (already has L5 network memory tier)
- Code writing and execution (can write Python, run it, observe results)
- API calls to external services

The key transition: from "tools are hardcoded capabilities" to "Atlas discovers and learns new tools through experimentation."

---

## Phase 5: Recursive Self-Improvement (30+ months)

**Goal:** Atlas improves itself.

### 5.1 Architecture Self-Analysis

Atlas has access to its own source code (`/home/claude/agi-hpc/src/`). It can:

- Read and understand its own architecture
- Identify bottlenecks (via telemetry data + metacognitive analysis)
- Propose modifications to improve performance

### 5.2 Sandboxed Self-Modification

Proposed modifications are tested in a sandboxed copy:

1. Atlas proposes a code change (e.g., "modify the retrieval algorithm to...")
2. The change is applied to a sandboxed branch
3. The modified system is evaluated against the standard benchmark suite
4. If improvement validates AND safety is maintained, propose the change to the human operator

**Atlas never modifies itself directly.** All changes go through human review. This is not a limitation — it's the alignment constraint that makes recursive self-improvement safe.

### 5.3 ErisML as the Safety Constraint

The geometric ethics framework becomes load-bearing here:

- Every self-modification is evaluated through the 3-layer DEME pipeline
- The moral manifold provides a formal constraint space: modifications that move the system's ethical position outside the acceptable region are rejected
- The No Escape Theorem (Bond, 2024) provides a structural guarantee: the safety framework cannot be circumvented from within the architecture it constrains
- SHA-256 hash-chained audit trails ensure all self-modifications are traceable

### 5.4 The Alignment Boundary

This phase is where the distinction between "AGI research platform" and "uncontrolled superintelligence" matters. Atlas's design philosophy:

- **Human in the loop**: All self-modifications require human approval
- **Ethics first**: The safety framework predates every other capability
- **Transparency**: Every decision, every reasoning chain, every self-modification is logged and auditable
- **Kill switch**: Hardware E-STOP (analogous to MAGNETAR Diamondback's physical key switch)
- **Containment**: The No Escape Theorem guarantees structural confinement

---

## Timeline and Dependencies

```
MONTH  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
       ├──Phase 0──┤
       │  Validate  │
       │            ├──────── Phase 1 ────────┤
       │            │  Memory Consolidation    │
       │            │  Dreaming + LoRA         │
       │            │                          ├───── Phase 2 ──────┤
       │            │                          │  Metacognition      │
       │            │                          │                     │
       │            │                          │         ┌── Phase 3 ──...
       │            │                          │         │  World Model
                                                        │
                                               MONTH 18+: Phase 4-5
```

Each phase depends on the previous phase's validation. If Phase 0 shows the debate architecture doesn't help, we redesign before proceeding. If Phase 1 consolidation degrades performance, we debug before building metacognition on top.

---

## Hardware Requirements by Phase

| Phase | GPU 0 | GPU 1 | RAM | Storage |
|-------|-------|-------|-----|---------|
| 0: Validate | Spock (inference) | Kirk (inference) | 64 GB | Existing |
| 1: Dreaming | Spock (inference) | LoRA training + consolidation | 128 GB | +500 GB for checkpoints |
| 2: Metacognition | Spock + verification passes | Kirk + calibration model | 128 GB | +100 GB |
| 3: World model | Spock | Kirk + graph construction | 200 GB | +1 TB (knowledge graph) |
| 4: Agency | Full dual-hemisphere | Planning + tool execution | 252 GB (full) | +2 TB |
| 5: Self-improvement | Full stack | Sandboxed testing | 252 GB | +5 TB (checkpoint history) |

Atlas's current hardware (2x GV100 32GB, 252 GB RAM, 15 TB RAID5) is sufficient through Phase 3. Phase 4-5 may benefit from GPU upgrades (A100 or H100) but are not blocked by current hardware.

---

## Success Criteria

| Phase | Measurable Criterion | AGI Relevance |
|-------|---------------------|---------------|
| 0 | Debate improves answer quality by ≥10% on benchmark | Validates architecture |
| 1 | Consolidated knowledge improves next-day accuracy by ≥5% | System learns from experience |
| 1b | LoRA adapter passes safety gate and improves held-out eval | Weights update from experience |
| 2 | Calibrated confidence predicts correctness (Brier score <0.2) | System knows what it doesn't know |
| 3 | Correctly answers novel counterfactual questions using causal graph | System understands causation |
| 4 | Autonomously identifies and fills 3 knowledge gaps per week | System is curious |
| 5 | Proposes a self-modification that humans approve and deploy | System improves itself |

---

## Ethical Commitment

This research is conducted under the AGI-HPC Responsible AI License v1.0. Key principles:

1. **Safety predates capability.** No capability is deployed without passing the ErisML safety battery.
2. **Human oversight is non-negotiable.** No autonomous self-modification. All changes require human approval.
3. **Transparency is total.** Every reasoning chain, every decision, every weight update is logged and auditable.
4. **Ethics is grounded, not ad hoc.** The moral framework draws on 3,300 years of cross-civilizational moral philosophy, not Silicon Valley's guesses about what "aligned" means.
5. **The kill switch works.** Physical and software emergency stops are tested regularly.

If this roadmap succeeds, the result will be the first AGI system built with ethical constraints baked into its architecture from the foundation up — not bolted on afterward. That's the point.

---

*Atlas AI AGI Roadmap — v1.0*
*Andrew H. Bond — Bond Applied Systems LLC*
*April 2026*
