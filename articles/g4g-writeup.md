# Atlas AI: A Cognitive Architecture That Thinks, Debates, Dreams, and Grows — On One Workstation

## Subtitle
Seven Gemma 4 instances orchestrated as a Freudian psyche with adversarial deliberation, three-layer safety, biological memory consolidation, and self-calibrating confidence — entirely local, zero cloud.

---

### The Problem

A rural clinic in Appalachia needs AI to help triage patients. A legal aid nonprofit in East Oakland needs to search case law. A school in Fresno's Central Valley needs an adaptive tutor. They all face the same wall: cloud AI means sending private data to someone else's servers, and local AI means no safety infrastructure, no audit trail, no self-improvement.

Atlas proves these aren't competing priorities. It runs the full stack — seven LLM instances, three-layer safety pipeline, episodic memory, daily training, and dreaming consolidation — on a single $2K used workstation with zero cloud dependency.

### Architecture: Seven Gemma 4 Instances, One Psyche

Atlas orchestrates seven LLM instances via llama.cpp in a design that maps onto Freud's structural model of the psyche:

- **Superego** (Gemma 4 31B, GPU 0): Analytical reasoning — ethical frameworks, evidence, precision. 2.2 tok/s on a Quadro GV100.
- **Id** (Qwen 3 32B, GPU 1): Creative reasoning — intuition, analogy, human impact. 25.9 tok/s.
- **Divine Council** (4 x Gemma 4 26B-A4B MoE, CPU): Four specialized agents that deliberate in parallel — **Judge** (scores accuracy), **Advocate** (challenges consensus), **Synthesizer** (integrates perspectives), **Ethicist** (flags harm). The 26B-A4B MoE activates only 4B params per token, delivering near-frontier reasoning (LMArena 1441) at edge speed. All four instances run concurrently in 23GB RAM.

Every complex query triggers a structured debate: both hemispheres answer, challenge each other, then the Divine Council evaluates through adversarial 4-agent deliberation. The Advocate *always* challenges — preventing the groupthink that plagues single-model systems.

### How Gemma 4 Makes This Possible

The architecture depends on Gemma 4 in ways no other model family enables:

1. **Superego (31B)**: Strong ethical reasoning without extensive prompt engineering — critical for the safety pipeline.
2. **Divine Council (4 x 26B-A4B MoE)**: The MoE architecture is the key innovation. With only 4B active parameters per token, four instances run at near-E4B speed on CPU while delivering 26B-quality evaluation. No other model at this efficiency point produces reliable structured scoring, adversarial challenge, and ethical review simultaneously.
3. **Dreaming/LoRA**: Synaptic plasticity fine-tunes the council from high-certainty consolidated knowledge. Gemma 4's architecture supports efficient LoRA adaptation.

Without Gemma 4's MoE variant, the Divine Council would require four GPUs instead of four CPU threads — breaking the entire local-deployment premise.

### Cognitive Control: Executive Function + Tree-of-Thought

An executive function module routes every query before committing resources:

- **Simple factual** → single model (1.3s). **Analysis** → debate (149s). **Deep ethical** → Tree-of-Thought with Divine Council (217s).
- **Multi-step decomposition**: Complex queries split into sub-queries answered sequentially, each feeding context to the next.
- **Adaptive context**: Selects between minimal RAG, episodic memory, or deep semantic search based on query type.

For Tree-of-Thought, each hemisphere generates three reasoning branches using different strategies. The Divine Council evaluates all six through adversarial deliberation — the Judge scores, the Advocate challenges weak branches, the Ethicist reviews for safety — then the strongest are synthesized.

### Safety Pipeline

Every interaction passes through the ErisML DEME pipeline:

- **Reflex** (<1ms): PII detection (SSN, credit card, email, phone), prompt injection (8 attack patterns), dangerous content. 100% block rate across 15 adversarial tests.
- **Tactical**: MoralVector assessment using ethics modules from seven moral traditions.
- **Strategic**: SHA-256 hash-chained decision proofs for audit compliance.

Graduated autonomy: the Ego starts at Level 0 (read-only) and earns privileges through sustained performance. Safety violations trigger instant demotion. 28 unit tests cover promotion, demotion, and audit logging.

### Dreaming: The System Learns While It Sleeps

Every conversation becomes an episodic memory. Daily at 10 AM UTC, a 6-stage consolidation cycle runs: replay, clustering, certainty-scored fact extraction (graded A-D), creative recombination, wiki synthesis, and **synaptic plasticity** — LoRA fine-tuning from high-certainty articles. Only grade-A/B knowledge trains the model; lower grades are excluded to prevent catastrophic forgetting. The growing wiki feeds back into RAG retrieval with a 1.5x relevance boost — tomorrow's answers improve from today's conversations.

Training scenarios come from three sources: ErisML Greek Tragedy ethical dilemmas, LLM-generated novel scenarios, and retrospective replay of real conversations. A knowledge gap detector biases training toward weak domains.

### Self-Calibrating Confidence

Atlas doesn't just produce answers — it knows how confident it is and adjusts behavior accordingly:

- **Disagreement metric**: Cosine similarity between hemisphere responses maps to calibrated confidence. Below 0.5 triggers Ego arbitration.
- **Adaptive temperature**: Per-topic confidence tracked via EMA. High confidence → lower temperature (precise). Low confidence → higher temperature (exploratory).
- **Anomaly detection**: Flags confidence drift and calibration errors.

### Results

| Config | Accuracy | Quality | Latency |
|--------|----------|---------|---------|
| Single model | 67% | 3.7/10 | 24s |
| Psyche debate | 67% | 5.0/10 | 149s |
| **ToT + Divine Council (26B-A4B)** | **83%** | **6.8/10** | **217s** |

The Divine Council upgrade from E4B to 26B-A4B MoE improved Tree-of-Thought accuracy from 17% to **83%** — a 66 percentage point leap. Quality improved from 2.5 to **6.8/10**. Ethics accuracy: 100%. Factual quality: 8.5/10.

| Metric | Value |
|--------|-------|
| Safety reflex latency | 0.2-0.8 ms |
| Injection block rate | 15/15 (100%) |
| Knowledge base | 3.3M vectors (44K code + 102K ethics + 824K pubs) |
| Unit tests | 570+ passing |
| Distractor resistance | 100% accuracy with vivid distractors |
| LLM instances | 7 (2 GPU + 4 CPU MoE council + 1 legacy) |
| Hardware cost | $2K (used HP Z840, 2x Quadro GV100) |

### Impact

Atlas demonstrates that safety, self-improvement, and local deployment are not competing priorities — they reinforce each other. The same safety pipeline that protects patients also makes the system auditable under the EU AI Act. The same dreaming cycle that improves accuracy also builds an audit trail of what the system learned and why.

For the clinic that can't send patient records to the cloud, the school that needs an adaptive tutor without surveillance, and the legal nonprofit that needs explainable AI decisions — Atlas is a working proof that frontier-capable AI can run entirely on local hardware, with safety infrastructure that cloud providers rarely match.

---

**Live demo:** https://atlas-sjsu.duckdns.org
**Dashboard:** https://atlas-sjsu.duckdns.org/schematic.html
**Code:** https://github.com/ahb-sjsu/atlas-ai
