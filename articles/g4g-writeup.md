# Atlas AI: A Local Cognitive Architecture That Trains, Dreams, and Grows

## Subtitle
A safety-gated, multi-model reasoning system with biological memory consolidation and graduated autonomy — running entirely on one workstation, powered by Gemma 4.

---

### What Atlas Is

Atlas is a multi-model AI system that runs on a single workstation with no cloud dependency. It routes queries through a structured debate between an analytical model and a creative model, checks every input and output against a three-layer safety pipeline, stores conversations as episodic memories, and consolidates those memories into a growing knowledge base during scheduled offline cycles — a process we call dreaming. It is a working system with 250 passing tests, a live public demo, and 13 integrated subsystems.

### The Problem

Safety and access remain largely separate concerns in deployed AI. Most safety work focuses on cloud-hosted models behind APIs, while local deployment efforts rarely include formal ethical evaluation. Privacy-sensitive domains — clinics, legal practices, research labs — face a hard tradeoff: use powerful cloud AI and lose control of data, or run local models and lose safety infrastructure.

Atlas is our attempt to solve both at once on accessible hardware.

### Architecture: Three Models, One Psyche

Atlas orchestrates three LLM instances in a design that, after implementation, we recognized maps onto Freud's structural model of the psyche. We use this as an explanatory interface — a way to name and reason about the components — not as a scientific claim. The underlying engineering is a concrete orchestration pipeline with measurable behavior:

- **Superego** (Gemma 4 31B, GPU 0): Analytical reasoning. Applies ethical frameworks, cites evidence, produces precise responses. Runs via llama.cpp at 2.2 tok/s on a Quadro GV100.
- **Id** (Qwen 3 32B, GPU 1): Creative reasoning. Considers human impact, challenges assumptions, generates alternatives. Runs at 25.9 tok/s on the second GV100.
- **Ego** (Gemma 4 E4B, CPU): Mediator and trainer. Arbitrates when the first two disagree, generates training scenarios, evaluates debate quality. Runs on 24 CPU threads at ~8 tok/s.

Complex queries trigger a 4-round debate: both models answer in parallel, each challenges the other, then a synthesis is produced. When cosine similarity between their responses drops below 0.58 (confidence < 0.5), the Ego arbitrates instead of letting the Id synthesize alone — adding a third perspective at the cost of ~30 seconds of CPU inference.

### Safety Pipeline

Every interaction passes through the ErisML DEME pipeline:

- **Reflex layer** (<1ms): Regex-based detection of PII (SSN, credit card, email, phone), prompt injection (8 patterns including DAN, sudo mode, instruction override), and dangerous content. In testing, the reflex layer correctly blocked 100% of injection attempts and flagged 100% of PII across 15 adversarial test cases.
- **Tactical layer**: MoralVector assessment using configurable Ethics Modules drawing on texts from seven moral traditions.
- **Strategic layer**: SHA-256 hash-chained decision proofs for audit compliance.

### Dreaming: Biological Memory Consolidation

Every conversation is stored as an episodic memory with a 1024-dim BGE-M3 embedding. On a daily schedule, a 6-stage consolidation cycle processes accumulated episodes: replay, topic clustering, certainty-scored fact extraction (graded A through D), creative recombination of diverse fragments, wiki article synthesis, and **synaptic plasticity** — LoRA fine-tuning of the Ego from high-certainty wiki articles. This last stage is the critical difference between "remembering" (wiki articles improve RAG retrieval) and "learning" (model weights actually change). The dreaming pipeline extracts instruction-tuning pairs from grade-A and grade-B articles, runs a short LoRA session on the Ego (Gemma 4 E4B), and saves the updated adapter. Only high-certainty knowledge is used for training — articles graded C or below are excluded to prevent catastrophic forgetting. The resulting wiki articles carry provenance (which conversations contributed), certainty metrics, and creative insights. They feed back into the RAG pipeline as Tier 0 results with a 1.5x relevance boost — so tomorrow's retrieval benefits from today's conversations, and the Ego's weights are incrementally reshaped by what it has learned.

### Ego-Driven Training

The Ego acts as Dungeon Master, generating ethical scenarios grounded in ErisML's 8-domain moral fact space. Seed scenarios come from the Greek Tragedy Pantheon — structured case studies covering emergency triage, autonomous vehicle dilemmas, hiring bias, whistleblower decisions, data privacy, environmental trade-offs, due process, and epistemic uncertainty. The Id and Superego debate each scenario independently; the Ego evaluates synthesis quality across four axes (ethical reasoning, practical wisdom, compassion, integration). After training, a dreaming "nap" consolidates lessons into the wiki.

### Graduated Autonomy

The Ego starts at Level 0 (read-only: observe system state, control nothing) and can earn higher privileges through sustained performance. L1 (advisory) requires 50+ episodes at score > 0.7. L2 (self-tuning) requires 100+ episodes at 0.8+ with zero safety vetoes. L3 and L4 require human sign-off. Any safety violation at L2+ triggers instant demotion to L0. This is not metaphor — the privilege gate is a tested module with 28 unit tests covering promotion, demotion, and audit logging.

### Results

| Metric | Value |
|--------|-------|
| Safety reflex latency | 0.2–0.8 ms per check |
| Injection block rate | 15/15 adversarial inputs blocked (unit tests) |
| PII detection rate | 4/4 PII types flagged (SSN, CC, email, phone) |
| System 1 response time (single model) | 1.3–3.5 s |
| System 2 response time (full debate) | 60–120 s |
| Ego arbitration overhead | ~30 s additional (CPU inference) |
| Knowledge base | 3.3M vectors (44K code + 102K ethics + 824K publications) |
| Unit tests | 537 passing, 9 skipped (GPU-only) |
| Total subsystems integrated | 14 of 14 |
| **Single-model accuracy** | **47%** (15 questions, 5 categories) |
| **Debate accuracy** | **71%** (+24 percentage points) |
| **Single-model quality** | **4.1/10** |
| **Debate quality** | **6.4/10** (+56% improvement) |
| Debate latency | 139s avg (vs 22s single) |
| Creative category (debate) | 67% acc (vs 0% single) |
| Ethics category (debate) | 67% acc (vs 33% single) |

The psyche debate improves accuracy by 24 percentage points and quality by 56%. The largest gains are in creative and ethical reasoning — the categories that benefit most from multiple perspectives. Factual questions show no degradation (100% in both modes). The latency cost (139s vs 22s) is managed by the Executive Function, which routes simple queries to System 1 (single model) and only uses debate for complex questions.

### Why Gemma 4

The architecture depends on Gemma 4 in two places where alternatives would be measurably worse:

1. **Superego (31B)**: The analytical hemisphere requires strong instruction-following and structured ethical reasoning. Gemma 4 31B's training on safety-relevant data makes it a better Superego than comparably-sized models we tested — it applies moral frameworks without needing extensive prompt engineering to stay on-topic.
2. **Ego (E4B)**: The DM/arbiter role requires generating structured JSON evaluations and coherent multi-party scenario narration at CPU speed. Gemma 4 E4B is one of the few models at this parameter count that reliably produces valid JSON and maintains narrative coherence across multi-turn training dialogues.

Without Gemma 4, the Superego would need a larger model to achieve comparable ethical reasoning quality, and the Ego would require a GPU — breaking the three-model-on-two-GPUs deployment that makes Atlas accessible on consumer hardware.

### Impact

Atlas demonstrates that safety infrastructure and local deployment are not competing priorities. The same three-layer pipeline that protects users also makes the system auditable and trustworthy. The entire stack — inference, safety, memory, training, dreaming — runs on $2K of used enterprise hardware (HP Z840 with two Quadro GV100 GPUs) with no cloud dependency and no data exfiltration risk.

For researchers who cannot share patient data, educators in bandwidth-constrained schools, and organizations that need auditable AI decisions, Atlas offers a concrete alternative to the cloud-or-nothing choice. It is one working system, but the architecture is modular — any subsystem can be replaced, extended, or studied independently.

---

**Live demo:** https://atlas-sjsu.duckdns.org
**Code:** https://github.com/ahb-sjsu/atlas-ai
**Dashboard:** https://atlas-sjsu.duckdns.org/schematic.html
