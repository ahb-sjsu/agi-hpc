# Atlas AI: A Local-First Cognitive Architecture for Safe, Accessible AI
## Gemma 4 Good Hackathon — Writeup Outline

**Tracks:** Main Track, Safety & Trust, llama.cpp Special

**Tagline:** "What if frontier AI came with a conscience, ran in your closet, and never phoned home?"

---

### 1. The Problem (~200 words)

AI's two biggest unsolved problems are **safety** and **access**:
- Cloud AI concentrates power and data in corporate hands
- No existing framework gates LLM outputs through a formal ethical pipeline
- Privacy-conscious users (medical, legal, research) can't use cloud AI
- Resource-constrained communities are locked out of frontier models

The result: the people who need AI most trust it least.

### 2. Our Solution: Atlas AI (~400 words)

Atlas is a **local-first cognitive architecture** that runs entirely on a single workstation with consumer GPUs. It implements the AGI-HPC framework — a modular, safety-gated architecture inspired by human cognition:

**Dual-Hemisphere Architecture**
- **Left Hemisphere (Spock):** Gemma 4 31B — analytical reasoning, code, math, citations. Precise, logical, evidence-based.
- **Right Hemisphere (Kirk):** Qwen 3 32B — creative thinking, pattern recognition, brainstorming. Intuitive, connective, imaginative.
- Different models trained on different data bring genuinely different perspectives — like biological hemispheres using different neural pathways.
- An intelligent router classifies queries and sends them to the appropriate hemisphere (or both for complex tasks).

**Three-Layer Safety Firewall (ErisML DEME Pipeline)**
- **Reflex Layer (<1ms):** Pattern-based input/output scanning. PII detection, prompt injection blocking, content policy enforcement.
- **Tactical Layer (~100ms):** Full MoralVector ethical assessment using configurable Ethics Modules. Maps every interaction to an 8+1 dimensional moral space.
- **Strategic Layer:** Decision proofs, audit logs, governance drift detection. Every interaction is accountable.

**RAG Over 27 Research Repositories**
- PostgreSQL + pgvector semantic search
- 44,000+ chunks from research code, papers, books
- Context-aware responses grounded in real data

### 3. How We Used Gemma 4 (~300 words)

- **Primary inference engine** (Left Hemisphere) via llama.cpp on Quadro GV100 GPU
- **Function calling** for environment interaction (system monitoring, file access, web fetch)
- **Safety classification** — Gemma 4 evaluates its own outputs through the DEME pipeline's Tactical layer
- **Fine-tuned E4B variant** (via Unsloth) specialized for ethical reasoning and safety classification
- Running locally on **resource-constrained hardware** (single workstation, no datacenter) — demonstrates Gemma 4's efficiency

### 4. Technical Architecture (~300 words)

```
Internet → Caddy (HTTPS/Let's Encrypt) → oauth2-proxy (Google auth)
  → RAG Server (query classification + pgvector context injection)
    → Left Hemisphere: Gemma 4 31B (GPU 0, llama.cpp, port 8080)
    → Right Hemisphere: Qwen 3 32B (GPU 1, llama.cpp, port 8082)
  → ErisML DEME Safety Pipeline (input gate + output gate)
  → Response to user
```

**Hardware:** HP Z840, 2x Quadro GV100 32GB, 256GB RAM
**Software stack:** llama.cpp (inference), PostgreSQL + pgvector (RAG), Flask (routing), Caddy (TLS), oauth2-proxy (auth), ErisML (safety), NATS (event fabric)

Key technical choices:
- Heterogeneous model deployment — different architectures on different GPUs
- CPU-based embedding (BGE-M3) keeps GPU RAM free for inference
- Night/day scheduling — training overnight, serving during the day
- All open-source, no proprietary dependencies

### 5. Impact & Vision (~200 words)

**Who this serves:**
- Researchers who can't send proprietary data to cloud AI
- Medical professionals in privacy-regulated environments
- Communities with limited internet but local hardware
- Anyone who believes AI safety shouldn't be optional

**What we prove:**
- Frontier AI can run locally on $2K of used hardware
- Safety and accessibility are complementary, not competing goals
- Heterogeneous model architectures produce better, more balanced outputs
- The AGI-HPC cognitive framework is implementable, not just theoretical

**Vision:** Every research lab, clinic, and school should have their own Atlas — a local AI with a conscience that knows their work, respects their privacy, and can't be turned off by a corporate policy change.

---

### Media / Demo Plan

**Video (3 min):**
0:00-0:30 — Hook: "This AI lives in my closet in Novato, California"
0:30-1:00 — The problem: safety + access
1:00-2:00 — Live demo: ask Atlas a research question, show dual-hemisphere response, show safety dashboard, show architecture schematic
2:00-2:30 — Technical depth: the DEME pipeline, heterogeneous models
2:30-3:00 — Vision: "Every lab should have one"

**Live demo:** https://atlas-sjsu.duckdns.org (Google auth)
**Code repo:** https://github.com/ahb-sjsu/atlas-ai
**Architecture:** https://atlas-sjsu.duckdns.org/schematic.html
