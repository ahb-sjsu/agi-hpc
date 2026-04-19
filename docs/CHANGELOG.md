# Changelog — Atlas AI / AGI-HPC

## 2026-04-19 — The Primer, vMOE, and regression guards

**Major: The Primer shipped.** Always-on Claude-style tutor for Erebus, inspired by Stephenson's *Young Lady's Illustrated Primer*. Watches the help queue, reads wiki + episodic memory for context, calls a **virtual Mixture-of-Experts** ensemble of frontier LLMs in parallel, verifies each returned `transform(grid)` against `task.train`, and publishes verified sensei notes to the wiki with git commit + push. See [`docs/THE_PRIMER.md`](THE_PRIMER.md).

Enforces the invariant **verified-only publishing**: a wrong mentor note is worse than no note at all (motivated by the task056 incident where my hand-written sensei note misdirected 30+ attempts before being caught and corrected).

### New subsystem: `src/agi/primer/`

- `vmoe.py` — virtual MoE router / cascade / ensemble over NRP's four frontier services plus Atlas local Kirk. Three orchestration policies:
  - `route(hint)` — single-expert by role tag
  - `cascade(hint)` — priority-ordered fallback
  - `ensemble(experts, verify)` / `first_verified()` — parallel fan-out with cancel-on-win
- `validator.py` — sandboxed subprocess runner. SIGALRM per-example timeout on Unix; outer subprocess wall-clock on Windows. Extracts candidate code from JSON, fenced-markdown, or raw Python.
- `service.py` — the daemon loop. Polls memory for stuck tasks, tiers them (partial progress first, then full-zero stuck), processes top 3 per tick, respects 6 h per-task cooldown, persists expert-health snapshots for dashboard consumption.
- `health.py` — per-expert rolling health window. If ≥ 3 of last 5 calls were timeouts or slow (> 180 s), expert enters a 1 h cooldown; automatically retried afterward.
- Systemd unit at `deploy/systemd/atlas-primer.service` — CPU-only, `MemoryMax=4G`, `CPUQuota=200%`, `EnvironmentFile=-/home/claude/.primer.env`.
- 32 new unit tests across vmoe / validator / health — all green.

See [`docs/VMOE.md`](VMOE.md) for the virtual-MoE design rationale and default expert pool.

### Cognitive architecture

- **Kirk is the Ego** — corrected labels across `schematic.html` (topology synthetic node, GPU 1 caption, bottom-bar stat) and the test suite. The systemd service name `atlas-id.service` is still historical; functional role is now Ego. Star Trek mapping is now correct: Kirk = captain = balanced decision-maker = Ego. The Id slot is unfilled at the LLM level (served by local-GPU procedural memory).
- Added The Primer as a synthetic node in the NATS topology. Colored by `atlas-primer.service` process liveness.

### Dashboard reliability

- **Dynamic UI version stamp.** Footer was a static `ui:2026-04-17a` literal for 16 commits. Now `telemetry_server.py` intercepts HTML serving and substitutes `{{UI_VERSION}}` with the current `git rev-parse --short HEAD · <file mtime UTC>`. Cached 15 s to avoid per-request git forks. New `/api/version` JSON endpoint exposes the same.
- **Dashboard-drift post-mortem.** Previously the CI deploy step ran `cp -f atlas-chat-*.html $STATIC_PATH/*.html`. The destination was a symlink into the git tree, so `cp` followed the symlink and wrote stale content *back* into the working tree after every push. Sixteen commits of dashboard features were silently reverted before anyone noticed. Fix: replaced `cp -f` with `ln -sfn`, deleted stale root-level `atlas-chat-*.html` duplicates. Post-mortem in `feedback_dashboard_deploy_drift` (personal memory).
- **NRP Burst Jobs + Worker Pools card.** Previously only queried K8s `Jobs`; the `erebus-workers` Deployment was invisible. Now queries `deployments` too, tags entries with `kind: Job | Deployment`, renders as one combined card with a Kind column and active/desired counts for deployments.
- **Layout re-order.** NATS Live + NRP Burst Jobs now share a row; NATS Topology is a full-width panel below.
- **Erebus — NeuroGolf 2026 card populated.** Previously had element IDs but no poller. Now shows live Score (unique tasks solved / 400), lifetime efficiency (attempts per solve), cycle/in-cycle counters, current task, strategy win-rate bars, vision-pool count, help-queue count, recent-solves list. `/api/erebus/status` enriched with parsed cycle/attempt/current/this_cycle/recent_solves/vision_pool/help_queue_count.
- **Headline metric fix** — Erebus card was showing solves / total_attempts (efficiency rate), making 92 / 978 read as "9.4 %". The competition-relevant denominator is the 400 tasks in the dataset: 92 / 400 = 23 %. Now the green headline is coverage; attempts-per-solve is the sidecar metric.

### CI / CD

- **Deploy-smoke workflow** (`.github/workflows/deploy-smoke.yaml`) — push-triggered + 30-min cron. Waits 2 min for deploy, fetches `/api/version`, compares live SHA to `git rev-parse HEAD`. Warns with commit-count delta if drift; asserts key widgets are present on the live page.
- **Dashboard-render workflow** (`.github/workflows/dashboard-render.yaml`) — Playwright headless Chromium. Opens `schematic.html`, asserts topology SVG populated (g/circle/text/rect children > 0), burst table has rows, version stamp matches `ui:[a-f0-9]{6,} · \d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z`, no console errors.
- `tests/dashboard/` — Playwright project with `@playwright/test ^1.48.0`, separate Node package to keep Python CI decoupled.
- Both workflows are `continue-on-error: true` — they're drift detectors, not PR gates.

### Chat / Erebus

- Chat timeouts raised **45 → 120 s agentic**, **30 → 90 s simple** in `telemetry_server.py`. Kimi on NRP had been retrying on upstream slowness and cumulative retries busted the old 45 s cap, producing "Request timed out" errors in chat.
- Raw response log peek on verify-fail so we can debug format / truncation without another smoke cycle.

### Sensei wiki (auto + human-written)

Human-written notes pre-Primer:

- `sensei_task_018.md` — color-pair lookup
- `sensei_task_030.md` — horizontal centering
- `sensei_task_056.md` — symmetry-class lookup (4-way classifier). **Rewrote** after discovering my earlier "any symmetry → 2 else 1" rule was wrong; the actual rule has 4 output classes disambiguated by `nz[0,0]`. Erebus solved the task on the next attempt with the corrected note.
- `sensei_task_077.md` — observe-mode nudge (no concrete rule)
- `sensei_task_078.md` — global translation for hole-fill
- `sensei_task_175.md` — diagonal stripe reconstruction
- `sensei_task_381.md` — matched rectangle pair gap fill
- `sensei_meta_task_typology.md` — output-shape-class taxonomy (CLASSIFICATION / TRANSFORMATION / EXTRACTION / EXPANSION / PARAMETERIZED)
- `sensei_meta_symmetry_classifiers.md` — symmetry-classifier family (sub-class of CLASSIFICATION)
- `sensei_meta_primitives_guided.md` — explains why the `primitives_guided` strategy is 0 / 118 (broken prompt, not low signal)
- `sensei_meta_strategy_allocation.md` — where Erebus's attempts are wasted and what to shift

### Vision pool

- 4× GLM-4.1V-9B-Thinking K8s Job burst fires when ≥ 10 perception-error tasks accumulate. nodeAffinity prefers L40 / L40S / A10 / V100. Manifest at `scripts/render_erebus_vision_job.py`. Dispatch hook in `arc_scientist._dispatch_vision_burst`.

### Rename + housekeeping

- Sensei notes moved from `/archive/neurogolf/mentor_notes.json` sidecar into `wiki/sensei_task_*.md` — same place RAG pulls Tier-1 knowledge from. `arc_scientist.ARCScientist.__init__` loads them at init, stripped of YAML frontmatter, prepended to matching task prompts.
- Deleted stale root-level `atlas-chat-*.html` files (5,271 lines removed).
- `deploy_to_atlas.sh` rewritten to use `ln -sfn` for dashboard files.
- 12 stale `git stash` entries on Atlas cleared.

### Competition progress

- **100 / 400 tasks solved (25.0 %)** — up from 92 at start of session.
- 1,364 lifetime attempts; strategy performance:
  - `direct`            50 / 456 (11.0 %)
  - `example_chain`     22 / 219 (10.0 %)
  - `failure_aware`     16 / 212 ( 7.5 %)
  - `diagnostic`        12 / 359 ( 3.3 %)
  - `primitives_guided`  0 / 118 ( 0.0 %)  ← flagged for removal

---

## 2026-04-15 → 2026-04-18 — nats-bursting, vision, chat, dreaming

- **nats-bursting 0.2.0.** Persistent worker pools via NATS. `erebus-workers` Deployment (8× pods on NRP) consumes `agi.tasks.*` subjects. HTTP webhook bridge for result-upload into arc_scientist memory (workaround for NATS leaf-node sub-propagation asymmetry: hub→spoke works, spoke→hub doesn't). `exit_on_idle_s` field lets Job-shape workers terminate cleanly when a burst dries up.
- **Vision pool.** GLM-4.1V-9B-Thinking (9 B total / A10-A100 compatible). Job shape (not Deployment) with idle-exit so it respects NRP's > 40 % GPU utilization rule. Dispatch from arc_scientist when perception-errors ≥ 10.
- **Dreaming schedule.** Runs whenever there's idle time, not at fixed 2–4 am. QLoRA fine-tuning during dreaming produces adapters on `/archive`, cached on NRP PV. Pipeline is intact but currently producing adapters with no base-model slot to attach to — resolved later by the Ego architecture plan in [`docs/AGI_ROADMAP.md`](AGI_ROADMAP.md).
- **Chat UI live-activity sidebar.** Right-side panel on `erebus.html` polls `/api/erebus/activity` every 2 s, shows a ring buffer of the last 500 log lines with kind filters (attempt / solved / help / meta / dream / error / info).
- **Erebus help queue.** `/api/erebus/help` backing stuck-task markers that the Primer now consumes.
- **Thrash detection + deprioritization.** Tasks with ≥ 15 attempts and no score improvement in the last 5 get deprioritized until a sensei note arrives.

---

## 2026-04-03 → 2026-04-14 — Phase-7 metacognitive loop, DHT, observability

### Phase 7 (Metacognitive loop)

- Hemisphere disagreement metric
- Adaptive temperature routing (LH 0.3, RH 0.8)
- Audio-diary-style periodic self-reflection

### DHT Service Registry

- `ServiceRegistry` with PostgreSQL-backed `service_registry` table, HTTP health probing, 60 s stale detection.
- `ConfigStore` for versioned config distribution.
- NATS service: `agi.dht.{register,deregister,lookup,heartbeat}`.
- 8 services registered, 7 config entries seeded, 5 integration tests.

### Observability

- Visitor logging to PostgreSQL.
- Training results table with auto-promotion / demotion.
- Curriculum manager.
- Mobile-responsive chat UI with debate collapsible.

---

## 2026-04-02 → 2026-04-03 — Atlas AI Launch

### Phase 0: Event Fabric

- NATS JetStream (`:4222` data, `:8222` monitoring).
- `AGI_EVENTS` stream with `agi.>` wildcard, 1 GB max, 7-day retention.
- `Event` dataclass + `NatsEventFabric` async wrapper.

### Phase 1: Left Hemisphere + LLM integration

- `LLMClient` async wrapper for OpenAI-compatible APIs.
- `InferenceConfig` (LH preset temp=0.3, RH preset temp=0.8).
- `PromptTemplateRegistry` with 4 built-in templates.
- `RAGSearcher` with pgvector semantic search.

### Phase 2: Memory subsystem

- Episodic memory (sessions + episodes with embeddings).
- Procedural memory (SQLite; learned behaviors + success tracking; 5 seed procedures).
- Semantic memory (wrapper around pgvector).
- Memory Service Broker on NATS (5 subjects).

### Phase 3: Safety gateway

- `SafetyAdapter` converts chat interactions to EthicalFacts.
- `SafetyGateway` with the DEME 3-layer pipeline:
  - Reflex (< 20 µs): PII, prompt injection, dangerous content
  - Tactical: ErisML MoralVector assessment
  - Strategic: SHA-256 hash-chained decision proofs
- Input gate + output gate via NATS.

### Phase 4: Right Hemisphere + Integration

- `RHNatsService` — Qwen 3 32B on GPU 1, creative prompts.
- Integration Orchestrator — query classification, dual-hemisphere merge.
- 4-round debate mode: parallel opening → mutual challenge → captain's call.

### Phase 5: Metacognition + Environment

- Monitor: latency p50 / p95 / p99, hemisphere ratio, veto rate, throughput.
- Reflector: periodic self-assessment every 10 interactions.
- Adjuster: auto-tunes max_tokens, safety thresholds, routing balance.
- System sensor — GPU / CPU / RAM / disk via NATS.
- Repo sensor — watches `/archive` for git changes.

### Phase 6: DHT service registry

(See 2026-04-03 → 2026-04-14 block above.)

### Infrastructure

- Dual-hemisphere LLMs: Gemma 4 31B (Spock / GPU 0) + Qwen 3 32B (Kirk / GPU 1).
- HTTPS via Caddy + Let's Encrypt (`atlas-sjsu.duckdns.org`).
- Google OAuth via `oauth2-proxy`.
- Custom chat UI with debate collapsible, thinking spinner, mobile responsive.
- Operations dashboard with GPU gauges, sparklines, job monitoring.
- Event log page with NATS activity.
- Visitor logging (PostgreSQL).
- `start_atlas.sh` with `--health` and `--stop` modes.
- Cron: train midnight–8 am, chat 8 am–midnight.

### RAG / Search

- Hybrid search: BM25 + dense vector + HyDE + Reciprocal Rank Fusion.
- Repo-aware filtering.
- 27 GitHub repos indexed (44 K+ chunks).
- tsvector + GIN index for full-text search.

### Knowledge base

- 102 K ethics chunks from 7 traditions, 37 languages, 3,300 years.
- 824 K publications catalog with full-text search.
- Wikipedia English dump (24 GB).
- Project Gutenberg (syncing).
- arXiv CS metadata (1.7 GB).
- Common Crawl WAT sample (2.9 GB).
- PostGIS: 258 countries, cities, coastlines.
- Kaggle datasets: Dear Abby, Reddit AITA, Philosophy, Jeopardy, arXiv.

### Training

- AtlasGym: 5 environments (ethics, reasoning, coding, debate, memory).
- Curriculum manager with auto-promotion / demotion.
- Unsloth Gemma 4 E4B ethics fine-tune script.

### Authorship fix

- Corrected A.H. Bowers → A.H. Bond across 28 files in geometric book series.

### Housekeeping

- Deleted 444 GB stale duplicates in `/newhome` (90 % → 39 % usage).
- Fixed `fstab` to use UUIDs + `nofail`.
