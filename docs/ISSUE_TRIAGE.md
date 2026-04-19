# Issue Triage Plan — 2026-04-19

**Status:** snapshot of where every non-trivial open issue on `ahb-sjsu/agi-hpc` stands and the plan to address it.

Refresh quarterly or whenever a cluster closes.

---

## Completed in this session

| # | Title | Resolution |
|---|---|---|
| **#35** | Deployment runbooks + on-call playbook | `docs/DEPLOYMENT_RUNBOOK.md` + `docs/ONCALL_PLAYBOOK.md` shipped (ee7f588) |
| **#36** | Inventory existing metrics, logs, evaluation artifacts | `docs/METRICS_INVENTORY.md` shipped (ee7f588) |
| **#37** | Define core SLOs / KPIs | `docs/SLOS_AND_KPIS.md` shipped (ee7f588) |
| **#38** | Instrument core job lifecycle paths | `agi.common.structured_log` + `arc_scientist` + `Primer` instrumented + `/api/jobs/recent` (7728724) |
| **#39** | Basic evaluation harness | `evals/` package + 9 tests (d96e394) |
| **#40** | Dashboard design for ops + evaluation | `/api/trends/erebus` + 30-day sparkline (484736a) |
| **#42** | Document Evaluation & Metrics practices | `docs/METRICS_CONTRIBUTOR_GUIDE.md` shipped (ee7f588) |
| **#34** | Deployment observability (umbrella) | Split into #78 / #79 / #80 / #81 (see Cluster A below) |
| **#45–#65** | bip-fusion sprints 1–6 (21 issues) | **Transferred to [`ahb-sjsu/bip-fusion`](https://github.com/ahb-sjsu/bip-fusion)** (issues #1–#21 there). Tracking stub: agi-hpc#82. |

---

## Cluster A — Remaining observability (#38, #41, #78, #79, #80, #81)

After the 2026-04-19 session, three of the original five closed (#39, #40, and #34-split). Remaining:

- **#38** — Instrument job lifecycle paths
- **#41** — Evaluation hooks for cognitive architecture
- **#78** — Structured JSON logging (child of #34)
- **#79** — OpenTelemetry tracing (child of #34)
- **#80** — Alerting pipeline (child of #34)
- **#81** — Error-budget automation (child of #34)

Each of #78–#81 is a self-contained 1–3-session piece of work with its own acceptance criteria, which is why splitting the umbrella #34 made sense. See the individual issue bodies.

### #40 — Design dashboards for ops and evaluation

**Scope:** small. Most is already built — the schematic dashboard at `atlas-sjsu.duckdns.org/schematic.html` covers ops comprehensively. What's missing is an explicit *evaluation* dashboard (Erebus solve trends over time, Primer verification rate per week, strategy win-rate deltas).

**Plan:**
1. Add a `Trends` tab or separate page (`trends.html`) showing time-series.
2. Backend: `/api/trends/erebus` returning `{"solves_by_day":[...], "solves_by_week":[...]}` from memory-file snapshots.
3. Snapshot job: cron on Atlas dumps `arc_scientist_memory.json.solves` every night to a compact per-day JSON in `/archive/neurogolf/trends/`.
4. Frontend: sparklines / line charts of the per-day series.

**Estimate:** 1 focused session. Closes #40.

### #38 — Instrument core job lifecycle paths

**Scope:** medium. Touches arc_scientist, dreaming, vision-burst dispatch, possibly nats-bursting workers. Requires adding structured logs + rough metrics across five subsystems.

**Plan (phased):**
1. Adopt the log-format convention from `METRICS_CONTRIBUTOR_GUIDE.md` §3 across services. First pass: `arc_scientist.py` attempt log lines (currently partially structured).
2. Add a `job_lifecycle` log category: `job_submitted` → `job_running` → `job_complete | job_failed | job_timeout`, with duration fields.
3. Optional: a `/api/jobs/recent` endpoint aggregating recent lifecycle events for a dashboard panel.

**Estimate:** 2–3 sessions. Partial progress per session is fine; doesn't need to be atomic.

**Prereq:** none; can start immediately.

### #39 — Build evaluation harness

**Scope:** medium. A repeatable offline test runner for ARC-like workloads. Overlaps with existing `tests/` — distinction is that this runs a *portfolio* of stuck tasks through a fixed strategy set and reports per-task outcomes.

**Plan:**
1. Directory `evals/` with task-selection config YAML + harness script.
2. Harness drives arc_scientist in "evaluate-only" mode (no memory updates).
3. Outputs per-run JSON: `{"timestamp":..., "git_sha":..., "tasks":{...}, "aggregate":{...}}`.
4. Optional: CI nightly runs a small portfolio and posts diff vs. previous run.

**Estimate:** 1–2 sessions.

**Prereq:** none; useful independently.

### #41 — Evaluation hooks for cognitive arch / ErisML

**Scope:** larger. Needs decision-point instrumentation inside the cognitive stack (Ego, Superego, Divine Council) and a judging function for decision quality.

**Plan:**
1. Identify 3–4 decision types to instrument (e.g. Superego veto, Council final call, ego routing choice).
2. Per-decision log record: `{decision_type, inputs_hash, output, alternatives_considered, time_s}`.
3. Offline eval pipeline: human or LLM judge compares `alternatives_considered` to `output`; produces quality signal.
4. Aggregate into a weekly "decision quality" panel.

**Estimate:** 3+ sessions. Partly research, partly code.

**Prereq:** #38 (lifecycle logging) is useful foundation.

### #34 — Deployment observability (status-ed already)

Already has long comment describing what's shipped vs. to do. Recommend splitting into child issues:
- `#34a` Structured JSON logs
- `#34b` OpenTelemetry tracing
- `#34c` Alerting pipeline
- `#34d` Error-budget automation

Each is standalone 1–2 sessions. Splitting gets them off the "too big to start" pile.

---

## Cluster B — DCGM attestation (#74, #75, #77)

Hardware-level verification that GPU work actually happened. Three issues; the first is data-collection, the other two depend on it.

### #74 — Power-profile fingerprinting

**Scope:** research + code. Must collect power traces on Atlas GPUs during representative workloads (Qwen forward pass, BGE batch, idle, replayed output) and build a classifier.

**Plan:**
1. Instrument a data-collection script: `nvidia-smi --query-gpu=power.draw --format=csv -lms 100` during scripted workloads.
2. Label each trace by workload type; export to `evals/dcgm_profiles/`.
3. Train a threshold-based or small-sklearn classifier.
4. Wrap the classifier behind `DCGMAttestor.classify(trace)`.

**Estimate:** 1 session for data collection + baseline classifier.

**Prereq:** Atlas access, GPU idle window, no concurrent work contending for GPU 0 or 1.

### #75 — Integrate DCGM attestation with DEME safety gateway

**Scope:** code. Wire `DCGMAttestor.snapshot()` before/after the forward pass in `src/agi/safety/deme_gateway.py`; NATS publish `agi.safety.attestation`.

**Plan:**
1. Patch the gateway's inference path.
2. Emit attestation event.
3. Add a gateway test that simulates a failed attestation (e.g. by mocking no power delta).
4. Dashboard panel: attestation pass/fail rate.

**Estimate:** 1 session.

**Prereq:** #74 (or at least a working `DCGMAttestor.attest()` with the coarse heuristic).

### #77 — Verify GPU utilization on NRP burst pods

**Scope:** code + NRP investigation.

**Plan:**
1. `kubectl describe node <nrp-gpu-node> | grep -i dcgm` — check if NRP exposes DCGM.
2. If yes: add a DCGM sidecar to burst-pod specs (vision pool + any future GPU pod).
3. If no: fall back to `nvidia-smi` polling from inside the pod during its own work.
4. Either way: publish a post-job attestation summary to `agi.safety.attestation`.

**Estimate:** 1 session once the kubectl investigation is done.

**Prereq:** live NRP connectivity.

**Dependency graph:** `#74 → #75 → #77` (attestor → gateway integration → NRP generalization).

---

## Cluster C — BIP-fusion (extracted 2026-04-19)

**Resolved by extraction.** The twenty-one fusion-reactor issues (original `#45`–`#65`) have been moved to a dedicated repository:

**New home:** https://github.com/ahb-sjsu/bip-fusion

Issue-number mapping is documented in the tracking stub `agi-hpc#82`. Sprint 1–6 work continues there at its own cadence.

The extraction was ~30 minutes of admin (create repo, seed README, transfer 21 issues, write tracking stub) and removed the biggest single distraction from the agi-hpc roadmap. The two programs share Atlas workstation + NRP account + operational memory but have independent issue queues now.

---

## Cluster D — Student quest issues (#66, #67, #68, #69, #70, #73, #76)

**Explicitly excluded from this plan** per user direction. These are for student contributors. PR #71 (adarshm11) addresses #67; PR #72 addresses #68. Both reviewed separately. Other student quests remain open for future contributors.

---

## Execution plan — recommended order

If attacking remaining issues in a single chunk of sessions:

### Session 1 — closable observability wins

- **#40** — Trends dashboard (existing schematic already covers ops; evaluation tab is the gap)
- **#39** — Evaluation harness skeleton (independent; unblocks #41 later)

**Deliverable:** both closed.

### Session 2 — job-lifecycle logging

- **#38** first pass: convert arc_scientist logs to structured format; add lifecycle log category.

**Deliverable:** #38 partial; #34 moved closer (JSON log migration is the big one).

### Session 3 — bip-fusion extraction

- Create `ahb-sjsu/bip-fusion` repo.
- Transfer #45–65 to the new repo.
- Leave tracking issue on `agi-hpc`.
- Close #45–65 here.

**Deliverable:** 21 issues off the agi-hpc queue; bip-fusion has its own room to grow.

### Session 4 — DCGM data collection

- **#74** — collect power traces, baseline classifier.

**Deliverable:** #74 closed; #75 and #77 unblocked.

### Session 5+ — whatever's left

Remaining: #34 subsections, #38 completion, #41, #75, #77. Attack in whatever order the operational needs dictate.

---

## Out-of-scope for this plan

- **PR review #71 / #72** — tracked separately; #71 has follow-up fix, #72 has posted review.
- **Student quest issues** — per explicit user direction.
- **GPU reservation / capacity expansion** — not a code issue; requires ops negotiation with NRP admins.
- **Chat ego cutover (Phase 3)** — tracked in `AGI_ROADMAP.md` status snapshot; not a Github issue.
- **Self-host ego pod (Phase 2)** — same as above; session-scoped task, waits on capacity.

---

## Keeping this plan fresh

1. When an issue closes, strike it from the plan and add to "Completed in this session" at the top.
2. When a new issue arrives, classify it into one of the four clusters (A/B/C/D) — or open a new cluster if it doesn't fit.
3. Re-read quarterly; delete any cluster that's stayed stale for two quarters (signals the work isn't real and the issue should close).
