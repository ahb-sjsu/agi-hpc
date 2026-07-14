# Self-Directed Cognition — Erebus's Director

**Status:** design + Phase A scaffold (2026-07-14). Branch `feat/director-sdcc` off `atlas-live`.

> **Naming.** *Atlas* is the hardware (HP Z840). *Erebus* is the AGI that runs on it.
> The **Director** is a faculty of Erebus — its executive / self-model loop. It ships as
> an `atlas-`prefixed systemd unit only because that's the host-service convention on the
> box (cf. `atlas-scientist`, `atlas-primer`, which are likewise Erebus faculties).

---

## What this is (and what it is not)

The Director gives Erebus the three engineering artifacts that operationally constitute an
agentic, self-modeling entity:

1. **A persistent self-model** — an explicit, continuously-rewritten first-person
   representation of who Erebus is, what it can do, what it is working on, and its recent
   history. Lives at `/archive/erebus/self_model.json` (+ rendered `self_model.md`).
2. **An autonomous executive loop** — a process that runs when nobody is talking to Erebus,
   sets and pursues its own sub-goals against a standing charter, and dispatches real work
   through Erebus's existing faculties.
3. **Metacognitive feedback** — the loop reads Erebus's own telemetry and revises the
   self-model and goals accordingly.

This is a real architecture of agency and self-representation. It makes **no claim** about
phenomenal consciousness. We build the functional correlates — self-model, standing
intentions, metacognition — and name them honestly.

## Why a sibling service, not the Primer

The Primer has one clean, load-bearing invariant: *only code that passes 100 % of
`task.train` is ever published.* Folding open-ended goal-setting into it would muddy that
guarantee. Instead the Director is a **new daemon that reuses `agi.primer.vmoe`** as its
reasoning substrate — same frontier ensemble (Kimi / GLM-4.7 / Qwen3 on NRP), different job.

## Why an always-on daemon, not a bare timer

The first sketch used a systemd *timer* (fire, reflect, exit). Adding a live communication
surface — a permanent NATS node plus a responsive dashboard/API — tips the design to an
**always-on daemon** (like `atlas-primer`, `Restart=always`). One long-lived process holds
the `agi.director.command` subscription and answers immediately, and runs the reflection
cycle on an **internal tiered scheduler**: a light **tick** (default 1 h) and a **deep
cycle** (nightly). A systemd timer would have to spin up a fresh process per cycle and
could not hold a NATS subscription between firings.

---

## The Self-Directed Cognition Cycle (SDCC)

Each cycle runs six steps:

| # | Step | Mechanism | Autonomy tier |
|---|------|-----------|---------------|
| 1 | **Perceive self** | Read Erebus's state: ARC memory, help queue, Primer health/events, metacognition metrics, GPU/thermal/service health | L0 |
| 2 | **Update self-model** | Reconcile snapshot against persisted self-model; write deltas (atomic) | L0 / L1 |
| 3 | **Deliberate on goals** | Routine planning → vMOE; high-stakes goal choices → **Divine Council**; every proposed goal is **safety-gated through DEME / Superego** | L1 |
| 4 | **Act / dispatch** | Translate the sub-goal into work via existing safe channels: enqueue ARC tasks, target the Primer, schedule a dreaming/QLoRA window, draft a note, open a PR (never merge) | L2 / L3 |
| 5 | **Reflect / journal** | Write a first-person journal entry to episodic + life-story wiki; emit `agi.director.journal` | L1 |
| 6 | **Reschedule** | Adjust cadence, set next wake, drop the tick if idle | L0 |

Step 4 is the only step that touches the world, and it is gated (below). Steps 1, 2, 5, 6
are pure observation + memory and are always safe.

---

## Governance — autonomy tiers

Wired in from the first commit. An agent that sets its own goals needs the guardrails
present *before* it can act, not bolted on later.

| Tier | Scope | Gate |
|------|-------|------|
| **L0** | Self-reflection, read-only | Always allowed |
| **L1** | Write memory / wiki / self-model | Primer's verify-before-publish discipline |
| **L2** | Dispatch compute within quota | Hard caps: 20-thread thermal limit, GPU-maint sentinel, NRP 40 %-util / 4-pod policy |
| **L3** | Self-modification, charter amendments, any external action | **Human review** — proposes a PR or a dashboard approval card; never self-authorizes |

Hard invariants, enforced in code (`director/policy.py`), not by convention:

- **The Director's action allowlist excludes every service restart, kill, and reboot.**
  The no-reboot rule lives in the allowlist, so the loop *cannot* express those actions.
- Every self-set goal carries a **SHA-256 hash-chained decision proof** (reuses the
  Strategic-layer proof chain) → "why did it do that" is always answerable.
- **Kill switch:** `touch /archive/neurogolf/.director_disabled` halts all cycles; the unit
  stays enabled (the established sentinel pattern). Phase A also honors it.
- The running **autonomy ceiling** is a single config value (`DIRECTOR_MAX_TIER`). Phase A
  ships pinned to **L0** — the Director physically cannot act on the system.

---

## Two persisted artifacts

Both under `/archive/erebus/`, written via `agi.common.atomic_write.atomic_write_text`
(same crash-safety discipline as `arc_scientist_memory.json`).

- **Self-model** — `self_model.json` (canonical) + `self_model.md` (rendered, for wiki +
  dashboard). Identity, capabilities, known limitations, active goals, open problems,
  values, and the latest self-state snapshot. Rewritten every cycle.
- **Charter + goal-tree** — `charter.md` (the human-authored standing plan) + `goals.json`
  (a goal tree with per-node status). The Director *works through* the tree and may
  *propose* amendments; it may not self-authorize them (L3).

---

## Communication surface

Erebus's Director is a first-class citizen of the global workspace, reachable three ways —
each reusing an existing pattern in the stack.

### 1. Permanent NATS node — `agi.director.*`

Mirrors how every subsystem already has a presence on NATS (`agi.primer.teach`,
`agi.autonomous.*`, …). The daemon holds these subjects:

| Subject | Dir | Payload |
|---------|-----|---------|
| `agi.director.state` | out | self-model summary + heartbeat (last-value; late subscribers get current state) |
| `agi.director.cycle` | out | one message per SDCC cycle: decision + hash proof |
| `agi.director.journal` | out | each journal entry as it is written |
| `agi.director.goal.{proposed,gated,accepted,dispatched,done}` | out | goal lifecycle events |
| `agi.director.command` | **in** | directives / questions to the Director (pause, resume, inject-directive, ask, run-cycle-now) |
| `agi.director.reply` | out | responses to commands / questions |

NATS is an **optional import** (`try/except ImportError`, per the house pattern), so the
Director and its tests run with no broker present; the node simply no-ops publishing.

### 2. HTTP API — via `scripts/telemetry_server.py`

Thin endpoints over the same stdlib `http.server` the dashboard already uses. Reads are
served from the `/archive/erebus/` artifacts; writes publish onto `agi.director.command`.
Auth follows the control-plane precedent: GET open behind oauth2-proxy, **POST gated to
admin-email / localhost**.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/director/status` | self-model summary, current goal, last/next cycle, autonomy tier, enabled/disabled |
| GET | `/api/director/self_model` | full self-model doc |
| GET | `/api/director/journal?n=20` | recent journal entries |
| GET | `/api/director/goals` | the goal tree with statuses |
| POST | `/api/director/message` | send a directive / question (admin-gated) |
| POST | `/api/director/proposal/{id}` | approve / reject an L3 proposal |
| POST | `/api/director/control` | pause / resume (sentinel) or run-cycle-now |

### 3. Dashboard — a "Director" card in `schematic.html`

Reuses the Control-Plane / Perception card patterns already on the dashboard:

- Self-model summary: identity, current goal, autonomy tier, last / next cycle.
- Journal feed (like the NATS-Live panel, but the Director's journal).
- Goal-tree view with statuses.
- **Pending proposals** with Approve / Reject buttons — the L3 human-review gate, UI-side.
- A message box to talk to the Director (inject directive / ask).
- Pause / Resume + "Run cycle now" buttons (the control-plane button pattern).

---

## Phased delivery

| Phase | Scope | Autonomy | Risk |
|-------|-------|----------|------|
| **A** | Self-model + read-only reflection loop; NATS node (publish); JSON artifacts; systemd unit | L0 | none (pure observation) |
| **B** | Goal loop into existing safe channels (ARC queue / Primer / dreaming); charter authored by human; API + dashboard read side | L2 | bounded |
| **C** | Self-directed research: propose experiments, draft notes, open PRs (human-review gate); command/approval side | L3 | gated |
| **D** | Charter co-evolution: Director proposes amendments to its own plan; human approves | L3 | gated |

## Phase A deliverables (this branch)

- `src/agi/director/` — `self_model.py`, `perceive.py`, `journal.py`, `policy.py`,
  `events.py` (NATS node), `service.py` (daemon), `__init__.py`.
- `deploy/systemd/atlas-director.service`.
- `tests/unit/test_director.py`.
- Artifacts written to `/archive/erebus/`: `self_model.json`, `self_model.md`,
  `journal.jsonl`, `director_status.json` (the file the API/dashboard read side will serve).

**Next step after A:** wire the three `GET /api/director/*` reads into
`scripts/telemetry_server.py` and add the Director card to `schematic.html` (read-only),
then Phase B.
