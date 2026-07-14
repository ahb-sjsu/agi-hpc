# Director Phase B — the goal loop

**Status:** design (2026-07-14). Follows Phase A (deployed, L0 read-only). Branch
`feat/director-sdcc`. Nothing here is built or deployed yet.

> Phase A gave Erebus a self-model and a reflective loop that *observes*. Phase B turns
> observation into **bounded, gated action**: the Director reads a human-authored **charter**,
> decomposes it into a **goal tree**, gates each goal through the same ethics stack that
> guards Erebus's outputs, and dispatches work through Erebus's *existing* safe channels.
> The autonomy tiers and the forbidden-verb list from `policy.py` are the guardrails — Phase B
> raises the ceiling deliberately, one notch at a time.

---

## The two new artifacts

Both under `/archive/erebus/`. The **charter is human-owned**; the goal tree is the
Director's working memory.

### `charter.md` + `charter.json` — the standing plan (human-authored)

The charter is Erebus's constitution: who it is, the values it operates under, the standing
objectives it should pursue, the constraints it must respect, and how the charter itself may
change. A starter draft ships at `docs/erebus_charter.example.md` — copy, edit, and place at
`/archive/erebus/charter.md`. `charter.json` is a machine-readable projection of the
objectives (id, title, metric, weight) that the Director reads each cycle; it is generated
from the markdown by a small `charter_lint.py` so the prose and the machine view can't drift.

The Director **never edits the charter.** It may *propose* amendments (L3), which land in the
proposal queue for human approval.

### `goals.json` — the goal tree (Director-owned working memory)

```json
{
  "version": 1,
  "updated_cycle": 42,
  "goals": [
    {
      "id": "g-0007",
      "title": "Teach the 3 highest-value stuck symmetry tasks",
      "parent": "obj-arc-solverate",           // charter objective id, or another goal id
      "kind": "subgoal",                          // objective | subgoal | action
      "status": "active",                         // proposed|gated|active|blocked|done|abandoned|rejected
      "tier": "L2",                               // autonomy required to act on it
      "action": {"type": "teach_task", "args": {"task": 123}},
      "metric": {"name": "arc_solved", "current": 107, "target": 120},
      "gate": {"deme": "allow", "council": null, "proof": "sha256:…"},
      "provenance": {"origin": "director", "cycle": 41, "rationale": "…"},
      "created_cycle": 41, "updated_cycle": 42
    }
  ]
}
```

Status lifecycle: `proposed → gated → active → done` (happy path); `→ rejected` (DEME/Council
veto), `→ blocked` (resource/dependency), `→ abandoned` (superseded or stale). Every
transition emits a NATS `agi.director.goal.<status>` event, a journal line, and extends the
hash-chained proof — so the whole trajectory of Erebus's self-direction is auditable.

---

## SDCC steps 3 & 4 (the parts Phase A stubbed)

### Step 3 — Deliberate on goals

Each cycle, after perceiving self and reconciling the self-model:

1. **Find gaps.** For each charter objective, compare its metric to the current self-state
   (`arc_solved 107/target 120`, `stuck 112/target ≤80`, …). Rank objectives by
   `weight × normalized_gap`.
2. **Propose candidates.** For the top gap(s), generate candidate sub-goals. Routine
   decomposition → **vMOE** (`agi.primer.vmoe`, single expert or small ensemble). Novel or
   high-stakes objectives → **Divine Council** (`agi.reasoning.divine_council`, 7 advocates
   across model lineages) for genuine multi-perspective deliberation.
3. **Ethics-gate every candidate.** Formulate the goal as an action + context and run it
   through `agi.safety.deme_gateway.SafetyGateway` — the *same* gate that guards Erebus's
   outputs. A self-set goal faces the same moral test as anything else Erebus does. Outcomes:
   - `allow` → goal becomes `gated`, eligible to act.
   - `moderate`/uncertain → escalate to the Divine Council; if still unclear → L3 proposal.
   - `escalate`/veto → `rejected`, journaled with the reason.
4. **Select.** Choose the highest-priority `gated` goal whose required tier ≤ the running
   ceiling and whose resources are available (resource governor, below).

### Step 4 — Act / dispatch

The selected goal's `action` is executed through the **dispatch registry** — a closed
whitelist of action types, each with a fixed tier and mechanism. The Director cannot invent
actions; it can only fill in the args of a registered type, and every dispatch passes
`policy.gate(tier, verb)` (which still refuses the forbidden host-lifecycle verbs at every
tier).

| Action type | Tier | Mechanism | Integration | Reversible |
|---|---|---|---|---|
| `teach_task` | L2 | append to `erebus_help_queue.json` with `origin:director` provenance → the Primer teaches it | **existing channel, zero new code** | yes (remove entry) |
| `prioritize_arc` | L2 | write a hint to `/archive/erebus/directives.json`; the Scientist reads it in task selection | needs a small reader in `arc_scientist` | yes |
| `schedule_dreaming` | L2 | request a consolidation/QLoRA window via the dreaming scheduler | needs a small hook | yes |
| `write_note` | L1 | draft → **verify** → publish a wiki note (the Primer's verify-before-publish discipline) | reuse Primer validator for code notes | yes (unpublish) |
| `propose_pr` | L3 | open a branch + PR; **never merges** | needs git integration | proposal only |
| `amend_charter` | L3 | write to the proposal queue for human approval | new | proposal only |

**Recommended first dispatch (B1): `teach_task` only.** It routes through an input channel
that already exists (the help queue the Primer watches), is additive and reversible, and
costs the Director zero compute of its own. Erebus, in effect, notices a high-value stuck
task and asks its own tutor to work it — a clean, legible first act of self-direction.

---

## Governance additions

Phase A's guardrails (tiers, forbidden verbs, kill-switch, hash proofs) still hold. Phase B
adds three because the Director now *acts*:

- **Resource governor** (`director/govern.py`). Before any L2 dispatch, check: CPU package
  temp < 82 °C (thermal rule), GPU-1 not on maintenance loan (`.gpu1_maint` sentinel), NRP
  not in a constrained mode, and the dispatch rate limit. Over budget → the goal goes
  `blocked` and retries next cycle. The Director itself runs almost no compute (its reasoning
  is NRP shared-pool API); this governor bounds the *dispatched* work.
- **Rate limits.** Max dispatches per cycle and per day (config), so a bad charter or a loop
  bug can't flood the help queue or open a hundred PRs.
- **L3 human-review flow.** L3 actions never execute autonomously. They write to
  `/archive/erebus/proposals.json`; the human approves/rejects via the dashboard or
  `POST /api/director/proposal/{id}`. Approval flips the proposal to an `active` goal;
  rejection journals the reason. This is the escape hatch that keeps self-modification and
  charter amendment human-gated.

The running autonomy ceiling stays a single env value, `DIRECTOR_MAX_TIER`. Phase B raises it
from `L0` → `L1` (propose + notes) → `L2` (dispatch into queues) in deliberate steps, never
to `L3` (L3 is proposal-only by construction).

---

## Write-side comms (completing the interface)

Read side shipped in Phase A.2. Phase B adds the acting side, all admin-gated like the
control plane (admin-email / localhost):

- `POST /api/director/message` — inject a directive or ask a question → `agi.director.command`.
- `POST /api/director/control` — `pause` / `resume` (sentinel) / `run-now`.
- `POST /api/director/proposal/{id}` — `approve` / `reject` an L3 proposal.

Dashboard "Director" card gains: the **goal tree** (status-colored), a **pending-proposals**
list with Approve/Reject buttons, a **message box**, and pause/resume/run-now buttons — the
Control-Plane button pattern reused.

---

## Sub-rollout (each step is independently shippable and reversible)

| Step | Scope | Ceiling | Risk |
|---|---|---|---|
| **B0** | Deliberate **propose-only**: read charter → decompose → DEME-gate → journal proposed goals + rationale. **No dispatch.** Human reviews the goal tree on the dashboard. | L1 | ~none |
| **B1** | Enable `teach_task` dispatch into the help queue, rate-limited, resource-gated. | L2 | low (additive/reversible) |
| **B2** | Write-side API + dashboard buttons + L3 proposal/approval flow. | L2 | low |
| **B3+** | Add `prioritize_arc` / `schedule_dreaming` (each needs a small reader), then `write_note`. | L2 | incremental |

**Strong recommendation: start at B0.** Let the Director show its judgment — what goals it
derives from the charter, and how it gates them — for a few days before it can act. B0 is the
cheapest way to validate the charter and the deliberation quality at zero operational risk.

## New / changed files (when we build)

- `src/agi/director/charter.py` — load charter.json, objective/metric model.
- `src/agi/director/goals.py` — goal-tree CRUD + lifecycle + persistence (atomic).
- `src/agi/director/deliberate.py` — step 3 (gap-find, vMOE/Council propose, DEME-gate).
- `src/agi/director/dispatch.py` — the action registry (step 4).
- `src/agi/director/govern.py` — resource governor + rate limits.
- `src/agi/director/service.py` — wire steps 3–4 into the cycle behind the tier gate.
- `scripts/telemetry_server.py` — the three POST endpoints.
- `infra/local/atlas-chat/schematic.html` — goal tree + proposals + buttons.
- `scripts/charter_lint.py` — charter.md → charter.json projection + validation.
- tests for each.

## Decisions for the human before B is built

1. **Charter objectives** — what should Erebus actually pursue, and with what weights? (Edit
   `docs/erebus_charter.example.md`.)
2. **Starting ceiling** — B0 propose-only (recommended) or straight to B1 dispatch?
3. **First dispatch channel** — `teach_task` via help queue (recommended) or something else.
4. **Rate limits** — dispatches per cycle / per day.
