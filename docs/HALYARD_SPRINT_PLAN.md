# HALYARD TABLE — SPRINT PLAN

*Companion to [`HALYARD_TABLE.md`](HALYARD_TABLE.md). This is the
execution plan for building the Halyard Table live-play runtime.*

**Format.** Each sprint has: goal, deliverables, acceptance
criteria, risks, and a gate-to-next. Sprints are sized for roughly
1–3 days of focused engineering each. Status is updated in-place
as each sprint completes.

**Working branches.** Each sprint lives on its own branch,
`halyard/sprint-NN`, off main. Merges are squash-merged after
all acceptance criteria are green.

**Commit hygiene.** All commits follow the agi-hpc convention:
typed prefix (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`),
scope (`halyard`, `halyard-state`, `halyard-web`, `sigma4`),
imperative subject, wrapped at 72.

---

## Sprint 0 — Plan + Scaffolding

**Status:** IN PROGRESS (this commit).

**Goal.** Get the arch doc, sprint plan, and package skeletons
committed so every subsequent sprint has a clear place to live.

**Deliverables.**

- [x] `docs/HALYARD_TABLE.md` — architecture of record.
- [x] `docs/HALYARD_SPRINT_PLAN.md` — this document.
- [x] `src/agi/halyard/__init__.py` — top-level namespace.
- [x] `src/agi/halyard/sigma4/` — SIGMA-4 package skeleton.
- [x] `src/agi/halyard/state/` — character state service skeleton.
- [x] `src/agi/halyard/keeper/` — Keeper backend skeleton.
- [x] `infra/local/livekit-sfu/` — LiveKit deploy directory with
      placeholder README.
- [x] `deploy/docker/halyard-web/` — web client deploy directory
      with placeholder README.
- [x] `web/halyard-table/` — Next.js source tree placeholder.
- [x] `tests/unit/halyard/` — test root for new packages.
- [x] Initial skeleton unit tests that assert package shape.

**Acceptance criteria.**

- `pytest tests/unit/halyard` green.
- `ruff check src/agi/halyard` clean.
- Commit lands on `halyard/sprint-0` with a clear message.

**Risks.**

- Naming collision with existing packages — checked, none.
- Misalignment with `src/agi/primer/artemis/` conventions —
  deliberately mirror the ARTEMIS shape.

**Gate to Sprint 1.** Scaffolding commit merged to main; branch
`halyard/sprint-1` created off main.

---

## Sprint 1 — LiveKit SFU on Atlas

**Status:** ARTIFACTS COMPLETE; PENDING ATLAS DEPLOYMENT.

*Config, runbook, smoke test, and static-config tests are all
committed and green. The Sprint-1 acceptance gate requires bringing
the SFU up on Atlas and running the smoke test from two real
browsers — scheduled as a coordinated Atlas rollout after the code
sprints (2–7) complete.*

**Goal.** Stand up a self-hosted LiveKit SFU on Atlas that two
browsers can connect to over TLS and hold a working video call.

**Deliverables.**

- `infra/local/livekit-sfu/docker-compose.yml` with:
  - `livekit/livekit-server` pinned to a known-good tag.
  - `coturn/coturn` sidecar for TURN.
  - Host networking or explicit port map for `7880`, `7881`,
    `50000-60000/udp`, `3478`.
- `infra/local/livekit-sfu/livekit.yaml` — config:
  - `keys:` one dev key/secret (stored in sops-encrypted
    `.env` for non-dev).
  - `rtc:` external-IP hint to Atlas's public/duckdns address.
  - `room.auto_create: true`, `room.empty_timeout: 86400`.
- `infra/local/livekit-sfu/turn.yaml` — coturn minimal config
  with shared-secret auth matching LiveKit's `turn.credentials`.
- `infra/local/livekit-sfu/README.md` — runbook: bring up, bring
  down, smoke test, logs, port notes.
- `atlas-caddy` snippet: add a `halyard.atlas-sjsu.duckdns.org`
  site block reverse-proxying LiveKit's `:7880` WebSocket.
- `systemd --user` unit file `livekit-sfu.service` that wraps the
  `docker-compose up -d` / `down` pair.
- Smoke test script `scripts/halyard/sfu_smoke.sh` — mints two
  JWTs, uses `livekit-cli` or a tiny Node script to connect two
  fake participants and verify presence events.

**Acceptance criteria.**

- Two browsers on different networks can join the same room at
  `wss://halyard.atlas-sjsu.duckdns.org` with JWTs minted from
  the Python `mint_participant_token()` helper and see each
  other's presence.
- `systemctl --user status livekit-sfu.service` reports active.
- `journalctl --user -u livekit-sfu` shows clean startup.
- TURN test: one browser behind a symmetric NAT can still
  connect.
- `scripts/halyard/sfu_smoke.sh` exits 0.

**Risks.**

- **NAT / firewall surprises** on Atlas's router — TURN should
  mitigate, but the first out-of-LAN connection may need
  router-level config changes. Flag early.
- **Caddy WebSocket** — confirm `reverse_proxy` handles
  upgrade headers; add `transport http { }` if needed.
- **TLS cert for duckdns subdomain** — Caddy auto-provisions
  via ACME; verify DNS-01 if HTTP-01 blocked.

**Gate to Sprint 2.** SFU runnable end-to-end from two external
browsers; one 15-minute dry-run call held without disconnects.

---

## Sprint 2 — SIGMA-4 Skeleton

**Status:** PERSONA + TRIGGER + MODE HANDLER COMPLETE; FULL
INTEGRATION PENDING.

*`prompt.py`, `trigger.py`, and `mode.py` (with parallel
`handle_turn`) are committed and green (63 tests covering
persona invariants, trigger policy branches, and handle_turn
happy/sad paths with injected fake LLM). NATS wiring
(`nats_handler.py`) and the LiveKit agent (`livekit_agent/`) are
deferred to the cross-agent refactor in post-sprint backlog — the
current ARTEMIS wiring already shares substrate we can cleanly
re-use once both agents are proven at this layer. Bible chunks
(`bible/`) are a content task that can proceed in parallel with
the web client work.*

**Goal.** A functional SIGMA-4 reasoning service that can answer
a turn, parallel to ARTEMIS. NATS plumbing, LiveKit agent fork,
and bible split included.

**Deliverables.**

- `src/agi/halyard/sigma4/mode.py` — `handle_turn()` entry,
  SIGMA-specific trigger policy and persona.
- `src/agi/halyard/sigma4/prompt.py` — `_SIGMA4_SYSTEM_PROMPT`
  constant with SIGMA's in-fiction identity, voice, hard rules,
  and `[INTERFACE_SILENT]` fallback token.
- `src/agi/halyard/sigma4/context.py` — bible split loader;
  SIGMA knows ship history and logs, not Halverson's private
  files.
- `src/agi/halyard/sigma4/trigger.py` — trigger policy: name
  mention `"sigma"` or `"sig"`, plus Keeper cue.
- `src/agi/halyard/sigma4/validator.py` — inherits the ARTEMIS
  validator architecture; SIGMA-specific forbidden n-grams.
- `src/agi/halyard/sigma4/nats_handler.py` — subscribes to
  `agi.rh.sigma4.heard`, publishes `agi.rh.sigma4.say`.
- `src/agi/halyard/sigma4/livekit_agent/` — LiveKit agent twin
  to ARTEMIS's, distinct identity `sigma-4` and display name
  `SIGMA-4`. Shares `token.py`.
- `src/agi/halyard/sigma4/bible/` — SIGMA-specific bible chunks
  (ship history, technical specs, SIGMA's own logs) tagged
  `sigma4_known` / `sigma4_unknown` / `sigma4_forbidden`.
- Unit tests — at least 20 cases across:
  - `test_sigma4_mode.py` — happy path, in-character rejection,
    silence fallback, length overrun, secret leak.
  - `test_sigma4_validator.py` — each check-reply rule.
  - `test_sigma4_trigger.py` — name mention, Keeper cue,
    diminutive, negative cases.
  - `test_sigma4_prompt.py` — prompt assembly + caching shape.
  - `test_sigma4_livekit.py` — agent I/O bridge analogue.

**Acceptance criteria.**

- `pytest tests/unit/halyard/sigma4` green (≥ 20 cases).
- Offline dry-run notebook shows a coherent 10-turn SIGMA-4
  dialogue against synthetic transcripts.
- NATS integration test: a fake bot publishes 10 turns on
  `sigma4.heard`; correct `sigma4.say` responses come back with
  valid DecisionProofs.
- LiveKit agent unit tests green (mirror of the ARTEMIS agent's
  test file).

**Risks.**

- **Persona collision** — SIGMA and ARTEMIS must sound
  *different*. Voice calibration is the main art here.
- **Bible contamination** — an item tagged
  `sigma4_known` that should be `sigma4_forbidden`. A careful
  editorial pass is required.
- **Both AIs speaking at once** — room etiquette requires a
  policy. v1: both AIs may be triggered by the same turn; both
  reply; Keeper moderates.

**Gate to Sprint 3.** SIGMA-4 handles a full 10-turn synthetic
dialogue through NATS with all validator checks passing.

---

## Sprint 3 — halyard-state service

**Status:** CODE COMPLETE; SYSTEMD + DOCKER PACKAGING PENDING.

*JSON Schema character sheet, field-level access control, JSON-Patch
applier, on-disk Store with append-only log, NATS bridge, and
aiohttp REST+WS API are all committed with 113 new unit tests
(216 total in tests/unit/halyard/). Two-subscriber WS fan-out test
passes; cross-session isolation verified. Systemd unit and
Dockerfile for Atlas deployment are mechanical wrappers that will
land alongside the broader Atlas rollout.*

**Goal.** A running service that owns character-sheet state for
a session, writes are accepted over NATS and REST, reads are
live over WebSocket.

**Deliverables.**

- `src/agi/halyard/state/schema/character_sheet.schema.json` —
  JSON Schema for the full sheet.
- `src/agi/halyard/state/schema/patch.py` — JSON-Patch
  validator, with field-level Keeper-restriction checks.
- `src/agi/halyard/state/store.py` — on-disk JSON storage, one
  file per PC, append-only log in `log.jsonl`.
- `src/agi/halyard/state/bridge.py` — NATS bridge: subscribes
  `halyard.sheet.*.patch`, applies, publishes `...update`.
- `src/agi/halyard/state/api/app.py` — FastAPI app:
  - `GET /api/sheets/<session>/<pc_id>`
  - `GET /api/sheets/<session>` — list
  - `POST /api/sheets/<session>/<pc_id>/patch` — HTTP entry
  - `WS  /ws/sheets/<session>` — live fan-out
- `deploy/systemd/halyard-state.service` — user-level unit.
- `deploy/docker/halyard-state/Dockerfile` — containerized
  version (portable to K8s if ever needed).
- Unit tests: ≥ 25 cases.
  - Schema validation (good and bad sheets).
  - Patch application (good / ill-formed / authorization fail).
  - Keeper-only write paths.
  - WS subscription and broadcast shape.
  - NATS bridge end-to-end with fake NATS.
  - Log append idempotence.

**Acceptance criteria.**

- Service starts via `systemctl --user start halyard-state`.
- An `httpx` integration test: two simulated clients, one
  patches, both see the update on the WS within 100 ms.
- A malformed patch is rejected with a structured error.
- A player-origin patch to a Keeper-restricted field is
  rejected with 403.
- Stress test: 100 patches/second sustained for 60 seconds, no
  dropped updates on the WS channels.

**Risks.**

- **Schema drift** between Python, JSON, and TypeScript client
  types — generate TS types from the JSON Schema in the web
  client sprint.
- **Concurrent writes** — single-writer semantics per PC per
  session keeps this simple. Multiple writers on the same PC
  at the same millisecond are an edge case we accept losing.

**Gate to Sprint 4.** State service running on Atlas, 24-hour
soak test in dry-run mode passes.

---

## Sprint 4 — halyard-web player client v0

**Status:** CODE COMPLETE; LIVE DRY-RUN PENDING KEEPER BACKEND.

*Next.js 14 App Router scaffold committed at `web/halyard-table/`.
412 npm deps, `npm run typecheck` clean, `npm run build` produces
the standalone bundle (session route: 244 kB first-load,
LiveKit-dominated). Components: VideoGrid (LiveKit React tiles,
AI-identity highlight), AiChatPanel × 2 (DataChannel-filtered
chat for ARTEMIS and SIGMA-4), CharacterSheetDrawer (live via
halyard-state WS, keyboard-toggled with `c`).
`lib/types.ts` mirrors the halyard-state JSON Schema; `lib/state.ts`
exposes `useSessionSheets()` backed by react-use-websocket.
Dockerfile + .dockerignore in `deploy/docker/halyard-web/`. End-to-
end live-play requires halyard-keeper-backend's token-mint
endpoint (Sprint 6) — until then the landing page shows a
structured error instead of an empty token.*

**Goal.** Players can visit `halyard.atlas-sjsu.duckdns.org`,
join a named session, see each other on video, and see ARTEMIS
replies arrive as chat.

**Deliverables.**

- `web/halyard-table/` — Next.js 14 App Router scaffold.
- `web/halyard-table/package.json` — pinned deps:
  - `next@^14`
  - `react@^18`
  - `livekit-client@^2`
  - `@livekit/components-react@^2`
  - `@livekit/components-styles@^1`
  - `tailwindcss@^3`
  - `react-use-websocket@^4`
- `web/halyard-table/app/layout.tsx` — global chrome (matches
  schematic.html aesthetic).
- `web/halyard-table/app/page.tsx` — landing (name, session,
  role).
- `web/halyard-table/app/session/[id]/page.tsx` — the player
  view.
- `web/halyard-table/components/VideoGrid.tsx` — LiveKit tiles
  with separate "AI NPC" tiles for ARTEMIS (and placeholder
  SIGMA-4).
- `web/halyard-table/components/ArtemisChat.tsx` — DataChannel-
  backed chat panel, filtered by `envelope.kind === "artemis.say"`.
- `web/halyard-table/lib/livekit.ts` — connection helper, token
  fetch from `/api/livekit/token`.
- `web/halyard-table/lib/types.ts` — TS types generated from the
  character sheet JSON Schema.
- `deploy/docker/halyard-web/Dockerfile` — multi-stage build,
  final image is `node:20-alpine` + built `.next/standalone`.
- `atlas-caddy` snippet: reverse-proxy
  `halyard.atlas-sjsu.duckdns.org` → container :3000.
- Token-minting route on the Keeper backend
  (`POST /api/livekit/token`) wired to `livekit_agent/token.py`.

**Acceptance criteria.**

- Cold visit to `halyard.atlas-sjsu.duckdns.org/session/test`
  from two separate browsers produces two video tiles with live
  A/V and ARTEMIS chat-panel message arrival within 5 seconds
  of a test publish.
- Page passes Lighthouse accessibility ≥ 90.

**Risks.**

- **Token minting in a cold dev loop** — need a fast local
  testing path that doesn't hit the real Atlas backend.
  Provide a `dev` flag in `livekit.ts` that mints tokens via a
  throwaway endpoint.
- **Safari / iPhone** compatibility — LiveKit supports it, but
  test explicitly on at least one iOS device.
- **Caddy reverse proxy for WS + HTTP on the same hostname** —
  route `/ws/*` to state service, `/api/*` to Keeper backend,
  everything else to the web container.

**Gate to Sprint 5.** Three players and the Keeper hold a
15-minute dry-run session with working video, audio, and AI
chat.

---

## Sprint 5 — halyard-web player client v1

**Status:** NOT STARTED.

**Goal.** Feature-complete player client: SIGMA-4 chat, live
character sheet drawer, dice roller, scene indicator.

**Deliverables.**

- `web/halyard-table/components/Sigma4Chat.tsx` — second chat
  panel for SIGMA-4, `envelope.kind === "sigma4.say"`.
- `web/halyard-table/components/CharacterSheet.tsx` — drawer
  that opens on keystroke `c`, shows the player's own sheet,
  subscribes to `/ws/sheets/<session>`.
- `web/halyard-table/components/DiceRoller.tsx` — local roller
  with `1d100`, bonus/penalty dice, and a "broadcast" button
  that publishes to the Keeper console.
- `web/halyard-table/components/SceneIndicator.tsx` — small
  top-bar indicator showing the current scene name.
- `web/halyard-table/components/EnvelopeRouter.tsx` — dispatcher
  for DataChannel envelopes to the right sub-component.
- Settings pane: mute AI, toggle chat sound, accessibility
  options.

**Acceptance criteria.**

- Sheet drawer shows current HP / SAN / Luck for the player's
  own PC and updates within 200 ms of a NATS patch.
- Dice roller computes correctly (verified by unit test).
- Scene indicator updates on `scene.trigger`.
- All components pass a11y audit.

**Risks.**

- **Mobile layout** — the sheet drawer needs to be usable on a
  phone. Test on real hardware.
- **Envelope routing complexity** — add unit tests for every
  `kind` value.

**Gate to Sprint 6.** Player client is done except for Keeper
affordances.

---

## Sprint 6 — halyard-keeper-backend + Keeper console v0

**Status:** BACKEND CODE COMPLETE; CONSOLE UI + NATS WIRING
PENDING SPRINT 7.

*Backend lands with 41 new unit tests (263 total in halyard suite).
aiohttp app factory with public `/healthz` + `/api/livekit/token`
and Keeper-only session lifecycle + approval queue routes
(HTTP Basic + IP allow-list). HTTP Basic uses `hmac.compare_digest`
for constant-time compare. Token-mint wraps the existing
`livekit_agent/token.py`. Session registry with 3-state transition
(open/paused/closed) + append-only JSONL log. Approval queue with
deterministic ids, per-session fan-out to WS listeners. The Keeper
console UI (app/keeper/[id]/page.tsx, ApprovalQueue / KillSwitchPanel
/ SessionControl components) is scoped into Sprint 7 since the
backend is usable from curl/manual test until then, and NATS
wiring (subscribing ARTEMIS/SIGMA `.say` with the approval flag)
is the same refactor mentioned in Sprint 2's deferrals.*

**Goal.** Keeper can log in, see pending AI replies, approve /
reject / edit them, and operate kill-switches.

**Deliverables.**

- `src/agi/halyard/keeper/app.py` — FastAPI app for Keeper
  operations.
- `src/agi/halyard/keeper/auth.py` — HTTP Basic for v1; IP
  allow-list.
- `src/agi/halyard/keeper/approvals.py` — in-memory pending
  queue per session, subscribed from NATS.
- `src/agi/halyard/keeper/sessions.py` — session lifecycle:
  create, start, end, pause.
- `src/agi/halyard/keeper/livekit.py` — JWT minting, room
  admin.
- `web/halyard-table/app/keeper/[id]/page.tsx` — Keeper view,
  gated by basic-auth reverse-proxy rule in Caddy.
- `web/halyard-table/components/ApprovalQueue.tsx` — two
  queues, ARTEMIS and SIGMA-4; each item shows the proposed
  reply, the triggering turn, validator proof summary, and
  three buttons: ✓ release, ✗ drop, ✏︎ edit.
- `web/halyard-table/components/KillSwitchPanel.tsx` — three
  buttons: silence ARTEMIS, silence SIGMA-4, silence both.
- `web/halyard-table/components/SessionControl.tsx` — start,
  end, pause.
- Unit tests: ≥ 20 cases across auth, approvals queue,
  session lifecycle, JWT minting.

**Acceptance criteria.**

- Keeper login works; unauthorized routes 401.
- ARTEMIS reply arrives in the approval queue within 500 ms of
  `artemis.say` publish (with approval-required flag).
- ✓ release publishes the reply to DataChannel; players see it.
- ✗ drop logs the rejection and silences that turn.
- Silence button suppresses ARTEMIS for the rest of the session
  until explicit resume.
- Full E2E test: a fake turn → ARTEMIS reply → Keeper approves
  → player DataChannel receives → assertion passes.

**Risks.**

- **Latency under load** — the approval gate adds a human
  round-trip to every AI reply. v1 accepts this; v2 may
  calibrate.
- **Keeper-side keybindings** — three-key shortcuts for
  ✓ / ✗ / ✏︎ are important; test with a real keyboard.

**Gate to Sprint 7.** Keeper console is usable for a full dry-run
session; feedback collected from Keeper.

---

## Sprint 7 — Keeper console v1

**Status:** NOT STARTED.

**Goal.** Stat overrides, scene triggers, dice broadcast,
recording control.

**Deliverables.**

- `web/halyard-table/components/StatOverride.tsx` — per-PC
  panel with every sheet field editable. Writes via
  `POST /api/sheets/<session>/<pc_id>/patch` with
  `author=keeper`.
- `web/halyard-table/components/SceneTrigger.tsx` — dropdown
  of pre-configured scenes; publishes
  `agi.rh.halyard.scene.trigger`.
- `web/halyard-table/components/DiceBroadcast.tsx` — roll a
  dice expression and publish the result to all players.
- `web/halyard-table/components/RecordingToggle.tsx` —
  start / stop LiveKit Egress recording.
- `web/halyard-table/components/OverrideLog.tsx` — Keeper-
  visible history of all overrides (with reason).
- Pre-configured scene library:
  `web/halyard-table/data/scenes.json` — all the major scenes
  from the Campaign Guide's chapter 12 pacing table.
- LiveKit Egress service added to docker-compose (Sprint 1
  container can be extended without disruption).

**Acceptance criteria.**

- Keeper can change any PC's SAN, HP, Luck, or arbitrary field,
  and the player's sheet drawer reflects it within 200 ms.
- Scene triggers appear in the SceneIndicator and are
  acknowledged by both AIs in their context.
- Dice broadcasts appear in the DataChannel and are rendered
  distinctly in the chat panes.
- Recording toggles write a file under
  `/archive/halyard/recordings/<session>/`.

**Risks.**

- **Scene-context integration with AIs** — the AIs need to
  receive the scene.trigger as a context update before the next
  turn. Requires a small change to the ARTEMIS and SIGMA-4
  `context.py` to read the current scene.
- **Egress resource usage** — recording a video pushes Atlas's
  disk I/O and ffmpeg CPU; verify sustained performance.

**Gate to Sprint 8.** Feature-complete Keeper console; Keeper
signs off on usability.

---

## Sprint 8 — Integration, observability, dry-run, runbook

**Status:** NOT STARTED.

**Goal.** The whole system is green, observable, documented, and
ready for Session One.

**Deliverables.**

- `tests/integration/test_halyard_e2e.py` — end-to-end test:
  - Spin up LiveKit + NATS + state + keeper backend + both AI
    agents in Docker Compose.
  - Simulate 20 turns of player speech via Whisper's
    streaming-transcription-mock mode.
  - Assert: AI replies arrive, are approved via a mocked
    Keeper, character sheets updated correctly, session record
    written.
- `infra/local/atlas-chat/schematic.html` — add the Halyard
  Table tile with all metrics from `HALYARD_TABLE.md` §9.
- `docs/HALYARD_KEEPER_RUNBOOK.md` — operator manual for the
  Keeper:
  - Session prep.
  - Starting a session.
  - Mid-session operations.
  - Ending a session.
  - Troubleshooting.
  - Sealed envelope procedure.
- `docs/HALYARD_PLAYER_ONBOARDING.md` — one-pager for players:
  how to join, browser requirements, safety-tool UX.
- Load test: 5 concurrent players + Keeper + both AIs for 2
  hours, no dropped packets, no service restarts.
- **Dry-run session**: 1-hour live play with two human players
  and the Keeper, full system, no scripted interventions.

**Acceptance criteria.**

- E2E test passes in CI.
- Dashboard tile renders all metrics correctly.
- Runbook is followed by the Keeper in a dry-run without
  consulting the engineer.
- Dry-run session completes without a system-level incident.

**Risks.**

- **CI environment for LiveKit** — LiveKit needs UDP ports
  that GitHub Actions does not expose. Plan: run the E2E test
  with a `livekit-mock` or gate it to `pytest -m e2e_local`.
- **Real-world networking** — the dry-run is the first time the
  full stack is exercised end-to-end; fix-forward is expected.

**Gate to production use.** Session One is scheduled; Keeper
signs off on go/no-go; sealed envelope prepared.

---

## Post-sprint backlog (not scheduled)

- **EPU FPGA validator swap** for both AIs.
- **v2 proactive trigger** policy (relevance classifier).
- **TTS voice** for one or both AIs (uncanny-valley
  considerations apply).
- **Session-record at-rest encryption.**
- **Nightly backup of session archives** to NRP persistent
  storage.
- **Multi-session concurrency** (if ever needed).
- **Mobile-first redesign** of the player client (if tables
  routinely play from phones).

---

## Working conventions

### Branch and PR model

- One sprint = one branch = one PR.
- Branch name: `halyard/sprint-NN` (0-padded).
- Squash-merge to main once acceptance criteria green.
- Rebase on main before squash-merging.

### Commit message style

```
<type>(<scope>): <subject, imperative, ≤72 chars>

<body — why, not what; wrap at 72>

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`,
`perf`, `ci`.

Scopes: `halyard`, `halyard-state`, `halyard-web`,
`halyard-keeper`, `sigma4`, `artemis`, `infra`, `docs`.

### Local CI gates

Before push:
1. `ruff check src/agi/halyard`.
2. `pytest tests/unit/halyard`.
3. `pytest tests/integration -m "not e2e_local"` (E2E gated).
4. For web client: `pnpm -C web/halyard-table typecheck &&
   pnpm -C web/halyard-table lint`.

### Definition of done

- All acceptance criteria green.
- Unit + integration tests written.
- Documentation updated (`HALYARD_TABLE.md` if the contract
  changed; runbook if operator-visible; inline docstrings for
  non-obvious code).
- Commit messages clean.
- No TODO left behind without a GitHub issue filed.

---

*Sprint status is canonical only in this file. Do not track
sprint state in the README, in GitHub Issues, or in commit
messages — they drift. This file is single-writer: whoever is
currently running a sprint updates this file on completion and
commits it alongside the sprint's closing PR.*
