# HALYARD TABLE

> *"The crew on the bridge, the AIs on the handheld and in the walls,
> the character sheets on a clipboard beside the Keeper's elbow —
> all one system, all one session."*

**Halyard Table** is the live-play runtime for the *Beyond the
Heliopause* Call of Cthulhu campaign. It is a self-hosted Zoom-style
conference room, an in-client interface to the campaign's AI NPCs,
a live character-sheet service, and a Keeper console — all served
from Atlas, all stitched together through the Atlas NATS fabric.

This document is the plan-of-record. It is the companion to
[`ARTEMIS.md`](ARTEMIS.md) — ARTEMIS is the first AI NPC in the
Halyard Table, but the table is bigger than any single NPC. Read
`ARTEMIS.md` first if you have not; many of the safety, validator,
and NATS conventions there carry through here unchanged.

---

## 1 · Scope

Halyard Table is:

- **A self-hosted conference server.** LiveKit SFU on Atlas,
  TLS-fronted by the existing `atlas-caddy`, TURN-served for NATs-
  traversal. The table can seat a Keeper, 3–5 players, and two AI
  NPCs in a single session.
- **An in-client interface to the ship AIs.** Two AI NPCs run as
  LiveKit "participants" with text-only output into the room's
  DataChannel: **ARTEMIS** (the handheld, Protogen research
  assistant) and **SIGMA-4** (the ship-mind of the MKS *Halyard*).
  Both are driven by NRP-hosted LLMs through the Primer substrate,
  gated by the ErisML validator, logged with SHA-chained
  DecisionProofs.
- **A live character-sheet service.** Player character sheets are
  JSON-backed, Git-trackable, and broadcast live over NATS. Stat
  changes — Sanity loss, Luck spend, HP damage, skill improvement
  checks — propagate to all seated players and to the Keeper
  console in near-real-time.
- **A Keeper console.** A separate mode of the same web client
  that exposes: approval queues for both AIs, kill-switches, per-PC
  stat overrides, scene triggers, dice-roll broadcast, session
  start/stop, and recording control.
- **A session record.** Every Halyard Table session writes a
  complete audit trail — transcripts, AI replies, validator
  proofs, stat deltas, Keeper actions — suitable for post-session
  replay, sealed-envelope verification, and campaign continuity.

**Halyard Table is not:**

- A general-purpose videoconference product. It runs one specific
  campaign. Everything is tuned for that campaign.
- The AIs themselves. The AIs live in their own packages under
  `src/agi/primer/artemis/` and `src/agi/halyard/sigma4/`. The
  Halyard Table is the room they speak in.
- The Primer. The Primer teaches Erebus. Halyard Table hosts a
  game. They share substrate (NATS, vMOE, the validator
  architecture) but not purpose.
- A replacement for the human Keeper. The Keeper is always in
  authority. Every AI reply, every stat override, every scene
  trigger is either Keeper-initiated or Keeper-approved.

---

## 2 · Design Principles

Halyard Table inherits the five ARTEMIS principles and adds two
specific to a multi-participant live-play setting:

1. **Cortex = frontier LLMs (NRP), subcortical = local GPUs.**
   ARTEMIS and SIGMA-4 are NRP-hosted. Whisper ASR runs on a
   single burst A10. LiveKit SFU runs on Atlas itself (cheap,
   self-hosted, no per-minute billing).
2. **Global workspace = NATS.** All cross-component communication
   is NATS subjects. No direct RPC between components. Every
   subject is documented; every payload is versioned.
3. **Multiple agents, explicit policy.** Two AI NPCs with distinct
   persona, validator rules, and bible splits. They do not talk
   to each other except through the room.
4. **Verified before published.** No AI reply reaches the table
   without passing the validator. No character-sheet write reaches
   the players without passing the schema gate.
5. **Never surprise the Keeper.** Every AI turn is held for Keeper
   approval by default in v1. Kill-switches are three-layered
   (Keeper hotkey, safety-tool flag, validator-streak cooldown).
6. **Session = room.** A Halyard Table session is identified by
   its LiveKit room name, which is the `session_id` end-to-end.
   NATS subjects include the session_id as a payload filter, so
   the same fabric can host multiple simultaneous sessions without
   cross-contamination. *(In practice we expect one active session
   at a time — this isolation is belt-and-braces.)*
7. **The record is the artifact.** Each session produces a
   deterministic, replayable record. Post-session debugging,
   sealed-envelope verification, and future-campaign continuity
   all depend on the record being complete and tamper-evident.

---

## 3 · System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      PLAYER / KEEPER BROWSERS                      │
│                                                                    │
│   Next.js client (web/halyard-table/)                              │
│   ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│   │ Video tiles    │  │ ARTEMIS chat   │  │ Character sheet    │   │
│   │ (LiveKit SDK)  │  │ SIGMA-4 chat   │  │ drawer (live)      │   │
│   └────────┬───────┘  └────────┬───────┘  └─────────┬──────────┘   │
│            │ WebRTC            │ DataChannel        │ WebSocket    │
└────────────┼───────────────────┼────────────────────┼──────────────┘
             │                   │                    │
             ▼                   ▼                    ▼
┌─────────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ LIVEKIT SFU         │  │                 │  │ halyard-state       │
│ (Atlas self-hosted) │  │                 │  │ (WebSocket bridge)  │
│ :7880 WS / :443 TLS │  │                 │  │ :8090               │
│ :7881 TCP / :50000- │  │                 │  │                     │
│  60000 UDP turn     │  │                 │  │  Subscribes NATS    │
│                     │  │                 │  │  agi.rh.halyard.    │
│  Rooms:             │  │                 │  │  sheet.*; pushes    │
│   halyard-sNN       │  │                 │  │  updates to clients │
└──────────┬──────────┘  │                 │  └──────────┬──────────┘
           │ audio tracks│                 │             │
           ▼             │                 │             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ATLAS NATS FABRIC (:4222)                     │
│                                                                     │
│   agi.rh.artemis.heard  /  agi.rh.artemis.say                       │
│   agi.rh.sigma4.heard   /  agi.rh.sigma4.say                        │
│   agi.rh.halyard.sheet.<pc_id>.{update,patch}                       │
│   agi.rh.halyard.session.{start,end,tick,silence}                   │
│   agi.rh.halyard.scene.{trigger,cue}                                │
│   agi.rh.halyard.keeper.{approve,reject,override,dice}              │
└──────────────┬────────────────────┬─────────────────┬───────────────┘
               │                    │                 │
               ▼                    ▼                 ▼
┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│ artemis-livekit-    │  │ atlas-primer +   │  │ halyard-state-      │
│ agent               │  │ artemis_mode     │  │ service             │
│ (joins room as      │  │ (NRP LLM calls,  │  │ (JSON sheets on     │
│  "ARTEMIS";         │  │  validator,      │  │  disk; CRUD NATS    │
│  Whisper → NATS;    │  │  DecisionProof)  │  │  + WS fan-out)      │
│  NATS → DataChannel)│  │                  │  │                     │
│                     │  │ sigma4_mode (new,│  │                     │
│                     │  │  parallel logic) │  │                     │
└─────────────────────┘  └──────────────────┘  └─────────────────────┘
               │                    │
               ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          NRP ellm GATEWAY                           │
│   https://ellm.nrp-nautilus.io/{v1,anthropic}                       │
│   Kimi-1T (long context) · Qwen-3 (short-turn cascade)              │
└─────────────────────────────────────────────────────────────────────┘
```

### The single source of truth for every flow

| Flow                          | Origin                  | Sink(s)                 |
|-------------------------------|-------------------------|-------------------------|
| Player / Keeper speech        | Browser mic             | Whisper → `artemis.heard`, `sigma4.heard` |
| AI reply                      | Atlas primer service    | `*.say` → DataChannel → Browser |
| Character stat change         | Keeper console or rule  | `halyard.sheet.*` → WS → Browsers |
| Scene trigger                 | Keeper console          | `halyard.scene.*` → AI context, SFX, sheet hooks |
| Dice broadcast                | Any client, Keeper-auth | `halyard.keeper.dice` → WS → Browsers |
| Kill-switch                   | Keeper console hotkey   | `halyard.session.silence` → all AIs |

---

## 4 · Components

### 4.1 LiveKit SFU (Selective Forwarding Unit)

Self-hosted [LiveKit](https://livekit.io) server running on Atlas.
LiveKit is the open-source WebRTC SFU that replaces Zoom for the
table. It handles audio and video routing between participants
and exposes a DataChannel for text side-traffic (which is where
the AI replies land).

- **Container**: official `livekit/livekit-server` image, deployed
  via docker-compose.
- **Port mapping**: `:7880` (WebSocket signaling, fronted by
  `atlas-caddy` at `:443`), `:7881` TCP for RTC fallback,
  `:50000-60000/udp` for RTC media.
- **TURN**: `coturn` sidecar on `:3478`; same ports fronted by
  Caddy for TLS where relevant.
- **Auth**: API-key / API-secret, with JWT tokens minted by the
  Halyard Table backend (one JWT per participant per session,
  short-lived, room-scoped).
- **Rooms**: one active room per session, named `halyard-sNN`
  (session number). Auto-expire after 24 hours of inactivity.
- **Recording**: optional Egress container writes a per-session
  WebM to `/archive/halyard/recordings/<session_id>/` if enabled.
- **Operator**: `systemd --user` unit `livekit-sfu.service` on
  Atlas; logs to `journalctl --user -u livekit-sfu`.

### 4.2 ARTEMIS (existing)

Already implemented. See [`ARTEMIS.md`](ARTEMIS.md).

For Halyard Table purposes, ARTEMIS is a black box that:
- Subscribes to `agi.rh.artemis.heard`.
- Publishes on `agi.rh.artemis.say`.
- Obeys `agi.rh.halyard.session.silence` as a kill-switch.
- Participates in the room as the LiveKit participant identity
  `artemis` with display name `ARTEMIS`.

### 4.3 SIGMA-4

The *Halyard*'s ship-mind. A new AI NPC, built as a near-twin of
ARTEMIS with distinct persona, bible split, and validator rules.

**Key differences from ARTEMIS:**

| Axis                   | ARTEMIS                         | SIGMA-4                              |
|------------------------|---------------------------------|--------------------------------------|
| In-fiction role        | Protogen handheld AI            | Ship-mind of the MKS *Halyard*       |
| Voice                  | Terse, clinical-but-warm        | Measured, formal, UNN-officer cadence|
| Knows                  | Halverson's private files, some protomolecule-adjacent data | Full ship systems, 28 years of ship logs, the carrier-frequency traffic |
| Does not know          | Anything outside Halverson's handheld | Contents of the captain's sealed case, Chamber's true history |
| Can volunteer          | Yes — ARTEMIS has drifted and will offer | No — SIGMA answers direct questions, rarely volunteers |
| Trigger policy (v1)    | Name-mention + Keeper cue       | Name-mention + Keeper cue + "SIG"/"Sig" diminutive |
| NATS lane              | `agi.rh.artemis.*`              | `agi.rh.sigma4.*`                    |
| LiveKit identity       | `artemis`                       | `sigma-4`                            |

**Package**: `src/agi/halyard/sigma4/` — same module shape as
`src/agi/primer/artemis/` (mode, prompt, context, trigger,
validator, nats_handler). The bible split, prompt text, and
persona configuration are in the SIGMA-4 package; the substrate
(vMOE, LLM routing, validator framework) is shared with ARTEMIS.

### 4.4 halyard-state-service

A small service that owns live character-sheet state for a
session and fans it out to all clients.

**Responsibilities:**

- Hold the canonical character sheets for the active session in
  JSON, one file per PC under
  `/archive/halyard/sheets/<session_id>/<pc_id>.json`.
- Accept writes over NATS (`agi.rh.halyard.sheet.<pc_id>.patch`).
- Broadcast updates (`agi.rh.halyard.sheet.<pc_id>.update`).
- Serve the current state to newly-connected clients via HTTP
  (`GET /api/sheets/<session_id>/<pc_id>`).
- Push live updates over WebSocket
  (`WS /ws/sheets/<session_id>`).
- Maintain an append-only log of all stat changes per session
  under `/archive/halyard/sheets/<session_id>/log.jsonl` for
  post-session replay.

**Schema**: a JSON-Schema-governed character sheet with sections
for identity, characteristics, derived stats, skills, Bonds,
current status (HP/SAN/Luck/MP/temporary conditions), equipment,
and campaign-specific extensions (faction loyalty, chassis type,
personal hook). The schema lives in
`src/agi/halyard/state/schema/character_sheet.schema.json`.

**Patch format**: JSON-Patch (RFC 6902) applied server-side, with
Keeper-authorization required for any write to a
Keeper-restricted field (faction loyalty, chassis secrets, hidden
hooks). A player client can write to its own character's
public-visible fields (HP spent in combat, Luck spent on a roll)
but not to another character's sheet and not to its own
Keeper-restricted fields.

**Package**: `src/agi/halyard/state/` with FastAPI routes under
`src/agi/halyard/state/api/` and the NATS bridge under
`src/agi/halyard/state/bridge.py`.

### 4.5 halyard-web (player client)

A Next.js web application that is the player- and Keeper-facing
interface to the Halyard Table.

**Routes:**

- `/` — session landing page (enter name, session ID, role).
- `/session/<session_id>` — the player view.
- `/keeper/<session_id>` — the Keeper view (separate auth).
- `/about` — documentation and attribution.

**Player view components:**

- **VideoGrid** — LiveKit React SDK tiles for every human
  participant plus two AI "avatar" tiles for ARTEMIS and SIGMA-4.
- **ArtemisChat** and **Sigma4Chat** — separate chat panels for
  each AI, subscribing to the LiveKit DataChannel filtered by
  `envelope.kind`.
- **CharacterSheet** — a drawer showing the current player's own
  sheet, live-updated via WebSocket.
- **DiceRoller** — a local dice roller with "broadcast" option
  that publishes to `agi.rh.halyard.keeper.dice` (Keeper can
  moderate).
- **SceneIndicator** — shows the current scene/chapter, cued by
  the Keeper.

**Keeper view components** — on top of the player view:

- **ApprovalQueue** — two panels, one per AI. Pending replies
  shown with "✓ release" / "✗ drop" / "✏︎ edit" buttons.
- **KillSwitch** — three buttons: silence ARTEMIS, silence
  SIGMA-4, silence both.
- **StatOverride** — per-PC panel for direct Keeper writes to
  any character sheet field, logged with reason.
- **SceneTrigger** — dropdown of pre-configured scene cues that
  publish `agi.rh.halyard.scene.trigger`.
- **DiceBroadcast** — broadcast a roll with result visible to
  all players.
- **SessionControl** — start session, end session, pause, save
  record.
- **RecordingControl** — toggle Egress recording.

**Tech**:
- Next.js 14 (App Router) + React 18.
- `livekit-client` and `@livekit/components-react` for WebRTC.
- `react-use-websocket` for halyard-state WS.
- Tailwind CSS (matches existing `atlas-chat/schematic.html`
  aesthetic — dark theme, monospace accents).

**Deployment**: `deploy/docker/halyard-web/` Dockerfile, served
from Atlas behind `atlas-caddy` at a dedicated subdomain like
`halyard.atlas-sjsu.duckdns.org`.

### 4.6 halyard-keeper-backend

A small FastAPI service that backs the Keeper console's
server-side concerns.

**Responsibilities:**

- Mint LiveKit participant JWTs (delegates to the existing
  `livekit_agent/token.py`).
- Authenticate the Keeper (single-user HTTP basic in v1;
  session-based OAuth in v2).
- Authorize Keeper actions before bridging them to NATS.
- Serve the approval queue's underlying state (pending replies,
  DecisionProofs, latency metrics).
- Persist session records and session configuration.

**Package**: `src/agi/halyard/keeper/` with FastAPI app in
`src/agi/halyard/keeper/app.py`.

### 4.7 Existing pieces we reuse

| Component              | Role                                                |
|------------------------|-----------------------------------------------------|
| `atlas-caddy`          | TLS termination + reverse proxy for all HTTP(S)     |
| `atlas-nats`           | NATS server (already running on :4222)              |
| `atlas-primer.service` | Runs ARTEMIS (and, after Sprint 2, SIGMA-4) reasoning |
| `atlas-chat/schematic.html` | Ops dashboard — adds a Halyard Table tile       |
| `vMOE` + `validator`   | Shared substrate across ARTEMIS and SIGMA-4         |

---

## 5 · Interface Contracts

### 5.1 NATS subjects

All subjects under `agi.rh.` namespace. Payloads are UTF-8 JSON
unless noted.

#### Existing (ARTEMIS, unchanged)

- `agi.rh.artemis.heard` — transcript in.
- `agi.rh.artemis.say` — reply out.

#### New (SIGMA-4, parallel to ARTEMIS)

- `agi.rh.sigma4.heard` — same schema as `artemis.heard`.
- `agi.rh.sigma4.say` — same schema as `artemis.say`.

#### New (Halyard Table session)

- `agi.rh.halyard.session.start`
  ```json
  { "session_id": "halyard-s01", "started_at": 1714934400, "roster": [...] }
  ```
- `agi.rh.halyard.session.end`
  ```json
  { "session_id": "halyard-s01", "ended_at": 1714938000, "reason": "keeper_stop" }
  ```
- `agi.rh.halyard.session.tick` — 1/second heartbeat from
  halyard-state for clock sync.
- `agi.rh.halyard.session.silence` — global AI kill-switch; any
  publish silences all AIs until `session.resume`.
- `agi.rh.halyard.session.resume` — release silence.

#### New (Halyard Table character sheet)

- `agi.rh.halyard.sheet.<pc_id>.patch` — a JSON-Patch write,
  intended consumer is halyard-state.
  ```json
  {
    "session_id": "halyard-s01",
    "pc_id":      "imogen-roth",
    "author":     "keeper" | "player:<pc_id>" | "system",
    "patch":      [{"op": "replace", "path": "/status/san/current", "value": 58}],
    "reason":     "SAN 1D4 loss — Bailey diary",
    "ts":         1714935200
  }
  ```
- `agi.rh.halyard.sheet.<pc_id>.update` — the post-patch state,
  broadcast to all listeners. Same envelope minus `patch`, plus
  `state: {... full sheet ...}`.

#### New (Halyard Table scene / Keeper actions)

- `agi.rh.halyard.scene.trigger` — Keeper cues a pre-configured
  scene.
  ```json
  { "session_id": "...", "scene_id": "ch1-mi_go_contact", "note": "..." }
  ```
- `agi.rh.halyard.scene.cue` — a smaller in-scene prompt (e.g.,
  environmental change, ambient audio cue).
- `agi.rh.halyard.keeper.approve` / `.reject` / `.edit` — actions
  on pending AI replies.
- `agi.rh.halyard.keeper.override` — direct stat override (maps
  to a `sheet.patch`).
- `agi.rh.halyard.keeper.dice` — a Keeper-broadcast dice roll.

### 5.2 REST endpoints (halyard-state + halyard-keeper)

- `GET /api/sessions` — list known sessions.
- `GET /api/sessions/<session_id>` — session metadata and roster.
- `POST /api/sessions` — create a session (Keeper-only).
- `POST /api/sessions/<session_id>/end` — end a session.
- `GET /api/sheets/<session_id>` — list all sheets.
- `GET /api/sheets/<session_id>/<pc_id>` — full sheet JSON.
- `POST /api/sheets/<session_id>/<pc_id>/patch` — apply a
  JSON-Patch (authenticated; Keeper or sheet owner).
- `POST /api/livekit/token` — mint a LiveKit JWT for a named
  participant in a named room (server-enforced expiry).
- `GET /api/keeper/approvals/<session_id>` — the Keeper's
  pending-reply queue.

### 5.3 WebSocket channels

- `WS /ws/sheets/<session_id>` — subscribes to all sheet updates
  for the session.
- `WS /ws/session/<session_id>` — session-level events (ticks,
  silences, scene triggers, dice broadcasts).
- `WS /ws/keeper/<session_id>` — Keeper-only channel with the
  approval queue, validator telemetry, and override log. Auth
  via short-lived token from the keeper backend.

### 5.4 LiveKit DataChannel envelopes

All DataChannel frames are UTF-8 JSON objects with a `kind`
discriminator:

```json
{ "kind": "artemis.say",  "text": "...", "turn_id": "...", "proof_hash": "...", "ts": 1714935400 }
{ "kind": "sigma4.say",   "text": "...", "turn_id": "...", "proof_hash": "...", "ts": 1714935410 }
{ "kind": "scene.trigger","scene_id":"...","note":"...","ts": 1714935420 }
{ "kind": "dice.roll",    "author":"keeper","expr":"1d100","result":47,"ts":1714935430 }
```

---

## 6 · Session Lifecycle

```
  ┌──────────────────────────────────────────────────────────────┐
  │  PREP                                                        │
  │   Keeper creates session via POST /api/sessions              │
  │   Roster, scene order, recording pref set                    │
  │   System pre-loads character sheets                          │
  └─────────┬────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  OPEN                                                        │
  │   LiveKit room created, JWTs minted                          │
  │   ARTEMIS agent + SIGMA-4 agent joined                       │
  │   Players join via emailed session link                      │
  │   agi.rh.halyard.session.start published                     │
  └─────────┬────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  LIVE                                                        │
  │   Human speech → Whisper → NATS heard                        │
  │   AIs reply via Keeper-approved queue → DataChannel          │
  │   Stat changes → sheet.patch → sheet.update → WS → clients   │
  │   Scene triggers from Keeper → scene.trigger → AI context    │
  │   Dice broadcasts → keeper.dice → DataChannel                │
  │                                                              │
  │   (For as long as the session lasts — typically 3–4 hours)   │
  └─────────┬────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  CLOSE                                                       │
  │   Keeper ends session                                        │
  │   agi.rh.halyard.session.end published                       │
  │   AIs disconnect cleanly                                     │
  │   Recording finalizes if enabled                             │
  │   Session record (transcripts + proofs + sheet-log) sealed   │
  └─────────┬────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  ARCHIVE                                                     │
  │   /archive/halyard/<session_id>/                             │
  │   ├─ recording.webm (optional)                               │
  │   ├─ transcripts.jsonl                                       │
  │   ├─ artemis/ (proofs, replies)                              │
  │   ├─ sigma4/  (proofs, replies)                              │
  │   ├─ sheets/                                                 │
  │   │    ├─ <pc_id>.json (final state)                         │
  │   │    └─ log.jsonl (all patches in order)                   │
  │   └─ metadata.json                                           │
  └──────────────────────────────────────────────────────────────┘
```

---

## 7 · Deployment Topology

### On Atlas

| Service                  | Port     | Systemd / Container          | TLS (Caddy)              |
|--------------------------|----------|------------------------------|--------------------------|
| atlas-caddy              | 443      | systemd                      | terminates for all below |
| atlas-nats               | 4222     | systemd                      | NATS internal            |
| atlas-primer.service     | internal | systemd                      | —                        |
| livekit-sfu              | 7880/443 | docker-compose user-level    | `halyard.atlas-sjsu...`  |
| livekit-turn (coturn)    | 3478     | docker-compose               | —                        |
| halyard-state-service    | 8090     | systemd --user               | `halyard-state.atlas...` |
| halyard-keeper-backend   | 8091     | systemd --user               | `halyard-keeper.atlas...`|
| halyard-web              | 3000     | docker container             | `halyard.atlas-sjsu...`  |
| artemis-livekit-agent    | —        | systemd --user, per session  | —                        |
| sigma4-livekit-agent     | —        | systemd --user, per session  | —                        |

The agent processes are launched on demand when a session opens
and torn down when the session closes. They are not always-on.

### On NRP

| Service                  | Path                        |
|--------------------------|-----------------------------|
| Kimi 1T (long context)   | `ellm.nrp-nautilus.io/anthropic` or `/v1` |
| Qwen 3 (short turn)      | same                        |
| EPU FPGA (future)        | Coder-workspace-pending     |

---

## 8 · Security Model

### 8.1 Authentication

- **Players**: a session link with an embedded **session token**
  (short-lived, single-use on first load; exchanged server-side
  for a LiveKit JWT). Session tokens are minted by the Keeper
  backend.
- **Keeper**: HTTP Basic in v1, sealed with a very strong
  password and IP-scoped to the Keeper's home IP / Tailscale
  address. Upgraded to session-based OAuth in a later phase.
- **AI agents**: LiveKit JWTs minted with restricted grants
  (`can_publish=false`, `can_publish_data=true`). The JWTs are
  held only by the agent process and never leave Atlas.

### 8.2 Authorization

- **Sheet writes** are authorized at the state service:
  - Keeper may write anything.
  - Player may write only their own PC's "player-writable"
    fields (current HP, current Luck, current SAN, certain
    counters).
  - "Keeper-restricted" fields (faction loyalty, hidden hooks,
    secret backstory) can never be written by a player.
- **Scene triggers** and dice broadcasts from the Keeper console
  require a valid Keeper session.
- **AI kill-switches** are Keeper-only.

### 8.3 Safety

- **Validator** gates every AI reply. See `ARTEMIS.md` §7.
- **Safety flag** — any non-null `meta.safety_flag` on a turn
  payload forces the AIs to silence. The web UI does not
  surface a control for this; the mechanism is reserved for
  keeper-side or programmatic use.
- **Kill-switches**: three independent layers (Keeper hotkey,
  programmatic safety flag, validator-streak cooldown).
- **Never-surprise invariant**: every AI reply is held for
  Keeper approval in v1. v2 may relax to a calibrated
  trust-after-sessions-N policy.

### 8.4 Sealed record

Every session produces a hash-chained DecisionProof chain (one
per AI). The chain's head is printed and sealed at session end
and may be included in the campaign's sealed envelope (see
`ARTEMIS.md` §9 / `project_nithon_coc_campaign.md` in memory).

---

## 9 · Observability

- All services emit structured JSON logs via the existing Atlas
  logging conventions.
- A new **Halyard Table panel** is added to
  `infra/local/atlas-chat/schematic.html`, showing:
  - Active session (ID, start time, participant count).
  - AI turn rate (per minute, both AIs).
  - Validator pass rate (both AIs).
  - Pending approvals queue depth.
  - Current kill-switch state.
  - Current recording state.
- Prometheus metrics on the FastAPI services' `/metrics`
  endpoints:
  - `halyard_session_active`
  - `halyard_artemis_turns_total`, `halyard_sigma4_turns_total`
  - `halyard_validator_pass_total`, `..._fail_total`
  - `halyard_sheet_patches_total`
  - `halyard_ws_connections`

---

## 10 · Open Questions

Deferred to sprint-level decisions or the Keeper's call:

1. **Recording default** — on or off for regular sessions? Off
   by default; opt-in via the recording toggle.
2. **Proactive AI triggers** — when (if ever) to enable the
   "relevance classifier" v2 trigger for either AI. Likely
   after 5+ sessions.
3. **Player custom soundboards** — players triggering in-fiction
   SFX from their side. Deferred.
4. **Multi-session runtime** — we design for isolation, but we
   do not plan to run two sessions concurrently. Revisit if
   ever needed.
5. **TTS voice for AIs** — deferred. Text-only via DataChannel
   for the foreseeable future; handheld-typing is in-fiction
   and easier to validate.
6. **Session-record encryption at rest** — session archives are
   currently filesystem-only on Atlas. A future sprint adds
   at-rest encryption with a Keeper-held key.
7. **Backup strategy** — Atlas disks are local; session records
   should be mirrored to NRP persistent storage or a cloud
   backup nightly. Deferred.

---

## 11 · Phased Delivery

See [`HALYARD_SPRINT_PLAN.md`](HALYARD_SPRINT_PLAN.md) for the
sprint-by-sprint execution plan. In summary:

- **Sprint 0** — Plan + scaffolding (this commit).
- **Sprint 1** — LiveKit SFU on Atlas.
- **Sprint 2** — SIGMA-4 skeleton.
- **Sprint 3** — halyard-state service.
- **Sprint 4** — halyard-web player client v0.
- **Sprint 5** — halyard-web v1 (SIGMA chat, character drawer, dice).
- **Sprint 6** — halyard-keeper-backend + Keeper console v0.
- **Sprint 7** — Keeper console v1 (overrides, scene triggers, dice broadcast).
- **Sprint 8** — Integration, observability, dry-run, runbook.

Each sprint ends with green tests, a merged PR, and a runnable
system. No sprint is declared done until the next can be started
cleanly.

---

## 12 · Kill-switch for this document

If at any phase gate the answer is "this is a worse idea than I
thought," Halyard Table can be shelved cleanly:

- No production Atlas service is modified in Sprints 0–2 (all
  new files under `src/agi/halyard/` and `docs/HALYARD_*`).
- Sprint 1 (LiveKit SFU) is a self-contained docker-compose;
  deletion is `docker-compose down -v`.
- Sprint 3 (halyard-state) is a new service; systemctl stop and
  delete the unit files.
- Sprints 4–7 (web client, Keeper console) are all subpaths of
  `atlas-caddy`; removing the site blocks is a Caddy-reload.

Revert is always a valid outcome.

---

*"The table is a room. The room is a session. The session is a
record. The record is the campaign."*

— engineering maxim, Halyard Table, Sprint 0.
