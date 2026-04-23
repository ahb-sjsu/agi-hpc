# ARTEMIS Avatar Roadmap

This document plans the evolution of the ARTEMIS in-room presence from
the scaffolded Phase 3 LiveKit agent into the production vision: a
browser-rendered humanoid 3D avatar + custom game-table UI + on-demand
player artifacts.

Companion to [`ARTEMIS.md`](ARTEMIS.md). The Phase 1 handler, Phase 2
NATS wiring, and Phase 3 LiveKit-agent skeleton are already on `main`.
This roadmap picks up from there.

---

## Status (2026-04-23)

**Already landed:**
- Phase 1 — offline `handle_turn`, validator, DecisionProof chain (72 unit tests)
- Phase 2 — `ArtemisService` NATS handler, systemd unit, 20-turn e2e
- Phase 3 — LiveKit agent package skeleton, JWT minting, Dockerfile
  scaffolding

**Experiment running tonight on Atlas:**
- v4 avatar agent (`artemis_avatar.py`) publishes an icosahedron-HUD
  video track + Piper TTS audio + canned-response responder. Out-of-tree
  under `/home/claude/` pending the Phase 3.5 commit below.

**Validated end-to-end:**
- Browser can reach Atlas's LiveKit via `wss://atlas-sjsu.duckdns.org/livekit`
- TTS (Piper + Amy female voice) publishes cleanly
- Room participants see ARTEMIS as a video participant

---

## Phase 3.5 — Consolidate the experiment into the repo

**Goal:** move the working experiment out of `/home/claude/` and into
version control so nothing's lost and the Phase 3 skeleton is replaced
by something that actually runs.

**Scope:**
- `src/agi/primer/artemis/livekit_agent/avatar_hud.py` — the HUD-based
  avatar agent (replaces the Zoom-era stub).
- `deploy/systemd/atlas-artemis-avatar.service` — systemd unit.
- Unit tests for the rendering + responder logic (no LiveKit in tests).
- Update `docs/ARTEMIS.md` §11 Phase 3 to reflect that the agent is
  real code now, not scaffolding.

**Done when:**
- Tests pass on CI
- `atlas-artemis-avatar.service` can be enabled on Atlas to reproduce
  the running experiment
- No code lives outside the repo

---

## Phase A — Custom game-table web UI

**Goal:** replace `meet.livekit.io` with our own browser UI at
`https://atlas-sjsu.duckdns.org/table/<session_id>`. Three-card layout:
avatar (left), info/HUD (top-right), artifacts (bottom-right).

**Scope:**

```
deploy/web/table/
├── index.html        — three-card layout, query-string token
├── table.css         — amber-on-black terminal styling
├── table.js          — LiveKit connection, HUD state, chat
└── vendor/
    └── livekit-client.umd.min.js   — bundled locally, not CDN
```

**Cards:**

- **AVATAR** (left, ~55 % width) — subscribes to the `artemis` video
  track from LiveKit. In v1 this is the Python-rendered HUD; in Phase
  B it will be replaced by a browser-rendered 3D scene.
- **INFO / HUD** (top-right) — participant list, status, event log,
  session clock. Rendered in HTML/CSS, updated from LiveKit room
  events and DataChannel messages.
- **ARTIFACTS** (bottom-right) — list of PDFs available for the
  session; each entry is a download link hitting `/artifacts/...`.
  Dynamic entries can be generated on-demand.

**LiveKit integration:** client uses `livekit-client` UMD bundle
pulled from npm once, checked into `vendor/`. Connects via the same
wss URL we already have. Token is passed as `?t=<jwt>` in the URL
(mint via `Fuller/artemis-tokens/mint.py` or a /mint endpoint later).

**Caddy route:**

```caddy
handle /table/* {
    uri strip_prefix /table
    root * /home/claude/atlas-web/table
    try_files {path} /index.html
    file_server
}
```

**Done when:**
- Two players open `/table/<session>?t=<token>` in browsers
- They see each other's audio levels, ARTEMIS's video, and can speak
- ARTEMIS posts canned responses when a player stops speaking
- All three cards render with correct data

---

## Phase C — PDF artifact service

**Goal:** serve player handouts on-demand, generated from the campaign
bible. Supports both static (pre-built) and dynamic (per-session) PDFs.

**Scope:**

```
src/agi/primer/artemis/artifacts/
├── __init__.py
├── service.py        — small Flask app, one endpoint per handout
├── templates/        — Jinja templates for dynamic PDFs
└── static/           — CSS for PDF styling
```

**Endpoints:**

| Path | Source | Type |
|---|---|---|
| `/artifacts/contract.pdf` | Handout 1 from `Beyond_the_Heliopause_Campaign_Guide.md` | Static (cached) |
| `/artifacts/halyard-deck.pdf` | §9 ship chapter | Static |
| `/artifacts/keeper-tightbeam.pdf` | Handout 2 | Static |
| `/artifacts/unborn-petition.pdf` | Handout 3 | Static |
| `/artifacts/bailey-diary.pdf` | Handout 4 | Static |
| `/artifacts/pdc-discrepancy.pdf` | Handout 5 | Static |
| `/artifacts/pregens.pdf` | Appendix E | Static |
| `/artifacts/session-summary.pdf?session=X` | Session log + proof chain | Dynamic |

**Generation:** pandoc on Atlas (already installed per
`atlas_runner.py` memory). Service shells out to `pandoc <md> -o <pdf>
--template=<tex>` on first request and caches results.

**Systemd unit:** `atlas-artifacts.service` on port 8086.

**Caddy route:**

```caddy
handle /artifacts/* {
    reverse_proxy 127.0.0.1:8086
}
```

**Done when:**
- Every static handout renders as a PDF
- Dynamic session-summary endpoint produces a PDF for a running session
- Links from the ARTIFACTS card in Phase A open the PDFs in a new tab

---

## Phase B — 3D humanoid avatar (gated, next sprint)

**Goal:** replace the Python-rendered HUD video track with a
browser-rendered 3D humanoid. Female Max-Headroom-adjacent aesthetic —
stylized, not photoreal. Lip-sync from phoneme stream.

**Architecture pivot:**

Current:
```
Piper (Atlas) → PIL render (Atlas) → video track → browser displays pixels
```

Phase B:
```
Piper (Atlas) → audio track + phoneme events via DataChannel
                      ▼
                   browser Three.js scene
                      ├─ Ready Player Me humanoid model
                      ├─ viseme blendshapes driven by phoneme events
                      └─ renders locally at browser framerate
```

**Tech pick: Ready Player Me (RPM).**

- Reason: has the right aesthetic (stylized, not photoreal); ships
  with viseme blendshapes (jawOpen, mouthFunnel, mouthPucker, etc.)
  that map 1:1 to Piper's IPA phonemes; loads via a URL we control;
  Three.js integration is documented and maintained.
- Alternatives considered: VRM/VRoid (anime-styled, less good for
  sci-fi feel); custom glTF + manual rig (too much content work for
  this project).

**Phoneme → viseme mapping:** already have this in
`artemis_avatar.py`'s `phoneme_to_viseme`. Extend from 4 shapes to
the full RPM viseme set (15 blendshapes).

**Data channel protocol:** ARTEMIS agent publishes
`agi.rh.artemis.say` with `{text, proof_hash, timings[]}` where
`timings[]` is a list of `{phoneme, start_ms, duration_ms}`. Browser
maps each to a blendshape, interpolates at framerate.

**Server-side work:**
- Compute phoneme timings from Piper's output (currently uniform; can
  be improved with forced alignment via gentle or whisper-alignments
  if quality demands it)
- Publish timings alongside audio

**Browser-side work:**
- Load Three.js + RPM loader
- Render scene with lighting, background matching the table UI
- Subscribe to DataChannel speech events
- Drive blendshapes in the animation loop

**Estimated scope:** 1-2 days of focused work. Not shipping tonight.

**Done when:**
- Browser renders a humanoid avatar in the AVATAR card (not the HUD
  video track)
- Mouth lip-syncs readably when ARTEMIS speaks
- Head has subtle idle motion (breathing, micro-head-tilts)
- No perceptible latency between audio and lip movement

---

## Phase D — ASR + real responses (next sprint)

**Goal:** replace canned responses with actual reasoning. Whisper on
Atlas listens to room audio, routes via NATS to Phase 2
`ArtemisService`, Primer+vMOE generates a real reply, pipeline plays
it through ARTEMIS's voice.

**Scope:**
- Add Whisper ASR to `avatar_hud.py`: subscribe to room audio,
  stream through faster-whisper, publish finalized utterances on
  `agi.rh.artemis.heard`.
- Atlas `atlas-artemis.service` (already deployed per Phase 2) is the
  other end of this pipe.
- Replies from `agi.rh.artemis.say` feed into the TTS queue.

**Done when:**
- Addressing ARTEMIS by name produces a contextually-relevant reply
  (not "Acknowledged")
- DecisionProof chain grows with each turn
- End-to-end latency ≤ 6 seconds (ASR + LLM + TTS)

---

## Phase E — ErisML game-scenario modeling (deferred)

**Goal:** route game decisions through ErisML's Hohfeldian analysis so
ARTEMIS reasons with formal ethics structure rather than free-form
LLM output.

**Out of scope for now.** Captured here because the user asked.
Needs its own design doc once A-D land.

---

## Phase F — Camera / participant face detection (deferred)

**Goal:** let ARTEMIS "see" players via OpenCV face detection on their
subscribed video tracks, feed signals into the reasoning context
("three players visible, imogen's face shows surprise").

**Out of scope for now.** Low marginal value vs. ASR (which gives
linguistic signal, much richer than face-detection boolean).

---

## Ordering

| # | Phase | Effort | Blocks |
|---|---|---|---|
| 1 | 3.5 consolidate | 30 min | A |
| 2 | A web UI | 1 h | deploy |
| 3 | C artifacts | 30 min | deploy |
| 4 | deploy both | 30 min | user eval |
| 5 | B 3D avatar | 1-2 days | dedicated sprint |
| 6 | D ASR + reasoning | 4 h | next session |
| 7 | E ErisML | TBD | designs first |
| 8 | F face detect | 3 h | low priority |

Tonight: 1 → 2 → 3 → 4. B-F by appointment.
