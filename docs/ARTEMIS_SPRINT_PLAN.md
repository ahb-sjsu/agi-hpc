# ARTEMIS Multi-Sprint Plan

> Consolidated plan across every feature discussed so far for ARTEMIS.
> For your review before execution.

**Date:** 2026-04-23 (rev 2)
**Status:** S0 landed (PR #88); S1 broken into four slices (1a-d) per
user review; executing 1a next.

This supersedes the phase-oriented [`ARTEMIS_AVATAR_ROADMAP.md`](ARTEMIS_AVATAR_ROADMAP.md)
by reorganizing into discrete, shippable sprints with dependencies and
honest effort estimates.

---

## Current state (checkpoint)

**Already landed on `main`:**
- Phase 1 — offline `handle_turn`, validator, DecisionProof chain (PR #85)
- Phase 2 — NATS handler, systemd unit, 20-turn round-trip test (PR #86)
- Phase 3 — LiveKit agent skeleton, JWT minting, scaffolding (PR #87)

**Running on Atlas but not yet committed:**
- v4 HUD avatar agent at `/home/claude/artemis_avatar.py` — icosahedron
  wireframe HUD + Piper TTS (Amy) + canned responder + 960×720 @ 30 fps
  with simulcast

**Written in worktree `artemis/phase-3.5-avatar` but not committed / deployed:**
- Phase 3.5 consolidation (avatar_hud.py → repo, systemd unit)
- `docs/ARTEMIS_AVATAR_ROADMAP.md` (older planning doc)
- Phase A `deploy/web/table/` — custom table UI (index.html, table.css,
  table.js, vendor readme)
- This sprint plan you are reading

**Atlas infrastructure:**
- LiveKit server (docker container, dev-mode keys)
- Caddy route `/livekit/*` → `127.0.0.1:7880` (TLS via duckdns cert)
- UDP 50000-50100 router-forwarded
- Two test tokens (keeper + player) working end-to-end

---

## Design principles — kept across all sprints

1. **Ship shippable slices.** Every sprint's output is usable at the table, not a half-built stepping stone.
2. **Atlas-hosted by default.** No cloud dependencies unless explicitly justified.
3. **Reuse the Phase 1/2 substrate.** The NATS subjects, validator, proof chain stay as they are.
4. **Browser = thin client.** Heavy lifting stays on Atlas.
5. **Roll-back friendly.** Each sprint is revertable without cascading into prior work.

---

## Sprint 0 — Consolidation (IN PROGRESS)

**Goal:** land the running experiment into version control + give the
table a real web UI + basic artifact downloads. Get everything off
`/home/claude/` and onto `main`.

**Effort:** ~2 hrs

**Items:**

- [x] **S0-1** Roadmap doc and sprint plan (this file) — done
- [ ] **S0-2** Move `artemis_avatar.py` → `src/agi/primer/artemis/livekit_agent/avatar_hud.py`
- [ ] **S0-3** `deploy/systemd/atlas-artemis-avatar.service`
- [ ] **S0-4** Phase A table UI (`deploy/web/table/` — done in worktree)
- [ ] **S0-5** Phase C PDF artifact service (`src/agi/primer/artemis/artifacts/`)
- [ ] **S0-6** Caddy additions — `/table/*`, `/artifacts/*`
- [ ] **S0-7** Deploy + verify end-to-end on Atlas
- [ ] **S0-8** PR + CI + merge

**Done when:**
- Players open `https://atlas-sjsu.duckdns.org/table/<session>?t=<jwt>`
- Three cards visible: avatar (HUD), info/HUD, artifacts
- At least 3 static PDFs (contract, tightbeam, pregens) downloadable
- ARTEMIS joins as 3rd participant with HUD + TTS
- Unit tests green on CI, PR merged

---

## Sprint 1 — Real responsiveness (split into 1a/b/c/d)

**Goal:** replace canned "Acknowledged" with actual LLM-generated
replies, in parallel with a top-tier voice + live HUD. Split into
four shippable slices so each PR is reviewable on its own.

### S1a — Expanse theming + stubbed HUD widgets

**Effort:** ~1 hr. No backend changes.

- Rework `deploy/web/table/table.css` to the Expanse corporate-vessel
  aesthetic: slate-black base, cyan accent with amber alerts, angular
  clip-path frames, Rajdhani + Share Tech Mono typography, subtle
  hex-grid overlay.
- Add HUD widget scaffolding to `table.js` / `index.html` — PC stat
  cards (name + SAN + HP + Luck + MP bars + status), scene-header
  strip, proof-chain ticker. Data is placeholder constants in v1.

**Done when:** UI visibly looks like an Expanse ship's tactical display
instead of a generic dark terminal; PC stat cards render with static
placeholder data.

### S1b — Google Sheets CSV → live HUD stats

**Effort:** ~2 hrs.

- New `src/agi/primer/artemis/sheets/` package:
  - `poller.py` — polls a "publish-to-web" CSV URL every 30 s, parses
    rows into character dicts, emits diffs on NATS
    `agi.rh.artemis.sheet.<sheet_name>`.
  - `__main__.py` — systemd entry point.
- `deploy/systemd/atlas-artemis-sheets.service` — systemd unit.
- Avatar agent: subscribe to `agi.rh.artemis.sheet.*` → publish diffs
  to LiveKit DataChannel with kind `"artemis.sheet"`.
- `table.js`: receive DataChannel events, update stat widgets.

Config model:
```
ARTEMIS_SHEET_URLS='characters=<csv-url>;npcs=<csv-url>'
ARTEMIS_SHEET_POLL_S=30
```

**Done when:** editing a cell in your published Google Sheet updates
the HUD widget within 30 seconds.

### S1c — Coqui XTTS-v2 voice upgrade

**Effort:** ~2 hrs. User-stated priority for feel.

- Install `TTS` (Coqui) into the artemis-avatar venv.
- Download XTTS-v2 weights (~2 GB).
- A/B benchmark vs Piper Amy using the same reference sentence; save
  both clips to `Fuller/artemis-tokens/voice-benchmark/` for
  listen-comparison.
- Add `ARTEMIS_TTS_BACKEND=xtts|piper` env switch in `avatar_hud.py`.
- Default to XTTS; Piper stays as fallback.
- Optional: voice-clone from a 6-30 s reference clip — user provides
  or I pick one from public-domain readings that matches the
  "clinical-but-warm" ARTEMIS system-prompt voice spec.

**Done when:** ARTEMIS's voice is decisively better than Piper and
the user confirms it clears the "mostly-talking-game" bar.

### S1d — Whisper ASR + NATS + Primer + Keeper approval gate

**Effort:** ~4 hrs. The big one — real responsiveness.

- Subscribe to remote-participant audio tracks in the avatar agent.
  Stream each into `faster-whisper`.
- Publish finalized utterances to `agi.rh.artemis.heard` with the
  TurnRequest schema.
- Verify `atlas-artemis.service` (Phase 2, on main) picks them up;
  configure `NRP_LLM_TOKEN` on Atlas.
- Consume `agi.rh.artemis.say` in avatar agent → XTTS queue → played.
- **Keeper approval gate UI** in the browser: when the bot's say
  carries `approval:"keeper_pending"`, surface a tile only the
  Keeper sees with ✓/✗ buttons. Approved → broadcast to the room;
  rejected → drop + log.

**Done when:** ask ARTEMIS a question → relevant reply within ~6 s
end-to-end; Keeper sees approval prompt; DecisionProof chain grows.

### Depends on
- S0 is in.
- S1a → unlocks S1b (widgets exist to fill).
- S1b → unlocks S1d (stat context available to Primer if desired).
- S1c is independent of the others; can run in parallel.

### Sprint gate
- All four PRs landed on main.
- Latency p50 < 5 s, p95 < 10 s on the ASR → LLM → TTS loop.
- Voice clears the "really good" bar.
- Stats update live from Sheets.

---

## Sprint 2 — LiveKit production polish

**Goal:** fix the media-quality rough edges so this is usable in a
real 4-hour session.

**Effort:** ~1 day

**Items:**

- **S2-1** **Noise cancellation** — integrate Krisp `TrackProcessor`
  (LiveKit plugin). Auto-applied to inbound participant audio.
- **S2-2** **Push-to-talk** — toggle in avatar card, `M` key binding.
  Default off for player mics, on-demand.
- **S2-3** **Live transcription card** — show a rolling text of what
  ARTEMIS said (already have text via TTS path) + optional ASR
  transcriptions of players.
- **S2-4** **X-card button** — top-right of every card. On click:
  publishes `agi.rh.artemis.silence` via NATS (kill-switch, already
  supported by Phase 2 handler). 5-min cooldown default.
- **S2-5** **Voice quality upgrade** — try Coqui XTTS for a
  higher-fidelity voice alternative; keep Piper as fallback. Env-var
  selectable.
- **S2-6** Reconnect handling — graceful UX when a player's WiFi
  blips.

**Depends on:** S0

**Done when:**
- Full 60-min test with 3 humans + ARTEMIS, no quality complaints
- X-card triggers immediate ARTEMIS silence
- Voice sounds noticeably better than Piper Amy medium

---

## Sprint 3 — 3D humanoid avatar (Phase B proper)

**Goal:** replace the Python-rendered HUD with a browser-rendered
female humanoid 3D avatar. Viseme lip-sync, idle animations.

**Effort:** ~2 days

**Stack pick (for review):**
- **Three.js** for rendering
- **Ready Player Me (RPM)** for the avatar model (stylized, has
  viseme blendshapes built in, URL-loadable, free tier)
- **Piper forced-alignment** for phoneme timing (or gentle-aligner
  if better precision needed)
- **LiveKit DataChannel** for audio timing cues

**Items:**

- **S3-1** Generate an RPM avatar (user picks via their web UI).
  Pin the URL. Design note: avoid photorealism — stylized matches
  ARTEMIS.
- **S3-2** Integrate Three.js into `table.js` — scene, camera,
  lighting, RPM model loader.
- **S3-3** Viseme blendshape driver — map Piper IPA → RPM blendshape
  names. 15 shapes total.
- **S3-4** **Phoneme timing publisher** on Atlas — extend Piper TTS
  pipeline to emit `{phoneme, start_ms, dur_ms}` timings via
  DataChannel alongside audio.
- **S3-5** Avatar card switches from `<video>` track to Three.js
  canvas.
- **S3-6** Idle animations — subtle head tilt, blinking, breathing
  (linear combination of small blendshapes over time).
- **S3-7** Expression cues — slight eyebrow lift when "listening,"
  neutral when "speaking." Drive from NATS events.
- **S3-8** Retire the HUD video track publisher (or relegate to a
  backup-mode env var).

**Depends on:** S0, S1 (for real speech events + timings)

**Done when:**
- Browser renders humanoid avatar, mouth moves in readable sync with
  TTS
- Latency between audio and lip-movement < 100ms
- 60fps rendering on target hardware (Atlas → desktop browsers)
- Old HUD video track removed

---

## Sprint 4 — Game-table features

**Goal:** actual TTRPG session companion features.

**Effort:** ~1 day

**Items:**

- **S4-1** **Dice roller** — UI control in the avatar card; supports
  d4/d6/d8/d10/d12/d20/d100 + modifiers. Each roll broadcast via
  DataChannel, logged in INFO card, visible to all players.
- **S4-2** **Shared roll log** — persistent across the session,
  exportable to session replay.
- **S4-3** **Handout push** — Keeper clicks a handout in ARTIFACTS,
  triggers all players' browsers to open it.
- **S4-4** **Scene state** — room.metadata field `"scene": "..."`.
  Changes broadcast to all. ARTEMIS's system prompt updates when
  scene changes (new scene → extra context block in handle_turn).
- **S4-5** **Character sheets** — live-updatable SAN, HP, Luck, MP
  per PC. Stored in room.metadata, rendered in a collapsible card
  under ARTIFACTS.
- **S4-6** **Keeper panel** — sidebar only Keeper can see. Scene
  controls, approval queue, kill-switch, quick-silence-ARTEMIS.

**Depends on:** S0

**Done when:**
- Full session playable without external tools (dice, sheets,
  handouts all in-app)
- Scene transitions make ARTEMIS context-aware
- Keeper has an ergonomic control surface

---

## Sprint 5 — Atmosphere + experience

**Goal:** production value — ambient sound, keyboard shortcuts,
accessibility polish.

**Effort:** ~0.5 days

**Items:**

- **S5-1** **Ambient soundscape per scene** — a second low-volume
  audio track ARTEMIS publishes that loops the scene's ambience
  (ship hum, vault silence, Mi-go installation buzz). Swaps on
  scene change. Pre-made ~5 min WAVs per scene.
- **S5-2** **Keyboard shortcuts** — M mute, Space PTT, X x-card,
  D dice, ESC leave.
- **S5-3** **Accessibility** — ARIA labels, keyboard nav, sufficient
  contrast in amber-on-black (audit WCAG AA).
- **S5-4** **Mobile-friendly** layout breakpoints (cards stack on
  narrow screens).

**Depends on:** S4 (scene state)

**Done when:**
- Every interaction keyboard-accessible
- Mobile browsers render usably (even if not ideal)
- Ambient sound fades between scenes

---

## Sprint 6 — ErisML integration

**Goal:** route every ARTEMIS reply through ErisML's Hohfeldian
moral-vector analysis before it's posted. The validator's current
`erisml_check` is a Phase-1 stub — this sprint makes it real.

**Effort:** ~1 day

**Items:**

- **S6-1** Import `erisml-lib` into the artemis validator.
- **S6-2** **Scenario encoder** — convert a proposed ARTEMIS reply +
  current game state into the rank-6 Coo6 tensor that ErisML expects.
- **S6-3** Run the tensor through ErisML's Python reference (bit-exact
  with the eventual EPU hardware path).
- **S6-4** Reject replies whose invariant-vector indicates moral
  concern beyond a configurable threshold; DecisionProof records
  the full vector.
- **S6-5** Keeper-visible "why rejected" explainer using ErisML's
  Hohfeldian correlative trace.
- **S6-6** Extended tests: 50 hand-crafted scenarios with expected
  verdicts.

**Depends on:** S1 (need real replies to reason about)

**Done when:**
- Every posted reply has ErisML verdict in its DecisionProof
- Borderline-unethical canned scenarios get correctly rejected
- The explainer surface is useful (not just "rejected because bad")

---

## Sprint 7 — Recording + ops

**Goal:** session archival + observability.

**Effort:** ~1 day

**Items:**

- **S7-1** **LiveKit Egress** — server-side composite MP4 recording
  of each session. Stored under `/archive/artemis/sessions/<id>/`.
- **S7-2** **Session replay export** — single ZIP bundle: MP4 +
  proof-chain + session log + handout PDFs. Downloadable from the
  ARTIFACTS card after session ends.
- **S7-3** **Dashboard tile** in `schematic.html` — ARTEMIS panel
  showing: active sessions, turn count, validator reject rate,
  p50/p95 turn latency.
- **S7-4** **Prometheus metrics** exported from atlas-artemis.service
  for historical graphing.
- **S7-5** **Backup inclusion** — session logs + proofs in
  atlas-backup.timer.

**Depends on:** S0

**Done when:**
- Each completed session is archived automatically
- Dashboard shows live metrics
- Keeper can hand a player a ZIP of "the whole session I just played in"

---

## Sprint 8 — Camera / face detection

**Goal:** let ARTEMIS visually "notice" participants.

**Effort:** ~0.5 days

**Items:**

- **S8-1** Subscribe to remote participants' video tracks in the
  avatar agent.
- **S8-2** Sample 1fps, run OpenCV haar-cascade face detection (or
  MediaPipe face landmarker).
- **S8-3** Publish presence events (`{speaker: x, face_visible: bool,
  bbox}`) on NATS `agi.rh.artemis.sees.*`.
- **S8-4** Feed into reasoning context — available to Primer as
  an extra context block.

**Depends on:** S1

**Low priority — text + speech signal is much richer than "face visible
boolean." Listed for completeness because user asked.**

**Done when:**
- Atlas logs face-visible events per frame
- ARTEMIS's context includes "3 of 4 players visible on camera"

---

## Sprint 9 — EPU FPGA validator swap

**Goal:** replace the ErisML Python reference in the validator with
the FPGA-hardware path. Same bit-exact math, different backend.

**Effort:** ~0.5 days (once unblocked)

**Blocked on:** external — the NRP Coder workspace auto-restart issue
preventing the Phase 3 `v++ link` for the TPU kernel. Once that ships,
Phase 5 Integration service is the call target.

**Items:**

- **S9-1** Swap validator backend: HTTP call to the EPU Phase 5
  Docker-compose service.
- **S9-2** Bit-exact smoke test: 10k scenarios, Python ref vs FPGA
  must agree.
- **S9-3** DecisionProof chains into EPU's SHA chain — both chains
  cross-verifiable.
- **S9-4** Metrics: FPGA latency vs Python reference.

**Done when:**
- Every reply's ErisML check runs on real U55C hardware
- SHA chains verify end-to-end
- Latency improvement measurable (expected µs vs ms)

---

## Sprint 10 — Atlas diagnostics in-table (widgets from `schematic.html`)

**Goal:** surface interesting panels from the existing Atlas dashboard
inside the game table UI so ARTEMIS feels like it's showing you its
actual mind. Diegetic framing: these are "the handheld's diagnostic
telemetry" that the in-fiction character would expose.

**Effort:** ~1 day

**Which panels to pull (ranked by in-fiction fit + info density):**

| Dashboard panel | In-fiction framing | Card placement |
|---|---|---|
| NATS Live (message stream) | "neural pathway traffic" | NEW: DIAGNOSTICS card |
| Primer health / vMOE expert status | "active cognitive cores" | DIAGNOSTICS |
| GPU temperature + utilization | "thermal / processing load" | DIAGNOSTICS |
| Memory tier L0/L1/L2/L3/L4 stats | "episodic recall depth" | DIAGNOSTICS |
| NRP burst jobs table | "offloaded computation" | DIAGNOSTICS |
| Recent DecisionProof rejects | "ethical veto stream" | INFO card (existing) |
| ARC scientist progress | (too meta, skip) | — |

**Items:**

- **S10-1** Add a **DIAGNOSTICS card** (fourth card in the right-column
  stack, collapsible). Default collapsed; Keeper can expand.
- **S10-2** Each widget is a small Svelte-less HTML component that
  polls an existing `/api/*` endpoint on Atlas (already served by
  `atlas-telemetry.service`):
  - `/api/nats/live?subject=agi.rh.artemis.*` — filter to only
    ARTEMIS-relevant subjects
  - `/api/primer/health` — vMOE expert summary
  - `/api/gpu/status` — temps + utilization
  - `/api/memory/tiers` — L0-L4 cardinality
  - `/api/burst/jobs` — NRP burst list
- **S10-3** Styling — amber-on-black to match the table UI. Tiny
  sparklines where appropriate (one line of SVG each).
- **S10-4** Keeper-only toggle — hide Diagnostics from players if
  distracting. Store in `room.metadata`.
- **S10-5** **Redacted mode** — when a player is viewing, some panels
  (e.g. "ethical veto stream") show redacted text so we don't leak
  what ARTEMIS suppressed. Keeper view = full.
- **S10-6** **Caddy route** already exposes `/api/*` → 8085, so CORS
  is a non-issue as long as the table page is on the same origin
  (`atlas-sjsu.duckdns.org`).

**Depends on:** S0 (needs the card layout), S4 (scene state for
Keeper-only toggle), S1 (for the decision-proof stream to have content)

**Done when:**
- DIAGNOSTICS card renders with at least 3 live widgets
- Widget data updates at 1-2 Hz without impacting the main
  game-session performance
- Keeper can toggle visibility per-session
- Redacted mode works for player views

**Why this is cool:**
- The Atlas dashboard already exists; this is mostly glue.
- It makes ARTEMIS feel *real* in a way that pre-rendered animations
  can't match — you're literally watching its thought process during
  the session.
- Post-session, an archived diagnostic feed becomes part of the session
  replay bundle (S7).

---

## Sprint 11 — Security + consent

**Goal:** real-session safety posture beyond dev-mode.

**Effort:** ~0.5 days

**Items:**

- **S11-1** **E2EE** — LiveKit end-to-end encryption with shared
  key per session. Tokens carry the key. Media not decrypted by
  SFU.
- **S11-2** Rotate LiveKit API key/secret off `devkey`/`secret` —
  proper secrets in Atlas `.env` files.
- **S11-3** **Recording consent opt-in** — per-participant banner
  + checkbox before joining a recorded session.
- **S11-4** **Token mint service** with user:read:token-style scope
  so tokens don't all pre-exist as files in `Fuller/artemis-tokens/`.

**Depends on:** S7 (recording exists to consent to)

**Done when:**
- No default/weak credentials in production
- Encryption verified via LiveKit UI indicator
- Consent banner blocks join until acknowledged

---

## Sprint order (dependency-aware)

```
S0 (current) ──┬── S1 ──┬── S3 ─── S6
               │        │    │
               ├── S2   │    │
               ├── S4   │    │
               ├── S5 (after S4)
               ├── S7 ──┬── S11
               │        │
               ├── S8 ──┤
               ├── S9 (blocked externally)
               └── S10 (after S0, S1, S4)
```

Critical path to a shippable tabletop: **S0 → S1 → S3 → S4 → S7**
(~5-6 focused days of work). S2, S5, S6, S10 can interleave. S8/S9/S11
are parallel-later.

---

## Effort total

| Sprint | Effort | Value | Priority |
|---|---|---|---|
| S0 consolidation | 2 hrs | foundation | NOW |
| S1 real responsiveness | 1 day | must-have | next |
| S2 LiveKit polish | 1 day | high | next |
| S3 3D avatar | 2 days | high | after S1 |
| S4 game features | 1 day | high | after S0 |
| S5 atmosphere | 0.5 day | medium | after S4 |
| S6 ErisML integration | 1 day | distinguishing | after S1 |
| S7 recording + ops | 1 day | high | after S0 |
| S8 face detection | 0.5 day | low | after S1 |
| S9 FPGA swap | 0.5 day | distinguishing | blocked |
| S10 Atlas diagnostics in-table | 1 day | cool + meta | after S0+S4 |
| S11 security + consent | 0.5 day | required for prod | after S7 |

**Total:** ~11 days of focused engineering, spread across as many
sessions as the calendar allows.

---

## Questions for you before we start

1. **Do you want the full sprint plan landed in this PR (S0 +
   this doc), or split — land S0 now, add this plan in a follow-up?**
2. **Any sprint you want to reorder?** E.g., if the 3D avatar (S3) is
   urgent-cool and you'd rather have that before ASR (S1), that's a
   valid tradeoff but means ARTEMIS still gives canned replies for
   longer.
3. **Any sprint you want to strike entirely?** E.g., S8 face-detection
   is low-leverage; I'd deprioritize or drop it.
4. **Stack picks to review:**
   - Ready Player Me vs VRM/VRoid for 3D avatar?
   - Coqui XTTS vs higher-quality Piper voices for voice upgrade?
   - `livekit-egress` for recording — OK to add as a sibling container?

Mark up this file directly in review, or reply with a consolidated
"S0 go, S1 first, swap S3 and S4" and I'll update and execute.
