# halyard-table — web client

Next.js 14 (App Router) client for the **Halyard Table** live-play
runtime. Serves the player view (video grid + AI chat panels +
character sheet drawer + safety bar) and — starting in Sprint 6 —
the Keeper console.

See [`../../docs/HALYARD_TABLE.md`](../../docs/HALYARD_TABLE.md) §4.5
for the component's place in the architecture,
[`../../docs/HALYARD_SPRINT_PLAN.md`](../../docs/HALYARD_SPRINT_PLAN.md)
§Sprints 4–7 for the phased delivery plan.

## Current status (Sprint 4)

Scaffolded + first player-view components. What ships:

- App-router structure (`app/`), global dark theme matching
  `atlas-chat/schematic.html`.
- Landing page (`/`) collects session id, display name, optional PC id.
- Session page (`/session/<id>`) — LiveKit room with:
  - **VideoGrid** (LiveKit React tiles).
  - **AiChatPanel** × 2 — ARTEMIS and SIGMA-4, listening on the
    DataChannel for `{artemis,sigma4}.say` envelopes.
  - **CharacterSheetDrawer** — live-updating sheet via halyard-state
    WebSocket. Toggle with the `c` key.
- Types (`lib/types.ts`) mirror the halyard-state JSON Schema and
  the DataChannel / NATS envelope contracts.
- LiveKit + state hooks (`lib/livekit.ts`, `lib/state.ts`) —
  testable in isolation from the UI.

Pending for Sprint 5: Dice roller, scene-indicator top-bar tile,
envelope router separation, per-PC mute controls.

## Dev loop

```bash
cd web/halyard-table
cp .env.example .env.local     # edit if needed
npm install
npm run dev                    # http://localhost:3000
```

For full end-to-end local dev you need three tunnels open to
Atlas (OOB path — Tailscale is admin-only):

```bash
# LiveKit signaling
ssh -L 7880:127.0.0.1:7880 claude@100.68.134.21 &

# halyard-state REST + WS
ssh -L 8090:127.0.0.1:8090 claude@100.68.134.21 &

# halyard-keeper-backend token mint (Sprint 6+; stub until then)
ssh -L 8091:127.0.0.1:8091 claude@100.68.134.21 &
```

## Build

```bash
npm run lint        # ESLint (Next defaults)
npm run typecheck   # tsc --noEmit
npm run build       # Next.js standalone build → .next/standalone
npm run start       # serve built app
```

The production build is `output: "standalone"` so the Dockerfile
copies just `.next/standalone/` + `.next/static/` + `public/` — no
`node_modules` at runtime.

## Deployment target

Lands as a container behind `atlas-caddy` at
`halyard.atlas-sjsu.duckdns.org`. Dockerfile + Caddy site block
are in [`../../deploy/docker/halyard-web/`](../../deploy/docker/halyard-web/).

## Keyboard

- `c` — toggle character sheet drawer.
- `Escape` — close the sheet drawer.

## Accessibility

The components aim for WCAG 2.1 AA on the main flows:

- Every interactive element has an accessible name (`aria-label`
  or visible label).
- Focus rings are visible (`focus:ring-2 focus:ring-accent`).
- Live regions (`aria-live="polite"`) on message logs and
  connection indicators.
- Colors are not the only signal for safety-bar state (ring,
  emoji glyph, and text label all flip together).

Lighthouse a11y target ≥ 90 per Sprint-4 acceptance criteria.
