# halyard-table — Next.js web client

Next.js 14 (App Router) source for the Halyard Table player
and Keeper interface.

**Status: placeholder (Sprint 0).** The Next.js scaffold and all
client components land progressively across **Sprints 4–7**:

- **Sprint 4** — scaffold, session landing, video grid, ARTEMIS
  chat panel, safety bar.
- **Sprint 5** — SIGMA-4 chat, character sheet drawer, dice
  roller, scene indicator.
- **Sprint 6** — Keeper view: approval queue, kill-switches,
  session control.
- **Sprint 7** — Keeper view: stat overrides, scene triggers,
  dice broadcast, recording control.

See [`../../docs/HALYARD_SPRINT_PLAN.md`](../../docs/HALYARD_SPRINT_PLAN.md)
for sprint-by-sprint deliverables.

## Planned shape (at end of Sprint 7)

```
web/halyard-table/
├── package.json
├── next.config.mjs
├── tailwind.config.ts
├── tsconfig.json
├── app/
│   ├── layout.tsx
│   ├── page.tsx                    # landing
│   ├── session/[id]/page.tsx       # player view
│   └── keeper/[id]/page.tsx        # Keeper view
├── components/
│   ├── VideoGrid.tsx
│   ├── ArtemisChat.tsx
│   ├── Sigma4Chat.tsx
│   ├── EnvelopeRouter.tsx
│   ├── CharacterSheet.tsx
│   ├── DiceRoller.tsx
│   ├── SceneIndicator.tsx
│   ├── SafetyBar.tsx
│   ├── ApprovalQueue.tsx           # Keeper-only
│   ├── KillSwitchPanel.tsx         # Keeper-only
│   ├── SessionControl.tsx          # Keeper-only
│   ├── StatOverride.tsx            # Keeper-only
│   ├── SceneTrigger.tsx            # Keeper-only
│   ├── DiceBroadcast.tsx           # Keeper-only
│   ├── RecordingToggle.tsx         # Keeper-only
│   └── OverrideLog.tsx             # Keeper-only
├── lib/
│   ├── livekit.ts                  # connection helpers
│   ├── ws.ts                       # halyard-state WS client
│   └── types.ts                    # generated from JSON Schema
└── data/
    └── scenes.json                 # pre-configured scene library
```

## Dependencies

Pinned in `package.json` when the scaffold lands:

- `next@^14`
- `react@^18`
- `livekit-client@^2`
- `@livekit/components-react@^2`
- `@livekit/components-styles@^1`
- `tailwindcss@^3`
- `react-use-websocket@^4`

## Dev loop

```bash
pnpm install
pnpm dev            # localhost:3000
pnpm typecheck
pnpm lint
pnpm build          # production build
pnpm start          # serve built app
```

## Deployment

Containerized via `deploy/docker/halyard-web/Dockerfile`.
Served behind `atlas-caddy` at `halyard.atlas-sjsu.duckdns.org`.

## See also

- [`../../docs/HALYARD_TABLE.md`](../../docs/HALYARD_TABLE.md)
  §4.5 — player client component.
- [`../../docs/HALYARD_TABLE.md`](../../docs/HALYARD_TABLE.md)
  §5.4 — DataChannel envelope contract.
