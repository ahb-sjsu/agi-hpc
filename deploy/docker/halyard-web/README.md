# halyard-web — player and Keeper client

Container deployment for the Halyard Table web client. The client
source lives at `web/halyard-table/` (Next.js 14, App Router); this
directory holds the build + deploy artifacts.

**Status: placeholder (Sprint 0).** The `Dockerfile` and
associated build scripts land in **Sprint 4**. Keeper console
features land progressively in **Sprints 6–7**. See
[`../../../docs/HALYARD_SPRINT_PLAN.md`](../../../docs/HALYARD_SPRINT_PLAN.md)
for full phasing.

## Planned shape

```
deploy/docker/halyard-web/
├── Dockerfile            # multi-stage, node:20-alpine + .next/standalone
├── .dockerignore
├── entrypoint.sh         # passes through to `node server.js` on :3000
└── README.md             # build + push + deploy
```

## Deployment target

- **Host:** Atlas (single container under `atlas-caddy`).
- **TLS:** `atlas-caddy` fronts `halyard.atlas-sjsu.duckdns.org`
  → container `:3000`.
- **Auth:**
  - Player routes: JWT-minted session tokens (see
    `halyard-keeper-backend`).
  - Keeper routes (`/keeper/*`): HTTP Basic in v1, enforced by
    Caddy; upgrade to session-based OAuth in a later sprint.

## Routes

- `/` — landing (name, session, role).
- `/session/<session_id>` — player view.
- `/keeper/<session_id>` — Keeper view.

## See also

- [`docs/HALYARD_TABLE.md`](../../../docs/HALYARD_TABLE.md) §4.5
  (player client) and §4.6 (Keeper backend) — component spec.
- `web/halyard-table/README.md` — source-level README
  (populated in Sprint 4).
