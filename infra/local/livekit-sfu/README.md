# LiveKit SFU — Halyard Table conference server

Self-hosted [LiveKit](https://livekit.io) SFU deployment for the
**Halyard Table** live-play runtime. This is the conference-room
engine that replaces Zoom for the *Beyond the Heliopause* CoC
campaign.

**Status: placeholder (Sprint 0).** The `docker-compose.yml`,
`livekit.yaml`, `turn.yaml`, and `systemd` unit file for this
service land in **Sprint 1**. See
[`../../../docs/HALYARD_SPRINT_PLAN.md`](../../../docs/HALYARD_SPRINT_PLAN.md)
§Sprint-1 for acceptance criteria and deliverables.

## Planned shape

```
infra/local/livekit-sfu/
├── docker-compose.yml    # livekit-server + coturn + (optional) egress
├── livekit.yaml          # SFU config (keys, rooms, rtc)
├── turn.yaml             # coturn minimal config
├── systemd/
│   └── livekit-sfu.service   # user-level unit
├── scripts/
│   └── sfu_smoke.sh      # two-fake-participant connection test
└── README.md             # runbook: up, down, logs, smoke test
```

## Deployment target

- **Host:** Atlas (`atlas-sjsu.duckdns.org` / LAN `192.168.0.7`
  / Tailscale `100.68.134.21`).
- **TLS:** `atlas-caddy` fronts `halyard.atlas-sjsu.duckdns.org`
  → SFU `:7880` WebSocket.
- **Ports:** `7880` (signaling, proxied), `7881/tcp` (RTC
  fallback), `50000-60000/udp` (RTC media), `3478` (TURN).

## Auth

Participant JWTs are minted by
`agi.primer.artemis.livekit_agent.token.mint_participant_token`
against a shared API key/secret stored sops-encrypted in
`infra/local/livekit-sfu/.env.sops`.

## Rooms

- Convention: `halyard-sNN` (zero-padded session number).
- Auto-create on first join, auto-expire after 24 hours idle.
- One active room per session; the SFU can host multiple rooms
  but we do not plan to run concurrent sessions.

## See also

- [`docs/HALYARD_TABLE.md`](../../../docs/HALYARD_TABLE.md) §4.1
  — component description.
- [`docs/HALYARD_SPRINT_PLAN.md`](../../../docs/HALYARD_SPRINT_PLAN.md)
  §Sprint-1 — planned deliverables.
