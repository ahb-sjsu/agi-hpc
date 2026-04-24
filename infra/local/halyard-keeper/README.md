# halyard-keeper — deployment artifacts

The Keeper-console backend. Mints LiveKit JWTs (for players and
for the Keeper), tracks session lifecycle, holds the pending-AI-
reply approval queue.

See [`../../docs/HALYARD_TABLE.md`](../../../docs/HALYARD_TABLE.md)
§4.6 and
[`../../docs/HALYARD_SPRINT_PLAN.md`](../../../docs/HALYARD_SPRINT_PLAN.md)
§Sprint-6.

## Install on Atlas

```bash
# On Atlas, as user 'claude':
cd ~/src/agi-hpc
git fetch origin
git checkout halyard/sprint-3
git pull
# (or, from a dedicated worktree if in use:
#    cd ~/agi-hpc-halyard && git fetch && git reset --hard origin/halyard/sprint-3)

# Install into the venv (no new deps beyond PyJWT, which ARTEMIS
# already needs).
/home/claude/env/bin/pip install -e . --quiet

# Install the systemd unit.
mkdir -p ~/.config/systemd/user
ln -sfn ~/src/agi-hpc/infra/local/halyard-keeper/systemd/halyard-keeper.service \
        ~/.config/systemd/user/halyard-keeper.service
systemctl --user daemon-reload
systemctl --user enable --now halyard-keeper.service

# Verify (loopback only; use SSH port-forward for off-box).
curl -sS http://127.0.0.1:8091/healthz | jq .
```

## Environment

| Var                       | Default                  | Purpose                                  |
|---------------------------|--------------------------|------------------------------------------|
| `HALYARD_KEEPER_HOST`     | `127.0.0.1`              | Bind address — loopback only.             |
| `HALYARD_KEEPER_PORT`     | `8091`                   |                                           |
| `LIVEKIT_URL`             | `ws://127.0.0.1:7880`    | Where browsers should connect.            |
| `LIVEKIT_API_KEY`         | `devkey`                 | Matches the SFU's keys file.              |
| `LIVEKIT_API_SECRET`      | `secret`                 | HS256 signing key for minted JWTs.        |
| `LIVEKIT_TOKEN_TTL_SECONDS` | `21600` (6h)           | Token lifetime.                           |
| `KEEPER_USERNAME`         | *(unset)*                | HTTP Basic user for Keeper routes.        |
| `KEEPER_PASSWORD`         | *(unset)*                | HTTP Basic password. **Unset = no auth.** |
| `KEEPER_IP_ALLOWLIST`     | *(empty)*                | Comma-separated CIDRs.                    |
| `HALYARD_ARCHIVE_ROOT`    | `/archive/halyard`       | Session append-log root.                  |
| `LOG_LEVEL`               | `info`                   |                                           |

## Routes

- **Public** (no auth — session-scoped):
    - `GET /healthz`
    - `POST /api/livekit/token` → mint a LiveKit JWT for a player.
- **Keeper-only** (HTTP Basic when `KEEPER_USERNAME`/`KEEPER_PASSWORD` are set):
    - `GET/POST /api/sessions[/...]`
    - `POST /api/livekit/keeper-token`
    - `GET/POST /api/keeper/approvals[/...]`
    - `GET  /ws/keeper/{session_id}`

## Smoke test

```bash
# On Atlas or via SSH port-forward to 127.0.0.1:8091:
curl -sS http://127.0.0.1:8091/healthz | jq .

# Mint a token (dev mode — no auth).
curl -sS -X POST \
     -H 'content-type: application/json' \
     -d '{"session_id":"halyard-s01","identity":"cross"}' \
     http://127.0.0.1:8091/api/livekit/token | jq .
```

## Binding policy

Same rule as halyard-state: the service binds `127.0.0.1` and
sits behind atlas-caddy. Never `0.0.0.0`. Tailscale is the
operator's out-of-band admin path, not a service channel.
