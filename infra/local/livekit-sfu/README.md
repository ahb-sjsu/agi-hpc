# LiveKit SFU — Halyard Table conference server

Self-hosted [LiveKit](https://livekit.io) SFU deployment for the
**Halyard Table** live-play runtime. This is the conference-room
engine that replaces Zoom for the *Beyond the Heliopause* CoC
campaign.

**Status: Sprint 1 deliverable.** All configuration and
operator tooling lands here. Deployment verification (bringing
the SFU up on Atlas and running the smoke test against two real
browsers) is the Sprint 1 acceptance gate. See
[`../../../docs/HALYARD_SPRINT_PLAN.md`](../../../docs/HALYARD_SPRINT_PLAN.md)
§Sprint-1 for the full acceptance criteria.

## Files

```
infra/local/livekit-sfu/
├── docker-compose.yml        # livekit-server + coturn, host networking
├── livekit.yaml              # SFU config (ports, rooms, logging)
├── livekit-keys.yaml.example # copy → livekit-keys.yaml; API key/secret
├── turn/
│   └── turnserver.conf       # coturn config (shared-secret auth)
├── systemd/
│   └── livekit-sfu.service   # user-level systemd unit
├── scripts/
│   └── sfu_smoke.sh          # see scripts/halyard/sfu_smoke.sh
├── caddy.snippet             # atlas-caddy reverse-proxy block
├── .env.example              # copy → .env; EXTERNAL_IP, TURN_SECRET
├── .gitignore                # keeps secrets + runtime data out of git
└── README.md                 # this file
```

## Prerequisites

- Atlas running Docker 24+ (`docker compose version` works).
- Atlas running `atlas-caddy` with ACME TLS.
- DNS: `halyard.atlas-sjsu.duckdns.org` resolves to Atlas's
  public-facing address (or Tailscale IP for restricted
  deployments).
- Firewall: UDP `50000-60000`, TCP `7881`, UDP/TCP `3478`
  open to the internet (or to the Tailscale network).

## First-time install

```bash
# On Atlas, from your checkout of agi-hpc:
cd ~/src/agi-hpc/infra/local/livekit-sfu

# 1. Populate secrets.
cp .env.example .env
cp livekit-keys.yaml.example livekit-keys.yaml

# Generate a fresh key/secret pair.
docker run --rm livekit/livekit-server:v1.7.2 generate-keys
# Paste the output into livekit-keys.yaml.

# Generate a TURN shared secret.
openssl rand -base64 32
# Paste into .env as LIVEKIT_TURN_SECRET=...

# Set LIVEKIT_EXTERNAL_IP — the IP Atlas advertises.
# Public duckdns: use `dig +short halyard.atlas-sjsu.duckdns.org`.
# Tailscale-only: use 100.68.134.21.

# 2. Install the systemd unit.
mkdir -p ~/.config/systemd/user
ln -s ~/src/agi-hpc/infra/local/livekit-sfu/systemd/livekit-sfu.service \
      ~/.config/systemd/user/livekit-sfu.service
systemctl --user daemon-reload

# 3. Enable lingering so the service starts at boot.
sudo loginctl enable-linger "$USER"

# 4. Install the Caddy snippet.
sudo cp caddy.snippet /etc/caddy/sites/halyard.conf
sudo systemctl reload caddy

# 5. Start the SFU.
systemctl --user enable --now livekit-sfu.service

# 6. Verify.
systemctl --user status livekit-sfu
docker ps | grep halyard-
```

## Smoke test

The smoke test lives at [`scripts/halyard/sfu_smoke.sh`](../../../scripts/halyard/sfu_smoke.sh).

```bash
# From a machine with agi-hpc checked out (Atlas or your laptop):
export LIVEKIT_API_KEY="APIkeyHalyardDev01"  # from livekit-keys.yaml
export LIVEKIT_API_SECRET="..."              # from livekit-keys.yaml
./scripts/halyard/sfu_smoke.sh
```

Expected output:

```
═══════════════════════════════════════════════════════════
  Halyard Table — SFU smoke test
  Host: halyard.atlas-sjsu.duckdns.org
  Room: halyard-smoke-1714934400
═══════════════════════════════════════════════════════════
[1/4] TLS reachability check...
  ✓ TLS reachable
[2/4] Minting participant JWT...
  ✓ Token minted (412 chars)
[3/4] Validating token against SFU...
  ✓ Token validated
[4/4] TURN reachability...
  ✓ TURN UDP :3478 reachable
═══════════════════════════════════════════════════════════
  ✓ ALL CHECKS PASSED
  SFU is ready for the Halyard Table.
═══════════════════════════════════════════════════════════
```

## Two-browser sanity

After the smoke test passes:

1. Open [LiveKit Meet](https://meet.livekit.io) in Browser A.
2. Paste the server URL: `wss://halyard.atlas-sjsu.duckdns.org`.
3. Paste a minted JWT (use `scripts/halyard/mint_token.py`).
4. Join room `halyard-sanity`.
5. Repeat in Browser B with a different identity.
6. Verify both see each other's video and can hear each other.
7. Check `docker logs halyard-livekit-server` for clean
   candidate negotiation.

## Operating

### Logs

```bash
# SFU + coturn combined (systemd journal):
journalctl --user -u livekit-sfu -f

# Just the SFU container:
docker logs -f halyard-livekit-server

# Just coturn:
docker logs -f halyard-coturn
```

### Bring down / restart

```bash
systemctl --user restart livekit-sfu.service
systemctl --user stop    livekit-sfu.service
systemctl --user start   livekit-sfu.service

# Or, directly:
docker compose down
docker compose --env-file .env up -d --wait
```

### Upgrading the SFU version

1. Update the `image:` tag in `docker-compose.yml`.
2. Test locally on a throwaway LiveKit instance.
3. On Atlas: `systemctl --user stop livekit-sfu` → `docker pull`
   → `systemctl --user start livekit-sfu`.
4. Re-run smoke test.
5. Commit the tag change with scope `halyard-sfu`.

## Troubleshooting

| Symptom                                | Likely cause                     | Fix                            |
|----------------------------------------|----------------------------------|--------------------------------|
| Smoke test `[1/4]` fails               | Caddy not running / DNS / TLS    | `systemctl status caddy`; dig  |
| Smoke test `[3/4]` fails HTTP 401      | Key/secret mismatch              | Rebuild `livekit-keys.yaml`    |
| Smoke test `[4/4]` warns               | Router UDP filtering             | Often benign; try from 4G     |
| Participants can see but not hear      | UDP media port blocked           | Open 50000-60000/udp           |
| Participants cannot connect from 4G    | TURN not reachable               | Open 3478; check coturn logs   |
| Container flapping                     | Port conflict on host network    | Check `ss -tulnp` on Atlas     |
| Caddy TLS fails to provision           | ACME rate-limited or DNS 404     | Check `caddy.log`; try DNS-01  |

## Security notes

- `livekit-keys.yaml` and `.env` are secrets. They are gitignored;
  never commit. Rotate annually and on any suspected compromise.
- The Keeper-minted participant JWTs are short-lived (default 6h).
  Long-lived tokens are a security footgun.
- The Caddy site block enforces HSTS and disables host headers
  from being forwarded beyond LiveKit.
- LiveKit recordings (when enabled in Sprint 7) land under
  `/archive/halyard/recordings/<session_id>/` with 0640 perms.
  Transcode or encrypt before moving off-host.

## See also

- [`docs/HALYARD_TABLE.md`](../../../docs/HALYARD_TABLE.md) §4.1
  — component description.
- [`docs/HALYARD_SPRINT_PLAN.md`](../../../docs/HALYARD_SPRINT_PLAN.md)
  §Sprint-1 — acceptance criteria.
- [LiveKit self-hosting docs](https://docs.livekit.io/home/self-hosting/deployment/).
- [coturn project](https://github.com/coturn/coturn).
