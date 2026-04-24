# atlas-caddy — Halyard Table public front

Configuration for `halyard.atlas-sjsu.duckdns.org` on the **existing
atlas-caddy** instance. The Halyard Table site block is shipped as
an importable Caddyfile fragment (`halyard.caddy.conf`) so it
coexists with the other hostnames atlas-caddy serves and can be
removed with a single-line edit.

atlas-caddy terminates TLS (ACME via Let's Encrypt) and reverse-
proxies to each backend on loopback:

| Path        | Upstream         | Service           |
|-------------|------------------|-------------------|
| `/rtc/*`    | `127.0.0.1:7880` | LiveKit SFU (signaling, WebSocket) |
| `/twirp/*`  | `127.0.0.1:7880` | LiveKit Twirp admin API            |
| `/keeper/*` | `127.0.0.1:8091` | halyard-keeper (path prefix stripped) |
| `/state/*`  | `127.0.0.1:8090` | halyard-state (path prefix stripped) |
| `/*`        | `127.0.0.1:3030` | halyard-web (Next.js bundle; :3000 is taken by a security-radar target) |

## Install

```bash
# On Atlas, as user 'claude' (who has sudo).
# Copy the site-block fragment somewhere atlas-caddy can import.
cp ~/agi-hpc-halyard/infra/local/halyard-caddy/halyard.caddy.conf \
   /home/claude/halyard.caddy.conf

# Add one line to the top of /home/claude/Caddyfile (outside any
# existing site block):
#     import /home/claude/halyard.caddy.conf

# Reload the running atlas-caddy (graceful, no dropped connections):
sudo systemctl reload atlas-caddy

# Logs — atlas-caddy.service, NOT the stock caddy.service.
journalctl -u atlas-caddy -f
```

## Removal

```bash
# Edit /home/claude/Caddyfile, remove the import line, then:
sudo systemctl reload atlas-caddy
rm /home/claude/halyard.caddy.conf
```

## First-request TLS provisioning

Caddy defers ACME until the first TLS handshake against the new
hostname. The first external request to
`https://halyard.atlas-sjsu.duckdns.org/` is therefore slightly
slower (a few seconds); subsequent requests hit the cached cert.
If you see a TLS error on the first visit, wait 10 seconds and
retry — Caddy is usually finishing the challenge.

Let's Encrypt requires port 80 reachable from the public internet
for the HTTP-01 challenge. The existing atlas-caddy already has
this working for the base `atlas-sjsu.duckdns.org` host, so the
halyard subdomain should Just Work.

## Troubleshooting

| Symptom                      | Likely cause                  | Fix                          |
|------------------------------|-------------------------------|------------------------------|
| 502 bad gateway              | upstream service down         | `systemctl --user status halyard-*` |
| 504 timeout                  | LiveKit WS not upgrading      | check `transport http { versions 1.1 }` block |
| ACME challenge fails         | :80 not reachable publicly    | switch to DNS-01 (needs duckdns API token) |
| 404 at `/`                   | halyard-web not bound `:3030` | `docker ps` on Atlas         |
| `/state/*` 404s              | halyard-state not bound       | `systemctl --user status halyard-state` |
| `/keeper/*` 401s             | keeper auth enabled           | supply `KEEPER_USERNAME`/`PASSWORD` |

Do **not** start the stock `caddy.service` — it collides with
atlas-caddy's admin port (`127.0.0.1:2019`). Only atlas-caddy
should run.

## Security

- HSTS enforced for 1 year with `includeSubDomains`.
- `X-Content-Type-Options: nosniff`.
- `Referrer-Policy: strict-origin-when-cross-origin`.
- `Server` header stripped.
- CSP is intentionally not set yet — Sprint 8 tightens once the
  client's resource graph stabilizes.
