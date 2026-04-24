# atlas-caddy — Halyard Table public front

Caddy configuration that serves
**`halyard.atlas-sjsu.duckdns.org`** as the single public origin
for the whole Halyard Table runtime.

Caddy terminates TLS (ACME via Let's Encrypt, auto-provisioned on
first request to the hostname) and reverse-proxies to each
backend on loopback:

| Path        | Upstream         | Service           |
|-------------|------------------|-------------------|
| `/rtc/*`    | `127.0.0.1:7880` | LiveKit SFU (signaling, WebSocket) |
| `/twirp/*`  | `127.0.0.1:7880` | LiveKit Twirp admin API            |
| `/keeper/*` | `127.0.0.1:8091` | halyard-keeper (path prefix stripped) |
| `/state/*`  | `127.0.0.1:8090` | halyard-state (path prefix stripped) |
| `/*`        | `127.0.0.1:3000` | halyard-web (Next.js bundle)       |

## Install

```bash
# Copy the Caddyfile into the system location. Caddy is a
# system-level service; claude has sudo per atlas-operations.md.
sudo cp ~/agi-hpc-halyard/infra/local/halyard-caddy/Caddyfile \
        /etc/caddy/Caddyfile

# Validate the config before reloading.
sudo caddy validate --config /etc/caddy/Caddyfile

# First-time bring-up:
sudo systemctl enable --now caddy

# On subsequent edits, reload — no dropped connections:
sudo systemctl reload caddy

# Logs.
journalctl -u caddy -f
```

## First-request TLS provisioning

Caddy waits until the first TLS handshake against the hostname to
request a certificate. So the first external request is slower
(a few seconds); subsequent requests hit the cached cert. If you
see a TLS error on the first visit, wait 10 seconds and retry.

Let's Encrypt requires port 80 reachable from the public internet
for the HTTP-01 challenge. Confirm with:

```bash
curl -sS http://halyard.atlas-sjsu.duckdns.org/
# Should redirect to https:// (Caddy auto-enforces).
```

If your router is NATing and :80 isn't forwarded to Atlas, switch
to DNS-01 challenge in the Caddyfile (not covered here — adds a
duckdns API token dependency).

## Operating

```bash
systemctl status caddy
systemctl reload caddy        # graceful reload on config change
systemctl restart caddy       # harder — drops connections
```

## Troubleshooting

| Symptom                      | Likely cause               | Fix                          |
|------------------------------|----------------------------|------------------------------|
| 502 bad gateway              | upstream service down      | `systemctl --user status halyard-*` |
| 504 timeout                  | LiveKit WS not upgrading   | check `transport http { versions 1.1 }` block |
| ACME challenge fails         | :80 not reachable publicly | switch to DNS-01 or open :80 |
| Everything 404s at /         | halyard-web not bound :3000| check `docker ps` on Atlas   |
| `/state/*` 404s              | halyard-state not bound    | `systemctl --user status halyard-state` |
| `/keeper/*` 401s             | keeper auth enabled        | supply `KEEPER_USERNAME` / `PASSWORD` |

## Security

- HSTS enforced for 1 year with `includeSubDomains`.
- `X-Content-Type-Options: nosniff`.
- `Referrer-Policy: strict-origin-when-cross-origin`.
- Server header stripped.
- CSP is intentionally not set yet — Sprint 8 tightens once the
  client's resource graph stabilizes.
