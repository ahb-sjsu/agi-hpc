# halyard-state — deployment artifacts

Runtime packaging for the Halyard Table character-sheet service.
Source lives at `src/agi/halyard/state/`. See
[`../../docs/HALYARD_TABLE.md`](../../../docs/HALYARD_TABLE.md) §4.4
for the component description and
[`../../docs/HALYARD_SPRINT_PLAN.md`](../../../docs/HALYARD_SPRINT_PLAN.md)
§Sprint-3 for the acceptance contract.

## Files

- `systemd/halyard-state.service` — user-level systemd unit.
- `env.example` — env template (copy to `.env`, gitignored).

## Deploy on Atlas

```bash
# On Atlas, logged in as 'claude':
cd ~/src/agi-hpc

# Ensure the Halyard branch is checked out and up-to-date.
git fetch origin
git checkout halyard/sprint-3
git pull

# Install the package into the existing venv. The state module
# has no new runtime deps beyond jsonschema (already present).
/home/claude/env/bin/pip install -e . --quiet

# Install the systemd user unit.
mkdir -p ~/.config/systemd/user
ln -sfn ~/src/agi-hpc/infra/local/halyard-state/systemd/halyard-state.service \
        ~/.config/systemd/user/halyard-state.service
systemctl --user daemon-reload

# One-time: enable linger so the service runs without a login.
sudo loginctl enable-linger claude

# Start the service.
systemctl --user enable --now halyard-state.service

# Verify.
systemctl --user status halyard-state --no-pager
curl -sS http://127.0.0.1:8090/healthz | jq .
journalctl --user -u halyard-state -n 30 --no-pager
```

## Environment

The service reads (in priority order): CLI flags → env vars →
built-in defaults.

| Var                     | Default                   | Purpose                                  |
|-------------------------|---------------------------|------------------------------------------|
| `HALYARD_STATE_HOST`    | `127.0.0.1`               | Bind address — **loopback only by default**. See note below. |
| `HALYARD_STATE_PORT`    | `8090`                    | Bind port                                 |
| `HALYARD_ARCHIVE_ROOT`  | `/archive/halyard`        | Sheet file root                           |
| `NATS_URL`              | (unset → no NATS bridge)  | e.g. `nats://127.0.0.1:4222`              |
| `LOG_LEVEL`             | `info`                    | Standard Python logging levels            |

With `NATS_URL` unset the service runs REST+WS only, which is the
recommended posture for initial deployment — it lets you smoke-
test the HTTP surface before the NATS fabric is wired in.

### A note on binding

The service binds to `127.0.0.1` by default. It is designed to sit
behind `atlas-caddy`, which provides TLS and public fronting.
Tailscale is the operator's **out-of-band admin path**, not a
service channel — binding on `0.0.0.0` would expose the service on
the tailnet and defeat that isolation.

If you want to reach the service from outside Atlas, the right
answer is to add a Caddy site block that reverse-proxies
`halyard-state.atlas-sjsu.duckdns.org` → `127.0.0.1:8090`. Do not
rebind the service itself.

## Smoke test

On Atlas itself (loopback — what the service binds to):

```bash
ssh claude@100.68.134.21
curl -sS http://127.0.0.1:8090/healthz | jq .
# → {"ok": true, "service": "halyard-state"}
curl -sS http://127.0.0.1:8090/api/sheets/test-session | jq .
# → {"session_id": "test-session", "pc_ids": []}
```

Or run the full end-to-end smoke from a workstation after Caddy
is fronting the service — tunneling over SSH is fine for dev
until then:

```bash
ssh -L 8090:127.0.0.1:8090 claude@100.68.134.21
# In another shell, against localhost:
python scripts/halyard/atlas_state_smoke.py --base http://127.0.0.1:8090
```

## Operating

```bash
systemctl --user status halyard-state
systemctl --user restart halyard-state
journalctl --user -u halyard-state -f

# Where the data lives (or will live):
ls /archive/halyard/sheets/
```

## Teardown

```bash
systemctl --user disable --now halyard-state.service
rm ~/.config/systemd/user/halyard-state.service
systemctl --user daemon-reload
# Session archives under /archive/halyard/ are preserved.
```
