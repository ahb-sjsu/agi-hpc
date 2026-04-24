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
| `HALYARD_STATE_HOST`    | `0.0.0.0`                 | Bind address                              |
| `HALYARD_STATE_PORT`    | `8090`                    | Bind port                                 |
| `HALYARD_ARCHIVE_ROOT`  | `/archive/halyard`        | Sheet file root                           |
| `NATS_URL`              | (unset → no NATS bridge)  | e.g. `nats://127.0.0.1:4222`              |
| `LOG_LEVEL`             | `info`                    | Standard Python logging levels            |

With `NATS_URL` unset the service runs REST+WS only, which is the
recommended posture for initial deployment — it lets you smoke-
test the HTTP surface before the NATS fabric is wired in.

## Smoke test

From any host on the tailnet (including the Atlas box itself):

```bash
# 1. Health.
curl -sS http://100.68.134.21:8090/healthz | jq .
# → {"ok": true, "service": "halyard-state"}

# 2. Empty session lists zero PCs.
curl -sS http://100.68.134.21:8090/api/sheets/test-session | jq .
# → {"session_id": "test-session", "pc_ids": []}

# 3. Create a sheet (substitute a real, validated sheet body).
curl -sS -X POST \
     -H 'content-type: application/json' \
     -d @examples/sheet.cross.json \
     http://100.68.134.21:8090/api/sheets/test-session/cross \
  | jq .identity.name
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
