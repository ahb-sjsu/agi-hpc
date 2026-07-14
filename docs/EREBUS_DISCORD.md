# Erebus's Discord faculty

Lets Erebus host a channel (e.g. **#erebus-agi** on the ErisML/DEME server) so people can
talk to it and it can learn. Reactive, gated, honest, auditable.

## How it works

```
Discord message  →  filter (bot? right channel? addressed?)  →  rate-limit  →
  Erebus cognition (/api/erebus/chat: Ego / vMOE / RAG)  →  length-cap  →
  first-contact AI disclosure  →  DEME output gate (check_output)  →  audit  →
  post  |  draft (log only)  |  suppress (stay silent)
```

- **Reactive only (v1):** responds when addressed; never initiates.
- **Reuses cognition:** a new mouth on the existing chat brain — no separate model.
- **Every reply is DEME-gated** before posting. **Fail-safe:** no gate ⇒ silent.
- **Honest:** discloses it is an AI on first contact + an intro message on connect.
- **Bounded:** channel-scoped, per-user cooldown + per-channel window, length cap,
  full audit (`/archive/erebus/discord_audit.jsonl`), learning corpus
  (`/archive/erebus/discord_conversations.jsonl`), sentinel kill-switch
  (`/archive/neurogolf/.discord_disabled`).

## Layout

| File | Role |
|---|---|
| `agi/discord/handler.py` | message-handling core (no discord.py — unit-tested) |
| `agi/discord/safety.py` | DEME output gate (`check_output`); NullOutputGate = fail-safe deny |
| `agi/discord/ratelimit.py` | per-user cooldown + per-channel window |
| `agi/discord/cognition.py` | adapter to `/api/erebus/chat` (stdlib urllib) |
| `agi/discord/audit.py` | audit + conversation JSONL |
| `agi/discord/config.py` | env config |
| `agi/discord/bot.py` | discord.py wiring + `main()` |
| `deploy/systemd/atlas-erebus-discord.service` | the unit |

## Setup (operator)

1. Create a bot at <https://discord.com/developers/applications> → Bot → **enable the
   MESSAGE CONTENT intent**. Invite it to the server with **Send Messages** + **Read Message
   History** in #erebus-agi.
2. Put the secrets in `/home/claude/.erebus_discord.env` on Atlas (never committed):
   ```
   EREBUS_DISCORD_TOKEN=<bot token>
   EREBUS_DISCORD_CHANNELS=<channel id of #erebus-agi>
   # optional: EREBUS_DISCORD_MODE=draft   # log replies without posting, for first runs
   ```
3. `pip install -U discord.py` in `/home/claude/env` if not present.
4. Install the unit (`sudo cp deploy/systemd/... /etc/systemd/system/ && daemon-reload`),
   then `systemctl enable --now atlas-erebus-discord`.

## Controls

- **Pause responding:** `touch /archive/neurogolf/.discord_disabled` (stays connected,
  stops replying). Remove to resume.
- **Draft mode:** set `EREBUS_DISCORD_MODE=draft` — replies are gated + logged to the
  conversation file but not posted, so you can review the first exchanges.
- **Stop entirely:** `sudo systemctl stop atlas-erebus-discord`.

## Config (env)

`EREBUS_DISCORD_TOKEN`, `EREBUS_DISCORD_CHANNELS` (csv of channel ids), `EREBUS_DISCORD_MODE`
(`autonomous`|`draft`), `EREBUS_DISCORD_REQUIRE_MENTION` (`0`/`1`; forced on when no channel
is set), `EREBUS_DISCORD_MAXLEN`, `EREBUS_DISCORD_USER_COOLDOWN`, `EREBUS_DISCORD_CHAN_MAX`,
`EREBUS_DISCORD_CHAN_WINDOW`, `EREBUS_COGNITION_URL`.
