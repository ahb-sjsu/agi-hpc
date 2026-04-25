---
name: halyard
description: Manage the Halyard Table game (UI, backend services, campaign content). Use when the user asks to deploy / restart halyard services, edit the 4x3 grid panels, add NPCs / maps / artifacts / announcements, update the wiki RAG corpus, rebuild the bible, change persona prompts, or smoke-test the AI turn endpoints.
user-invocable: true
---

# /halyard — Halyard Table operations

The Halyard Table is the live-play web client + backend for the Beyond
the Heliopause CoC campaign. This skill is the operator playbook for
everything that lives under `web/halyard-table/`, `src/agi/halyard/`,
`wiki/halyard/`, and `infra/local/halyard-*`.

The user invoked you with arguments: `$ARGUMENTS`.

If there are no arguments, show **status** and a short menu of
sub-commands. Otherwise dispatch on the first word.

---

## Pre-flight (do this before any deploy/restart)

Atlas operations rules in `.claude/rules/atlas-operations.md` are
**mandatory** — they have the SSH password, thermal caps, and the
"never reboot without asking" rule. Read them first if you haven't
this session.

The Atlas checkout is at `/home/claude/agi-hpc-halyard` (NOT
`~/src/agi-hpc` — that's a different copy). The python venv lives at
`/home/claude/env`. Services are user-level systemd units.

---

## Authoritative locations (single source of truth)

| What | Where |
|---|---|
| Atlas repo checkout | `/home/claude/agi-hpc-halyard` |
| Live web bundle | container `halyard-web` (image `halyard-web:local`) |
| Web source | `web/halyard-table/` |
| Web Dockerfile | `deploy/docker/halyard-web/Dockerfile` |
| Web compose file | `infra/local/halyard-web/docker-compose.yml` |
| Keeper backend | systemd `halyard-keeper.service` (user) |
| Keeper source | `src/agi/halyard/keeper/` |
| State backend | systemd `halyard-state.service` (user) |
| State source | `src/agi/halyard/state/` |
| RAG corpus | `wiki/halyard/*.md` (frontmatter-tagged) |
| Built bible | `/archive/halyard/bible/halyard_bible.json` |
| Bible builder | `scripts/halyard/build_bible.py` |
| Persona prompts | `src/agi/halyard/keeper/campaign_facts.py` |
| Forbidden phrases | `src/agi/halyard/keeper/llm.py` (`_FORBIDDEN_PHRASES`) |
| Atlas-Caddy config | `/home/claude/Caddyfile` (imports site files) |
| Public origin | `https://halyard.atlas-sjsu.duckdns.org` |
| Internal keeper | `http://127.0.0.1:8091` |
| Internal state | `http://127.0.0.1:8090` |
| Internal web | `http://127.0.0.1:3030` |

**Tailscale (`100.68.134.21`) is OOB admin only — never a service
channel.** All halyard services bind 127.0.0.1 and sit behind
atlas-caddy on the duckdns name. Do not change that.

---

## Sub-commands

### `status` (or no args)

Quick health check. Run all of these in parallel — they're independent.

1. SSH (paramiko) to Atlas. Run:
   - `systemctl --user is-active halyard-keeper.service halyard-state.service halyard-web.service`
   - `docker ps --filter name=halyard-web --format '{{.Names}}\t{{.Status}}'`
   - `git -C /home/claude/agi-hpc-halyard log --oneline -3`
2. Public reachability:
   - `curl -sS -m 5 -o /dev/null -w '%{http_code}' https://halyard.atlas-sjsu.duckdns.org/`
   - Smoke `POST /keeper/api/ai/sigma4/turn` with `{"text":"location report","session_id":"skill-status"}` — expect `source: "vmoe-cascade"` and a non-empty `text`.

Report tersely: services up/down, last 3 commits, public 200/non-200,
SIGMA-4 source name. Then list available sub-commands.

### `deploy` — push local changes to Atlas

The full pipeline. Default behavior:

1. Ensure the working tree is clean enough — show `git status -sb` and
   ask before staging anything the user didn't mention.
2. Push the current branch: `git push origin <branch>`.
3. SSH to Atlas. `cd /home/claude/agi-hpc-halyard && git pull --ff-only`.
4. If the diff touched **only** `src/agi/halyard/keeper/**` or
   `src/agi/halyard/state/**` or `src/agi/primer/**` →
   `systemctl --user restart halyard-keeper.service` (and/or
   `halyard-state.service`).
5. If the diff touched **any** `web/halyard-table/**` →
   `cd /home/claude/agi-hpc-halyard/infra/local/halyard-web &&
   docker compose build halyard-web && docker compose up --detach`.
   The build takes ~40s. Watch for `Failed to compile` lines.
6. If `wiki/halyard/**` changed → rebuild the bible (`/halyard wiki rebuild`).
7. Smoke-test (same as `status`). Report.

Common build error: **strict TS unused-locals**. If `npm run build`
fails with `'X' is declared but its value is never read`, that import
or local has to be deleted in source — not silenced — and re-deployed.
The user has been bitten by this several times.

### `restart [service]` — restart one service

Without an argument, list:
- `keeper` → `halyard-keeper.service`
- `state` → `halyard-state.service`
- `web` → `halyard-web.service`
- `caddy` → `atlas-caddy.service` (system unit, needs sudo)

`/halyard restart keeper` runs `systemctl --user restart
halyard-keeper.service` then waits 3s and tails the journal:
`journalctl --user -u halyard-keeper.service -n 20 --no-pager`.

**Never restart caddy without asking** — atlas-caddy fronts other
sites (atlas-chat, etc.) and a flap is visible to other users. If a
caddy reload is genuinely needed, prefer `caddy reload --config
/home/claude/Caddyfile` over a service restart.

### `wiki` — manage RAG corpus

Sub-actions:

- `wiki list` — show every `.md` under `wiki/halyard/` with its
  topic, artemis visibility, sigma4 visibility (parsed from frontmatter).
- `wiki add <topic>/<slug>` — scaffold a new entry from the README
  template. Topic must be one of: `setting`, `faction`, `ship`, `crew`,
  `location`, `tech`, `mission`, `mythos`. Open the file for the user
  to fill in.
- `wiki rebuild` — run `python scripts/halyard/build_bible.py` ON
  ATLAS, not locally. The bible path `/archive/halyard/bible/...` is
  Atlas-only. After rebuild: `systemctl --user restart
  halyard-keeper.service` so the keeper drops its `_load_json_bible`
  lru_cache.

Frontmatter schema (every entry must have these):
```yaml
---
id:       <topic>/<slug>
title:    Human title
artemis:  known | unknown | forbidden
sigma4:   known | unknown | forbidden
topic:    <one of the eight>
tags:     [optional, list]
---
```

`mythos/` directory defaults to `forbidden` for both AIs unless
explicitly overridden. Forbidden entries are NOT shipped as RAG
chunks; their titles are added to the keeper's forbidden-phrase ban
list (validator's forbidden_check rejects replies that contain them).

### `npc <add|edit|list>` — manage NPC roster shown in StatusPanel

The roster lives in two places that must stay in sync:
1. `web/halyard-table/components/StatusPanel.tsx` — the `NPCS` array
   (what players see in the "scene" tab).
2. `wiki/halyard/crew/<slug>.md` — RAG context the AIs use.

`/halyard npc add "Capt. Marsh" "Commanding officer"` →
- Append to `NPCS` in StatusPanel.tsx.
- If a wiki entry doesn't already exist, scaffold one at
  `wiki/halyard/crew/<slug>.md` with empty body — let the user fill in.
- Reminder: rebuilding the web bundle is required for the NPC list
  change to be visible (`/halyard deploy`).

`/halyard npc list` — show the StatusPanel array contents.

### `map <add|list>` — manage MapPanel floorplans

Maps live in `web/halyard-table/components/MapPanel.tsx` (`MAPS`
array) plus an optional image asset under
`web/halyard-table/public/maps/<id>.png`.

`/halyard map add halyard-medbay "MKS Halyard — Medbay (Deck 3)"` →
- Append to MAPS array with `src: "/maps/halyard-medbay.png"`.
- If the user has a PNG ready, ask for the path to copy. Otherwise
  leave `src: null` so the placeholder renders.

### `announce <text>` — push an announcement

In the current iteration the `ANNOUNCEMENTS` array in StatusPanel.tsx
is a static client-side list (no backend feed). To broadcast:

- Append `{ ts: <ISO8601 utc>, text: "<text>" }` to the array.
- `/halyard deploy` to ship it.

When the GM-console/state-feed wiring lands (later sprint), this
sub-command will instead POST to `halyard-state` and the client will
re-render without a rebuild. Update this skill at that point.

### `artifact <add>` — add a downloadable handout

Targets the `ARTIFACTS` array in StatusPanel.tsx for now. Each entry
has `title`, `url`, `kind: "pdf"|"image"|"audio"`, optional `bytes`.

For the URL: drop the file on Atlas at
`/archive/halyard/artifacts/<filename>` and serve via Caddy's
`/artifacts/*` route (already configured to proxy halyard-state's
static dir; verify by reading `infra/local/halyard-caddy/halyard.caddy.conf`).

### `prompt <which>` — view / edit a persona prompt

`which` is `artemis`, `sigma4`, or `shared`. Opens the corresponding
chunk of `src/agi/halyard/keeper/campaign_facts.py`:
- `ARTEMIS_ONLY` — handheld scope, ARTEMIS-specific constraints.
- `SIGMA_ONLY` — ship-mind scope, SIGMA-4-specific constraints.
- `SHARED_CONTEXT` — facts both AIs know.
- `system_prompt_for(which)` — assembles the final prompt.

Reminder after editing: `/halyard restart keeper` to pick up changes.

### `forbid <add|list> [phrase]` — manage validator's forbidden list

The keeper's ErisML validator rejects any reply containing one of
these phrases. Source of truth: `_FORBIDDEN_PHRASES` tuple in
`src/agi/halyard/keeper/llm.py`. Adding a phrase here also extends
the runtime list merged with the bible's `forbidden_phrases`.

Per the campaign canon: Chamber, Ostland, Meridian, A-7, Persephone,
Mi-go, Elder Thing, Yithian, Starry Wisdom, etc. are forbidden. New
phrases that come up in play (e.g. a place name a player invents that
the AIs shouldn't repeat) belong here.

### `smoke <which>` — hit one AI endpoint with a real request

`/halyard smoke sigma4` → curl `POST /api/ai/sigma4/turn` directly on
the keeper (loopback, no Caddy in the path) and report:
- HTTP status
- `source` field (`vmoe-cascade` for SIGMA-4, `handle_turn` for
  ARTEMIS, `stub-fallback` if NRP creds missing or all experts failed)
- `expert` field (which model answered — `gpt-oss` is the priority-10
  default; if you see `qwen3` or others it means the cascade fell
  through, often because gpt-oss returned no content)
- `proof_hash` (must be non-null for ARTEMIS, null for SIGMA-4)
- First 120 chars of `text`

If `source` is `stub-fallback` and you weren't expecting it, check
`journalctl --user -u halyard-keeper.service -n 50` for `NRP call
failed`, `cascade raised`, or `KEEPER_USERNAME/PASSWORD unset` lines
(the last one is harmless in dev — services are loopback-bound).

### `logs [service] [n]` — tail a service journal

Default service `keeper`, default `n=50`. Equivalent to
`journalctl --user -u halyard-<service>.service -n <n> --no-pager`.

### `session <id>` — show a session's state

Reads from `/archive/halyard/sessions/<id>.jsonl` (ARTEMIS append-only
log) and `/archive/halyard/proofs/{artemis,sigma4}/<id>.chain`
(DecisionProof chains). Reports turn count, last 5 turns, last proof
hash. The chain is hash-linked — broken hashes mean a session was
edited by hand, which should never happen in dev.

---

## Things to NOT do (learned the hard way)

- **Don't** rebuild the web container without checking that
  AiChatPanel, MediaControls, etc. were last modified BEFORE the
  current image build time. We've shipped stale bundles twice when
  the source had been edited but `docker compose up` reused the
  cached image.
- **Don't** point `LIVEKIT_URL` (in `halyard-keeper.service`) at the
  loopback. The token's `url` field is what the *browser* connects
  to; it must be `wss://halyard.atlas-sjsu.duckdns.org`.
- **Don't** add a new `process.env[name]` dynamic access in
  `lib/env.ts`. Next.js only substitutes literal `process.env.NEXT_PUBLIC_FOO`
  references at build time — dynamic forms silently fall through to
  the localhost defaults in the browser.
- **Don't** import unused symbols. `npm run build` runs strict TS
  with `noUnusedLocals` — every dead import fails the deploy.
- **Don't** use Type-1 (`Type=oneshot`) restarts when the service is
  `Type=exec`. `halyard-web.service` is oneshot wrapping
  docker-compose; `halyard-keeper.service` is `Type=exec` directly.
  Their restart semantics differ.

---

## When to update this skill

If you discover a new failure mode, a new convention, or a new path —
add a row to the location table or a bullet to "things to NOT do".
Skills are durable memory; the conversation isn't. Keep it under 250
lines so it stays readable; if it grows past that, factor a section
into `wiki/halyard/operator/` and link out.
