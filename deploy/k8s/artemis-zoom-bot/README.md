# ARTEMIS Zoom Bot — Kubernetes Job

**Status: Phase 3 scaffolding (stubs).** Do not apply in production
without completing the build + deploy work outlined in
[`../../../docs/ARTEMIS.md`](../../../docs/ARTEMIS.md) §11 Phase 3.

## What this directory is for

The ARTEMIS character is driven by `atlas-primer.service` on Atlas
(see `src/agi/primer/artemis/`). That Atlas-side brain talks to the
Zoom meeting through a **per-session Kubernetes Job** that runs the
bot container defined under `deploy/docker/artemis-zoom-bot/`.

One Job per game session. The Job:

1. Joins a Zoom meeting via the Meeting SDK as a named participant.
2. Streams per-speaker audio into Whisper-large-v3 (on-pod A10 GPU).
3. Publishes transcripts on NATS subject `agi.rh.artemis.heard` via
   the leaf bridge (`NATS_URL=nats://atlas-nats:4222`).
4. Subscribes to `agi.rh.artemis.say` and posts replies as chat
   messages under the "ARTEMIS" participant.
5. Exits when the meeting ends. Job-shape, not Deployment — this
   respects NRP's > 40 % GPU utilization policy for the short active
   window rather than holding a pod idle between sessions.

## Files

| File | Purpose |
|---|---|
| `job.yaml` | Per-session K8s Job template. Parameterize `session_id`, `zoom_meeting_url`. |
| `configmap.yaml` | Per-session config (system-prompt overrides, validator thresholds). |
| `README.md` | This file. |

## Burst-slot accounting

NRP limits `ssu-atlas-ai` to 4 heavy-GPU burst pods at a time. The
ARC vision burst uses up to 4 during a vision sweep; a running
ARTEMIS session consumes one slot exclusively. **Do not run an
ARTEMIS session during an ARC vision burst unless the burst has
been paused.** Operator procedure is in the Phase-4 runbook.

## Submit

```bash
# Apply configmap (one-time per session — carries session-specific
# settings like keeper_approval_required, validator tweaks).
kubectl -n ssu-atlas-ai apply -f configmap.yaml

# Launch the Job.
# ZOOM_MEETING_URL and ARTEMIS_SESSION_ID are substituted at apply-time.
envsubst < job.yaml | kubectl -n ssu-atlas-ai apply -f -

# Follow logs.
kubectl -n ssu-atlas-ai logs -f job/artemis-zoom-bot-$ARTEMIS_SESSION_ID
```

Alternative: submit via `nats-bursting` on Atlas (preferred once the
Keeper dashboard tile is wired up — see Phase 4):

```bash
nats --server nats://localhost:4222 pub burst.submit \
  '{"job_id":"artemis-<session_id>","descriptor":{"image":"...","env":{...}}}'
```

## Teardown

```bash
kubectl -n ssu-atlas-ai delete job artemis-zoom-bot-$ARTEMIS_SESSION_ID
```

Or let the Job exit on its own when the meeting ends; the bot
container is responsible for calling `complete()` when it detects
Zoom meeting end.

## Gotchas (to be filled in during Phase 3 build)

*None yet — file exists as scaffolding. Record real-world findings
here as Phase 3 lands.*
