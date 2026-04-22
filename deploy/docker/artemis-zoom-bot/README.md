# ARTEMIS Zoom Bot — container build

**Status: Phase 3 scaffolding (stubs).** The Dockerfile here describes
the shape; the build is not yet wired up.

## What this container does

See [`../../../docs/ARTEMIS.md`](../../../docs/ARTEMIS.md) §11 Phase 3
for the plan. Short version: join a Zoom meeting as a named
participant, stream audio through Whisper, publish transcripts on
NATS, post Atlas-generated replies back to Zoom chat.

## Build

```bash
# From repo root:
docker buildx build \
  --platform linux/amd64 \
  -t ghcr.io/ahb-sjsu/artemis-zoom-bot:$(git describe --tags --always) \
  -f deploy/docker/artemis-zoom-bot/Dockerfile \
  .

# Push:
docker push ghcr.io/ahb-sjsu/artemis-zoom-bot:<tag>
```

## Components inside the image

| Component | Purpose |
|---|---|
| Zoom Meeting SDK (Linux C++) | Joins meetings, pulls raw audio, posts chat |
| Whisper-large-v3 (faster-whisper) | Streaming ASR on A10 GPU |
| nats.go client | Publishes `agi.rh.artemis.heard`, subscribes `.say` |
| Python entrypoint | Glues the above; no reasoning logic (that's Atlas) |

## External dependencies (to procure during build)

1. **Zoom Meeting SDK credentials** — SDK Key + Secret issued to the
   `agi-hpc@gmail.com` Zoom account. Mounted via
   `artemis-zoom-credentials` secret. Reference:
   <https://developers.zoom.us/docs/meeting-sdk/linux/>
2. **Whisper weights** — pre-staged on the `erebus-ego-models`
   CephFS PVC at `/models/whisper/large-v3/`. One-time copy from
   HuggingFace Hub on a worker node with internet.
3. **NATS user credentials** — issued via Atlas's leaf auth. Mounted
   from `atlas-nats-leaf-creds` secret.

## Gotchas captured during build (Phase 3 — placeholder)

*Record findings here as Phase 3 lands. Candidates:*

- *Zoom SDK on Linux requires a display; use `xvfb-run` in entrypoint.*
- *Zoom SDK is IP-restricted in some regions; verify NRP SDSC egress.*
- *Whisper streaming emits partial hypotheses; bot must flag
  `partial=true` on interim publishes to match the Primer's schema.*

## Test join (before first real session)

```bash
# Locally (with a scratch Zoom meeting):
docker run --rm --gpus all \
  -e ARTEMIS_SESSION_ID=local-test \
  -e ZOOM_MEETING_URL="<scratch meeting URL>" \
  -e ZOOM_SDK_KEY="<key>" \
  -e ZOOM_SDK_SECRET="<secret>" \
  -e NATS_URL="nats://host.docker.internal:4222" \
  ghcr.io/ahb-sjsu/artemis-zoom-bot:<tag>
```

Confirm: (a) "ARTEMIS" appears in the meeting's participant list,
(b) speaking into the meeting emits a `.heard` NATS message,
(c) publishing a `.say` message on NATS posts to Zoom chat.
