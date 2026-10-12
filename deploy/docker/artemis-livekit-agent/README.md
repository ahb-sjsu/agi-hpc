# ARTEMIS LiveKit Agent — container build

Lightweight Python container that runs `python -m
agi.primer.artemis.livekit_agent`. Joins a LiveKit room as the
`ARTEMIS` participant, runs streaming Whisper ASR, bridges to
Atlas's NATS bus.

Superseded the Zoom-Meeting-SDK approach on 2026-04-22. See
`docs/ARTEMIS.md` §11 Phase 3 and §13 decision #6 for rationale.

## Images (built by CI)

Production images are built and pushed by
[`.github/workflows/build-images.yaml`](../../../.github/workflows/build-images.yaml)
to `ghcr.io/ahb-sjsu/artemis-livekit-agent`. Pull a specific build:

```bash
# Pin to an immutable git SHA (recommended for production manifests)
docker pull ghcr.io/ahb-sjsu/artemis-livekit-agent:sha-<7char>

# Or a semver release
docker pull ghcr.io/ahb-sjsu/artemis-livekit-agent:v0.1.0
```

See [`docs/HPC_DEPLOYMENT.md`](../../../docs/HPC_DEPLOYMENT.md#container-image-publishing)
for the full tag scheme and which tag to use in which environment.

### Local build (dev only)

If you need to build locally for a smoke test before pushing, build is
straightforward CUDA + Python — no C++ SDK, no xvfb, no display:

```bash
# From repo root:
docker buildx build \
  --platform linux/amd64 \
  -t artemis-livekit-agent:dev \
  -f deploy/docker/artemis-livekit-agent/Dockerfile \
  .
```

Do not push manually-built images to GHCR — let CI own the published
tags so the registry stays in sync with git.

## Run (local smoke, against Atlas LiveKit)

```bash
docker run --rm --gpus all \
  -e ARTEMIS_SESSION_ID=local-smoke \
  -e LIVEKIT_URL=ws://host.docker.internal:7880 \
  -e LIVEKIT_API_KEY="$LK_KEY" \
  -e LIVEKIT_API_SECRET="$LK_SECRET" \
  -e NATS_URL=nats://host.docker.internal:4222 \
  ghcr.io/ahb-sjsu/artemis-livekit-agent:<tag>
```

Expected: agent connects to the LiveKit room, appears in the room's
participant list as `ARTEMIS` (no audio track published), logs
`"artemis agent online"`.

## Deployment options

| Option | Systemd unit / manifest | When to use |
|---|---|---|
| Atlas systemd | `deploy/systemd/atlas-artemis-agent.service` | Default — Atlas has spare GPU, NATS is local, latency is best |
| NRP K8s Job | `deploy/k8s/artemis-livekit-agent/job.yaml` | Atlas is busy with ARC bursts or training; offload to an A10 |
| Local dev | `docker compose -f deploy/compose/livekit-atlas.yml up` | Development and smoke tests; brings up LiveKit SFU + agent together |

## External dependencies

Resolved at build time by pip — no SDK gating:

- `livekit-agents` (>=0.9) — agent runtime
- `livekit-plugins-openai` or `livekit-plugins-faster-whisper` — STT
- `faster-whisper` — if we run Whisper locally (recommended)
- `nats-py` — Atlas fabric client
- `PyJWT` — room token minting

No Zoom developer account, no SDK download, no app registration, no
OAuth scopes.

## Gotchas captured during build

*None yet — populate as real-world findings accumulate.*
