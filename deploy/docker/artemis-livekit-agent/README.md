# ARTEMIS LiveKit Agent — container build

Lightweight Python container that runs `python -m
agi.primer.artemis.livekit_agent`. Joins a LiveKit room as the
`ARTEMIS` participant, runs streaming Whisper ASR, bridges to
Atlas's NATS bus.

Superseded the Zoom-Meeting-SDK approach on 2026-04-22. See
`docs/ARTEMIS.md` §11 Phase 3 and §13 decision #6 for rationale.

## Build

```bash
# From repo root:
docker buildx build \
  --platform linux/amd64 \
  -t ghcr.io/ahb-sjsu/artemis-livekit-agent:$(git describe --tags --always) \
  -f deploy/docker/artemis-livekit-agent/Dockerfile \
  .

docker push ghcr.io/ahb-sjsu/artemis-livekit-agent:<tag>
```

Build is straightforward CUDA + Python — no C++ SDK, no xvfb, no
display. One pip install and we're done.

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
