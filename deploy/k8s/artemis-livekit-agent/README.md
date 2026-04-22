# ARTEMIS LiveKit Agent — K8s manifests

Optional NRP deployment of the LiveKit agent. Primary deployment is
the systemd unit on Atlas (`deploy/systemd/atlas-artemis-agent.service`).
Use the K8s Job manifest to offload agent compute to an NRP A10 when
Atlas is busy with ARC bursts or training windows.

## Burst accounting

Consumes one of the 4 heavy-GPU burst slots in the `ssu-atlas-ai`
namespace while a session is active. Coordinate with ongoing ARC
vision bursts before launching.

## Submit

```bash
export ARTEMIS_SESSION_ID=halyard-2026-05-04-s01
export ARTEMIS_IMAGE_TAG=2026-05-01

kubectl -n ssu-atlas-ai apply -f configmap.yaml
envsubst < job.yaml | kubectl -n ssu-atlas-ai apply -f -

# Watch
kubectl -n ssu-atlas-ai get pods -l artemis.session=$ARTEMIS_SESSION_ID -w
kubectl -n ssu-atlas-ai logs -f job/artemis-livekit-agent-$ARTEMIS_SESSION_ID
```

## Credentials

The `artemis-livekit-credentials` secret must exist:

```bash
kubectl -n ssu-atlas-ai create secret generic artemis-livekit-credentials \
  --from-literal=url=wss://atlas-sjsu.duckdns.org/livekit \
  --from-literal=api_key=<LIVEKIT_API_KEY> \
  --from-literal=api_secret=<LIVEKIT_API_SECRET>
```
